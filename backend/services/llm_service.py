import os
import re
import json
import time
import asyncio
import traceback
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ── Key Rotation Pool ──────────────────────────────────────────────────────────
_raw_keys = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
]
API_KEYS = [k for k in _raw_keys if k]  # Filter out unset vars
_current_key_idx = 0

def _get_client() -> genai.Client:
    """Get the current active Gemini client."""
    return genai.Client(
        api_key=API_KEYS[_current_key_idx]
    )

def _rotate_key() -> bool:
    """Rotate to next available key index globally. Returns False if all keys are exhausted."""
    global _current_key_idx
    if _current_key_idx + 1 < len(API_KEYS):
        _current_key_idx += 1
        return True
    return False

# ──────────────────────────────────────────────────────────────────────────────

def _extract_retry_delay(error_msg: str) -> int:
    """Extract the retryDelay value in seconds from Google's 429 error message."""
    match = re.search(r"retryDelay.*?(\d+)s", error_msg)
    if match:
        return int(match.group(1)) + 2
    return 30

async def _call_with_retry(model_name: str, contents: str, json_mode: bool = False, max_retries: int = 4):
    """
    Call the Gemini API with automatic sequential key rotation.
    Tries each available key once.
    """
    global _current_key_idx
    
    # We try at most as many times as we have keys (up to max_retries)
    total_attempts = min(len(API_KEYS), max_retries)
    
    for attempt in range(total_attempts):
        try:
            # Create client with the current key index
            current_key = API_KEYS[_current_key_idx]
            print(f"🔑 Using API key #{_current_key_idx + 1} (Attempt {attempt + 1}/{total_attempts})")
            
            client = genai.Client(api_key=current_key)
            config = types.GenerateContentConfig(response_mime_type="application/json") if json_mode else None
            
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=config
            )
            
            return response.text
            
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️ Key #{_current_key_idx + 1} failed: {error_msg}")
            
            # If we have more keys to try, rotate and continue
            if attempt < total_attempts - 1:
                if _rotate_key():
                    print(f"🔄 Switched to Key #{_current_key_idx + 1} for next attempt...")
                    continue
            
            # If all else fails or no more keys
            raise Exception(f"All {total_attempts} Gemini API keys failed or exhausted. Last error: {error_msg}")


async def generate_literature_review(topic: str, papers: list = None, language: str = "English"):
    try:
        start_time = time.time()
        
        # Re-add paper context processing
        context_parts = []
        if papers:
            # Limit to top 6 papers for stability and truncate long abstracts
            for i, paper in enumerate(papers[:6], 1):
                raw_abstract = paper.abstract if paper.abstract else "No abstract available."
                # Truncate to 500 chars to avoid prompt bloat and parsing issues
                abstract = (raw_abstract[:500] + '...') if len(raw_abstract) > 500 else raw_abstract
                
                authors_str = ", ".join(paper.authors) if paper.authors else "Unknown Authors"
                year = paper.year if paper.year else "Unknown Year"
                context_parts.append(
                    f"Paper {i}:\nTitle: {paper.title}\nAuthors: {authors_str} ({year})\nAbstract: {abstract}\n"
                )
        
        papers_context = "\n".join(context_parts) if context_parts else "No specific papers found."

        prompt = f"""
        Generate a structured literature review on: {topic}

        Use the following provided research papers as your primary source material:
        {papers_context}

        IMPORTANT: You MUST write all the internal content (the strings/arrays inside the JSON) in the **{language}** language.
        HOWEVER, the JSON Keys MUST critically remain exactly as defined in English.
        You MUST respond entirely in valid raw JSON format matching this specific schema, with strictly NO markdown wrappers or codeblocks.

        {{
            "introduction": "Introductory paragraph here (in {language})",
            "key_themes": ["Theme 1", "Theme 2"],
            "comparative_analysis": "Markdown table comparing methodologies/findings here (in {language})",
            "research_gaps": ["Gap 1", "Gap 2"],
            "conclusion": "Final conclusion here",
            "key_takeaways": ["Takeaway 1", "Takeaway 2", "Takeaway 3", "Takeaway 4", "Takeaway 5"],
            "ai_idea": "A novel AI-generated idea based on these gaps",
            "confidence_level": "High", 
            "complexity_level": "Expert"
        }}
        """

        # Call with automatic retry on rate-limit errors
        raw_text = await _call_with_retry(
            model_name="gemini-2.0-flash",
            contents=prompt,
            json_mode=True
        )

        end_time = time.time()
        generation_time_seconds = round(end_time - start_time, 2)
        
        # Robust cleaning of JSON response
        clean_text = raw_text.strip().strip('`').removeprefix('json').strip()
        
        # Remove internal control characters that JSON doesn't like (like literal newlines in strings)
        # We replace them with spaces or escaped versions
        import re
        # This removes non-printable control characters except newline and tab
        clean_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', clean_text)
        
        try:
            # strict=False allows control characters like literal newlines in strings
            parsed_json = json.loads(clean_text, strict=False)
        except json.JSONDecodeError:
            # Final attempt: try to escape newlines within strings
            # Note: This is an expensive regex, only run if standard parsing fails
            repaired_text = re.sub(r'([^\\])\n', r'\1\\n', clean_text)
            parsed_json = json.loads(repaired_text, strict=False)
        
        return {
            "structured_review": parsed_json,
            "generation_time_seconds": generation_time_seconds
        }

    except Exception as e:
        error_msg = str(e)
        print("🔥 ERROR TRACEBACK ENCOUNTERED:")
        traceback.print_exc()
        
        if "429" in error_msg or "ResourceExhausted" in error_msg or "quota" in error_msg.lower():
            return {"error": "Your Gemini API Key has exceeded its daily quota limit. All 3 retry attempts failed. Please try again tomorrow or switch to a different API key."}
            
        return {"error": f"LLM Generation Failed: {error_msg}"}

async def answer_question(topic: str, question: str, chat_history: list = None, language: str = "English"):
    try:
        dialogue = ""
        if chat_history:
            for msg in chat_history:
                dialogue += f"\n{msg.role.capitalize()}: {msg.content}"

        prompt = f"""
        You are an expert AI answering questions about the research topic: "{topic}".
        The user asks: "{question}"
        
        Previous conversation context:
        {dialogue}

        INSTRUCTIONS:
        - Think OUTSIDE of specific research papers. Use your broad knowledge base to answer the user as an expert.
        - Answer directly, clearly, and concisely.
        - Your final output MUST be evaluated and written natively in the **{language}** language.
        """

        # Call with automatic retry on rate-limit errors
        answer = await _call_with_retry(
            model_name="gemini-2.0-flash",
            contents=prompt
        )
        return answer

    except Exception as e:
        error_msg = str(e)
        print("🔥 QA ERROR TRACEBACK ENCOUNTERED:")
        traceback.print_exc()
        if "429" in error_msg or "ResourceExhausted" in error_msg or "quota" in error_msg.lower():
            return "API rate limit hit after 3 retries. Please try again in a few minutes."
        return "An error occurred while answering."