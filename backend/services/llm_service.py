import os
import re
import json
import time
import asyncio
import traceback
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def _get_groq_client():
    """Safely initialize the Groq client, ensuring the API key is present."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "gsk_your_groq_api_key_here":
        raise ValueError("Missing or invalid GROQ_API_KEY. Please set a valid Groq API key in your environment variables.")
    return Groq(api_key=api_key)

# ──────────────────────────────────────────────────────────────────────────────

def _extract_retry_delay(error_msg: str) -> int:
    """Extract the retryDelay value in seconds from Google's 429 error message."""
    match = re.search(r"retryDelay.*?(\d+)s", error_msg)
    if match:
        return int(match.group(1)) + 2
    return 30

async def _call_with_retry(contents: str, json_mode: bool = False, max_retries: int = 3, max_tokens: int = 1200, temperature: float = 0.2):
    """Call the Groq API with Llama 3 with automatic retry on failures and token limits."""
    for attempt in range(max_retries):
        try:
            # Initialize client only inside the function to prevent import-time crashes
            client = _get_groq_client()
            
            # Groq chat completion call
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You MUST return ONLY valid JSON. Do NOT include any explanation, markdown, or extra text. Return strictly JSON format only. All keys must be in double quotes. Ensure valid JSON syntax."},
                    {"role": "user", "content": contents}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️ Groq attempt {attempt + 1} failed: {error_msg}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            raise Exception(f"Groq API failed after {max_retries} attempts. Last error: {error_msg}")


async def generate_literature_review(topic: str, papers: list = None, language: str = "English"):
    try:
        start_time = time.time()
        
        # Re-add paper context processing (Optimized to 4 papers to avoid token limits)
        context_parts = []
        if papers:
            # Limit to top 4 papers for stability and to prevent output truncation
            for i, paper in enumerate(papers[:4], 1):
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
        
        {f"Use the following provided research papers as your primary source material:\n{papers_context}" if papers else "No specific recent papers were found for this topic. Please generate a high-quality literature review based on your internal knowledge of the research landscape in this field."}

        IMPORTANT:
        - You MUST return ONLY valid JSON.
        - KEEP RESPONSE CONCISE: Total length should not exceed 800 words.
        - Ensure your response fits within the 1200 token limit.
        - Use double quotes for ALL keys and strings.
        - No extra text, no explanations, no markdown wrappers, no code blocks.
        - The output must be directly parsable by json.loads() in Python.
        - Write all internal content (strings/arrays inside the JSON) in **{language}**.
        - The JSON Keys MUST remain exactly as defined in English.

        {{
            "introduction": "... (in {language})",
            "key_themes": ["..."],
            "comparative_analysis": "Markdown table... (in {language})",
            "research_gaps": ["..."],
            "conclusion": "...",
            "key_takeaways": ["...", "...", "...", "...", "..."],
            "ai_idea": "...",
            "confidence_level": "High", 
            "complexity_level": "Expert"
        }}
        """

        # Call Groq with Llama 3 (Strict 1200 token limit)
        raw_text = await _call_with_retry(
            contents=prompt,
            json_mode=True,
            max_tokens=1200
        )

        end_time = time.time()
        generation_time_seconds = round(end_time - start_time, 2)
        
        # 1. Safely extract JSON using regex (handles extra text/markdown)
        import re
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            clean_text = json_match.group(0)
        else:
            clean_text = raw_text.strip().strip('`').removeprefix('json').strip()
        
        # 2. Remove internal control characters
        clean_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', clean_text)
        
        try:
            # 3. First attempt: standard parse
            parsed_json = json.loads(clean_text, strict=False)
        except json.JSONDecodeError:
            try:
                # 4. Fallback repair: Fix missing quotes around keys and internal newlines
                fixed_text = re.sub(r'(\b\w+\b):', r'"\1":', clean_text)
                repaired_text = re.sub(r'([^\\])\n', r'\1\\n', fixed_text)
                parsed_json = json.loads(repaired_text, strict=False)
            except Exception as e:
                print(f"❌ Critical JSON Parsing Failure: {str(e)}")
                raise Exception("The AI returned a malformed JSON response that could not be repaired.")
        
        # 5. Normalize fields based on expected Pydantic schema types
        if parsed_json:
            def ensure_list(value):
                if isinstance(value, list):
                    return [str(v) for v in value]
                if isinstance(value, str):
                    return [v.strip() for v in value.strip().split("\n") if v.strip()]
                return []

            def ensure_string(value):
                if isinstance(value, list):
                    return "\n".join(str(v) for v in value)
                return str(value) if value is not None else ""

            # Define which fields should be lists vs strings
            list_fields = ["key_themes", "research_gaps", "key_takeaways"]
            
            for key in parsed_json:
                if key in list_fields:
                    parsed_json[key] = ensure_list(parsed_json[key])
                else:
                    parsed_json[key] = ensure_string(parsed_json[key])

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

        # Call Groq with Llama 3
        answer = await _call_with_retry(
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