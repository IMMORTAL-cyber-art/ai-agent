import os
import re
import json
import time
import asyncio
import traceback
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ──────────────────────────────────────────────────────────────────────────────

def _extract_retry_delay(error_msg: str) -> int:
    """Extract the retryDelay value in seconds from Google's 429 error message."""
    match = re.search(r"retryDelay.*?(\d+)s", error_msg)
    if match:
        return int(match.group(1)) + 2
    return 30

async def _call_with_retry(contents: str, json_mode: bool = False, max_retries: int = 3):
    """Call the Groq API with Llama 3 with automatic retry on failures."""
    for attempt in range(max_retries):
        try:
            # Groq chat completion call
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant and an expert scientist."},
                    {"role": "user", "content": contents}
                ],
                # If json_mode is required, we can specify it if the model supports it, 
                # or just rely on the prompt. Llama 3 70B is very good at JSON.
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

        # Call Groq with Llama 3
        raw_text = await _call_with_retry(
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