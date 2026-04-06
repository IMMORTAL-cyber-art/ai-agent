import asyncio
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

async def test():
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content("test")
        print("SUCCESS:", response.text)
    except Exception as e:
        print("EXCEPTION TYPE:", type(e))
        print("EXCEPTION STR:", str(e))

if __name__ == "__main__":
    asyncio.run(test())
