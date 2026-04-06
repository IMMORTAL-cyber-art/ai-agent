import asyncio
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

async def test():
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "gsk_your_groq_api_key_here":
            raise ValueError("Missing or invalid GROQ_API_KEY. Please set a valid Groq API key in your environment variables.")
        
        client = Groq(api_key=api_key)
        
        response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "user", "content": "test"}
            ]
        )
        print("SUCCESS:", response.choices[0].message.content)
    except Exception as e:
        print("EXCEPTION TYPE:", type(e))
        print("EXCEPTION STR:", str(e))

if __name__ == "__main__":
    asyncio.run(test())
