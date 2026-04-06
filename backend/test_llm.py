import asyncio
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={'api_version': 'v1beta'}
)

async def test():
    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.5-pro",
            contents="test"
        )
        print("SUCCESS:", response.text)
    except Exception as e:
        print("EXCEPTION TYPE:", type(e))
        print("EXCEPTION STR:", str(e))

if __name__ == "__main__":
    asyncio.run(test())
