import asyncio
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

async def test():
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
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
