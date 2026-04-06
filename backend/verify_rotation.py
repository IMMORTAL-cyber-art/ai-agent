import asyncio
import os
from backend.services.llm_service import _call_with_retry

async def test_rotation():
    print("🚀 Starting Rotation Verification Test...")
    try:
        # This will trigger the rotation logic if Key #1 is exhausted
        response = await _call_with_retry(
            model_name="gemini-2.0-flash",
            contents="Say 'Rotation Success' if you are working.",
            max_retries=4
        )
        print(f"\n✅ FINAL RESPONSE: {response}")
    except Exception as e:
        print(f"\n❌ ALL KEYS FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(test_rotation())
