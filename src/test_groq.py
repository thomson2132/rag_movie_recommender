import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv('GROQ_API_KEY')

if not api_key:
    print("❌ ERROR: GROQ_API_KEY not found in .env file")
    print("Please add: GROQ_API_KEY=gsk_your_key_here to .env")
    exit(1)

print(f"✓ API Key found: {api_key[:10]}...{api_key[-5:]}")

# Test connection with updated model
try:
    client = Groq(api_key=api_key)

    # Simple test request with current model
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # ← Updated model
        messages=[
            {"role": "user", "content": "Say 'Hello, Groq is working!'"}
        ],
        max_tokens=50
    )

    result = response.choices[0].message.content
    print("\n✓ Groq API is working!")
    print(f"Response: {result}")
    print(f"\n✓ Model used: llama-3.3-70b-versatile")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nPossible issues:")
    print("1. Invalid API key")
    print("2. Network connectivity problem")
    print("3. Groq service is down")
    print("4. API key quota exceeded")
