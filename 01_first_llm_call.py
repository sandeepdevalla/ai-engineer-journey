import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROK_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROK_API_KEY)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "Explain the concept of recursion in programming with a simple example for 5 years boy with simple 5 steps."
        }
    ],
)
print(response.choices[0].message.content)
