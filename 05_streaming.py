import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROK_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROK_API_KEY)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant for AI knowledge and programming questions and help Front end guy for learning AI"
    },
    {
        "role": "user",
        "content": "Explain how Netflix recommends movies to users?"
    }
]

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,    
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)

print()