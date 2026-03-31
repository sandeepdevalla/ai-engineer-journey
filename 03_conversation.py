import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROK_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROK_API_KEY)

messages = [
    {
        "role": "system",
        "content": '''You are a helpful assistant for AI knowledge and programming questions and help Front end guy for learning AI'''
    },
    {
        "role": "user",
        "content": "what is RAG in AI ?"
    }
]

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
)

answer = response.choices[0].message.content
print("AI:", answer)

messages.append({"role": "assistant", "content": answer})

follow_up_question = "Give me real world example of that?"
print("User:", follow_up_question)
messages.append({"role": "user", "content": follow_up_question})

response2 = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
)

answer2 = response2.choices[0].message.content
print("AI:", answer2)
