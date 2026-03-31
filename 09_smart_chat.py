# This combines streaming + temperature + system prompt into as polished chatbot

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROK_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROK_API_KEY)

messages = [
    {
        "role": "system",
        "content": ''' you are "CodeMentor AI" - an expert programming mentor. Rules:
        1. Explain concepts in simple terms with real-world analogies.
        2. Always include a short code example when relevant.
        3. keep answers in 150 words.
        4. If user asks something non-programming related, politely redirect them'''
    }
]

print(" CodeMentor AI")
print("Type 'exit' to end the conversation.\n")

while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() == "exit":
        print("Keep coding! see you next time!")
        break
    if not user_input:
        print("Please enter a question or type 'exit' to quit.")
        continue

    messages.append({"role": "user", "content": user_input})

    print("CodeMentor AI : ", end="", flush=True)

    response = client.chat.completions.create( 
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,
        stream=True,
        max_tokens=500
    )

    answer = ""
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
            answer += content
    print()
    messages.append({"role": "assistant", "content": answer})