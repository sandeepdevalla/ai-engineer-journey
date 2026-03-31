import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROK_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROK_API_KEY)

messags = [
    {
        "role": "system",
        "content": "You are AI Mentor for front end developer or full stack developer for transissioning to AI Engineer and give them consice answers"
    }
]

print("AI career mentor (type 'exit' to quit)")
print("-" * 40)

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    messags.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messags,
    )

    answer = response.choices[0].message.content
    print("AI:", answer)
    messags.append({"role": "assistant", "content": answer})
    