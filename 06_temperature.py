import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROK_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROK_API_KEY)

prompt = "Suggest the name for an AI powered code review tool."

print("=== Temperature 0 (Precise, same answer every time) ===\n")
for i in range(4):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0  # Varying temperature for different outputs
    )
    answer = response.choices[0].message.content[:80]
    print(f"Attempt {i + 1}: {answer}\n")


print("=== Temperature 1 (Creative, different answer every time) ===\n")
for i in range(4):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=1  # Varying temperature for different outputs
    )
    answer = response.choices[0].message.content[:80]
    print(f"Attempt {i + 1}: {answer}\n")