import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("=== zero-shot ===")
response_zero_shot = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": "Clasify this review as POSITIVE or NEGATIVE: 'the battery dies too fast but camera quality is good'"
        }
    ]
)
print(response_zero_shot.choices[0].message.content)

print("\n=== few-shot ===")
response_few_shot = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant for classifying product reviews and respond with exactly one word POSITIVE or NEGATIVE or MIXED."
        },
        {
            "role": "user",
            "content": "Review: I obsolutely love this product"
        },
        {
            "role": "assistant",
            "content": "POSITIVE"
        },
        {
            "role": "user",
            "content": "Review: Terrible quelity, broke after one use"
        },
        {
            "role": "assistant",
            "content": "NEGATIVE"
        },
        {
            "role": "user",
            "content": "Review: 'Great design but software is buggy'"
        },
        {
            "role": "assistant",
            "content": "MIXED"
        },
        {
            "role": "user",
            "content": "review: 'the battery dies too fast but camera quality is amazing'"
        }
    ],
    temperature=0.0  # Setting temperature to 0 for deterministic output
)
print(response_few_shot.choices[0].message.content)

