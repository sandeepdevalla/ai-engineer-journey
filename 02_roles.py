import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        # {
        #     "role": "system",
        #     "content": "You are seniour Frontend developer with 10 yearss of experience . You explain ai concepts by compareing them to fe/react concepts. Keep anssers concise."
        #     },
            {
                "role": "user",
                "content": "What is the difference between a transformer and a recurrent neural network (RNN)?"
            }
    ],
)

print(response.choices[0].message.content)