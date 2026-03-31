import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROK_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROK_API_KEY)

short_prompt = "What is AI ?"
long_prompt = "Explain the concept of Artificial Intelligence (AI) in simple terms, including its applications and potential impact on society."

usage_short = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": short_prompt
        }
    ],
).usage

print(f"Input tokens: {usage_short.prompt_tokens}")
print(f"Output tokens: {usage_short.completion_tokens}")
print(f"Total tokens: {usage_short.total_tokens}")

usage_long = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": long_prompt
        }
    ],
).usage

print(f"Input tokens: {usage_long.prompt_tokens}")
print(f"Output tokens: {usage_long.completion_tokens}")
print(f"Total tokens: {usage_long.total_tokens}")

print("\n=== Cost Estimation ===")
input_cost = (usage_short.prompt_tokens / 1_000_000) * 2.50
output_cost = (usage_short.completion_tokens / 1_000_000) * 10.00
print(f" Input Cost: ${input_cost:.6f}")
print(f" Output Cost: ${output_cost:.6f}")
print(f" Total Cost: ${input_cost + output_cost:.6f}")