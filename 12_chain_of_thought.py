import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


QUESTION = (
    "A store is running a sale. All items are 20% off. "
    "Riya buys a jacket that normally costs ₹4,000 and a pair of shoes that normally costs ₹3,000. "
    "She also uses a coupon for an additional ₹500 off the total after the discount. "
    "What is the final amount she pays?"
)


def answer_without_cot() -> str:
    """
    Ask the question normally, without explicitly requesting step-by-step reasoning.
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful math tutor. Do not show reasoning and give final answer only.",
            },
            {
                "role": "user",
                "content": QUESTION,
            },
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def answer_with_cot() -> str:
    """
    Ask the same question but explicitly request chain-of-thought (step-by-step) reasoning.
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful math tutor. "
                    "First think through the problem step by step, clearly showing your reasoning, "
                    "and then give the final answer at the end."
                ),
            },
            {
                "role": "user",
                "content": QUESTION + "\n\nPlease show your reasoning step by step.",
            },
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print("=== Chain-of-thought prompting demo ===")
    print("\nQuestion:")
    print(QUESTION)

    print("\n--- Answer WITHOUT explicit chain-of-thought ---")
    print(answer_without_cot())

    print("\n--- Answer WITH explicit chain-of-thought ---")
    print(answer_with_cot())

