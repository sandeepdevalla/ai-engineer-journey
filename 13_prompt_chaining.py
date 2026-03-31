import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

question = "We are looking for a software engineer with 5 years of experience in React, Node.js, and MongoDB. The role is based in the United States."

def summarize_job_description(job_description: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "Summarize this job description in 2–3 sentences and list the main required skills and location.",
            },
            {
                "role": "user",
                "content": job_description,
            },
        ],
    )
    return response.choices[0].message.content

def evaluate_fit(summary: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You evaluate job fit for a Senior Frontend Engineer. Given a job description summary, reply with exactly one word: STRONG or WEAK, based mainly on frontend (React/TypeScript/JS) requirements.",
            },
            {
                "role": "user",
                "content": (
                    "Here is a job description summary:\n"
                    f"{summary}\n\n"
                    "Rate fit for a Senior Frontend Engineer as STRONG or WEAK (one word only)."
                ),
            },
        ],
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print("== Question ==\n", question)
    print("\n== Job Description Summary ==")
    summary = summarize_job_description(question)
    print(summary)
    print("\n== Fit Evaluation ==")
    fit = evaluate_fit(summary)
    print(fit)
