import os
import json

from dotenv import load_dotenv
from groq import Groq


load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def analyze_job_description(job_description: str) -> dict:
    """
    Call the LLM and get a STRICT JSON response describing a job description.
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        # NOTE: We could also *try* to enforce JSON using only system prompts + few-shot examples,
        # but using `response_format={"type": "json_object"}` is more reliable because the API
        # itself guarantees a single valid JSON object.
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict JSON API that analyzes software engineering job descriptions.\n"
                    "You MUST respond with a single valid JSON object and nothing else.\n"
                    "The JSON schema is:\n"
                    "{\n"
                    '  "title": string,\n'
                    '  "seniority": "Junior" | "Mid" | "Senior" | "Staff" | "Principal",\n'
                    '  "location": string,\n'
                    '  "is_remote": boolean,\n'
                    '  "required_skills": string[],\n'
                    '  "nice_to_have_skills": string[],\n'
                    '  "salary_range": {"min": number | null, "max": number | null, "currency": string | null},\n'
                    '  "summary": string\n'
                    "}\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Analyze this job description and return JSON only, "
                    f"following the schema exactly: {job_description!r}"
                ),
            },
        ],
        temperature=0.0,
    )

    content = response.choices[0].message.content

    # Because we used response_format=json_object, the content SHOULD be valid JSON.
    # We still wrap in try/except so the script does not crash if something goes wrong.
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        print("Model did not return valid JSON, here is the raw content:")
        print(content)
        raise

    return data


if __name__ == "__main__":
    print("=== JSON structured output demo (Job Descriptions) ===")

    sample_job_descriptions = [
        "We are looking for a Senior Frontend Engineer with 7+ years of experience in React, TypeScript, and modern JavaScript tooling. "
        "You will work closely with designers and backend engineers to build scalable web applications for global clients. "
        # "Location: Bangalore or remote within India. Experience with Next.js and performance optimization is a plus. "
        # "Compensation: 45–55 LPA depending on experience.",
        # "Join our team as a Junior Software Engineer. You should be comfortable with Python and basic web development concepts. "
        # "This is an on-site role in Hyderabad. Familiarity with SQL and Git is required; knowledge of Docker is a bonus.",
        # "We need a Staff AI Engineer to lead our ML platform initiatives. Strong background in distributed systems, embeddings, and retrieval is required. "
        # "Remote-first, with a preference for candidates in Europe or India. Prior experience building RAG systems is highly desirable.",
    ]

    for i, jd in enumerate(sample_job_descriptions, start=1):
        print(f"\n--- Job Description {i} ---")
        print("Text:", jd)

        result = analyze_job_description(jd)

        print("Parsed JSON:")
        print(json.dumps(result, indent=2))

