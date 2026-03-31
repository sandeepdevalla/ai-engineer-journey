import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def explain_code(language: str, code: str, question: str | None = None) -> str:
    system_content = (
        f"You are a senior {language} engineer. "
        "Given a code snippet, explain it to a mid-level developer. "
        "Use clear, concise language. "
        "Structure your answer with these Markdown headings:\n"
        "- Summary\n"
        "- How it works\n"
        "- Complexity\n"
        "- Potential issues / improvements"
    )
    user_content = f"Language: {language}\n\nCode:\n```{language.lower()}\n{code.strip()}\n```\n\n"
    if question:
        user_content += f"Specific question: {question}"
    else:
        user_content += "Explain this code."

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    code = """
    def add(a, b):
        return a + b
    """
    question = "What is the purpose of this code?"
    print(explain_code("Python", code, question))