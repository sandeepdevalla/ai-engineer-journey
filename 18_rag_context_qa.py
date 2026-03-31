import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np


load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)


def chunk_text(text: str, *, max_chars: int = 500, overlap: int = 80) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def add(text: str, metadata: dict | None = None) -> dict:
    embedding = model.encode(text, normalize_embeddings=True)
    return {"text": text, "embedding": embedding, "metadata": metadata or {}}


def build_index(docs: list[dict]) -> tuple[list[dict], np.ndarray]:
    if not docs:
        return [], np.zeros((0, 0), dtype=np.float32)

    doc_matrix = np.stack([d["embedding"] for d in docs]).astype(np.float32)
    documents = [{"text": d["text"], "metadata": d.get("metadata", {})} for d in docs]
    return documents, doc_matrix


def retrieve(query: str, *, documents: list[dict], doc_matrix: np.ndarray, k: int = 4) -> list[dict]:
    if doc_matrix.size == 0:
        return []

    query_embedding = model.encode(query, normalize_embeddings=True).astype(np.float32)
    similarities = doc_matrix @ query_embedding

    k = min(k, len(documents))
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    results: list[dict] = []
    for idx in top_k_indices:
        results.append(
            {
                "score": float(similarities[idx]),
                "text": documents[idx]["text"],
                "metadata": documents[idx].get("metadata", {}),
            }
        )
    return results


def answer_question_with_context(question: str, context_chunks: list[dict]) -> str:
    # Label context chunks with stable identifiers so the LLM can cite them.
    context_text = "\n\n".join(
        (
            lambda c, idx: (
                f"[chunk_id {c.get('metadata', {}).get('chunk_id', idx)} | score={c['score']:.3f}]\n"
                f"{c['text'].strip()}"
            )
        )(c, idx)
        for idx, c in enumerate(context_chunks)
    )

    system_content = (
        "You are an AI Engineering mentor helping the user understand their own learning plan.\n"
        "You will be given context chunks from their personal CONTEXT.md file.\n"
        "Answer the question using only this context. If the answer is not present, say so clearly.\n"
        "Always end your response with a 'Sources:' section listing the chunk_id values you used."
    )

    user_content = (
        f"Question:\n{question.strip()}\n\n"
        "Context chunks:\n"
        f"{context_text}\n\n"
        "Now answer the question concisely for this user."
    )

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content or ""


def build_context_index_from_file() -> tuple[list[dict], np.ndarray]:
    ctx_path = Path(__file__).with_name("CONTEXT.md")
    raw_text = ctx_path.read_text(encoding="utf-8")

    doc_id = "learning-journey-context"
    source = str(ctx_path.name)
    title = "AI Engineering Learning Journey Context"

    chunks = chunk_text(raw_text, max_chars=420, overlap=60)

    docs: list[dict] = []
    for i, chunk in enumerate(chunks):
        metadata = {
            "doc_id": doc_id,
            "chunk_id": i,
            "source": source,
            "title": title,
        }
        docs.append(add(chunk, metadata=metadata))

    return build_index(docs)


if __name__ == "__main__":
    documents, doc_matrix = build_context_index_from_file()

    while True:
        try:
            question = input("\nAsk a question about your learning journey (or 'q' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question or question.lower() in {"q", "quit", "exit"}:
            break

        top_chunks = retrieve(question, documents=documents, doc_matrix=doc_matrix, k=4)

        print("\n--- Retrieved Chunks (top-k) ---")
        for i, c in enumerate(top_chunks):
            meta = c.get("metadata", {})
            chunk_id = meta.get("chunk_id")
            score = c.get("score", 0.0)
            source = meta.get("source", "")
            title = meta.get("title", "")
            print(f"{i+1}. chunk_id={chunk_id} | score={score:.3f} | source={source} | title={title}")
            preview = c["text"].strip().replace("\n", " ")
            print(f"   preview: {preview[:220]}{'...' if len(preview) > 220 else ''}")

        answer = answer_question_with_context(question, top_chunks)

        print("\n--- Answer ---")
        print(answer)

