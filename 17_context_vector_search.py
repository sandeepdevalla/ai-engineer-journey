from pathlib import Path

from sentence_transformers import SentenceTransformer
import numpy as np


MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)


def chunk_text(text: str, *, max_chars: int = 500, overlap: int = 80) -> list[str]:
    """
    Simple character-based chunking with overlap.
    """
    text = " ".join(text.split())
    if not text:
        return []

    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= max_chars:
        raise ValueError("overlap must be < max_chars")

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


def search(query: str, *, documents: list[dict], doc_matrix: np.ndarray, k: int = 3) -> list[dict]:
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


if __name__ == "__main__":
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

    documents, doc_matrix = build_index(docs)

    example_queries = [
        "What have I completed so far in my journey?",
        "What is planned for Day 4?",
        "What is my career target in terms of salary and company type?",
        "What is the roadmap for months 1 to 3?",
    ]

    for q in example_queries:
        print("\n" + "=" * 80)
        print("Query:", q)
        results = search(q, documents=documents, doc_matrix=doc_matrix, k=3)
        for r in results:
            meta = r["metadata"]
            print(f"\nScore: {r['score']:.4f}")
            print(f"Source: {meta.get('source')} | Doc: {meta.get('doc_id')} | Chunk: {meta.get('chunk_id')}")
            print(r["text"])

