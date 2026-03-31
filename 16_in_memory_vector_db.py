from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)


def chunk_text(text: str, *, max_chars: int = 500, overlap: int = 80) -> list[str]:
    """
    Simple character-based chunking with overlap.
    Good enough to learn the pipeline; later you can switch to token-based chunking.
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
    """
    Returns:
    - documents: list of {text, metadata}
    - doc_matrix: shape (n_docs, dim) with L2-normalized vectors
    """
    if not docs:
        return [], np.zeros((0, 0), dtype=np.float32)

    doc_matrix = np.stack([d["embedding"] for d in docs]).astype(np.float32)
    documents = [{"text": d["text"], "metadata": d.get("metadata", {})} for d in docs]
    return documents, doc_matrix


def search(query: str, *, documents: list[dict], doc_matrix: np.ndarray, k: int = 3) -> list[dict]:
    if doc_matrix.size == 0:
        return []

    query_embedding = model.encode(query, normalize_embeddings=True).astype(np.float32)
    similarities = doc_matrix @ query_embedding  # cosine similarity because vectors normalized

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
    # Real-ish example: imagine this is extracted from a PDF, Confluence, or a README.
    # In practice you'd load from files/web and keep doc_id + source URL/path.
    doc_id = "acme-handbook-v1"
    source = "handbook.pdf"
    title = "ACME Engineering Handbook"

    raw_document = """
    ACME Engineering Handbook

    Incident Response
    When an alert fires, acknowledge it and assess impact: customer-facing outages are P0.
    For P0 incidents, create an incident channel, assign an incident commander, and start a timeline.
    Communicate early and often. Update the status page at least every 30 minutes during a P0.

    Postmortems
    For every P0/P1 incident, write a blameless postmortem within 5 business days.
    Include: what happened, impact, detection, response, root cause, and action items with owners.
    Action items should be tracked to completion and reviewed in the weekly reliability meeting.

    API Guidelines
    Prefer explicit versioning in URLs (e.g., /v1/) and never break clients without a migration plan.
    Pagination should be cursor-based for large datasets.
    Use consistent error envelopes with a stable error code, message, and request id.
    """

    chunks = chunk_text(raw_document, max_chars=420, overlap=60)

    docs: list[dict] = []
    for i, chunk in enumerate(chunks):
        metadata = {
            "doc_id": doc_id,
            "chunk_id": i,
            "source": source,
            "title": title,
            # Pretend these came from your parser; for PDFs you'd have page numbers.
            "section_hint": "handbook",
        }
        docs.append(add(chunk, metadata=metadata))

    documents, doc_matrix = build_index(docs)

    queries = [
        "What should we do when there is a P0 outage?",
        "How often to update status page?",
        "What are the requirements for a postmortem?",
        "How should API pagination work?",
    ]

    for q in queries:
        print("\n" + "=" * 80)
        print("Query:", q)
        results = search(q, documents=documents, doc_matrix=doc_matrix, k=3)
        for r in results:
            meta = r["metadata"]
            print(f"\nScore: {r['score']:.4f}")
            print(f"Source: {meta.get('source')} | Doc: {meta.get('doc_id')} | Chunk: {meta.get('chunk_id')}")
            print(r["text"])