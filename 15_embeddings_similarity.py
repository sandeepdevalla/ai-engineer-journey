from sentence_transformers import SentenceTransformer
import numpy as np



query = "how do we represent text as vectors for semantic search?"
def calculate_similarity(documents, query):
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        doc_embeddings = model.encode(documents, normalize_embeddings=True)
        # encode([query]) returns shape (1, dim). We want a 1D vector (dim,)
        query_embedding = model.encode([query], normalize_embeddings=True)[0]
        # With normalized vectors, dot product == cosine similarity
        similarities = np.dot(doc_embeddings, query_embedding)  # shape: (num_docs,)
        k = 3
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        for idx in top_k_indices:
            print(f"Score: {similarities[idx]:.4f} - {documents[idx]}")
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    documents = [
    "LLM temperature controls creativity of responses.",
    "Embeddings map text to vectors in high-dimensional space.",
    "React is a JavaScript library for building UIs.",
    "Vector databases store and search embeddings efficiently.",
    "Python is often used for backend APIs and AI tooling.",
    ]

    query = "how do we represent text as vectors for semantic search?"
    # query = "What are skills i need to become full stack developer?"

    calculate_similarity(documents, query)