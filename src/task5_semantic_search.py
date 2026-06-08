"""
Task 5 - Semantic search over the Task 4 local index.
"""

from __future__ import annotations

import math

try:
    from .task4_chunking_indexing import embed_texts, load_local_index
except ImportError:
    from task4_chunking_indexing import embed_texts, load_local_index


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embed_query(query: str) -> list[float]:
    embeddings, _ = embed_texts([query])
    return embeddings[0]


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search chunks by vector similarity.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}, sorted desc.
    """
    if top_k <= 0:
        return []

    query_embedding = _embed_query(query)
    results = []
    for chunk in load_local_index():
        score = _cosine(query_embedding, chunk.get("embedding", []))
        if score <= 0:
            continue
        results.append(
            {
                "content": chunk["content"],
                "score": float(score),
                "metadata": chunk.get("metadata", {}),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in semantic_search("hinh phat cho toi tang tru ma tuy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
