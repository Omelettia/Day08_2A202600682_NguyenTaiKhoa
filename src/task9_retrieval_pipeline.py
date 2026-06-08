"""
Task 9 - Unified retrieval pipeline.

Flow: semantic search + lexical search -> RRF merge -> rerank -> PageIndex
fallback when the hybrid result is weak.
"""

from __future__ import annotations

try:
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search
    from .task7_reranking import rerank, rerank_rrf
    from .task8_pageindex_vectorless import pageindex_search
except ImportError:
    from task5_semantic_search import semantic_search
    from task6_lexical_search import lexical_search
    from task7_reranking import rerank, rerank_rrf
    from task8_pageindex_vectorless import pageindex_search

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"


def _normalize_scores(results: list[dict]) -> list[dict]:
    if not results:
        return results
    max_score = max(float(r.get("score", 0.0)) for r in results)
    if max_score <= 0:
        return results
    return [{**r, "score": float(r.get("score", 0.0)) / max_score} for r in results]


def _mark_source(results: list[dict], source: str) -> list[dict]:
    marked = []
    for result in results:
        marked.append({**result, "source": source})
    return marked


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieve top-k chunks using hybrid search and PageIndex fallback.
    """
    if top_k <= 0:
        return []

    dense_results = semantic_search(query, top_k=top_k * 3)
    sparse_results = lexical_search(query, top_k=top_k * 3)

    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 3)
    merged = _normalize_scores(merged)
    merged = _mark_source(merged, "hybrid")

    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        final_results = _normalize_scores(final_results)
        final_results = _mark_source(final_results, "hybrid")
    else:
        final_results = merged[:top_k]

    if not final_results or float(final_results[0].get("score", 0.0)) < score_threshold:
        fallback = pageindex_search(query, top_k=top_k)
        return _normalize_scores(fallback[:top_k])

    return final_results[:top_k]


if __name__ == "__main__":
    for q in [
        "Hình phạt cho tội tàng trữ trái phép chất ma túy",
        "Nghệ sĩ nào bị bắt vì sử dụng ma túy",
        "Luật phòng chống ma túy quy định gì về cai nghiện",
    ]:
        print(f"\nQuery: {q}")
        for i, result in enumerate(retrieve(q, top_k=3), 1):
            print(f"{i}. [{result['score']:.3f}] [{result['source']}] {result['content'][:90]}...")
