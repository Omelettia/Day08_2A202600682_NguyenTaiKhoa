"""
Task 7 - Reranking.

Primary path: Jina multilingual reranker when JINA_API_KEY is configured and
USE_JINA_RERANKER=true.

Reliable local path: term-overlap reranking and Reciprocal Rank Fusion (RRF).
RRF is especially useful here because Task 9 merges semantic and BM25 rankers.
"""

from __future__ import annotations

import os
import re
from math import sqrt

from dotenv import load_dotenv

load_dotenv()

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
JINA_RERANK_MODEL = os.getenv("JINA_RERANK_MODEL", "jina-reranker-v2-base-multilingual")


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _local_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    query_tokens = _tokens(query)
    scored = []
    for rank, candidate in enumerate(candidates, 1):
        content_tokens = _tokens(candidate.get("content", ""))
        overlap = len(query_tokens & content_tokens) / max(len(query_tokens), 1)
        original = float(candidate.get("score", 0.0))
        # Keep original retrieval signal, then add direct lexical overlap.
        score = 0.65 * original + 0.35 * overlap + 1 / (1000 + rank)
        item = {**candidate, "score": float(score)}
        scored.append(item)
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank candidates with Jina when explicitly enabled; otherwise local rerank.
    """
    if not candidates or top_k <= 0:
        return []

    api_key = os.getenv("JINA_API_KEY", "")
    use_jina = os.getenv("USE_JINA_RERANKER", "false").lower() in {"1", "true", "yes"}
    if not api_key or not use_jina:
        return _local_rerank(query, candidates, top_k)

    try:
        import requests

        response = requests.post(
            JINA_RERANK_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": JINA_RERANK_MODEL,
                "query": query,
                "documents": [candidate["content"] for candidate in candidates],
                "top_n": min(top_k, len(candidates)),
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        results = []
        for result in payload.get("results", []):
            item = {**candidates[result["index"]]}
            item["score"] = float(result.get("relevance_score", item.get("score", 0.0)))
            results.append(item)
        if results:
            return results[:top_k]
    except Exception as exc:
        print(f"Jina reranker skipped: {exc}")

    return _local_rerank(query, candidates, top_k)


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance: relevant to query while reducing duplication.
    """
    if not candidates or top_k <= 0:
        return []

    selected: list[int] = []
    remaining = list(range(len(candidates)))
    for _ in range(min(top_k, len(candidates))):
        best_idx = remaining[0]
        best_score = float("-inf")
        for idx in remaining:
            embedding = candidates[idx].get("embedding", [])
            relevance = _cosine(query_embedding, embedding) or float(candidates[idx].get("score", 0))
            diversity_penalty = 0.0
            for selected_idx in selected:
                diversity_penalty = max(
                    diversity_penalty,
                    _cosine(embedding, candidates[selected_idx].get("embedding", [])),
                )
            score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if score > best_score:
                best_idx = idx
                best_score = score
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [{**candidates[idx], "score": float(candidates[idx].get("score", 0.0))} for idx in selected]


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion.

    RRF(d) = sum(1 / (k + rank_r(d))) across rankers.
    """
    if top_k <= 0:
        return []

    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item.get("metadata", {}).get("path") or item.get("content", "")
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
            if key not in items:
                items[key] = item

    fused = []
    for key, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True):
        fused.append({**items[key], "score": float(score)})
    return fused[:top_k]


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """
    Unified reranking interface.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    if method == "mmr":
        return _local_rerank(query, candidates, top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma túy", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ bị bắt vì sử dụng ma túy", "score": 0.7, "metadata": {}},
        {"content": "Python programming", "score": 0.4, "metadata": {}},
    ]
    for result in rerank("hình phạt ma túy", dummy_candidates, top_k=2):
        print(f"[{result['score']:.3f}] {result['content']}")
