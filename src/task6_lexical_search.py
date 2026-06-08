"""
Task 6 - Lexical search with BM25.

Uses rank-bm25 when installed. A small built-in BM25 fallback keeps the project
working in minimal environments.
"""

from __future__ import annotations

import math
import re
from collections import Counter

try:
    from .task4_chunking_indexing import load_local_index
except ImportError:
    from task4_chunking_indexing import load_local_index

CORPUS: list[dict] = []
_BM25_INDEX = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class SimpleBM25:
    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.tokenized_corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(doc) for doc in tokenized_corpus]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        self.term_freqs = [Counter(doc) for doc in tokenized_corpus]
        doc_freq = Counter()
        for doc in tokenized_corpus:
            doc_freq.update(set(doc))
        total_docs = len(tokenized_corpus)
        self.idf = {
            term: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = []
        for idx, freqs in enumerate(self.term_freqs):
            score = 0.0
            doc_len = self.doc_lengths[idx]
            for term in query_tokens:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1))
                score += idf * (tf * (self.k1 + 1)) / denom
            scores.append(score)
        return scores


def _load_corpus() -> list[dict]:
    global CORPUS
    if not CORPUS:
        CORPUS = [
            {"content": chunk["content"], "metadata": chunk.get("metadata", {})}
            for chunk in load_local_index()
        ]
    return CORPUS


def build_bm25_index(corpus: list[dict]):
    """
    Build BM25 index from corpus.
    """
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    try:
        from rank_bm25 import BM25Okapi

        return BM25Okapi(tokenized_corpus)
    except Exception:
        return SimpleBM25(tokenized_corpus)


def _get_index():
    global _BM25_INDEX
    corpus = _load_corpus()
    if _BM25_INDEX is None:
        _BM25_INDEX = build_bm25_index(corpus)
    return _BM25_INDEX


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search chunks with BM25.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}, sorted desc.
    """
    if top_k <= 0:
        return []

    corpus = _load_corpus()
    if not corpus:
        return []

    tokenized_query = _tokenize(query)
    scores = _get_index().get_scores(tokenized_query)
    ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)

    results = []
    for idx, score in ranked[:top_k]:
        if float(score) <= 0:
            continue
        results.append(
            {
                "content": corpus[idx]["content"],
                "score": float(score),
                "metadata": corpus[idx].get("metadata", {}),
            }
        )
    return results


if __name__ == "__main__":
    for result in lexical_search("Dieu 248 tang tru trai phep chat ma tuy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
