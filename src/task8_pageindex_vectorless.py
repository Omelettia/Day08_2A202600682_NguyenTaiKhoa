"""
Task 8 - PageIndex vectorless RAG fallback.

The PageIndex SDK is optional in this project environment. If it is installed
and PAGEINDEX_API_KEY is set, the functions try to use it. Otherwise, the same
interface falls back to local lexical retrieval and marks results as
source='pageindex' so Task 9 fallback logic remains demonstrable.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _local_fallback_search(query: str, top_k: int) -> list[dict]:
    try:
        from .task6_lexical_search import lexical_search
    except ImportError:
        from task6_lexical_search import lexical_search

    results = lexical_search(query, top_k=top_k)
    if not results:
        try:
            from .task5_semantic_search import semantic_search
        except ImportError:
            from task5_semantic_search import semantic_search

        results = semantic_search(query, top_k=top_k)

    output = []
    for result in results[:top_k]:
        output.append(
            {
                "content": result["content"],
                "score": float(result.get("score", 0.0)),
                "metadata": result.get("metadata", {}),
                "source": "pageindex",
            }
        )
    return output


def upload_documents():
    """
    Upload markdown documents to PageIndex when the SDK is available.

    Returns a small status dict instead of crashing in local-only mode.
    """
    if not PAGEINDEX_API_KEY:
        return {"uploaded": 0, "mode": "local-fallback", "reason": "PAGEINDEX_API_KEY missing"}

    try:
        from pageindex import PageIndex

        client = PageIndex(api_key=PAGEINDEX_API_KEY)
        uploaded = 0
        for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
            content = md_file.read_text(encoding="utf-8")
            metadata = {"filename": md_file.name, "type": md_file.parent.name}
            if hasattr(client, "upload"):
                client.upload(content=content, metadata=metadata)
            elif hasattr(client, "add"):
                client.add(content=content, metadata=metadata)
            else:
                raise AttributeError("PageIndex client has no upload/add method")
            uploaded += 1
        return {"uploaded": uploaded, "mode": "pageindex"}
    except Exception as exc:
        return {"uploaded": 0, "mode": "local-fallback", "reason": str(exc)}


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval using PageIndex, with local fallback.
    """
    if top_k <= 0:
        return []

    if PAGEINDEX_API_KEY:
        try:
            from pageindex import PageIndex

            client = PageIndex(api_key=PAGEINDEX_API_KEY)
            raw_results = client.query(query=query, top_k=top_k)
            results = []
            for raw in raw_results:
                if isinstance(raw, dict):
                    content = raw.get("text") or raw.get("content") or ""
                    score = raw.get("score", 0.0)
                    metadata = raw.get("metadata", {})
                else:
                    content = getattr(raw, "text", None) or getattr(raw, "content", "") or ""
                    score = getattr(raw, "score", 0.0)
                    metadata = getattr(raw, "metadata", {})
                results.append(
                    {
                        "content": content,
                        "score": float(score or 0.0),
                        "metadata": metadata or {},
                        "source": "pageindex",
                    }
                )
            results = [result for result in results if result["content"]]
            if results:
                return results[:top_k]
        except Exception as exc:
            print(f"PageIndex search skipped: {exc}")

    return _local_fallback_search(query, top_k)


if __name__ == "__main__":
    print(upload_documents())
    for result in pageindex_search("hình phạt sử dụng ma túy", top_k=3):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
