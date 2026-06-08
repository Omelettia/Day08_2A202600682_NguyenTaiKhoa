"""
Task 4 - Chunking and indexing.

Default path: build a local JSON index from data/standardized so the project is
repeatable without Docker or cloud services.

Full path: the same chunks/embeddings can be synced to Weaviate when
WEAVIATE_URL and WEAVIATE_API_KEY are available.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.json"
INDEX_META_PATH = INDEX_DIR / "index_meta.json"

# Recursive character chunking is stable for mixed legal/news markdown. Legal
# paragraphs are long, so 800 chars keeps enough context; 120 overlap preserves
# definitions and citations that spill across chunk boundaries.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
CHUNKING_METHOD = "recursive"

# AITeamVN/Vietnamese_Embedding is fine-tuned from BGE-M3 for Vietnamese
# retrieval, which fits this Vietnamese legal/news corpus better than a generic
# multilingual model. If sentence-transformers is unavailable, we use
# deterministic hashing with the same configured dimension so tests and local
# retrieval still work.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "AITeamVN/Vietnamese_Embedding")
EMBEDDING_DIM = 1024

VECTOR_STORE = "local_json_with_optional_weaviate"
WEAVIATE_COLLECTION = "DrugLawDocs"
_SENTENCE_TRANSFORMER_MODEL = None


def _doc_type(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "legal" in parts:
        return "legal"
    if "news" in parts:
        return "news"
    return "unknown"


def load_documents() -> list[dict]:
    """
    Read all markdown files from data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name.startswith("."):
            continue
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        rel_path = md_file.relative_to(STANDARDIZED_DIR).as_posix()
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": rel_path,
                    "type": _doc_type(md_file),
                },
            }
        )
    return documents


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return [chunk for chunk in chunks if chunk]


def _recursive_split(text: str) -> list[str]:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        )
        return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]
    except Exception:
        pass

    chunks = []
    current = ""
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > CHUNK_SIZE:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_text(paragraph, CHUNK_SIZE, CHUNK_OVERLAP))
        elif not current:
            current = paragraph
        elif len(current) + len(paragraph) + 2 <= CHUNK_SIZE:
            current = f"{current}\n\n{paragraph}"
        else:
            chunks.append(current)
            overlap_text = current[-CHUNK_OVERLAP:].strip()
            current = f"{overlap_text}\n\n{paragraph}" if overlap_text else paragraph
            if len(current) > CHUNK_SIZE:
                chunks.extend(_split_long_text(current, CHUNK_SIZE, CHUNK_OVERLAP))
                current = ""
    if current:
        chunks.append(current)
    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents with recursive character splitting.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    chunks = []
    for doc_index, doc in enumerate(documents):
        for chunk_index, chunk_text in enumerate(_recursive_split(doc["content"])):
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "doc_index": doc_index,
                        "chunk_index": chunk_index,
                    },
                }
            )
    return chunks


def _hash_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    vector = [0.0] * dim
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _sentence_transformer_embeddings(texts: list[str]) -> list[list[float]] | None:
    if os.getenv("USE_SENTENCE_TRANSFORMERS", "false").lower() not in {"1", "true", "yes"}:
        return None
    try:
        global _SENTENCE_TRANSFORMER_MODEL
        from sentence_transformers import SentenceTransformer

        if _SENTENCE_TRANSFORMER_MODEL is None:
            _SENTENCE_TRANSFORMER_MODEL = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = _SENTENCE_TRANSFORMER_MODEL.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [embedding.tolist() for embedding in embeddings]
    except Exception as exc:
        print(f"SentenceTransformer embedding skipped: {exc}")
        return None


def embed_texts(texts: list[str]) -> tuple[list[list[float]], str]:
    embeddings = _sentence_transformer_embeddings(texts)
    if embeddings is not None:
        return embeddings, EMBEDDING_MODEL
    return [_hash_embedding(text) for text in texts], "hashing-fallback"


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add an embedding vector to each chunk.

    Uses BAAI/bge-m3 through sentence-transformers when available. Falls back to
    deterministic hashing so indexing/search is always runnable.
    """
    texts = [chunk["content"] for chunk in chunks]
    embeddings, embedding_method = embed_texts(texts)
    embedding_dim = len(embeddings[0]) if embeddings else EMBEDDING_DIM

    embedded = []
    for chunk, embedding in zip(chunks, embeddings):
        item = {**chunk, "embedding": embedding}
        item["metadata"] = {
            **item.get("metadata", {}),
            "embedding_model": embedding_method,
            "embedding_dim": embedding_dim,
        }
        embedded.append(item)
    return embedded


def _save_local_index(chunks: list[dict]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    slim_chunks = [
        {"content": chunk["content"], "metadata": chunk.get("metadata", {})}
        for chunk in chunks
    ]
    embeddings = [chunk["embedding"] for chunk in chunks]
    CHUNKS_PATH.write_text(json.dumps(slim_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    EMBEDDINGS_PATH.write_text(json.dumps(embeddings), encoding="utf-8")
    INDEX_META_PATH.write_text(
        json.dumps(
            {
                "chunk_count": len(chunks),
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "chunking_method": CHUNKING_METHOD,
                "embedding_model": chunks[0]["metadata"].get("embedding_model") if chunks else None,
                "embedding_dim": len(chunks[0]["embedding"]) if chunks else EMBEDDING_DIM,
                "vector_store": VECTOR_STORE,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _try_index_weaviate(chunks: list[dict]) -> bool:
    url = os.getenv("WEAVIATE_URL")
    api_key = os.getenv("WEAVIATE_API_KEY")
    if not url or not api_key:
        return False

    try:
        import weaviate
        from weaviate.auth import AuthApiKey
        from weaviate.classes.config import DataType, Property

        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=AuthApiKey(api_key),
        )
        try:
            if client.collections.exists(WEAVIATE_COLLECTION):
                client.collections.delete(WEAVIATE_COLLECTION)
            collection = client.collections.create(
                name=WEAVIATE_COLLECTION,
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="doc_type", data_type=DataType.TEXT),
                ],
            )
            with collection.batch.dynamic() as batch:
                for chunk in chunks:
                    metadata = chunk.get("metadata", {})
                    batch.add_object(
                        properties={
                            "content": chunk["content"],
                            "source": metadata.get("source", ""),
                            "doc_type": metadata.get("type", ""),
                        },
                        vector=chunk["embedding"],
                    )
        finally:
            client.close()
        return True
    except Exception as exc:
        print(f"Weaviate indexing skipped: {exc}")
        return False


def index_to_vectorstore(chunks: list[dict]):
    """
    Persist chunks locally and optionally sync to Weaviate.
    """
    _save_local_index(chunks)
    weaviate_synced = _try_index_weaviate(chunks)
    return {"local_index": str(INDEX_DIR), "weaviate_synced": weaviate_synced}


def load_local_index() -> list[dict]:
    if not CHUNKS_PATH.exists() or not EMBEDDINGS_PATH.exists():
        run_pipeline()
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    embeddings = json.loads(EMBEDDINGS_PATH.read_text(encoding="utf-8"))
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    return chunks


def run_pipeline():
    """Run load -> chunk -> embed -> index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\nLoaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks")

    result = index_to_vectorstore(chunks)
    print(f"Indexed locally at {result['local_index']}")
    print(f"Weaviate synced: {result['weaviate_synced']}")
    return result


if __name__ == "__main__":
    run_pipeline()
