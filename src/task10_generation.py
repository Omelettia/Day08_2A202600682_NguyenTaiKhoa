"""
Task 10 - Generation with citations.

Gemini is the preferred LLM for this repo when GEMINI_API_KEY is configured.
OpenAI remains supported as an alternative. If no LLM key is available, the
module returns an extractive answer with citations so the RAG pipeline remains
testable and demoable.
"""

from __future__ import annotations

import os
import re

from dotenv import load_dotenv

load_dotenv()

try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:
    from task9_retrieval_pipeline import retrieve

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """Answer the question in Vietnamese using only the provided context.
Every factual claim must include a citation in square brackets using the source
label shown in the context, for example [105_2021_ND-CP_496664.md, 2021].
If the context does not explicitly support the answer, say:
"Tôi không thể xác minh thông tin này từ nguồn hiện có."
"""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Avoid lost-in-the-middle by placing strong chunks at the beginning and end.

    Example: [1, 2, 3, 4, 5] -> [1, 3, 5, 4, 2]
    """
    if len(chunks) <= 2:
        return chunks
    front = chunks[::2]
    back = chunks[1::2][::-1]
    return front + back


def _citation_label(chunk: dict, index: int) -> str:
    metadata = chunk.get("metadata", {})
    source = metadata.get("source") or metadata.get("path") or f"Source {index}"
    text = f"{source} {chunk.get('content', '')}"
    year_match = re.search(r"\b(20\d{2}|19\d{2})\b", text)
    year = year_match.group(1) if year_match else "n.d."
    return f"{source}, {year}"


def format_context(chunks: list[dict]) -> str:
    """
    Format chunks with explicit source labels for citation.
    """
    context_parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        label = _citation_label(chunk, index)
        doc_type = metadata.get("type", "unknown")
        context_parts.append(
            f"[Document {index} | Citation: {label} | Type: {doc_type}]\n"
            f"{chunk.get('content', '')}"
        )
    return "\n\n---\n\n".join(context_parts)


def _build_prompt(query: str, context: str, conversation_context: str | None = None) -> str:
    history_block = ""
    if conversation_context:
        history_block = (
            "\n\nConversation context for resolving follow-up wording only. "
            "Do not cite this conversation as evidence:\n"
            f"{conversation_context}"
        )
    return f"{SYSTEM_PROMPT}{history_block}\n\nContext:\n{context}\n\nQuestion: {query}"


def _generate_with_gemini(prompt: str) -> str | None:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return None
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"temperature": TEMPERATURE, "top_p": TOP_P},
        )
        return getattr(response, "text", None)
    except Exception as exc:
        print(f"Gemini generation skipped: {exc}")
        return None


def _generate_with_openai(prompt: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "sk-xxx":
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content
    except Exception as exc:
        print(f"OpenAI generation skipped: {exc}")
        return None


def _extractive_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    sentences = []
    for index, chunk in enumerate(chunks[:3], 1):
        content = " ".join(chunk.get("content", "").split())
        parts = re.split(r"(?<=[.!?])\s+", content)
        sentence = next((part for part in parts if len(part) > 40), content[:260])
        citation = _citation_label(chunk, index)
        sentences.append(f"{sentence.strip()} [{citation}]")

    intro = f"Dựa trên các nguồn truy xuất được cho câu hỏi: {query}"
    return intro + "\n\n" + "\n\n".join(sentences)


def generate_with_citation(
    query: str,
    top_k: int = TOP_K,
    score_threshold: float | None = None,
    use_reranking: bool = True,
    use_llm: bool = True,
    conversation_context: str | None = None,
) -> dict:
    """
    End-to-end RAG generation with citations.
    """
    retrieve_kwargs = {"top_k": top_k, "use_reranking": use_reranking}
    if score_threshold is not None:
        retrieve_kwargs["score_threshold"] = score_threshold
    chunks = retrieve(query, **retrieve_kwargs)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    prompt = _build_prompt(query, context, conversation_context)

    answer = None
    if use_llm:
        answer = _generate_with_gemini(prompt)
        if not answer:
            answer = _generate_with_openai(prompt)
    if not answer:
        answer = _extractive_answer(query, reordered)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "none") if chunks else "none",
    }


if __name__ == "__main__":
    result = generate_with_citation("Hình phạt cho tội tàng trữ trái phép chất ma túy?")
    print(result["answer"])
    print(f"Sources: {len(result['sources'])} via {result['retrieval_source']}")
