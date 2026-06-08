"""
Streamlit RAG chatbot for the group project.

Run:
    streamlit run app.py
"""

from __future__ import annotations

from textwrap import shorten

import streamlit as st

from src.task10_generation import generate_with_citation


st.set_page_config(page_title="Drug Law RAG", page_icon="RAG", layout="wide")


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _conversation_context(turns: int) -> str | None:
    if turns <= 0:
        return None
    history = st.session_state.messages[-turns * 2 :]
    if not history:
        return None
    return "\n".join(f"{item['role']}: {item['content']}" for item in history)


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        st.info("No source chunks returned.")
        return
    for index, source in enumerate(sources, 1):
        metadata = source.get("metadata", {})
        title = metadata.get("source") or metadata.get("path") or f"Source {index}"
        score = float(source.get("score", 0.0))
        mode = source.get("source", "unknown")
        with st.expander(f"{index}. {title} | {mode} | score={score:.3f}"):
            st.caption(f"Type: {metadata.get('type', 'unknown')} | Path: {metadata.get('path', 'n/a')}")
            st.write(source.get("content", ""))


_init_state()

with st.sidebar:
    st.header("Retrieval")
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5)
    score_threshold = st.slider("Fallback threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    use_reranking = st.toggle("Use reranking", value=True)
    memory_turns = st.slider("Memory turns", min_value=0, max_value=5, value=2)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.title("Vietnam Drug Law & News RAG")
st.caption("Answers use the completed personal pipeline: retrieval, reranking, PageIndex fallback, and citation generation.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            _render_sources(message["sources"])

prompt = st.chat_input("Ask about Vietnamese drug law or related artist news...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving sources and generating an answer..."):
            result = generate_with_citation(
                prompt,
                top_k=top_k,
                score_threshold=score_threshold,
                use_reranking=use_reranking,
                conversation_context=_conversation_context(memory_turns),
            )
        answer = result["answer"]
        st.markdown(answer)
        st.caption(
            "Retrieval source: "
            f"{result.get('retrieval_source', 'unknown')} | "
            f"{len(result.get('sources', []))} source chunks"
        )
        _render_sources(result.get("sources", []))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": result.get("sources", []),
            "preview": shorten(answer, width=120),
        }
    )
