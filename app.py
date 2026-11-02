"""Streamlit UI for the RAG Document Q&A Chatbot.

This module orchestrates the src/rag_chatbot modules; it contains no
retrieval, chunking, embedding, or generation logic of its own. API keys
are never requested, displayed, or logged here — providers read them from
the environment, and the sidebar only shows whether a key is configured.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_chatbot.chunking import make_chunks
from rag_chatbot.config import (
    DEFAULT_LLM_MODELS,
    AppConfig,
    load_config,
)
from rag_chatbot.embeddings import create_embedder
from rag_chatbot.llm import create_llm_provider
from rag_chatbot.loaders import load_file, load_url, sha256_hex, validate_upload
from rag_chatbot.rag import INSUFFICIENT_ANSWER, answer_question
from rag_chatbot.vector_store import (
    EmbeddingProfileMismatchError,
    FAISSVectorStore,
)

LLM_MODELS = {
    "openai": ["gpt-4o-mini", "gpt-4o"],
    "anthropic": ["claude-haiku-4-5-20251001", "claude-sonnet-4-5-20250929"],
    "ollama": ["llama3.2", "llama3.1", "mistral"],
    "huggingface": ["meta-llama/Llama-3.2-1B-Instruct"],
    "gemini": ["gemini-3.8-flash"],
}

KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "huggingface": "HUGGINGFACEHUB_API_TOKEN",
    "gemini": "GEMINI_API_KEY",
    "ollama": None,
}


@st.cache_resource(show_spinner=False)
def get_embedder(provider: str, model: str, ollama_base_url: str = "http://localhost:11434"):
    """Cached so the heavy embedding model is built once per process."""
    return create_embedder(provider, model, ollama_base_url)


def get_store(config: AppConfig) -> FAISSVectorStore:
    return FAISSVectorStore.load(
        config.index_dir,
        config.embedding_provider,
        config.effective_embedding_model,
    )


_SECRET_ENV_KEYS = (
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_TEMPERATURE",
    "RETRIEVAL_TOP_K",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "OPENAI_EMBEDDING_MODEL",
    "OLLAMA_EMBEDDING_MODEL",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "MAX_UPLOAD_MB",
    "OLLAMA_BASE_URL",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "HUGGINGFACEHUB_API_TOKEN",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)


def _apply_streamlit_secrets() -> None:
    """Copy Streamlit secrets into os.environ when they are set.

    Streamlit Community Cloud stores secrets in ``st.secrets``, not in
    the process environment, while this app (and its provider modules)
    read configuration from ``os.environ``. Without this bridge, a
    Cloud deployment would silently ignore the keys set in its
    dashboard. Local runs without a secrets file are unaffected:
    any failure to read secrets is ignored and .env / environment
    values win, since existing values are never overwritten.
    """
    try:
        secrets = st.secrets
    except Exception:  # noqa: BLE001 - no secrets file is a normal local case
        return
    try:
        for key in _SECRET_ENV_KEYS:
            if key in secrets and key not in os.environ:
                os.environ[key] = str(secrets[key])
    except Exception:  # noqa: BLE001 - secrets must never break app startup
        return


def main() -> None:
    _apply_streamlit_secrets()
    st.set_page_config(page_title="RAG Document Q&A Chatbot", layout="wide")
    st.title("📄 RAG Document Q&A Chatbot")
    st.caption(
        "Upload your documents, index them locally, and ask questions "
        "grounded in retrieved passages. Scanned PDFs require OCR, which "
        "is not included — they cannot be indexed."
    )

    base = load_config()

    with st.sidebar:
        st.header("Settings")
        llm_provider = st.selectbox(
            "LLM provider",
            ["openai", "anthropic", "ollama", "huggingface", "gemini"],
            index=["openai", "anthropic", "ollama", "huggingface", "gemini"].index(base.llm_provider),
        )
        default_model = DEFAULT_LLM_MODELS[llm_provider]
        llm_model = st.text_input("Model", value=default_model)
        temperature = st.slider("Temperature", 0.0, 2.0, float(base.llm_temperature), 0.05)
        top_k = st.slider("Top-k retrieval", 1, 10, int(base.retrieval_top_k), 1)
        embedding_provider = st.selectbox(
            "Embedding provider",
            ["local", "openai", "ollama"],
            index=["local", "openai", "ollama"].index(base.embedding_provider)
            if base.embedding_provider in ("local", "openai", "ollama")
            else 0,
        )
        embedding_model = st.text_input(
            "Embedding model",
            value=(
                base.embedding_model
                if embedding_provider == "local"
                else base.openai_embedding_model
                if embedding_provider == "openai"
                else base.ollama_embedding_model
            ),
        )
        chunk_size = st.number_input("Chunk size", 100, 4000, int(base.chunk_size), 50)
        chunk_overlap = st.number_input("Chunk overlap", 0, 1000, int(base.chunk_overlap), 10)
        max_upload_mb = st.number_input("Max upload (MB)", 1, 100, int(base.max_upload_mb), 1)

        key_env = KEY_ENV[llm_provider]
        if key_env is None:
            st.success("Ollama: local — no API key needed.")
        elif os.environ.get(key_env):
            st.success(f"{llm_provider}: API key configured (value hidden).")
        else:
            st.warning(f"{llm_provider}: no API key configured ({key_env} not set).")

        st.divider()
        st.header("Knowledge base")
        config = AppConfig(
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_temperature=float(temperature),
            retrieval_top_k=int(top_k),
            embedding_provider=embedding_provider,
            embedding_model=embedding_model if embedding_provider == "local" else base.embedding_model,
            openai_embedding_model=embedding_model if embedding_provider == "openai" else base.openai_embedding_model,
            ollama_embedding_model=embedding_model if embedding_provider == "ollama" else base.ollama_embedding_model,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            max_upload_mb=int(max_upload_mb),
            ollama_base_url=base.ollama_base_url,
            index_dir=base.index_dir,
        )
        try:
            store = get_store(config)
            st.metric("Indexed chunks", len(store.chunks))
            st.metric("Indexed documents", len(store.doc_hashes))
            st.caption(f"Embeddings: {config.embedding_provider}/{config.effective_embedding_model}")
        except EmbeddingProfileMismatchError as exc:
            st.error(str(exc))
            store = None
        except Exception as exc:  # noqa: BLE001 - UI surfaces errors, never swallows
            st.error(f"Could not load the index: {exc}")
            store = None

        st.divider()
        st.header("Danger zone")
        confirm_clear = st.checkbox("I understand clearing deletes the local index.")
        if st.button("Clear index", disabled=not confirm_clear):
            if store is not None:
                store.clear()
            else:
                FAISSVectorStore(config.index_dir, config.embedding_provider, config.effective_embedding_model).clear()
            st.success("Index cleared. Re-upload and re-index to ask questions.")
            st.rerun()

    if store is None:
        st.info("Clear the index (sidebar) and re-index to continue with the new embedding profile.")
        return

    uploaded = st.file_uploader(
        "Upload documents (PDF, DOCX, TXT, MD, CSV)",
        type=["pdf", "docx", "txt", "md", "csv"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Index documents"):
        try:
            embedder = get_embedder(config.embedding_provider, config.effective_embedding_model, config.ollama_base_url)
        except Exception as exc:  # noqa: BLE001 - UI surfaces all errors to the user
            st.error(f"Could not load the embedding model: {exc}")
            return
        added_docs, skipped, errors = 0, 0, []
        for f in uploaded:
            data = f.getvalue()
            try:
                validate_upload(f.name, len(data), config.max_upload_bytes)
                source_hash = sha256_hex(data)
                if store.has_document(source_hash):
                    skipped += 1
                    continue
                chunks = make_chunks(
                    load_file(f.name, data), source_hash, config.chunk_size, config.chunk_overlap
                )
                added = store.add_chunks(chunks, embedder, source_hash)
                if added:
                    added_docs += 1
                else:
                    skipped += 1
            except Exception as exc:  # noqa: BLE001 - UI surfaces all errors to the user
                errors.append(f"{f.name}: {exc}")
        if added_docs:
            try:
                store.save()
                st.success(f"Indexed {added_docs} new document(s).")
            except Exception as exc:  # noqa: BLE001 - UI surfaces all errors to the user
                st.error(f"Indexing succeeded but saving failed: {exc}")
        if skipped:
            st.info(f"Skipped {skipped} duplicate/empty file(s) (SHA-256 match).")
        for err in errors:
            st.error(f"Failed to index — {err}")
        st.rerun()

    url = st.text_input(
        "Or index a webpage (http/https only)",
        placeholder="https://example.com/article",
    )
    if url and st.button("Index URL"):
        try:
            embedder = get_embedder(config.embedding_provider, config.effective_embedding_model, config.ollama_base_url)
            docs = load_url(url)
            source_hash = sha256_hex("\n".join(d.text for d in docs).encode("utf-8"))
            if store.has_document(source_hash):
                st.info("Skipped duplicate URL content (SHA-256 match).")
            else:
                chunks = make_chunks(docs, source_hash, config.chunk_size, config.chunk_overlap)
                if store.add_chunks(chunks, embedder, source_hash):
                    store.save()
                    st.success("Indexed URL content.")
                    st.rerun()
                else:
                    st.info("No chunks produced from this URL.")
        except Exception as exc:  # noqa: BLE001 - UI surfaces all errors to the user
            st.error(f"Failed to index URL — {exc}")

    st.divider()
    st.header("Ask a question")
    if "history" not in st.session_state:
        st.session_state.history = []
    question = st.text_input("Your question", placeholder="What does the document say about…?")
    if st.button("Ask") and question.strip():
        if store.is_empty:
            st.warning("The knowledge base is empty. Upload and index documents first.")
        else:
            try:
                embedder = get_embedder(config.embedding_provider, config.effective_embedding_model, config.ollama_base_url)
                llm = create_llm_provider(
                    config.llm_provider, config.llm_model, config.llm_temperature, config.ollama_base_url
                )
                result = answer_question(question, store, embedder, llm, top_k=config.retrieval_top_k)
            except Exception as exc:  # noqa: BLE001 - UI surfaces all errors to the user
                st.error(f"Could not answer: {exc}")
                result = None
            if result is not None:
                st.session_state.history.append((question, result))
                if result.answer == INSUFFICIENT_ANSWER:
                    st.warning(result.answer)
                else:
                    st.markdown(result.answer)
                st.subheader("Sources")
                if not result.sources:
                    st.caption("No sources retrieved.")
                for src in result.sources:
                    with st.expander(src.label()):
                        st.write(src.excerpt)

    if st.session_state.history:
        st.divider()
        st.subheader("Conversation history")
        for q, r in reversed(st.session_state.history):
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {r.answer}")


if __name__ == "__main__":
    main()
