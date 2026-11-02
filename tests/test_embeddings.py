import pytest

from rag_chatbot.embeddings import (
    LocalSentenceTransformerEmbedder,
    OpenAIEmbedder,
    create_embedder,
)


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError):
        create_embedder("bogus", "model")


def test_local_embedder_creation_does_not_load_model():
    # Lazy import: constructing must not require sentence-transformers.
    embedder = LocalSentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    assert embedder.model_name.endswith("all-MiniLM-L6-v2")


def test_openai_embedder_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbedder().embed_query("hello")
