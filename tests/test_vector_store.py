import pytest
from conftest import StubEmbedder

from rag_chatbot.chunking import Chunk
from rag_chatbot.vector_store import (
    EmbeddingProfileMismatchError,
    FAISSVectorStore,
)


def _store(tmp_path, provider="local", model="stub-v1"):
    return FAISSVectorStore(tmp_path / "index", provider, model)


def _chunks():
    return [
        Chunk(id="hash-0000", text="Cats are small animals.", filename="a.txt", source_hash="hash1"),
        Chunk(id="hash-0001", text="Dogs are loyal animals.", filename="a.txt", source_hash="hash1"),
        Chunk(id="hash-0002", text="Paris is in France.", filename="b.txt", source_hash="hash2"),
    ]


def test_add_search_and_persist_roundtrip(tmp_path):
    embedder = StubEmbedder()
    store = _store(tmp_path)
    chunks = _chunks()
    assert store.add_chunks(chunks[:2], embedder, "hash1") == 2
    store.save()

    reloaded = FAISSVectorStore.load(tmp_path / "index", "local", "stub-v1")
    assert len(reloaded.chunks) == 2
    assert reloaded.has_document("hash1")
    results = reloaded.search(embedder.embed_query("tell me about cats"), 1)
    assert len(results) == 1
    assert "Cats" in results[0].chunk.text


def test_deduplication_by_hash(tmp_path):
    embedder = StubEmbedder()
    store = _store(tmp_path)
    chunks = _chunks()
    assert store.add_chunks(chunks[:2], embedder, "hash1") == 2
    assert store.add_chunks(chunks[:2], embedder, "hash1") == 0  # duplicate
    assert len(store.chunks) == 2


def test_embedding_profile_change_requires_rebuild(tmp_path):
    embedder = StubEmbedder()
    store = _store(tmp_path)
    store.add_chunks(_chunks()[:1], embedder, "hash1")
    store.save()
    with pytest.raises(EmbeddingProfileMismatchError, match="different embedding"):
        FAISSVectorStore.load(tmp_path / "index", "openai", "other-model")


def test_search_empty_store_returns_empty(tmp_path):
    store = _store(tmp_path)
    assert store.search([0.0, 0.0, 0.0, 0.0], 4) == []


def test_clear_removes_persisted_index(tmp_path):
    embedder = StubEmbedder()
    store = _store(tmp_path)
    store.add_chunks(_chunks()[:1], embedder, "hash1")
    store.save()
    assert (tmp_path / "index" / "index.faiss").exists()
    store.clear()
    assert not (tmp_path / "index").exists()
    assert store.is_empty
