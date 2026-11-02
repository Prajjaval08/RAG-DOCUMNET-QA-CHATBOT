import pytest

from rag_chatbot.chunking import _fallback_split, make_chunks, split_text
from rag_chatbot.loaders import RawDocument


def test_empty_text_yields_no_chunks():
    assert split_text("", 100, 10) == []
    assert split_text("   ", 100, 10) == []


def test_overlap_must_be_less_than_size():
    with pytest.raises(ValueError):
        split_text("hello world", 10, 10)


def test_fallback_split_is_deterministic_and_overlapping():
    text = "abcdefghij" * 10
    first = _fallback_split(text, 20, 5)
    second = _fallback_split(text, 20, 5)
    assert first == second
    assert all(len(c) <= 20 for c in first)
    # Overlap: the tail of chunk 0 reappears at the head of chunk 1.
    assert first[0][-5:] == first[1][:5]


def test_make_chunks_preserves_metadata_and_ids():
    docs = [
        RawDocument(filename="a.pdf", text="Cats are great animals. " * 20, page=3),
        RawDocument(filename="a.pdf", text="Dogs are loyal. " * 20, page=4),
    ]
    chunks = make_chunks(docs, "a" * 64, 100, 20)
    assert chunks
    assert all(c.filename == "a.pdf" for c in chunks)
    assert all(c.source_hash == "a" * 64 for c in chunks)
    assert chunks[0].page == 3
    assert len({c.id for c in chunks}) == len(chunks)  # unique ids
    assert chunks[0].id.startswith("a" * 12 + "-")
    assert chunks[0].id == "a" * 12 + "-0000"


def test_chunking_roundtrip_covers_text():
    text = "word " * 200
    chunks = split_text(text, 100, 20)
    assert len(chunks) > 1
    assert "word" in chunks[0]
