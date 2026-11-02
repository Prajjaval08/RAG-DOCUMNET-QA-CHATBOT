import json

import pytest
from conftest import StubEmbedder

from rag_chatbot.embeddings import OllamaEmbedder, create_embedder
from rag_chatbot.evaluation import (
    answer_keyword_coverage,
    evaluate_cases,
    retrieval_recall,
)
from rag_chatbot.loaders import extract_text_from_html, load_url
from rag_chatbot.vector_store import FAISSVectorStore
from rag_chatbot.watch import scan_directory


def test_html_extraction_skips_script_and_style():
    html = (
        "<html><head><style>.x{color:red}</style>"
        "<script>alert(1)</script></head>"
        "<body><h1>Cats</h1><p>Cats are small animals.</p></body></html>"
    )
    text = extract_text_from_html(html)
    assert "Cats are small animals" in text
    assert "alert" not in text
    assert "color" not in text


def test_load_url_rejects_non_http():
    with pytest.raises(ValueError, match="only http"):
        load_url("file:///etc/passwd")
    with pytest.raises(ValueError, match="only http"):
        load_url("https://")


def test_load_url_html_and_plain_text_via_fake_fetcher():
    html_docs = load_url(
        "https://example.com/cats",
        fetcher=lambda url: ("text/html", b"<p>Cats are small.</p>"),
    )
    assert html_docs[0].filename == "https://example.com/cats"
    assert "Cats are small" in html_docs[0].text

    txt_docs = load_url(
        "https://example.com/notes.txt",
        fetcher=lambda url: ("text/plain", b"Plain notes."),
    )
    assert "Plain notes" in txt_docs[0].text


def test_load_url_rejects_unsupported_content_type():
    with pytest.raises(ValueError, match="Unsupported content type"):
        load_url(
            "https://example.com/f.bin",
            fetcher=lambda url: ("application/octet-stream", b"\x00\x01"),
        )


def test_ollama_embedder_factory_and_mocked_http(monkeypatch):
    embedder = create_embedder("ollama", "nomic-embed-text")
    assert isinstance(embedder, OllamaEmbedder)
    assert embedder.model_name == "nomic-embed-text"

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"embedding": [0.1, 0.2, 0.3]}).encode()

    monkeypatch.setattr(
        "rag_chatbot.embeddings.urllib.request.urlopen",
        lambda req, timeout=None: FakeResponse(),
    )
    assert embedder.embed_query("hello") == [0.1, 0.2, 0.3]


def test_evaluation_metrics():
    assert retrieval_recall(["a.txt", "b.txt"], ["a.txt", "c.txt"]) == 0.5
    assert retrieval_recall([], []) == 1.0
    assert answer_keyword_coverage("Cats are in France", ["cats", "france"]) == 1.0
    assert answer_keyword_coverage("Cats only", ["cats", "france"]) == 0.5
    report = evaluate_cases(
        [{"question": "q", "expected_sources": ["a.txt"], "expected_keywords": ["cats"]}],
        retrieve=lambda q: ["a.txt"],
        answer=lambda q: "Cats are discussed.",
    )
    assert report.mean_recall == 1.0
    assert report.hit_rate == 1.0
    assert report.mean_keyword_coverage == 1.0


def test_watch_scans_only_new_files(tmp_path):
    watch_dir = tmp_path / "incoming"
    watch_dir.mkdir()
    (watch_dir / "a.txt").write_text("Cats are small animals.")
    (watch_dir / "b.txt").write_text("Dogs are loyal animals.")
    store = FAISSVectorStore(tmp_path / "index", "local", "stub-v1")
    embedder = StubEmbedder()
    stats = scan_directory(watch_dir, store, embedder)
    assert stats.added_documents == 2
    assert stats.errors == []
    # Second run: both are duplicates, nothing new added.
    stats2 = scan_directory(watch_dir, store, embedder)
    assert stats2.added_documents == 0
    assert stats2.skipped_duplicates == 2
