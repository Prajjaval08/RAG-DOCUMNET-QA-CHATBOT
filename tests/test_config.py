import pytest

from rag_chatbot.config import AppConfig, load_config


def test_defaults_match_documented_values():
    cfg = AppConfig()
    assert cfg.llm_provider == "ollama"
    assert cfg.chunk_size == 1000
    assert cfg.chunk_overlap == 150
    assert cfg.retrieval_top_k == 4
    assert cfg.max_upload_mb == 20


def test_invalid_provider_rejected():
    with pytest.raises(ValueError):
        AppConfig(llm_provider="bogus")


def test_invalid_embedding_provider_rejected():
    with pytest.raises(ValueError):
        AppConfig(embedding_provider="bogus")


def test_temperature_range_enforced():
    with pytest.raises(ValueError):
        AppConfig(llm_temperature=2.5)
    with pytest.raises(ValueError):
        AppConfig(llm_temperature=-0.1)


def test_chunk_size_must_exceed_overlap():
    with pytest.raises(ValueError):
        AppConfig(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        AppConfig(chunk_size=100, chunk_overlap=150)


def test_top_k_at_least_one():
    with pytest.raises(ValueError):
        AppConfig(retrieval_top_k=0)


def test_max_upload_bytes():
    assert AppConfig(max_upload_mb=20).max_upload_bytes == 20 * 1024 * 1024


def test_load_config_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "50")
    cfg = load_config()
    assert cfg.llm_provider == "openai"
    assert cfg.llm_model == "gpt-4o-mini"
    assert cfg.chunk_size == 500


def test_load_config_bad_int_is_actionable(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "not-a-number")
    with pytest.raises(ValueError, match="CHUNK_SIZE"):
        load_config()
