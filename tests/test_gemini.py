import json

import pytest

from rag_chatbot.llm import GeminiChatProvider, create_llm_provider


def _fake_response(body):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(body).encode()

    return FakeResponse()


def test_gemini_factory_and_mocked_generate(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    provider = create_llm_provider("gemini", "gemini-3.8-flash")
    assert isinstance(provider, GeminiChatProvider)
    body = {
        "candidates": [
            {"content": {"parts": [{"text": "Paris, "}, {"text": "per the notes."}]}}
        ]
    }
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode())
        return _fake_response(body)

    monkeypatch.setattr(
        "rag_chatbot.llm.urllib.request.urlopen", fake_urlopen
    )
    assert provider.generate("What is the capital?") == "Paris, per the notes."
    assert "gemini-3.8-flash" in captured["url"]
    assert "test-key" not in json.dumps(captured["body"])
    assert captured["body"]["contents"][0]["parts"][0]["text"] == "What is the capital?"


def test_gemini_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiChatProvider("gemini-3.8-flash").generate("hello")


def test_gemini_accepts_google_api_key_fallback(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "fallback-key")
    monkeypatch.setattr(
        "rag_chatbot.llm.urllib.request.urlopen",
        lambda req, timeout=None: _fake_response(
            {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
        ),
    )
    assert GeminiChatProvider("gemini-3.8-flash").generate("hi") == "ok"


def test_gemini_rejects_unsafe_model_name(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="Invalid Gemini model"):
        GeminiChatProvider("evil/../../x?y=1").generate("hi")


def test_gemini_retries_once_on_timeout_then_succeeds(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls = {"n": 0}

    def flaky_urlopen(request, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("The read operation timed out")
        return _fake_response(
            {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
        )

    monkeypatch.setattr("rag_chatbot.llm.urllib.request.urlopen", flaky_urlopen)
    monkeypatch.setattr("rag_chatbot.llm.time.sleep", lambda _: None)
    assert GeminiChatProvider("gemini-3.8-flash").generate("hi") == "ok"
    assert calls["n"] == 2


def test_gemini_retries_once_on_503_then_succeeds(monkeypatch):
    import io
    import urllib.error

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls = {"n": 0}

    def flaky_urlopen(request, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            body = b'{"error": {"code": 503, "message": "high demand"}}'
            raise urllib.error.HTTPError(
                request.full_url, 503, "Unavailable", {}, io.BytesIO(body)
            )
        return _fake_response(
            {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
        )

    monkeypatch.setattr("rag_chatbot.llm.urllib.request.urlopen", flaky_urlopen)
    monkeypatch.setattr("rag_chatbot.llm.time.sleep", lambda _: None)
    assert GeminiChatProvider("gemini-3.8-flash").generate("hi") == "ok"
    assert calls["n"] == 2


def test_gemini_timeout_twice_raises_clear_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def always_timeout(request, timeout=None):
        raise TimeoutError("The read operation timed out")

    monkeypatch.setattr("rag_chatbot.llm.urllib.request.urlopen", always_timeout)
    monkeypatch.setattr("rag_chatbot.llm.time.sleep", lambda _: None)
    with pytest.raises(ConnectionError, match="timed out after 2 attempts"):
        GeminiChatProvider("gemini-3.8-flash").generate("hi")
