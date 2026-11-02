"""Provider-agnostic chat generation with lazy imports.

API keys are read only from the environment, never from UI input or code,
and are never logged. Each provider's SDK / HTTP call lives behind a
factory so swapping the generation provider never touches ingestion or
retrieval. Any object with ``generate(prompt) -> str`` works as a provider
(the tests use such a stub).
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from typing import Protocol


class ChatProvider(Protocol):
    def generate(self, prompt: str) -> str: ...


def _require_env(key: str, provider: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise ValueError(
            f"{key} is not set. Configure it in your environment (via .env) "
            f"to use the {provider} provider, or pick a different provider."
        )
    return value


class OpenAIChatProvider:
    def __init__(self, model: str, temperature: float = 0.1):
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        api_key = _require_env("OPENAI_API_KEY", "OpenAI")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI generation requires the 'openai' package. "
                "Install it with: pip install -r requirements.txt"
            ) from exc
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip()


class AnthropicChatProvider:
    def __init__(self, model: str, temperature: float = 0.1):
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        api_key = _require_env("ANTHROPIC_API_KEY", "Anthropic")
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic generation requires the 'anthropic' package. "
                "Install it with: pip install -r requirements.txt"
            ) from exc
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in resp.content if block.type == "text"
        ).strip()


class OllamaChatProvider:
    def __init__(
        self,
        model: str,
        temperature: float = 0.1,
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.temperature = temperature
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Could not reach Ollama at {self.base_url}. Is Ollama "
                f"running, and has the model {self.model!r} been pulled "
                f"(e.g. `ollama pull {self.model}`)? Original error: {exc}"
            ) from exc
        return (body.get("response") or "").strip()


class HuggingFaceChatProvider:
    def __init__(self, model: str, temperature: float = 0.1):
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        token = _require_env("HUGGINGFACEHUB_API_TOKEN", "Hugging Face")
        payload = json.dumps(
            {
                "inputs": prompt,
                "parameters": {
                    "temperature": self.temperature,
                    "max_new_tokens": 512,
                    "return_full_text": False,
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"https://api-inference.huggingface.co/models/{self.model}",
            data=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ConnectionError(
                f"Hugging Face Inference request failed ({exc.code}). Check "
                "that the model id is a supported text-generation model and "
                "your token has access. Original error: "
                f"{exc.read().decode('utf-8', errors='replace')[:300]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Could not reach Hugging Face Inference: {exc}") from exc
        if isinstance(body, list) and body and isinstance(body[0], dict):
            return (body[0].get("generated_text") or "").strip()
        if isinstance(body, dict) and body.get("generated_text"):
            return str(body["generated_text"]).strip()
        raise ValueError(f"Unexpected Hugging Face Inference response: {str(body)[:300]}")


class GeminiChatProvider:
    """Google Gemini via AI Studio (free tier, no credit card required).

    Uses the REST generateContent endpoint with the key as a query
    parameter, which is how the Gemini API authenticates key-based
    calls. The key is read from GEMINI_API_KEY, falling back to
    GOOGLE_API_KEY (the name Google's own client libraries use).
    The key is never logged; only HTTP status / response-body excerpts
    are surfaced on failure, and the response body never contains it.
    """

    def __init__(self, model: str, temperature: float = 0.1):
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set (GOOGLE_API_KEY is also accepted). "
                "Get a free key at Google AI Studio (aistudio.google.com), "
                "or pick a different provider."
            )
        payload = json.dumps(
            {
                "contents": [{"parts": [{"text": prompt}]}],
                # Cap the answer length: grounded answers to a document
                # question should be short, and a long free-form reply
                # is the most common reason a free-tier call runs long.
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": 512,
                },
            }
        ).encode("utf-8")
        # The model name is user-configured; only allow a safe shape so
        # a crafted model string cannot turn this into an arbitrary URL.
        if not self.model or "/" in self.model or "?" in self.model or "&" in self.model:
            raise ValueError(f"Invalid Gemini model name {self.model!r}.")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={api_key}"
        )
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # Retry once on a read timeout: on Streamlit Cloud free tier
        # a first call after waking up can stall even when the model
        # and key are correct, and a bare TimeoutError used to escape
        # here and surface in the UI as just "The read operation
        # timed out". Retrying also keeps a transient stall from
        # looking like a permanent failure.
        body = None
        for attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=180) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                # 429 (rate limit) and 503 (model overloaded) are
                # transient: Google itself says spikes are usually
                # temporary, so retry once instead of failing at once.
                if exc.code in (429, 503) and attempt == 0:
                    time.sleep(5)
                    continue
                raise ConnectionError(
                    f"Gemini request failed ({exc.code}). Check the model name "
                    f"and that the key has access to it. Original error: "
                    f"{exc.read().decode('utf-8', errors='replace')[:300]}"
                ) from exc
            except (TimeoutError, socket.timeout) as exc:
                if attempt == 1:
                    raise ConnectionError(
                        "Gemini request timed out after 2 attempts. "
                        "The free-tier model may be slow right now — "
                        "try asking again, or try a shorter question."
                    ) from exc
                time.sleep(2)
            except urllib.error.URLError as exc:
                if "timed out" in str(exc).lower() and attempt == 0:
                    time.sleep(2)
                    continue
                raise ConnectionError(
                    f"Could not reach the Gemini API: {exc}"
                ) from exc
        assert body is not None
        try:
            candidates = body["candidates"]
            parts = candidates[0]["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                f"Unexpected Gemini response: {str(body)[:300]}"
            ) from exc
        return text.strip()


def create_llm_provider(
    provider: str,
    model: str,
    temperature: float = 0.1,
    ollama_base_url: str = "http://localhost:11434",
) -> ChatProvider:
    if provider == "openai":
        return OpenAIChatProvider(model, temperature)
    if provider == "anthropic":
        return AnthropicChatProvider(model, temperature)
    if provider == "ollama":
        return OllamaChatProvider(model, temperature, ollama_base_url)
    if provider == "huggingface":
        return HuggingFaceChatProvider(model, temperature)
    if provider == "gemini":
        return GeminiChatProvider(model, temperature)
    raise ValueError(
        f"Unknown LLM provider {provider!r}; expected one of: "
        "openai, anthropic, ollama, huggingface, gemini."
    )
