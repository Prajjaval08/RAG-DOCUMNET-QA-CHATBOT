"""Embedding backends with lazy imports.

Creating an embedder never loads a model or SDK: heavy imports happen on
the first ``embed_*`` call, so a missing optional dependency produces a
focused error and tests run without model weights. Any object with
``embed_documents(texts) -> list[list[float]]`` and
``embed_query(text) -> list[float]`` can be used in place of these classes
(the tests use such a stub).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class LocalSentenceTransformerEmbedder:
    """Default local embeddings (sentence-transformers/all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "Local embeddings require 'sentence-transformers' (and "
                    "its torch dependency). Install it with: "
                    "pip install -r requirements.txt"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(texts, batch_size=32, normalize_embeddings=False)
        return [list(map(float, v)) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OpenAIEmbedder:
    """Optional OpenAI embeddings (requires OPENAI_API_KEY)."""

    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name

    def _client(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Set it in your environment (via "
                ".env) or switch EMBEDDING_PROVIDER back to 'local'."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI embeddings require the 'openai' package. "
                "Install it with: pip install -r requirements.txt"
            ) from exc
        return OpenAI(api_key=api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._client()
        response = client.embeddings.create(input=texts, model=self.model_name)
        return [list(map(float, item.embedding)) for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OllamaEmbedder:
    """Local Ollama embeddings, e.g. nomic-embed-text (no hosted key).

    Uses Ollama's POST /api/embeddings endpoint, one prompt per call.
    The model must already be pulled, e.g. ``ollama pull nomic-embed-text``.
    """

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for text in texts:
            payload = json.dumps(
                {"model": self.model_name, "prompt": text}
            ).encode("utf-8")
            request = urllib.request.Request(
                f"{self.base_url}/api/embeddings",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
            except urllib.error.URLError as exc:
                raise ConnectionError(
                    f"Could not reach Ollama at {self.base_url}. Is Ollama "
                    f"running, and has the model {self.model_name!r} been "
                    f"pulled (e.g. `ollama pull {self.model_name}`)? "
                    f"Original error: {exc}"
                ) from exc
            embedding = body.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise ValueError(
                    f"Unexpected Ollama embeddings response: {str(body)[:300]}"
                )
            vectors.append([float(v) for v in embedding])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def create_embedder(
    provider: str,
    model: str,
    ollama_base_url: str = "http://localhost:11434",
) -> Embedder:
    if provider == "local":
        return LocalSentenceTransformerEmbedder(model)
    if provider == "openai":
        return OpenAIEmbedder(model)
    if provider == "ollama":
        return OllamaEmbedder(model, ollama_base_url)
    raise ValueError(
        "Unknown embedding provider "
        f"{provider!r}; expected 'local', 'openai', or 'ollama'."
    )
