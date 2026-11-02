"""Environment-based configuration with validation.

No secrets are stored here: API keys are read from the environment at
runtime by :mod:`rag_chatbot.llm` / :mod:`rag_chatbot.embeddings` and are
never printed, logged, or rendered in the UI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

LLM_PROVIDERS = ("openai", "anthropic", "ollama", "huggingface", "gemini")
EMBEDDING_PROVIDERS = ("local", "openai", "ollama")

DEFAULT_LLM_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "ollama": "llama3.2",
    "huggingface": "meta-llama/Llama-3.2-1B-Instruct",
    "gemini": "gemini-3.8-flash",
}


@dataclass(frozen=True)
class AppConfig:
    """Validated application configuration."""

    llm_provider: str = "ollama"
    llm_model: str = "llama3.2"
    llm_temperature: float = 0.1
    retrieval_top_k: int = 4
    embedding_provider: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"
    ollama_embedding_model: str = "nomic-embed-text"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    max_upload_mb: int = 20
    ollama_base_url: str = "http://localhost:11434"
    index_dir: Path = field(default_factory=lambda: Path("data/faiss_index"))

    def __post_init__(self) -> None:
        if self.llm_provider not in LLM_PROVIDERS:
            raise ValueError(
                f"Invalid LLM_PROVIDER {self.llm_provider!r}; "
                f"expected one of {', '.join(LLM_PROVIDERS)}."
            )
        if self.embedding_provider not in EMBEDDING_PROVIDERS:
            raise ValueError(
                f"Invalid EMBEDDING_PROVIDER {self.embedding_provider!r}; "
                f"expected one of {', '.join(EMBEDDING_PROVIDERS)}."
            )
        if not self.llm_model:
            raise ValueError("LLM_MODEL must not be empty.")
        if not self.embedding_model:
            raise ValueError("EMBEDDING_MODEL must not be empty.")
        if not 0.0 <= self.llm_temperature <= 2.0:
            raise ValueError(
                f"LLM_TEMPERATURE {self.llm_temperature} out of range; "
                "expected 0.0 to 2.0."
            )
        if self.retrieval_top_k < 1:
            raise ValueError(
                f"RETRIEVAL_TOP_K {self.retrieval_top_k} is invalid; "
                "expected >= 1."
            )
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be > 0.")
        if self.chunk_overlap < 0:
            raise ValueError("CHUNK_OVERLAP must be >= 0.")
        if self.chunk_size <= self.chunk_overlap:
            raise ValueError(
                f"CHUNK_SIZE ({self.chunk_size}) must be greater than "
                f"CHUNK_OVERLAP ({self.chunk_overlap})."
            )
        if self.max_upload_mb <= 0:
            raise ValueError("MAX_UPLOAD_MB must be > 0.")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def effective_embedding_model(self) -> str:
        """Model name that identifies the embedding space in use."""
        if self.embedding_provider == "openai":
            return self.openai_embedding_model
        if self.embedding_provider == "ollama":
            return self.ollama_embedding_model
        return self.embedding_model

    @property
    def embedding_profile(self) -> dict[str, str]:
        """Profile saved in the index manifest to detect vector-space drift."""
        return {
            "provider": self.embedding_provider,
            "model": self.effective_embedding_model,
        }


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not a valid number.") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not a valid integer.") from exc


def load_config() -> AppConfig:
    """Build :class:`AppConfig` from environment variables."""
    provider = os.environ.get("LLM_PROVIDER", "ollama")
    model = os.environ.get("LLM_MODEL") or DEFAULT_LLM_MODELS.get(provider, "llama3.2")
    return AppConfig(
        llm_provider=provider,
        llm_model=model,
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.1),
        retrieval_top_k=_env_int("RETRIEVAL_TOP_K", 4),
        embedding_provider=os.environ.get("EMBEDDING_PROVIDER", "local"),
        embedding_model=os.environ.get(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        openai_embedding_model=os.environ.get(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        ),
        ollama_embedding_model=os.environ.get(
            "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
        ),
        chunk_size=_env_int("CHUNK_SIZE", 1000),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 150),
        max_upload_mb=_env_int("MAX_UPLOAD_MB", 20),
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        index_dir=Path(os.environ.get("INDEX_DIR", "data/faiss_index")),
    )
