"""Shared test helpers: a deterministic stub embedder and stub LLM.

No paid APIs, no model weights, no network. The stub embedder maps a few
keywords onto fixed vector coordinates so retrieval is predictable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class StubEmbedder:
    """Deterministic 4-D embedder: [cats, dogs, paris, length-bucket]."""

    DIM = 4

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(lowered.count("cat")),
            float(lowered.count("dog")),
            float(lowered.count("paris")),
            float(min(len(text), 100)) / 100.0,
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class StubLLM:
    """Returns a canned answer, recording the prompt it received."""

    def __init__(self, answer: str = "Cats are discussed in the retrieved passage."):
        self.answer = answer
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.answer
