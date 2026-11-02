"""FAISS vector store with persistence, deduplication, and profile guards.

Layout under ``index_dir``::

    index.faiss      FAISS index (vectors only)
    manifest.json    embedding profile, dimension, doc SHA-256 hashes, chunks

Security note: only load an index from your own trusted local directory.
Never load an index file supplied by an untrusted person: FAISS index
files (and any pickle-based sidecar) can execute code on load in some
configurations, so this store only ever loads what this app wrote locally.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .embeddings import Embedder


class EmbeddingProfileMismatchError(ValueError):
    """Raised when a stored index was built with a different embedding space."""


def _require_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "FAISS is required. Install it with: "
            "pip install -r requirements.txt (or faiss-cpu for tests)."
        ) from exc
    return faiss


@dataclass
class SearchResult:
    chunk: Chunk
    distance: float


class FAISSVectorStore:
    def __init__(
        self,
        index_dir: str | Path,
        embedding_provider: str,
        embedding_model: str,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.chunks: list[Chunk] = []
        self.doc_hashes: set[str] = set()
        self.dimension: int | None = None
        self._index = None

    # -- paths ---------------------------------------------------------
    @property
    def index_path(self) -> Path:
        return self.index_dir / "index.faiss"

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    @property
    def embedding_profile(self) -> dict[str, str]:
        return {"provider": self.embedding_provider, "model": self.embedding_model}

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    # -- persistence ---------------------------------------------------
    @classmethod
    def load(
        cls, index_dir: str | Path, embedding_provider: str, embedding_model: str
    ) -> FAISSVectorStore:
        store = cls(index_dir, embedding_provider, embedding_model)
        if not store.manifest_path.exists():
            return store  # fresh, empty store
        manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
        profile = manifest.get("embedding_profile", {})
        if profile != store.embedding_profile:
            raise EmbeddingProfileMismatchError(
                f"Stored index was built with embedding "
                f"{profile.get('provider')}/{profile.get('model')}, but the "
                f"current profile is {embedding_provider}/{embedding_model}. "
                "Clear the index and re-index your documents; querying "
                "vectors from a different embedding space would be wrong."
            )
        faiss = _require_faiss()
        if not store.index_path.exists():
            raise FileNotFoundError(
                f"Index manifest exists but {store.index_path} is missing. "
                "Clear the index and re-index."
            )
        store._index = faiss.read_index(str(store.index_path))
        store.dimension = int(manifest["dimension"])
        store.chunks = [Chunk(**c) for c in manifest.get("chunks", [])]
        store.doc_hashes = set(manifest.get("doc_hashes", []))
        return store

    def save(self) -> None:
        if self._index is None:
            raise ValueError("Nothing to save: the index has not been built yet.")
        faiss = _require_faiss()
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        manifest = {
            "embedding_profile": self.embedding_profile,
            "dimension": self.dimension,
            "doc_hashes": sorted(self.doc_hashes),
            "chunks": [asdict(c) for c in self.chunks],
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def clear(self) -> None:
        """Delete the persisted index and manifest (safe clear-index)."""
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
        self.chunks = []
        self.doc_hashes = set()
        self.dimension = None
        self._index = None

    # -- indexing ------------------------------------------------------
    def has_document(self, source_hash: str) -> bool:
        return source_hash in self.doc_hashes

    def add_chunks(
        self, chunks: list[Chunk], embedder: Embedder, source_hash: str
    ) -> int:
        """Embed and append chunks; returns the number added (0 if duplicate)."""
        if not chunks:
            return 0
        if self.has_document(source_hash):
            return 0
        vectors = embedder.embed_documents([c.text for c in chunks])
        if not vectors:
            return 0
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("Embedder must return a 2-D list of vectors.")
        if self._index is None:
            faiss = _require_faiss()
            self.dimension = int(matrix.shape[1])
            self._index = faiss.IndexFlatL2(self.dimension)
        elif int(matrix.shape[1]) != self.dimension:
            raise ValueError(
                f"Embedding dimension {matrix.shape[1]} does not match the "
                f"existing index dimension {self.dimension}."
            )
        self._index.add(matrix)
        self.chunks.extend(chunks)
        self.doc_hashes.add(source_hash)
        return len(chunks)

    # -- retrieval -----------------------------------------------------
    def search(self, query_vector: list[float], top_k: int) -> list[SearchResult]:
        if self._index is None or self.is_empty:
            return []
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")
        query = np.asarray([query_vector], dtype=np.float32)
        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Query vector dimension {query.shape[1]} does not match "
                f"index dimension {self.dimension}."
            )
        k = min(top_k, len(self.chunks))
        distances, indices = self._index.search(query, k)
        results: list[SearchResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            results.append(SearchResult(chunk=self.chunks[int(idx)], distance=float(dist)))
        return results
