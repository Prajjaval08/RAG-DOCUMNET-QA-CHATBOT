"""Directory-watch / scheduled re-index support.

``scan_directory`` is the core used by ``scripts/watch_and_reindex.py``
(and by any scheduler / Muse connector that invokes that script):

- walks a watch directory for supported files (PDF/DOCX/TXT/MD/CSV),
- skips files already in the FAISS store (SHA-256 dedup, same as the UI),
- indexes only new files, then saves the store,
- returns a stats dict and a list of per-file errors, never failing the
  whole run on one bad file.

A ``state.json`` next to the index is NOT needed: the store's own
doc-hash set is the source of truth, so a nightly run is idempotent
and a second run with no new files adds nothing. Deleted files are
not removed from the index (per-document delete is still on the
roadmap); the report notes this limitation explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunking import make_chunks
from .loaders import SUPPORTED_EXTENSIONS, load_file, sha256_hex, validate_upload
from .vector_store import FAISSVectorStore


@dataclass
class WatchStats:
    added_documents: int = 0
    skipped_duplicates: int = 0
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []

    def as_dict(self) -> dict:
        return {
            "added_documents": self.added_documents,
            "skipped_duplicates": self.skipped_duplicates,
            "errors": list(self.errors),
        }


def scan_directory(
    watch_dir: str | Path,
    store: FAISSVectorStore,
    embedder,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    max_upload_mb: int = 20,
) -> WatchStats:
    """Index every new supported file under ``watch_dir`` into ``store``."""
    watch_dir = Path(watch_dir)
    if not watch_dir.is_dir():
        raise ValueError(f"Watch directory does not exist: {watch_dir}")
    stats = WatchStats()
    changed = False
    files = sorted(
        p
        for p in watch_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    for path in files:
        try:
            data = path.read_bytes()
            validate_upload(path.name, len(data), max_upload_mb * 1024 * 1024)
            source_hash = sha256_hex(data)
            if store.has_document(source_hash):
                stats.skipped_duplicates += 1
                continue
            chunks = make_chunks(
                load_file(path.name, data), source_hash, chunk_size, chunk_overlap
            )
            added = store.add_chunks(chunks, embedder, source_hash)
            if added:
                stats.added_documents += 1
                changed = True
            else:
                stats.skipped_duplicates += 1
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
            stats.errors.append(f"{path}: {exc}")
    if changed:
        store.save()
    return stats
