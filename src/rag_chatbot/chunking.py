"""Chunking with metadata preservation.

Uses LangChain's RecursiveCharacterTextSplitter when available; otherwise
falls back to a simple deterministic sliding-window splitter so the core
pipeline (and tests) stay light. Either way, chunk ids are deterministic:
``<first 12 chars of source SHA-256>-<zero-padded chunk index>``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .loaders import RawDocument, sha256_hex


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    filename: str
    source_hash: str
    page: int | None = None


def _fallback_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = end - chunk_overlap
    return chunks


def _langchain_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str] | None:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore[no-redef]
        except ImportError:
            return None
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return [c for c in splitter.split_text(text) if c.strip()]


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size.")
    if not text.strip():
        return []
    chunks = _langchain_split(text, chunk_size, chunk_overlap)
    if chunks is not None:
        return chunks
    return _fallback_split(text, chunk_size, chunk_overlap)


def make_chunks(
    documents: list[RawDocument],
    source_hash: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Chunk documents, preserving filename/page and assigning stable ids."""
    chunks: list[Chunk] = []
    index = 0
    for doc in documents:
        for piece in split_text(doc.text, chunk_size, chunk_overlap):
            chunks.append(
                Chunk(
                    id=f"{source_hash[:12]}-{index:04d}",
                    text=piece,
                    filename=doc.filename,
                    source_hash=source_hash,
                    page=doc.page,
                )
            )
            index += 1
    return chunks


def chunk_bytes(
    filename: str, data: bytes, chunk_size: int, chunk_overlap: int
) -> tuple[str, list[Chunk]]:
    """Convenience helper: hash + load + chunk one uploaded file."""
    from .loaders import load_file

    source_hash = sha256_hex(data)
    return source_hash, make_chunks(
        load_file(filename, data), source_hash, chunk_size, chunk_overlap
    )
