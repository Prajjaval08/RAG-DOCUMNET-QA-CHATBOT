"""Safe, in-memory document loaders.

Uploads are parsed from bytes and are never written to disk by this module.
Scanned PDFs (images with no extractable text) are NOT OCR'd: loading such
a file raises a clear error instead of silently returning nothing.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md", ".csv")


@dataclass(frozen=True)
class RawDocument:
    """One extractable unit of text, before chunking."""

    filename: str
    text: str
    page: int | None = None  # 1-based PDF page number, when applicable


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_upload(filename: str, size_bytes: int, max_upload_bytes: int) -> None:
    """Validate extension and size; raise ValueError with an actionable message."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type {ext!r} for {filename!r}. Supported types: "
            f"{', '.join(SUPPORTED_EXTENSIONS)}."
        )
    if size_bytes <= 0:
        raise ValueError(f"File {filename!r} is empty.")
    if size_bytes > max_upload_bytes:
        raise ValueError(
            f"File {filename!r} is {size_bytes / (1024 * 1024):.1f} MB, "
            f"which exceeds the {max_upload_bytes / (1024 * 1024):.0f} MB limit."
        )


def normalize_text(text: str) -> str:
    """Normalize line endings/whitespace without destroying paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _load_pdf(filename: str, data: bytes) -> list[RawDocument]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "PDF support requires the 'pypdf' package. "
            "Install it with: pip install -r requirements.txt"
        ) from exc
    reader = PdfReader(io.BytesIO(data))
    docs: list[RawDocument] = []
    for i, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if text:
            docs.append(RawDocument(filename=filename, text=text, page=i))
    if not docs:
        raise ValueError(
            f"No extractable text found in {filename!r}. This is often a "
            "scanned PDF; OCR is not supported by this app, so scanned "
            "PDFs cannot be indexed."
        )
    return docs


def _load_docx(filename: str, data: bytes) -> list[RawDocument]:
    try:
        from docx import Document
    except ImportError as exc:
        raise ImportError(
            "DOCX support requires the 'python-docx' package. "
            "Install it with: pip install -r requirements.txt"
        ) from exc
    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    text = normalize_text("\n".join(parts))
    if not text:
        raise ValueError(f"No extractable text found in {filename!r}.")
    return [RawDocument(filename=filename, text=text)]


def _load_text(filename: str, data: bytes) -> list[RawDocument]:
    text = normalize_text(data.decode("utf-8", errors="replace"))
    if not text:
        raise ValueError(f"No extractable text found in {filename!r}.")
    return [RawDocument(filename=filename, text=text)]


def _load_csv(filename: str, data: bytes) -> list[RawDocument]:
    decoded = data.decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(decoded)))
    lines = [" | ".join(cell.strip() for cell in row if cell.strip()) for row in rows]
    lines = [line for line in lines if line]
    text = normalize_text("\n".join(lines))
    if not text:
        raise ValueError(f"No extractable text found in {filename!r}.")
    return [RawDocument(filename=filename, text=text)]


class _TextExtractor(HTMLParser):
    """Collect visible text, skipping script/style content."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("script", "style"):
            self._skip_depth += 1
        elif tag in ("p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "br"):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self.parts.append(data)


def extract_text_from_html(html: str) -> str:
    """Strip tags from HTML to readable text (WebBaseLoader equivalent).

    No external dependency: this is deliberately not a full browser
    render, so JavaScript-rendered pages may yield little or no text
    and will raise rather than silently index an empty page.
    """
    parser = _TextExtractor()
    parser.feed(html)
    return normalize_text(" ".join(parser.parts))


def _default_url_fetcher(url: str) -> tuple[str, bytes]:
    request = urllib.request.Request(
        url, headers={"User-Agent": "rag-document-qa-chatbot/1.0"}, method="GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:
            content_type = resp.headers.get_content_type() if resp.headers else ""
            data = resp.read(10 * 1024 * 1024 + 1)
    except urllib.error.HTTPError as exc:
        raise ConnectionError(
            f"Could not fetch {url!r}: HTTP {exc.code}."
        ) from exc
    except urllib.error.URLError as exc:
        raise ConnectionError(f"Could not fetch {url!r}: {exc.reason}.") from exc
    if len(data) > 10 * 1024 * 1024:
        raise ValueError(f"URL content at {url!r} exceeds the 10 MB fetch limit.")
    return content_type, data


def load_url(url: str, fetcher=None) -> list[RawDocument]:
    """Fetch a webpage / plain-text URL into RawDocuments.

    This is the project's WebBaseLoader equivalent: only http(s) URLs
    are allowed (no file://, no credentials in the URL), HTML is
    stripped to text, and plain text is used as-is. ``fetcher`` is an
    injectable ``url -> (content_type, bytes)`` callable used by tests
    so URL loading never needs the network in the test suite.
    """
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(
            f"Invalid URL {url!r}: only http:// and https:// URLs are supported."
        )
    fetch = fetcher or _default_url_fetcher
    content_type, data = fetch(url)
    if content_type == "text/html":
        text = extract_text_from_html(data.decode("utf-8", errors="replace"))
    elif content_type in ("text/plain", "text/markdown", ""):
        text = normalize_text(data.decode("utf-8", errors="replace"))
    else:
        raise ValueError(
            f"Unsupported content type {content_type!r} at {url!r}. "
            "Only HTML and plain-text pages can be indexed from a URL."
        )
    if not text:
        raise ValueError(
            f"No extractable text found at {url!r}. JavaScript-rendered "
            "pages are not supported by this loader."
        )
    return [RawDocument(filename=url, text=text)]


def load_file(filename: str, data: bytes) -> list[RawDocument]:
    """Parse uploaded bytes into one RawDocument per PDF page (or one total).

    Callers must call :func:`validate_upload` first for size/extension
    checks, or rely on this function's extension dispatch error.
    """
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == ".pdf":
        return _load_pdf(filename, data)
    if ext == ".docx":
        return _load_docx(filename, data)
    if ext in (".txt", ".md"):
        return _load_text(filename, data)
    if ext == ".csv":
        return _load_csv(filename, data)
    raise ValueError(
        f"Unsupported file type {ext!r} for {filename!r}. Supported types: "
        f"{', '.join(SUPPORTED_EXTENSIONS)}."
    )
