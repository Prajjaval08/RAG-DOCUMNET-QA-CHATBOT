import io

import pytest
from docx import Document
from pypdf import PdfWriter

from rag_chatbot.loaders import (
    load_file,
    normalize_text,
    sha256_hex,
    validate_upload,
)


def test_validate_rejects_bad_extension():
    with pytest.raises(ValueError, match="Unsupported"):
        validate_upload("evil.exe", 100, 20 * 1024 * 1024)


def test_validate_rejects_oversize():
    with pytest.raises(ValueError, match="exceeds"):
        validate_upload("big.pdf", 21 * 1024 * 1024, 20 * 1024 * 1024)


def test_validate_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        validate_upload("empty.txt", 0, 20 * 1024 * 1024)


def test_validate_accepts_supported():
    for name in ("a.pdf", "a.docx", "a.txt", "a.md", "a.csv"):
        validate_upload(name, 10, 20 * 1024 * 1024)


def test_txt_and_md_loading():
    docs = load_file("notes.md", b"# Title\n\nHello world.\n")
    assert len(docs) == 1
    assert "Hello world" in docs[0].text
    assert docs[0].page is None


def test_csv_loading_keeps_rows():
    docs = load_file("data.csv", b"name,age\nAda,36\nGrace,40\n")
    assert "Ada" in docs[0].text and "Grace" in docs[0].text


def test_docx_loading_paragraphs_and_tables():
    buf = io.BytesIO()
    doc = Document()
    doc.add_paragraph("Hello from docx.")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Cell A"
    table.rows[0].cells[1].text = "Cell B"
    doc.save(buf)
    docs = load_file("doc.docx", buf.getvalue())
    assert "Hello from docx" in docs[0].text
    assert "Cell A" in docs[0].text


def _minimal_text_pdf(page_texts: list[str]) -> bytes:
    """Build a minimal valid PDF with one text object per page (no deps)."""
    objects: list[bytes] = []
    n_pages = len(page_texts)
    # Object numbering: 1 catalog, 2 pages, 3.. pages, then contents, then font.
    font_id = 3 + 2 * n_pages
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{3 + i} 0 R" for i in range(n_pages))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    for i in range(n_pages):
        content_id = 3 + n_pages + i
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
    for text in page_texts:
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode()
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    return bytes(out)


def test_pdf_page_numbers_preserved():
    data = _minimal_text_pdf(["First page cats.", "Second page dogs."])
    docs = load_file("doc.pdf", data)
    assert len(docs) == 2
    assert docs[0].page == 1 and "cats" in docs[0].text
    assert docs[1].page == 2 and "dogs" in docs[1].text


def test_scanned_or_blank_pdf_raises_clear_error():
    writer = PdfWriter()
    writer.add_blank_page(200, 200)
    buf = io.BytesIO()
    writer.write(buf)
    # Blank pages carry no text, so the loader must raise the scanned-PDF
    # style error rather than silently returning empty documents.
    with pytest.raises(ValueError, match="No extractable text"):
        load_file("blank.pdf", buf.getvalue())


def test_sha256_is_stable_and_distinct():
    assert sha256_hex(b"abc") == sha256_hex(b"abc")
    assert sha256_hex(b"abc") != sha256_hex(b"abd")
    assert len(sha256_hex(b"abc")) == 64


def test_normalize_collapses_blank_lines():
    assert normalize_text("a\r\n\r\n\r\n\r\nb") == "a\n\nb"


def test_load_file_rejects_unknown_extension():
    with pytest.raises(ValueError, match="Unsupported"):
        load_file("x.exe", b"data")
