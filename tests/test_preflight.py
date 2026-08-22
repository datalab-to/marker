import pypdfium2 as pdfium
import pytest

from marker.providers.preflight import (
    UnreadablePdfError,
    preflight_pdf,
    unreadable_pdf_message,
)


@pytest.fixture
def minimal_pdf(tmp_path):
    """The smallest PDF PDFium will open, written by PDFium itself."""
    path = tmp_path / "minimal.pdf"
    doc = pdfium.PdfDocument.new()
    doc.new_page(200, 200)
    doc.save(str(path))
    doc.close()
    return path


def test_a_real_pdf_passes(minimal_pdf):
    assert preflight_pdf(str(minimal_pdf)) is None


def test_a_header_offset_into_the_file_passes(minimal_pdf, tmp_path):
    # PDFium reads a header that is not at byte zero, so refusing one here
    # would refuse a document marker can convert today.
    path = tmp_path / "offset.pdf"
    path.write_bytes(b"\x00" * 300 + minimal_pdf.read_bytes())

    assert preflight_pdf(str(path)) is None


def test_a_header_beyond_the_search_window_is_refused(minimal_pdf, tmp_path):
    # PDFium refuses this one too, verified against pypdfium2.
    path = tmp_path / "far.pdf"
    path.write_bytes(b"\x00" * 2000 + minimal_pdf.read_bytes())

    with pytest.raises(UnreadablePdfError, match="not a PDF"):
        preflight_pdf(str(path))


def test_a_missing_file_is_named(tmp_path):
    path = tmp_path / "absent.pdf"

    with pytest.raises(UnreadablePdfError, match=str(path)):
        preflight_pdf(str(path))


def test_a_directory_is_not_a_document(tmp_path):
    with pytest.raises(UnreadablePdfError):
        preflight_pdf(str(tmp_path))


def test_an_empty_file_says_so(tmp_path):
    path = tmp_path / "empty.pdf"
    path.touch()

    with pytest.raises(UnreadablePdfError, match="empty"):
        preflight_pdf(str(path))


def test_a_file_that_is_not_a_pdf_says_so(tmp_path):
    # A download that returned an error page and got saved with a .pdf name.
    path = tmp_path / "notapdf.pdf"
    path.write_text("<html><body><h1>404 Not Found</h1></body></html>")

    with pytest.raises(UnreadablePdfError, match="not a PDF"):
        preflight_pdf(str(path))


def test_preflight_does_not_validate_the_document(tmp_path):
    # Preflight refuses what cannot be a PDF; deciding whether a real PDF is
    # intact is PDFium's job, and this file gets to reach it.
    path = tmp_path / "truncated.pdf"
    path.write_bytes(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\ngarbage")

    assert preflight_pdf(str(path)) is None


def test_an_encryption_failure_is_reported_as_one():
    # Both messages come from pypdfium2's own error table, for
    # FPDF_ERR_PASSWORD and FPDF_ERR_SECURITY.
    for reported in [
        "Failed to load document (PDFium: Incorrect password error).",
        "Failed to load document (PDFium: Unsupported security scheme error).",
    ]:
        message = unreadable_pdf_message("/in/secret.pdf", RuntimeError(reported))

        assert "/in/secret.pdf" in message
        assert "encrypted" in message
        assert "password" in message


def test_any_other_failure_keeps_what_pdfium_said():
    message = unreadable_pdf_message(
        "/in/broken.pdf",
        RuntimeError("Failed to load document (PDFium: Data format error)."),
    )

    assert "/in/broken.pdf" in message
    assert "Data format error" in message


def test_the_error_is_catchable_as_a_value_error(tmp_path):
    path = tmp_path / "empty.pdf"
    path.touch()

    with pytest.raises(ValueError):
        preflight_pdf(str(path))
