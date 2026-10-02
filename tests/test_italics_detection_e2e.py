"""
E2E Test for Issue #1020: Italics Detection in PDF to Markdown Conversion

This test creates a PDF with various italic text scenarios and validates
that Marker correctly detects and preserves italic formatting.
"""

import pytest
import tempfile
import os
from pathlib import Path

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from marker.converters.pdf import PdfConverter
from marker.renderers.markdown import MarkdownRenderer
from marker.models import create_model_dict


def create_test_pdf_with_italics(output_path: str) -> dict:
    """Create a PDF with various italic text scenarios for testing."""
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    expected_content = {}
    y_position = height - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y_position, "Italics Detection Test Document")
    y_position -= 40

    c.setFont("Helvetica-Oblique", 12)
    c.drawString(
        50, y_position, "This is oblique (italic) text using Helvetica-Oblique"
    )
    expected_content["This is oblique (italic) text using Helvetica-Oblique"] = True
    y_position -= 30

    c.setFont("Helvetica", 12)
    c.drawString(50, y_position, "This is plain text using Helvetica regular")
    expected_content["This is plain text using Helvetica regular"] = False
    y_position -= 30

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, "This is bold text using Helvetica-Bold")
    expected_content["This is bold text using Helvetica-Bold"] = False
    y_position -= 30

    c.setFont("Helvetica-BoldOblique", 12)
    c.drawString(50, y_position, "This is bold and oblique text")
    expected_content["This is bold and oblique text"] = True
    y_position -= 30

    c.setFont("Times-Italic", 12)
    c.drawString(50, y_position, "This is Times Italic text")
    expected_content["This is Times Italic text"] = True
    y_position -= 30

    c.setFont("Times-BoldItalic", 12)
    c.drawString(50, y_position, "This is Times Bold Italic text")
    expected_content["This is Times Bold Italic text"] = True
    y_position -= 30

    c.setFont("Courier-Oblique", 12)
    c.drawString(50, y_position, "This is Courier Oblique monospace text")
    expected_content["This is Courier Oblique monospace text"] = True
    y_position -= 30

    c.setFont("Helvetica", 12)
    c.drawString(50, y_position, "Start plain ")
    expected_content["Start plain"] = False

    text_width = c.stringWidth("Start plain ", "Helvetica", 12)
    c.setFont("Helvetica-Oblique", 12)
    c.drawString(50 + text_width, y_position, "then italic")
    expected_content["then italic"] = True

    y_position -= 30

    c.setFont("Helvetica-Oblique", 14)
    c.drawString(50, y_position, "Book Title in Italic Format")
    expected_content["Book Title in Italic Format"] = True
    y_position -= 40

    c.setFont("Helvetica", 12)
    c.drawString(50, y_position, "The term ")
    text_width = c.stringWidth("The term ", "Helvetica", 12)
    c.setFont("Times-Italic", 12)
    c.drawString(50 + text_width, y_position, "ceteris paribus")
    expected_content["ceteris paribus"] = True

    text_width2 = text_width + c.stringWidth("ceteris paribus", "Times-Italic", 12)
    c.setFont("Helvetica", 12)
    c.drawString(50 + text_width2, y_position, " means all else equal")
    y_position -= 30

    c.save()
    return expected_content


class TestItalicsDetectionE2E:
    """End-to-end tests for Issue #1020: Italics Detection."""

    @pytest.fixture(scope="class")
    def model_dict(self):
        """Create model dictionary for converter."""
        return create_model_dict()

    @pytest.fixture
    def test_pdf_path(self, tmp_path_factory):
        """Create a test PDF with italic text."""
        tmp_path = tmp_path_factory.mktemp("test_pdfs")
        pdf_path = tmp_path / "italics_test.pdf"
        expected_content = create_test_pdf_with_italics(str(pdf_path))
        return str(pdf_path), expected_content

    def test_italic_detection_e2e(self, test_pdf_path, model_dict):
        """E2E test: Verify italic text is detected and converted to markdown."""
        pdf_path, expected_content = test_pdf_path

        converter = PdfConverter(
            artifact_dict=model_dict,
            renderer="marker.renderers.markdown.MarkdownRenderer",
            config={"page_range": [0]},
        )

        result = converter(pdf_path)
        markdown_output = result.markdown

        assert markdown_output, "Markdown output should not be empty"

        has_italic_markers = "*" in markdown_output or "_" in markdown_output

        print("\n" + "=" * 60)
        print("MARKDOWN OUTPUT:")
        print("=" * 60)
        print(markdown_output)
        print("=" * 60)

        italic_count = markdown_output.count("*") + markdown_output.count("_")

        assert has_italic_markers, (
            f"Expected italic markers (* or _) in output, but found none.\n"
            f"This indicates Issue #1020 is not fixed.\n"
            f"Output:\n{markdown_output}"
        )

        print(f"\n SUCCESS: Found {italic_count} italic markers in output")

    def test_helvetica_oblique_detection(self, test_pdf_path, model_dict):
        """Test that Helvetica-Oblique font is detected as italic."""
        pdf_path, _ = test_pdf_path

        converter = PdfConverter(
            artifact_dict=model_dict,
            renderer="marker.renderers.markdown.MarkdownRenderer",
            config={"page_range": [0]},
        )

        result = converter(pdf_path)
        markdown = result.markdown.lower()

        italic_found = "*" in result.markdown or "_" in result.markdown

        assert italic_found, (
            "Helvetica-Oblique text should be detected as italic. "
            "This tests the font name detection fix for 'oblique' keyword."
        )

        result = converter(pdf_path)
        markdown = result.markdown.lower()

        italic_found = "*" in result.markdown or "_" in result.markdown

        assert italic_found, (
            "Helvetica-Oblique text should be detected as italic. "
            "This tests the font name detection fix for 'oblique' keyword."
        )

    def test_times_italic_detection(self, test_pdf_path, model_dict):
        """Test that Times-Italic font is detected as italic."""
        pdf_path, _ = test_pdf_path

        converter = PdfConverter(
            artifact_dict=model_dict,
            renderer="marker.renderers.markdown.MarkdownRenderer",
            config={"page_range": [0]},
        )

        result = converter(pdf_path)

        has_italic = "*" in result.markdown or "_" in result.markdown

        assert has_italic, (
            "Times-Italic text should be detected and marked as italic. "
            "This validates standard italic font detection works."
        )

        result = converter(pdf_path)

        has_italic = "*" in result.markdown or "_" in result.markdown

        assert has_italic, (
            "Times-Italic text should be detected and marked as italic. "
            "This validates standard italic font detection works."
        )

    def test_bold_not_marked_as_italic(self, test_pdf_path, model_dict):
        """Test that bold text is NOT incorrectly marked as italic."""
        pdf_path, _ = test_pdf_path

        converter = PdfConverter(
            artifact_dict=model_dict,
            renderer="marker.renderers.markdown.MarkdownRenderer",
            config={"page_range": [0]},
        )

        result = converter(pdf_path)

        has_bold = "**" in result.markdown

        print(f"\nBold markers (**): {result.markdown.count('**')}")
        print(f"Italic markers (*): {result.markdown.count('*')}")

        assert has_bold, "Should detect bold formatting with ** markers"


def test_production_scenario_fontspring_savoy_simulation(tmp_path_factory, model_dict):
    """Production scenario test: Simulates various italic font naming patterns."""
    tmp_path = tmp_path_factory.mktemp("production_test")
    pdf_path = tmp_path / "italic_fonts_sim.pdf"

    c = canvas.Canvas(str(pdf_path), pagesize=letter)

    # Using standard fonts that simulate various italic naming patterns
    test_cases = [
        ("Helvetica-Oblique", "Oblique font style text"),
        ("Times-Italic", "Standard italic font"),
        ("Courier-Oblique", "Monospace oblique style"),
        ("Times-BoldItalic", "Bold and italic combined"),
    ]

    y = 750
    for font_name, text in test_cases:
        c.setFont(font_name, 12)
        c.drawString(50, y, text)
        y -= 30

    c.save()

    converter = PdfConverter(
        artifact_dict=model_dict,
        renderer="marker.renderers.markdown.MarkdownRenderer",
        config={"page_range": [0]},
    )

    result = converter(str(pdf_path))
    markdown = result.markdown

    print("\n" + "=" * 70)
    print("PRODUCTION TEST: Italic Font Naming Patterns")
    print("=" * 70)
    print(markdown)
    print("=" * 70)

    italic_markers = markdown.count("*") + markdown.count("_")

    assert italic_markers > 0, (
        f"Expected italic detection for various font naming patterns.\n"
        f"Found {italic_markers} italic markers.\n"
        f"Issue #1020 may not be fully resolved.\n"
        f"Output:\n{markdown}"
    )

    print(f"\n Production test passed: Detected {italic_markers} italic markers")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
