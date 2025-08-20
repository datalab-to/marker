import pytest

from marker.schema import BlockTypes
from marker.schema.text.line import Line


@pytest.mark.filename("thinkpython.pdf")
@pytest.mark.config({"page_range": [0]})
def test_document_builder(pdf_document):
    first_page = pdf_document.pages[0]
    assert first_page.structure[0] == "/page/0/SectionHeader/0"

    first_block = first_page.get_block(first_page.structure[0])
    assert first_block.block_type == BlockTypes.SectionHeader
    assert first_block.text_extraction_method == "pdftext"

    first_text_block: Line = first_page.get_block(first_block.structure[0])
    assert first_text_block.block_type == BlockTypes.Line

    first_span = first_page.get_block(first_text_block.structure[0])
    assert first_span.block_type == BlockTypes.Span
    assert first_span.text == "Think Python"
    assert first_span.font == "URWPalladioL-Roma"
    assert first_span.formats == ["plain"]


@pytest.mark.config({"page_range": [0]})
def test_document_builder_inline_eq(pdf_document):
    first_page = pdf_document.pages[0]
    assert first_page.structure[0] == "/page/0/SectionHeader/0"

    first_block = first_page.get_block(first_page.structure[0])
    assert first_block.block_type == BlockTypes.SectionHeader
    assert first_block.text_extraction_method == "surya"

    first_text_block: Line = first_page.get_block(first_block.structure[0])
    assert first_text_block.block_type == BlockTypes.Line

    first_span = first_page.get_block(first_text_block.structure[0])
    assert first_span.block_type == BlockTypes.Span
    assert first_span.text.strip() == "Subspace Adversarial Training"
    assert "bold" in first_span.formats
