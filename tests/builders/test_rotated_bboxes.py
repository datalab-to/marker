import pytest

from marker.schema import BlockTypes


@pytest.mark.config({"page_range": [0]})
@pytest.mark.filename("adversarial_rot.pdf")
def test_rotated_bboxes(pdf_document):
    first_page = pdf_document.pages[0]

    # Ensure we match all text lines up properly
    text_lines = first_page.contained_blocks(pdf_document, (BlockTypes.Line,))
    text_blocks = first_page.contained_blocks(
        pdf_document, (BlockTypes.Text, BlockTypes.TextInlineMath)
    )
    assert len(text_lines) == 84

    # Ensure the bbox sizes match up
    max_line_position = max([line.polygon.x_end for line in text_lines])
    max_block_position = max(
        [block.polygon.x_end for block in text_blocks if block.source == "layout"]
    )
    assert max_line_position <= max_block_position
