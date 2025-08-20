import pytest

from marker.builders.document import DocumentBuilder
from marker.builders.layout import LayoutBuilder
from marker.builders.line import LineBuilder
from marker.renderers.markdown import MarkdownRenderer
from marker.schema import BlockTypes
from marker.schema.registry import get_block_class


@pytest.mark.filename("thinkpython.pdf")
@pytest.mark.config({"page_range": [0]})
def test_layout_replace(
    request, config, doc_provider, layout_model, ocr_error_model, detection_model
):
    # The llm layout builder replaces blocks - this makes sure text is still merged properly
    layout_builder = LayoutBuilder(layout_model, config)
    line_builder = LineBuilder(detection_model, ocr_error_model, config)
    builder = DocumentBuilder(config)
    document = builder.build_document(doc_provider)
    layout_builder(document, doc_provider)
    page = document.pages[0]
    new_blocks = []
    for block in page.contained_blocks(document, (BlockTypes.Text,)):
        generated_block_class = get_block_class(BlockTypes.TextInlineMath)
        generated_block = generated_block_class(
            polygon=block.polygon,
            page_id=block.page_id,
            structure=block.structure,
        )
        page.replace_block(block, generated_block)
        new_blocks.append(generated_block)
    line_builder(document, doc_provider)

    for block in new_blocks:
        assert block.raw_text(document).strip()

    renderer = MarkdownRenderer(config)
    rendered = renderer(document)

    assert "Think Python" in rendered.markdown
