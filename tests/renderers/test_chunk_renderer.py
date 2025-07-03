import pytest

from marker.renderers.chunk import ChunkRenderer


@pytest.mark.config({"page_range": [0]})
def test_markdown_renderer_pagination(pdf_document):
    renderer = ChunkRenderer()
    blocks = renderer(pdf_document).blocks

    assert len(blocks) == 15
    assert blocks[0].block_type == "SectionHeader"
