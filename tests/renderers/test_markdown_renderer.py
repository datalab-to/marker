import pytest

from marker.renderers.markdown import MarkdownRenderer
from marker.schema import BlockTypes
from marker.schema.blocks import TableCell


@pytest.mark.config({"page_range": [0], "disable_ocr": True})
def test_markdown_renderer(pdf_document):
    renderer = MarkdownRenderer()
    md = renderer(pdf_document).markdown

    # Verify markdown
    assert "# Subspace Adversarial Training" in md


@pytest.mark.config({"page_range": [0]})
def test_markdown_renderer_auto_ocr(pdf_document):
    renderer = MarkdownRenderer()
    md = renderer(pdf_document).markdown

    # Verify markdown
    assert "Subspace Adversarial Training" in md


@pytest.mark.config({"page_range": [0, 1], "paginate_output": True})
def test_markdown_renderer_pagination(pdf_document):
    renderer = MarkdownRenderer({"paginate_output": True})
    md = renderer(pdf_document).markdown

    assert "\n\n{0}-" in md
    assert "\n\n{1}-" in md


@pytest.mark.config({"page_range": [0, 1], "paginate_output": True})
def test_markdown_renderer_pagination_blank_last_page(pdf_document):
    # Clear all children and structure from the last page to simulate a blank page
    last_page = pdf_document.pages[-1]
    last_page.children = []
    last_page.structure = []

    renderer = MarkdownRenderer({"paginate_output": True})
    md = renderer(pdf_document).markdown

    # Should end with pagination marker and preserve trailing newlines
    assert md.endswith("}\n\n") or md.endswith(
        "}------------------------------------------------\n\n"
    )


@pytest.mark.config({"page_range": [0, 1]})
def test_markdown_renderer_metadata(pdf_document):
    renderer = MarkdownRenderer({"paginate_output": True})
    metadata = renderer(pdf_document).metadata
    assert "table_of_contents" in metadata


@pytest.mark.config({"page_range": [0, 1]})
def test_markdown_renderer_images(pdf_document):
    renderer = MarkdownRenderer({"extract_images": False})
    markdown_output = renderer(pdf_document)

    assert len(markdown_output.images) == 0
    assert "![](" not in markdown_output.markdown


@pytest.mark.cpu
def test_markdown_ordered_list_ocr_numbering():
    md = MarkdownRenderer().md_cls.convert(
        "<ol><li>1. First</li><li>2) Second</li></ol>"
        "<ol start='7'><li>7. Seventh</li><li>3.5 kg of flour</li></ol>"
    )

    assert "1. First" in md
    assert "2. Second" in md
    assert "1. 1." not in md
    assert "7. Seventh" in md
    # Item text that merely starts with a number keeps the list numbering
    assert "8. 3.5 kg of flour" in md


@pytest.mark.config({"page_range": [5]})
def test_markdown_renderer_tables(pdf_document):
    table = pdf_document.contained_blocks((BlockTypes.Table,))[0]
    page = pdf_document.pages[0]

    cell = TableCell(
        polygon=table.polygon,
        text_lines=["54<i>.45</i>67<br>89<math>x</math>"],
        rowspan=1,
        colspan=1,
        row_id=0,
        col_id=0,
        is_header=False,
        page_id=page.page_id,
    )
    page.add_full_block(cell)
    table.structure = []
    table.add_structure(cell)

    renderer = MarkdownRenderer()
    md = renderer(pdf_document).markdown
    assert "54 <i>.45</i> 67<br>89 $x$" in md
