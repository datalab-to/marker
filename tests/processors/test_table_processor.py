from types import SimpleNamespace

import pytest
from bs4 import BeautifulSoup
from PIL import Image

from marker.renderers.markdown import MarkdownRenderer
from marker.schema import BlockTypes
from marker.processors.table import TableProcessor
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.polygon import PolygonBox
from marker.schema.registry import get_block_class


@pytest.mark.config({"page_range": [5]})
def test_table_processor(pdf_document, recognition_model):
    processor = TableProcessor(recognition_model)
    processor(pdf_document)

    tables = pdf_document.contained_blocks((BlockTypes.Table,))
    assert len(tables) == 2
    for block in tables:
        # Tables now carry HTML directly (pdftext recon or OCR), no cell blocks.
        assert block.html
        assert "<table" in block.html

    renderer = MarkdownRenderer()
    table_output = renderer(pdf_document)
    assert "Schedule" in table_output.markdown


@pytest.mark.filename("table_ex.pdf")
@pytest.mark.config({"page_range": [0], "force_ocr": True})
def test_avoid_double_ocr(pdf_document, recognition_model):
    tables = pdf_document.contained_blocks((BlockTypes.Table,))
    lines = tables[0].contained_blocks(pdf_document, (BlockTypes.Line,))
    assert len(lines) == 0

    processor = TableProcessor(recognition_model, config={"force_ocr": True})
    processor(pdf_document)

    renderer = MarkdownRenderer()
    table_output = renderer(pdf_document)
    assert "Participants" in table_output.markdown


@pytest.mark.filename("multicol-blocks.pdf")
@pytest.mark.config({"page_range": [3]})
def test_overlap_blocks(pdf_document, recognition_model):
    page = pdf_document.pages[0]
    assert "Cascading, and the Auxiliary Problem Principle" in page.raw_text(
        pdf_document
    )

    processor = TableProcessor(recognition_model)
    processor(pdf_document)

    assert "Cascading, and the Auxiliary Problem Principle" in page.raw_text(
        pdf_document
    )


@pytest.mark.filename("pres.pdf")
@pytest.mark.config({"page_range": [4]})
def test_ocr_table(pdf_document, recognition_model):
    processor = TableProcessor(recognition_model)
    processor(pdf_document)

    renderer = MarkdownRenderer()
    table_output = renderer(pdf_document)
    assert "1.2E-38" in table_output.markdown


@pytest.mark.config({"page_range": [11]})
def test_split_rows(pdf_document, recognition_model):
    processor = TableProcessor(recognition_model)
    processor(pdf_document)

    table = pdf_document.contained_blocks((BlockTypes.Table,))[-1]
    assert table.html
    rows = BeautifulSoup(table.html, "html.parser").find_all("tr")
    assert len(rows) >= 5


def _unresolved_table_document(rows):
    """One page holding one digital table whose text lines came from the PDF
    text layer, with no cached pdftext page, so the pdftext rebuild finds no
    grid and the table goes to the OCR fallback."""
    page = PageGroup(
        polygon=PolygonBox.from_bbox([0, 0, 600, 800]),
        page_id=0,
        lowres_image=Image.new("RGB", (600, 800), "white"),
        highres_image=Image.new("RGB", (1200, 1600), "white"),
    )
    table = page.add_block(
        get_block_class(BlockTypes.Table),
        PolygonBox.from_bbox([50, 50, 550, 60 + 20 * len(rows)]),
    )
    page.add_structure(table)
    for i, text in enumerate(rows):
        bbox = [60, 55 + 20 * i, 540, 70 + 20 * i]
        line = page.add_block(
            get_block_class(BlockTypes.Line), PolygonBox.from_bbox(bbox)
        )
        span = page.add_full_block(
            get_block_class(BlockTypes.Span)(
                polygon=PolygonBox.from_bbox(bbox),
                page_id=0,
                text=text,
                font="Helvetica",
                font_weight=400,
                font_size=10,
                minimum_position=0,
                maximum_position=len(text),
                formats=["plain"],
                text_extraction_method="pdftext",
            )
        )
        line.add_structure(span)
        table.add_structure(line)
    return Document(filepath="table.pdf", pages=[page])


class _RejectedTableOCR:
    """Stands in for the recognition model, answering with no <table>, which
    clean_table_html rejects ("Table OCR failed")."""

    disable_tqdm = False

    def __call__(self, images, layout_results, full_page=False):
        return [
            SimpleNamespace(
                blocks=[SimpleNamespace(html="<p>Contents</p>", error=False)]
            )
            for _ in images
        ]


@pytest.mark.cpu
@pytest.mark.parametrize(
    "recognition_model,config",
    [(None, {"disable_ocr": True}), (_RejectedTableOCR(), {"mode": "fast"})],
    ids=["ocr_disabled", "ocr_rejected"],
)
def test_unresolved_table_keeps_text(recognition_model, config):
    rows = ["Introduction 7", "Methodology 19", "Annex 42"]
    document = _unresolved_table_document(rows)

    TableProcessor(recognition_model, config=config)(document)

    table = document.contained_blocks((BlockTypes.Table,))[0]
    assert not table.html
    markdown = MarkdownRenderer()(document).markdown
    for row in rows:
        assert row in markdown
