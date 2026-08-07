import pytest

from marker.processors.marginalia import MarginaliaProcessor
from marker.schema import BlockTypes
from marker.schema.blocks import Text
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.polygon import PolygonBox

# Same rendered layout on every page: a running header at the very top, body
# text that starts just below it, a tall body paragraph, and a page number.
# Coordinates are 0-origin within the page, as block polygons always are.
PAGE_CONTENT_HEIGHT = 711.0
LAYOUT = [
    ("Chapter 3 - Results", [63.0, 8.0, 400.0, 22.0]),
    ("Introductory sentence.", [63.0, 83.2, 500.0, 103.0]),
    ("Body paragraph. " * 20, [63.0, 120.0, 500.0, 500.0]),
    ("12", [63.0, 690.0, 200.0, 704.0]),
]


def _build_document(page_bbox):
    page = PageGroup(page_id=0, polygon=PolygonBox.from_bbox(page_bbox))
    for text, bbox in LAYOUT:
        block = page.add_block(Text, PolygonBox.from_bbox(bbox))
        block.html = text
        page.add_structure(block)
    return Document(filepath="test.pdf", pages=[page])


def _ignored_texts(document):
    page = document.pages[0]
    return {
        block.raw_text(document).strip()
        for block in page.contained_blocks(document, [BlockTypes.Text])
        if block.ignore_for_output
    }


@pytest.mark.cpu
def test_marginalia_processor_ignores_page_polygon_origin():
    """A non-zero page-polygon origin must not shift the margin zones.

    PdfProvider assigns page bboxes as [0, 0, w, h] on the pdftext path but as
    the raw pdfium CropBox on the force_ocr path, so a page polygon can carry a
    non-zero y origin while its blocks stay 0-origin. Both must classify the
    same rendering identically.
    """
    zero_origin = _build_document([0.0, 0.0, 612.0, PAGE_CONTENT_HEIGHT])
    shifted_origin = _build_document([0.0, 81.0, 612.0, 81.0 + PAGE_CONTENT_HEIGHT])

    processor = MarginaliaProcessor()
    processor(zero_origin)
    processor(shifted_origin)

    # Only the running header and the page number are marginalia; the body
    # text just below the header must survive on both pages.
    expected = {"Chapter 3 - Results", "12"}
    assert _ignored_texts(zero_origin) == expected
    assert _ignored_texts(shifted_origin) == expected
