import pytest

from marker.builders.layout import LayoutBuilder
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.polygon import PolygonBox
from marker.schema.registry import get_block_class


def _build_page_with_table_and_form():
    """One region on a page detected as BOTH a Table and a Form with
    near-identical boxes (reproduces issue #1078)."""
    W, H = 600, 800
    page = PageGroup(page_id=0, polygon=PolygonBox.from_bbox([0, 0, W, H]))

    table_cls = get_block_class(BlockTypes.Table)
    form_cls = get_block_class(BlockTypes.Form)

    table = page.add_block(table_cls, PolygonBox.from_bbox([50, 100, 550, 400]))
    table.top_k = {BlockTypes.Table: 0.9}
    table.html = "<table><tr><td>x</td></tr></table>"
    page.add_structure(table)

    # Almost the same box, one pixel off - the Form detection.
    form = page.add_block(form_cls, PolygonBox.from_bbox([51, 101, 549, 399]))
    form.top_k = {BlockTypes.Form: 0.6}
    form.html = "<table><tr><td>x</td></tr></table>"
    page.add_structure(form)

    doc = Document(filepath="synthetic.pdf", pages=[page])
    return doc, page, table, form


@pytest.mark.cpu
def test_overlapping_table_and_form_deduped():
    doc, page, table, form = _build_page_with_table_and_form()

    table_family = {BlockTypes.Table, BlockTypes.Form, BlockTypes.TableOfContents}
    before = [
        page.get_block(bid)
        for bid in page.structure
        if page.get_block(bid).block_type in table_family
    ]
    assert len(before) == 2  # bug precondition: duplicate detections present

    builder = LayoutBuilder.__new__(LayoutBuilder)
    builder.table_merge_overlap_pct = 0.75
    builder.merge_overlapping_table_blocks(doc)

    after = [
        page.get_block(bid)
        for bid in page.structure
        if page.get_block(bid).block_type in table_family
    ]
    # Only the higher-confidence Table survives; the Form duplicate is removed.
    assert len(after) == 1
    assert after[0].id == table.id
    assert form.removed is True
    assert table.removed is False


@pytest.mark.cpu
def test_non_overlapping_tables_are_kept():
    W, H = 600, 800
    page = PageGroup(page_id=0, polygon=PolygonBox.from_bbox([0, 0, W, H]))
    table_cls = get_block_class(BlockTypes.Table)

    t1 = page.add_block(table_cls, PolygonBox.from_bbox([50, 100, 550, 300]))
    t1.top_k = {BlockTypes.Table: 0.9}
    page.add_structure(t1)
    t2 = page.add_block(table_cls, PolygonBox.from_bbox([50, 400, 550, 600]))
    t2.top_k = {BlockTypes.Table: 0.9}
    page.add_structure(t2)

    doc = Document(filepath="synthetic.pdf", pages=[page])

    builder = LayoutBuilder.__new__(LayoutBuilder)
    builder.table_merge_overlap_pct = 0.75
    builder.merge_overlapping_table_blocks(doc)

    assert len([bid for bid in page.structure]) == 2
    assert t1.removed is False and t2.removed is False
