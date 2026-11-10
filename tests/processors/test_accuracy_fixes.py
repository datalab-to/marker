import pytest
from bs4 import BeautifulSoup
from PIL import Image

pytestmark = pytest.mark.cpu

from marker.processors.blank_page import BlankPageProcessor
from marker.processors.ignoretext import IgnoreTextProcessor
from marker.processors.llm.llm_page_correction import LLMPageCorrectionProcessor
from marker.processors.llm.llm_sectionheader import LLMSectionHeaderProcessor
from marker.processors.sectionheader import SectionHeaderProcessor
from marker.renderers.markdown import Markdownify
from marker.schema import BlockTypes
from marker.schema.blocks import BlockId, Picture, SectionHeader, Text
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.polygon import PolygonBox
from marker.schema.text.line import Line
from marker.schema.text.span import Span


def test_sectionheader_bucket_headings_preserves_cluster_labels():
    proc = SectionHeaderProcessor({"level_count": 3, "merge_threshold": 0.25})
    # Distinct heading heights that KMeans assigns non-monotonic cluster IDs to
    heights = [24.0, 24.5, 23.8, 12.0, 12.2, 11.9, 18.0, 18.1, 17.9]
    ranges = proc.bucket_headings(heights)
    assert len(ranges) == 3
    # Ranges are sorted descending: (min_h, max_h) should not overlap across distinct clusters
    assert ranges[0][0] >= 23.8
    assert 17.9 <= ranges[1][0] <= 18.1
    assert ranges[2][1] <= 12.2


def test_markdownify_table_rowspan_column_count():
    html = """
    <table>
        <tr><th rowspan="2">A</th><th>B</th><th>C</th></tr>
        <tr><td>B2</td><td>C2</td></tr>
    </table>
    """
    md_conv = Markdownify(
        paginate_output=False,
        page_separator="---",
        inline_math_delimiters=("$", "$"),
        block_math_delimiters=("$$", "$$"),
        html_tables_in_markdown=False,
    )
    soup = BeautifulSoup(html, "html.parser")
    table_md = md_conv.convert_table(soup.find("table"), "", set())
    lines = [line.strip() for line in table_md.strip().splitlines() if line.strip()]
    # Both data rows must have 3 columns (4 pipe characters)
    for line in lines:
        assert line.count("|") == 4, f"Expected 3 columns in row: {line}"
    assert "B2" in lines[-1] and "C2" in lines[-1]


def test_span_and_line_combined_bold_italic_and_safe_url():
    poly = PolygonBox.from_bbox([0, 0, 100, 20])
    span = Span(
        polygon=poly,
        text="Important Link",
        font="Times",
        font_weight=700,
        font_size=12,
        minimum_position=0,
        maximum_position=14,
        formats=["bold", "italic", "underline"],
        url="https://datalab.to/docs?x=1&y=2",
        page_id=0,
        block_id=1,
    )
    rendered = span.assemble_html(None, [], None, None)
    assert "<b>" in rendered and "<i>" in rendered and "<u>" in rendered
    assert "href='https://datalab.to/docs?x=1&amp;y=2'" in rendered

    # Unsafe javascript: URL is stripped
    span_xss = Span(
        polygon=poly,
        text="Click",
        font="Times",
        font_weight=400,
        font_size=12,
        minimum_position=0,
        maximum_position=5,
        formats=["bold", "italic"],
        url="javascript:alert(1)",
        page_id=0,
        block_id=2,
    )
    rendered_xss = span_xss.assemble_html(None, [], None, None)
    assert "javascript:" not in rendered_xss
    assert "<a " not in rendered_xss
    assert "<b><i>Click</i></b>" in rendered_xss


def test_blank_page_processor_uses_intersection_pct():
    page_poly = PolygonBox.from_bbox([0, 0, 600, 800])
    pic_poly = PolygonBox.from_bbox([10, 10, 590, 790])  # >90% of page area
    blank_img = Image.new("RGB", (600, 800), color="white")

    pic = Picture(polygon=pic_poly, page_id=0, block_id=0)
    page = PageGroup(
        page_id=0,
        polygon=page_poly,
        children=[pic],
        structure=[pic.id],
        lowres_image=blank_img,
        highres_image=blank_img,
    )
    doc = Document(filepath="test.pdf", pages=[page])

    proc = BlankPageProcessor({"filter_blank_pages": True})
    proc(doc)
    assert pic.removed is True
    assert page.structure == []


def test_llm_page_correction_reorder_and_sectionheader_level_update():
    page_poly = PolygonBox.from_bbox([0, 0, 600, 800])
    b1 = Text(polygon= PolygonBox.from_bbox([0, 100, 200, 150]), page_id=0, block_id=1, html="<p>First</p>")
    b2 = SectionHeader(polygon=PolygonBox.from_bbox([0, 10, 200, 50]), page_id=0, block_id=2, html="<h1>Title</h1>", heading_level=1)
    page = PageGroup(
        page_id=0,
        polygon=page_poly,
        children=[b1, b2],
        structure=[b1.id, b2.id],
    )
    doc = Document(filepath="test.pdf", pages=[page])

    corr_proc = LLMPageCorrectionProcessor(llm_service=None, config={"use_llm": False})
    corr_proc.handle_reorder(
        [
            {"id": "/page/0/SectionHeader/1", "block_type": "", "html": ""},
            {"id": "/page/0/Text/0", "block_type": "", "html": ""},
        ],
        page,
    )
    #Wait: b1 has block_id=1, b2 has block_id=2; let's pass their actual IDs!
    corr_proc.handle_reorder(
        [
            {"id": str(b2.id), "block_type": "", "html": ""},
            {"id": str(b1.id), "block_type": "", "html": ""},
        ],
        page,
    )
    assert page.structure == [b2.id, b1.id]

    sh_proc = LLMSectionHeaderProcessor(llm_service=None, config={"use_llm": False})
    # Note: Document.get_block indexes page.children by block_id, so add a dummy block 0
    dummy0 = Text(polygon=page_poly, page_id=0, block_id=0)
    page.children = [dummy0, b1, b2]
    sh_proc.handle_rewrites([{"id": str(b2.id), "html": "<h3>Updated Subtitle</h3>"}], doc)
    assert b2.html == "<h3>Updated Subtitle</h3>"
    assert b2.heading_level == 3


def test_surya_compact_blocks_wrapping_and_postprocessors():
    from marker.processors.marginalia import MarginaliaProcessor
    from marker.processors.table import TableProcessor
    from marker.renderers.markdown import MarkdownRenderer
    from marker.schema.blocks import Equation, Form

    page_poly = PolygonBox.from_bbox([0, 0, 1000, 1000])
    sh = SectionHeader(
        polygon=PolygonBox.from_bbox([100, 20, 500, 50]),
        page_id=0,
        block_id=0,
        html="1. Introduction",
    )
    t1 = Text(
        polygon=PolygonBox.from_bbox([100, 100, 800, 150]),
        page_id=0,
        block_id=1,
        html="First paragraph.",
    )
    t2 = Text(
        polygon=PolygonBox.from_bbox([100, 160, 800, 210]),
        page_id=0,
        block_id=2,
        html="Second paragraph.",
    )
    eq = Equation(
        polygon=PolygonBox.from_bbox([200, 250, 600, 300]),
        page_id=0,
        block_id=3,
        html="<math>E = mc^2</math>",
    )
    form = Form(
        polygon=PolygonBox.from_bbox([50, 80, 900, 230]),
        page_id=0,
        block_id=4,
        html="<table><tr><td>Form header</td></tr></table>",
    )
    page = PageGroup(
        page_id=0,
        polygon=page_poly,
        children=[sh, t1, t2, eq, form],
        structure=[sh.id, t1.id, t2.id, eq.id, form.id],
        text_extraction_method="surya",
    )
    doc = Document(filepath="test.pdf", pages=[page])

    MarginaliaProcessor()(doc)
    assert sh.block_type == BlockTypes.SectionHeader

    tp = TableProcessor(recognition_model=None)
    tp.cleanup_contained_blocks(doc, {0: [form]})
    assert t1.id in page.structure and t2.id in page.structure

    md = MarkdownRenderer()(doc).markdown
    assert "## 1. Introduction" in md
    assert "First paragraph.\n\nSecond paragraph." in md
    assert "$$E = mc^2$$" in md

