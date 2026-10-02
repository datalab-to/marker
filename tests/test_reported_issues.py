import os
import tempfile
from PIL import Image

from marker.builders.structure import StructureBuilder
from marker.output import save_output
from marker.renderers.markdown import MarkdownRenderer, cleanup_text, MarkdownOutput
from marker.schema import BlockTypes
from marker.schema.blocks import Figure, Caption, Text
from marker.schema.blocks.caption import clean_caption_text, clean_caption_html
from marker.schema.document import Document
from marker.schema.groups.figure import FigureGroup
from marker.schema.groups.page import PageGroup
from marker.schema.polygon import PolygonBox
from marker.settings import settings


def test_caption_word_agglutination_and_missing_spaces():
    raw_caption = "Fig. 4.1 NormalrighthipX-ray:femoralshaftaxis(redline), lateral offset (blue line)..."
    cleaned = clean_caption_text(raw_caption)
    expected = "Fig. 4.1 Normal right hip X-ray: femoral shaft axis (red line), lateral offset (blue line)..."
    assert cleaned == expected

    html_caption = "<p>Fig. 4.1 <b>Normalrighthip</b>X-ray:femoralshaftaxis(redline), lateral offset (blue line)...</p>"
    cleaned_html = clean_caption_html(html_caption)
    assert "Normal right hip" in cleaned_html
    assert "femoral shaft axis" in cleaned_html
    assert "red line" in cleaned_html


def test_citation_anchor_escaping_and_sanitization():
    cases = {
        r"[\[52](#page-43-0)]": "[52](#page-43-0)",
        r"[[53\]](#page-43-0)": "[53](#page-43-0)",
        r"[\[56](#page-43-0)]": "[56](#page-43-0)",
        r"[[52](#page-43-0)]": "[52](#page-43-0)",
        r"[\[52\]](#page-43-0)": "[52](#page-43-0)",
        r"[.8,9](#page--1-0)": ".[8,9](#page--1-0)",
        r"sentence[.8,9](#page--1-0)": "sentence.[8,9](#page--1-0)",
        r'<span id="page-1-0">II. THEORETICAL FRAMEWORK': '<span id="page-1-0"></span>II. THEORETICAL FRAMEWORK',
        r'<span id="page-1-0"></span>II. THEORETICAL FRAMEWORK': '<span id="page-1-0"></span>II. THEORETICAL FRAMEWORK',
    }

    for input_text, expected_output in cases.items():
        assert cleanup_text(input_text) == expected_output, f"Failed on {input_text}"


def test_convert_a_citation_clean():
    renderer = MarkdownRenderer()
    md_cls = renderer.md_cls

    html = '<p>See <a href="#page-43-0">[52]</a> and <a href="#page-43-0">53]</a> for reference.</p>'
    converted = md_cls.convert(html)
    assert "[52](#page-43-0)" in converted
    assert "[53](#page-43-0)" in converted
    assert r"\[" not in converted
    assert r"\]" not in converted


def test_caption_no_gluing_to_next_block():
    renderer = MarkdownRenderer()
    # Simulating HTML output from Figure, Caption, and next Text block
    html = """
    <p><img src="fig.jpg"></p>
    <p>Fig. 3.2 Remodeling increases in states of disuse (yellow) and overuse (orange)... partially offsetting the bone loss induced by the increased bone remodeling</p>
    <p>increase in bone remodeling activity corresponds to a decrease in bone mass.</p>
    """
    md = renderer.md_cls.convert(html)
    assert "increased bone remodeling\n\nincrease in bone remodeling activity" in md
    assert "increased bone remodeling increase in bone remodeling activity" not in md


def test_defer_interrupted_images_block_ordering():
    page = PageGroup(
        page_id=0,
        polygon=PolygonBox(polygon=[[0, 0], [1000, 0], [1000, 1000], [0, 1000]]),
    )

    t1 = Text(
        page_id=0,
        polygon=PolygonBox(polygon=[[10, 10], [500, 10], [500, 100], [10, 100]]),
        html="<p>Mechanical loading influences bone remodeling following a U-shaped curve: As mentioned before, an</p>",
    )
    t1.structure = []
    page.add_full_block(t1)

    fig = Figure(
        page_id=0,
        polygon=PolygonBox(polygon=[[10, 110], [500, 110], [500, 400], [10, 400]]),
    )
    fig.structure = []
    page.add_full_block(fig)

    cap = Caption(
        page_id=0,
        polygon=PolygonBox(polygon=[[10, 410], [500, 410], [500, 450], [10, 450]]),
        html="Fig. 3.2 Remodeling increases in states of disuse (yellow)...",
    )
    cap.structure = []
    page.add_full_block(cap)

    t2 = Text(
        page_id=0,
        polygon=PolygonBox(polygon=[[10, 460], [500, 460], [500, 600], [10, 600]]),
        html="<p>increase in bone remodeling activity corresponds to a decrease in bone mass.</p>",
    )
    t2.structure = []
    page.add_full_block(t2)

    t3 = Text(
        page_id=0,
        polygon=PolygonBox(polygon=[[10, 610], [500, 610], [500, 700], [10, 700]]),
        html="<p>Furthermore, osteoporosis involves complex signaling.</p>",
    )
    t3.structure = []
    page.add_full_block(t3)

    page.structure = [t1.id, fig.id, cap.id, t2.id, t3.id]
    doc = Document(filepath="dummy.pdf", pages=[page])

    builder = StructureBuilder()
    builder(doc)

    # After grouping and deferral:
    # t1 and t2 should be healed into one continuous paragraph.
    # The figure (or FigureGroup) should be placed AFTER t1.
    assert t1.id in page.structure
    assert t2.id not in page.structure  # merged into t1
    assert "As mentioned before, an increase in bone remodeling activity" in t1.raw_text(doc)

    t1_idx = page.structure.index(t1.id)
    # The next block in structure should be the FigureGroup or Figure
    next_block = page.get_block(page.structure[t1_idx + 1])
    assert next_block.block_type in (BlockTypes.FigureGroup, BlockTypes.Figure)


def test_non_interrupted_image_not_deferred():
    page = PageGroup(
        page_id=0,
        polygon=PolygonBox(polygon=[[0, 0], [1000, 0], [1000, 1000], [0, 1000]]),
    )

    t1 = Text(
        page_id=0,
        polygon=PolygonBox(polygon=[[10, 10], [500, 10], [500, 100], [10, 100]]),
        html="<p>This is a complete sentence that concludes the thought.</p>",
    )
    t1.structure = []
    page.add_full_block(t1)

    fig = Figure(
        page_id=0,
        polygon=PolygonBox(polygon=[[10, 110], [500, 110], [500, 400], [10, 400]]),
    )
    fig.structure = []
    page.add_full_block(fig)

    t2 = Text(
        page_id=0,
        polygon=PolygonBox(polygon=[[10, 410], [500, 410], [500, 500], [10, 500]]),
        html="<p>However, another study showed a different response.</p>",
    )
    t2.structure = []
    page.add_full_block(t2)

    page.structure = [t1.id, fig.id, t2.id]
    doc = Document(filepath="dummy.pdf", pages=[page])

    builder = StructureBuilder()
    builder(doc)

    # Sentence was complete and next is capitalized: figure stays in place
    assert page.structure == [t1.id, fig.id, t2.id]


def test_save_output_jpg_and_jpeg():
    with tempfile.TemporaryDirectory() as tmpdir:
        img = Image.new("RGB", (20, 20), color="blue")
        output = MarkdownOutput(
            markdown="# Test Document\n\nSome text.",
            images={"fig1.jpg": img},
            metadata={"title": "Test"},
        )

        orig_fmt = settings.OUTPUT_IMAGE_FORMAT
        try:
            settings.OUTPUT_IMAGE_FORMAT = "JPG"
            save_output(output, tmpdir, "test_doc")
            assert os.path.exists(os.path.join(tmpdir, "test_doc.md"))
            assert os.path.exists(os.path.join(tmpdir, "fig1.jpg"))
            assert os.path.exists(os.path.join(tmpdir, "test_doc_meta.json"))
        finally:
            settings.OUTPUT_IMAGE_FORMAT = orig_fmt
