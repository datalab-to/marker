import tempfile
from pathlib import Path
import pytest
from PIL import Image, ImageDraw

pytestmark = pytest.mark.cpu

from marker.builders.ocr import OcrBuilder
from marker.converters.pdf import PdfConverter
import marker.models  # noqa: F401 - ensures Surya cache dir redirect in read-only HOME
from marker.models import create_model_dict
from marker.ocr.indra import IndraRecognitionPredictor, select_recognition_model
from marker.output import text_from_rendered
from marker.processors.equation import EquationProcessor
from marker.processors.table import TableProcessor
from surya.layout.schema import LayoutBox, LayoutResult

DATA_DIR = Path(__file__).parent.parent / "data" / "olmocr_bench" / "pdfs"


def _make_text_image(text: str = "Indra OCR High Performance Engine", size=(800, 200)) -> Image.Image:
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)
    draw.text((40, 60), text, fill="black", font_size=32)
    return img


def test_select_recognition_model():
    surya_sentinel = object()
    indra_sentinel = object()

    # Default (balanced mode) uses surya recognition_model for both full_page=True and False
    assert select_recognition_model(surya_sentinel, indra_sentinel, mode="balanced") is surya_sentinel
    assert select_recognition_model(surya_sentinel, indra_sentinel, mode="balanced", full_page=False) is surya_sentinel

    # Fast and indra modes select indra_model
    assert select_recognition_model(surya_sentinel, indra_sentinel, mode="fast") is indra_sentinel
    assert select_recognition_model(surya_sentinel, indra_sentinel, mode="indra") is indra_sentinel

    # Explicit ocr_backend overrides mode
    assert (
        select_recognition_model(
            surya_sentinel, indra_sentinel, mode="balanced", ocr_backend="indra"
        )
        is indra_sentinel
    )
    assert (
        select_recognition_model(
            surya_sentinel, indra_sentinel, mode="fast", ocr_backend="surya"
        )
        is surya_sentinel
    )


def test_indra_predictor_full_page():
    predictor = IndraRecognitionPredictor()
    img = _make_text_image("Marker with Indra OCR full page test")
    results = predictor(images=[img], full_page=True)
    assert len(results) == 1
    assert len(results[0].blocks) >= 1
    combined_html = " ".join(b.html for b in results[0].blocks)
    assert "Indra" in combined_html or "Marker" in combined_html


def test_indra_predictor_block_modes():
    predictor = IndraRecognitionPredictor()
    img = _make_text_image("Section 1: Introduction to Indra", size=(800, 300))

    layout = LayoutResult(
        image_bbox=[0, 0, 800, 300],
        bboxes=[
            LayoutBox(
                polygon=[[20, 40], [780, 40], [780, 140], [20, 140]],
                label="SectionHeader",
                raw_label="SectionHeader",
                position=0,
            ),
        ],
    )
    results = predictor(images=[img], layout_results=[layout], full_page=False)
    assert len(results) == 1
    assert len(results[0].blocks) == 1
    block = results[0].blocks[0]
    assert block.label == "SectionHeader"
    assert block.html.startswith("<h")


def test_builder_and_processors_prefer_indra_in_fast_mode():
    surya_sentinel = object()
    indra_predictor = IndraRecognitionPredictor()

    ocr_builder = OcrBuilder(
        surya_sentinel, config={"mode": "fast"}, indra_model=indra_predictor
    )
    assert ocr_builder.get_recognition_model() is indra_predictor

    table_proc = TableProcessor(
        surya_sentinel, config={"mode": "indra"}, indra_model=indra_predictor
    )
    assert table_proc.get_recognition_model() is indra_predictor

    eq_proc = EquationProcessor(
        surya_sentinel, config={"ocr_backend": "indra"}, indra_model=indra_predictor
    )
    assert eq_proc.get_recognition_model() is indra_predictor


def test_pdf_converter_indra_mode_digital_and_forced_ocr():
    pdf_path = DATA_DIR / "headers_footers_page1.pdf"
    artifact_dict = create_model_dict()

    # 1. Mode 1: no_ocr mode (pure text layer + lightweight Indra layout, zero OCR)
    converter_no_ocr = PdfConverter(
        artifact_dict=artifact_dict,
        config={
            "mode": "no_ocr",
            "disable_image_extraction": True,
            "disable_tqdm": True,
            "pdftext_workers": 1,
        },
    )
    assert converter_no_ocr.config["disable_ocr"] is True
    md_no_ocr, _, _ = text_from_rendered(converter_no_ocr(str(pdf_path)))
    assert len(md_no_ocr.strip()) > 100

    # 2. Mode 2: indra / fast mode (line-level Indra ONNX OCR)
    converter = PdfConverter(
        artifact_dict=artifact_dict,
        config={
            "mode": "indra",
            "disable_image_extraction": True,
            "disable_tqdm": True,
            "pdftext_workers": 1,
        },
    )
    md, _, _ = text_from_rendered(converter(str(pdf_path)))
    assert len(md.strip()) > 100

    # 3. Full-page Indra OCR on image input
    img = _make_text_image("Standalone image converted via Indra OCR")
    with tempfile.NamedTemporaryFile(suffix=".png") as tf:
        img.save(tf.name)
        tf.flush()
        md_img, _, _ = text_from_rendered(converter(tf.name))
        assert "Indra" in md_img or "Standalone" in md_img
