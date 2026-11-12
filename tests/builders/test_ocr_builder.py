from pathlib import Path
from unittest.mock import MagicMock
from surya.recognition.schema import BlockOCRResult, PageOCRResult

from marker.builders.ocr import OcrBuilder
from marker.converters.pdf import PdfConverter
from marker.ocr import select_recognition_model


def test_clean_html(recognition_model):
    builder = OcrBuilder(recognition_model)

    # Debug attributes are stripped, truncated tags are balanced
    html = '<p data-bbox="1 2 3 4" data-label="Text">Hello <b>world'
    cleaned = builder.clean_html(html)
    assert "data-bbox" not in cleaned
    assert "data-label" not in cleaned
    assert "</b>" in cleaned
    assert "Hello" in cleaned

    # Repetition loops are dropped
    looping = "<p>" + "same phrase " * 400
    assert builder.clean_html(looping) == ""

    assert builder.clean_html("") == ""


def test_vllm_mode_skips_pdftext_and_indra():
    pdf_path = next((Path(__file__).parent.parent / "data" / "olmocr_bench").rglob("*.pdf"))
    mock_rec = MagicMock()
    mock_rec.return_value = [
        PageOCRResult(
            blocks=[
                BlockOCRResult(
                    polygon=[[10, 10], [200, 10], [200, 50], [10, 50]],
                    label="Text",
                    raw_label="Text",
                    reading_order=0,
                    html="<p>Pure VLLM Output</p>",
                    skipped=False,
                    error=False,
                    confidence=1.0,
                )
            ],
            image_bbox=[0, 0, 612, 792],
        )
    ]
    mock_indra = MagicMock()
    artifact_dict = {
        "recognition_model": mock_rec,
        "layout_model": mock_indra,
        "fast_layout_model": mock_indra,
        "ocr_error_model": None,
        "indra_model": mock_indra,
    }
    converter = PdfConverter(
        artifact_dict=artifact_dict,
        config={"mode": "vllm", "page_range": [0]},
    )
    assert converter.mode == "vllm"
    assert converter.config["force_ocr"] is True
    assert select_recognition_model(mock_rec, mock_indra, mode="vllm", full_page=False) is mock_rec

    rendered = converter(str(pdf_path))
    assert "Pure VLLM Output" in rendered.markdown
    assert mock_rec.called
    assert mock_rec.call_args.kwargs.get("full_page") is True
    assert not mock_indra.called
    assert not mock_indra.convert_pdf_pages.called
    assert not mock_indra.detect_layout.called


def test_has_non_latin_script_gate():
    from marker.builders.line import has_non_latin_script

    assert not has_non_latin_script("Standard English prose with French accents café résumé.")
    assert not has_non_latin_script("Theorem 1: Let α, β, γ be parameters in a convex set.")
    assert has_non_latin_script("紧紧抓住重要战略机遇期加快推进现代农业发展")
    assert has_non_latin_script("Введение в теорию вероятностей и математическую статистику")
    assert has_non_latin_script("المقدمة في الذكاء الاصطناعي وتطبيقاته")

