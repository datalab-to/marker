"""Fast vs balanced conversion modes."""

import pytest

from marker.converters.pdf import PdfConverter
from marker.renderers.markdown import MarkdownOutput


def _convert(model_dict, temp_pdf, mode, extra=None):
    config = {"page_range": [0], "mode": mode, "disable_tqdm": True}
    if extra:
        config.update(extra)
    converter = PdfConverter(artifact_dict=model_dict, config=config)
    return converter(temp_pdf.name)


@pytest.mark.filename("adversarial.pdf")
def test_fast_mode_digital(model_dict, temp_doc):
    # Clean digital page in fast mode: Indra ONNX layout + pdftext, no VLM OCR.
    md: MarkdownOutput = _convert(model_dict, temp_doc, "fast")
    assert "Subspace Adversarial Training" in md.markdown
    # Nothing should have been full-page OCR'd in fast mode
    page = md.metadata["page_stats"][0]
    assert page["text_extraction_method"] == "pdftext"


@pytest.mark.filename("adversarial.pdf")
def test_balanced_mode_digital(model_dict, temp_doc):
    md: MarkdownOutput = _convert(model_dict, temp_doc, "balanced")
    assert "Subspace Adversarial Training" in md.markdown


@pytest.mark.filename("pres.pdf")
@pytest.mark.config({"page_range": [4]})
def test_fast_mode_table(pdf_document, recognition_model):
    # The table processor (fast geometric model) runs the same in both modes.
    from marker.processors.table import TableProcessor
    from marker.schema import BlockTypes

    processor = TableProcessor(recognition_model)
    processor(pdf_document)
    tables = pdf_document.contained_blocks((BlockTypes.Table,))
    assert len(tables) > 0
