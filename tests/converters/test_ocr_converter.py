import pytest

from marker.converters.ocr import OCRConverter
from marker.renderers.ocr_json import OCRJSONOutput, OCRJSONPageOutput


def _ocr_converter(config, model_dict, temp_pdf, line_count: int, eq_count: int):
    converter = OCRConverter(artifact_dict=model_dict, config=config)

    ocr_json: OCRJSONOutput = converter(temp_pdf.name)
    pages = ocr_json.children

    assert len(pages) == 1
    assert len(pages[0].children) == line_count
    eqs = [line for line in pages[0].children if line.block_type == "Equation"]
    assert len(eqs) == eq_count
    return pages


def check_bboxes(page: OCRJSONPageOutput, lines):
    page_size = page.bbox
    for line in lines:
        assert len(line.children) > 0
        for child in line.children:
            bbox = child.bbox
            assert all(
                [
                    bbox[0] >= page_size[0],
                    bbox[1] >= page_size[1],
                    bbox[2] <= page_size[2],
                    bbox[3] <= page_size[3],
                ]
            ), "Child bbox is outside page bbox"


@pytest.mark.config({"page_range": [0]})
def test_ocr_converter(config, model_dict, temp_doc):
    _ocr_converter(config, model_dict, temp_doc, 83, 2)


@pytest.mark.filename("pres.pdf")
@pytest.mark.config({"page_range": [1], "force_ocr": True, "keep_chars": True})
def test_ocr_converter_force(config, model_dict, temp_doc):
    pages = _ocr_converter(config, model_dict, temp_doc, 10, 0)
    lines = [line for line in pages[0].children if line.block_type == "Line"]
    check_bboxes(pages[0], lines)


@pytest.mark.filename("pres.pdf")
@pytest.mark.config({"page_range": [1], "keep_chars": True})
def test_ocr_converter_keep(config, model_dict, temp_doc):
    pages = _ocr_converter(config, model_dict, temp_doc, 9, 0)
    lines = [line for line in pages[0].children if line.block_type == "Line"]
    check_bboxes(pages[0], lines)
