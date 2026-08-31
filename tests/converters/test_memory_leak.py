import gc
import os
import psutil
import pytest

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser

TEST_PDF = os.path.join(os.path.dirname(__file__), "../test.pdf")


@pytest.fixture(scope="module")
def converter():
    config = ConfigParser({"disable_ocr": True}).generate_config_dict()
    model_dict = create_model_dict()
    converter = PdfConverter(artifact_dict=model_dict, config=config)
    yield converter
    del converter


def test_highres_images_freed_after_conversion(converter):
    """
    Regression test for https://github.com/datalab-to/marker/issues/1040
    highres_image should be None for all pages after build_document completes.
    """
    document = converter.build_document(TEST_PDF)

    for i, page in enumerate(document.pages):
        assert page.highres_image is None, (
            f"Page {i} still holds a highres_image after build_document — memory leak!"
        )


def test_memory_stable_across_multiple_pdfs(converter):
    """
    Regression test for https://github.com/datalab-to/marker/issues/1040
    Memory growth should be stable when reusing PdfConverter across multiple PDFs.
    """
    process = psutil.Process(os.getpid())

    # Warmup pass
    document = converter.build_document(TEST_PDF)
    del document
    gc.collect()

    baseline_rss = process.memory_info().rss / 1e6

    for i in range(3):
        document = converter.build_document(TEST_PDF)
        del document
        gc.collect()

    final_rss = process.memory_info().rss / 1e6
    growth = final_rss - baseline_rss

    assert growth < 50, (
        f"Memory grew by {growth:.1f} MB across 3 PDFs — possible memory leak!"
    )