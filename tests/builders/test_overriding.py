import multiprocessing as mp

import pytest

from marker.providers.pdf import PdfProvider
from marker.schema import BlockTypes
from marker.schema.blocks import SectionHeader
from marker.schema.document import Document
from marker.schema.registry import register_block_class
from marker.schema.text import Line
from tests.utils import setup_pdf_provider


class NewSectionHeader(SectionHeader):
    pass


class NewLine(Line):
    pass


@pytest.mark.config(
    {"page_range": [0], "override_map": {BlockTypes.SectionHeader: NewSectionHeader}}
)
def test_overriding(pdf_document: Document):
    assert (
        pdf_document.pages[0].get_block(pdf_document.pages[0].structure[0]).__class__
        == NewSectionHeader
    )


def get_lines(pdf: str, config=None):
    for block_type, block_cls in config["override_map"].items():
        register_block_class(block_type, block_cls)

    provider: PdfProvider = setup_pdf_provider(pdf, config)
    return provider.get_page_lines(0)


@pytest.mark.usefixtures("pdf_dataset")
def test_overriding_mp():
    config = {"page_range": [0], "override_map": {BlockTypes.Line: NewLine}}

    pdf_list = ["adversarial.pdf", "adversarial_rot.pdf"]

    with mp.Pool(processes=2) as pool:
        results = pool.starmap(get_lines, [(pdf, config) for pdf in pdf_list])
        assert all([r[0].line.__class__ == NewLine for r in results])
