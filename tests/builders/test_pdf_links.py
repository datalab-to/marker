import re

import pytest

from marker.converters.pdf import PdfConverter
from marker.renderers.markdown import MarkdownOutput
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.util import classes_to_strings


@pytest.mark.filename("arxiv_test.pdf")
@pytest.mark.output_format("markdown")
@pytest.mark.config({"disable_ocr": True})
def test_pdf_links(pdf_document: Document, config, renderer, model_dict, temp_doc):
    first_page = pdf_document.pages[1]

    processors = ["marker.processors.reference.ReferenceProcessor"]
    pdf_converter = PdfConverter(
        artifact_dict=model_dict,
        processor_list=processors,
        renderer=classes_to_strings([renderer])[0],
        config=config,
    )

    for section_header_span in first_page.contained_blocks(
        pdf_document, (BlockTypes.Span,)
    ):
        if "II." in section_header_span.text:
            assert section_header_span.url == "#page-1-0"
            break
    else:
        raise ValueError("Could not find II. in the first page")

    section_header_block = first_page.contained_blocks(
        pdf_document, (BlockTypes.SectionHeader,)
    )[0]
    assert section_header_block.raw_text(pdf_document) == "II. THEORETICAL FRAMEWORK\n"

    assert first_page.refs[0].ref == "page-1-0"

    markdown_output: MarkdownOutput = pdf_converter(temp_doc.name)
    markdown = markdown_output.markdown

    assert "[II.](#page-1-0)" in markdown
    assert '<span id="page-1-0"></span>II. THEORETICAL FRAMEWORK' in markdown

    for ref in set(
        [
            f'<span id="page-{m[0]}-{m[1]}">'
            for m in re.findall(r"\]\(#page-(\d+)-(\d+)\)", markdown)
        ]
    ):
        assert ref in markdown, f"Reference {ref} not found in markdown"
