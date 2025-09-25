from unittest.mock import MagicMock, Mock

import pytest
from marker.processors.llm.llm_complex import LLMComplexRegionProcessor
from marker.processors.llm.llm_equation import LLMEquationProcessor

from marker.processors.llm.llm_form import LLMFormProcessor
from marker.processors.llm.llm_image_description import LLMImageDescriptionProcessor
from marker.processors.llm.llm_meta import LLMSimpleBlockMetaProcessor
from marker.processors.llm.llm_strikethrough import LLMStrikethroughProcessor
from marker.processors.llm.llm_table import LLMTableProcessor
from marker.processors.table import TableProcessor
from marker.renderers.markdown import MarkdownRenderer
from marker.schema import BlockTypes
from marker.schema.blocks import ComplexRegion


@pytest.mark.filename("form_1040.pdf")
@pytest.mark.config({"page_range": [0]})
def test_llm_form_processor_no_config(pdf_document, llm_service):
    processor_lst = [LLMFormProcessor()]
    processor = LLMSimpleBlockMetaProcessor(processor_lst, llm_service)
    processor(pdf_document)

    forms = pdf_document.contained_blocks((BlockTypes.Form,))
    assert forms[0].html is None


@pytest.mark.filename("form_1040.pdf")
@pytest.mark.config({"page_range": [0]})
def test_llm_form_processor_no_cells(pdf_document, llm_service):
    config = {"use_llm": True, "gemini_api_key": "test"}
    processor_lst = [LLMFormProcessor(config)]
    processor = LLMSimpleBlockMetaProcessor(processor_lst, llm_service, config)
    processor(pdf_document)

    forms = pdf_document.contained_blocks((BlockTypes.Form,))
    assert forms[0].html is None


@pytest.mark.filename("form_1040.pdf")
@pytest.mark.config({"page_range": [0]})
def test_llm_form_processor(pdf_document, table_rec_model, recognition_model, detection_model):
    corrected_html = "<em>This is corrected markdown.</em>\n" * 100
    corrected_html = "<p>" + corrected_html.strip() + "</p>\n"

    mock_cls = Mock()
    mock_cls.return_value = {"corrected_html": corrected_html}

    cell_processor = TableProcessor(recognition_model, table_rec_model, detection_model)
    cell_processor(pdf_document)

    config = {"use_llm": True, "gemini_api_key": "test"}
    processor_lst = [LLMFormProcessor(config)]
    processor = LLMSimpleBlockMetaProcessor(processor_lst, mock_cls, config)
    processor(pdf_document)

    forms = pdf_document.contained_blocks((BlockTypes.Form,))
    assert forms[0].html == corrected_html.strip()



@pytest.mark.filename("table_ex2.pdf")
@pytest.mark.config({"page_range": [0]})
def test_llm_table_processor(pdf_document, table_rec_model, recognition_model, detection_model):
    corrected_html = """
<table>
    <tr>
        <td>Column 1</td>
        <td>Column 2</td>
        <td>Column 3</td>
        <td>Column 4</td>
    </tr>
    <tr>
        <td>Value 1 <math>x</math></td>
        <td>Value 2</td>
        <td>Value 3</td>
        <td>Value 4</td>
    </tr>
    <tr>
        <td>Value 5</td>
        <td>Value 6</td>
        <td>Value 7</td>
        <td>Value 8</td>
    </tr>
</table>
    """.strip()

    mock_cls = Mock()
    mock_cls.return_value = {"corrected_html": corrected_html}

    cell_processor = TableProcessor(recognition_model, table_rec_model, detection_model)
    cell_processor(pdf_document)

    processor = LLMTableProcessor(mock_cls, {"use_llm": True, "gemini_api_key": "test"})
    processor(pdf_document)

    tables = pdf_document.contained_blocks((BlockTypes.Table,))
    table_cells = tables[0].contained_blocks(pdf_document, (BlockTypes.TableCell,))
    assert table_cells[0].text == "Column 1"

    markdown = MarkdownRenderer()(pdf_document).markdown
    assert "Value 1 $x$" in markdown


@pytest.mark.filename("A17_FlightPlan.pdf")
@pytest.mark.config({"page_range": [0]})
def test_llm_caption_processor_disabled(pdf_document):
    config = {"use_llm": True, "gemini_api_key": "test"}
    mock_cls = MagicMock()
    processor_lst = [LLMImageDescriptionProcessor(config)]
    processor = LLMSimpleBlockMetaProcessor(processor_lst, mock_cls, config)
    processor(pdf_document)

    contained_pictures = pdf_document.contained_blocks((BlockTypes.Picture, BlockTypes.Figure))
    assert all(picture.description is None for picture in contained_pictures)

@pytest.mark.filename("A17_FlightPlan.pdf")
@pytest.mark.config({"page_range": [0]})
def test_llm_caption_processor(pdf_document):
    description = "This is an image description."
    mock_cls = Mock()
    mock_cls.return_value = {"image_description": description}

    config = {"use_llm": True, "gemini_api_key": "test", "extract_images": False}
    processor_lst = [LLMImageDescriptionProcessor(config)]
    processor = LLMSimpleBlockMetaProcessor(processor_lst, mock_cls, config)
    processor(pdf_document)

    contained_pictures = pdf_document.contained_blocks((BlockTypes.Picture, BlockTypes.Figure))
    assert all(picture.description == description for picture in contained_pictures)

    # Ensure the rendering includes the description
    renderer = MarkdownRenderer({"extract_images": False})
    md = renderer(pdf_document).markdown

    assert description in md


@pytest.mark.filename("A17_FlightPlan.pdf")
@pytest.mark.config({"page_range": [0]})
def test_llm_complex_region_processor(pdf_document):
    md = "This is some *markdown* for a complex region."
    mock_cls = Mock()
    mock_cls.return_value = {"corrected_markdown": md * 25}

    # Replace the block with a complex region
    old_block = pdf_document.pages[0].children[0]
    new_block = ComplexRegion(
        **old_block.dict(exclude=["id", "block_id", "block_type"]),
    )
    pdf_document.pages[0].replace_block(old_block, new_block)

    # Test processor
    config = {"use_llm": True, "gemini_api_key": "test"}
    processor_lst = [LLMComplexRegionProcessor(config)]
    processor = LLMSimpleBlockMetaProcessor(processor_lst, mock_cls, config)
    processor(pdf_document)

    # Ensure the rendering includes the description
    renderer = MarkdownRenderer()
    rendered_md = renderer(pdf_document).markdown

    assert md in rendered_md

@pytest.mark.filename("adversarial.pdf")
@pytest.mark.config({"page_range": [0]})
def test_multi_llm_processors(pdf_document):
    description = "<math>This is an image description.  And here is a lot of writing about it.</math>" * 10
    mock_cls = Mock()
    mock_cls.return_value = {"image_description": description, "corrected_equation": description}

    config = {"use_llm": True, "gemini_api_key": "test", "extract_images": False, "min_equation_height": .001}
    processor_lst = [LLMImageDescriptionProcessor(config), LLMEquationProcessor(config)]
    processor = LLMSimpleBlockMetaProcessor(processor_lst, mock_cls, config)
    processor(pdf_document)

    contained_pictures = pdf_document.contained_blocks((BlockTypes.Picture, BlockTypes.Figure))
    assert all(picture.description == description for picture in contained_pictures)

    contained_equations = pdf_document.contained_blocks((BlockTypes.Equation,))
    print([equation.html for equation in contained_equations])
    assert all(equation.html == description for equation in contained_equations)

@pytest.mark.filename("adversarial.pdf")
@pytest.mark.config({"page_range": [0]})
def test_llm_strikethrough_processor(pdf_document):
    corrected_html = "<p>This is regular text and <del>this text is struck through</del> and this is normal again.</p>"
    mock_cls = Mock()
    mock_cls.return_value = {
        "analysis": "The text 'this text is struck through' has a strikethrough line in the image.",
        "corrections_needed": True,
        "corrected_html": corrected_html
    }

    config = {"use_llm": True, "gemini_api_key": "test"}
    processor_lst = [LLMStrikethroughProcessor(config)]
    processor = LLMSimpleBlockMetaProcessor(processor_lst, mock_cls, config)
    processor(pdf_document)

    text_blocks = pdf_document.contained_blocks((BlockTypes.Text,))

    has_strikethrough = any(
        block.html and "<del>" in block.html
        for block in text_blocks
    )
    assert has_strikethrough, "No text blocks contain strikethrough HTML"

    renderer = MarkdownRenderer()
    rendered_md = renderer(pdf_document).markdown

    assert "~~this text is struck through~~" in rendered_md or "<del>" in rendered_md
