import json
import pytest

from marker.converters.extraction import ExtractionConverter
from marker.extractors.page import PageExtractionSchema
from marker.extractors.document import DocumentExtractionSchema
from marker.services import BaseService


class MockLLMService(BaseService):
    def __call__(self, prompt, image=None, page=None, response_schema=None, **kwargs):
        if response_schema == PageExtractionSchema:
            return {
                "description": "Mock extraction description",
                "detailed_notes": "Mock detailed notes for page extraction",
            }
        elif response_schema == DocumentExtractionSchema:
            return {
                "analysis": "Mock document analysis",
                "document_json": json.dumps({"test_key": "test_value"}),
            }
        return {}


@pytest.fixture
def mock_llm_service():
    return MockLLMService


@pytest.fixture
def extraction_converter(config, model_dict, mock_llm_service):
    test_schema = {
        "title": "TestSchema",
        "type": "object",
        "properties": {"test_key": {"title": "Test Key", "type": "string"}},
        "required": ["test_key"],
    }

    config["page_schema"] = json.dumps(test_schema)
    config["output_format"] = "markdown"
    model_dict["llm_service"] = mock_llm_service

    converter = ExtractionConverter(
        artifact_dict=model_dict, processor_list=None, config=config
    )
    converter.llm_service = mock_llm_service
    converter.default_llm_service = MockLLMService
    return converter


@pytest.mark.config({"page_range": [0]})
def test_extraction_converter(config, model_dict, mock_llm_service, temp_doc):
    config["page_schema"] = "invalid json"

    model_dict["llm_service"] = mock_llm_service
    converter = ExtractionConverter(
        artifact_dict=model_dict, processor_list=None, config=config
    )
    converter.artifact_dict["llm_service"] = mock_llm_service()

    results = converter(temp_doc.name)
    assert results.document_json == '{"test_key": "test_value"}'


@pytest.mark.config({"page_range": [0, 1]})
def test_extraction_converter_multiple_pages(extraction_converter, temp_doc):
    result = extraction_converter(temp_doc.name)

    assert result is not None
    assert result.document_json is not None
    assert json.loads(result.document_json) == {"test_key": "test_value"}
    assert result.analysis == "Mock document analysis"
