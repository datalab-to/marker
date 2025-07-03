import pytest

from marker.converters.pdf import PdfConverter
from marker.services.gemini import GoogleGeminiService
from marker.services.ollama import OllamaService
from marker.services.vertex import GoogleVertexService
from marker.services.openai import OpenAIService
from marker.services.azure_openai import AzureOpenAIService


@pytest.mark.output_format("markdown")
@pytest.mark.config({"page_range": [0]})
def test_empty_llm(pdf_converter: PdfConverter, temp_doc):
    assert pdf_converter.artifact_dict["llm_service"] is None
    assert pdf_converter.llm_service is None


def test_llm_no_keys(model_dict, config):
    with pytest.raises(AssertionError):
        PdfConverter(artifact_dict=model_dict, config={"use_llm": True})


@pytest.mark.output_format("markdown")
@pytest.mark.config({"page_range": [0], "use_llm": True, "gemini_api_key": "test"})
def test_llm_gemini(pdf_converter: PdfConverter, temp_doc):
    assert pdf_converter.artifact_dict["llm_service"] is not None
    assert isinstance(pdf_converter.llm_service, GoogleGeminiService)


@pytest.mark.output_format("markdown")
@pytest.mark.config(
    {
        "page_range": [0],
        "use_llm": True,
        "vertex_project_id": "test",
        "llm_service": "marker.services.vertex.GoogleVertexService",
    }
)
def test_llm_vertex(pdf_converter: PdfConverter, temp_doc):
    assert pdf_converter.artifact_dict["llm_service"] is not None
    assert isinstance(pdf_converter.llm_service, GoogleVertexService)


@pytest.mark.output_format("markdown")
@pytest.mark.config(
    {
        "page_range": [0],
        "use_llm": True,
        "llm_service": "marker.services.ollama.OllamaService",
    }
)
def test_llm_ollama(pdf_converter: PdfConverter, temp_doc):
    assert pdf_converter.artifact_dict["llm_service"] is not None
    assert isinstance(pdf_converter.llm_service, OllamaService)


@pytest.mark.output_format("markdown")
@pytest.mark.config(
    {
        "page_range": [0],
        "use_llm": True,
        "llm_service": "marker.services.openai.OpenAIService",
        "openai_api_key": "test",
    }
)
def test_llm_openai(pdf_converter: PdfConverter, temp_doc):
    assert pdf_converter.artifact_dict["llm_service"] is not None
    assert isinstance(pdf_converter.llm_service, OpenAIService)


@pytest.mark.output_format("markdown")
@pytest.mark.config(
    {
        "page_range": [0],
        "use_llm": True,
        "llm_service": "marker.services.azure_openai.AzureOpenAIService",
        "azure_endpoint": "https://example.openai.azure.com",
        "azure_api_key": "test",
        "deployment_name": "test-model",
        "azure_api_version": "1",
    }
)
def test_llm_azure_openai(pdf_converter: PdfConverter, temp_doc):
    assert pdf_converter.artifact_dict["llm_service"] is not None
    assert isinstance(pdf_converter.llm_service, AzureOpenAIService)
