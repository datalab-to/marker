import pytest
from unittest.mock import Mock, patch
import json

from marker.converters.pdf import PdfConverter
from marker.services.llama_cpp import LlamaCPPService
from marker.services.openai import OpenAIService
from marker.services.ollama import OllamaService
from marker.models import create_model_dict
from marker.processors.llm.llm_complex import LLMComplexRegionProcessor
from marker.processors.llm.llm_meta import LLMSimpleBlockMetaProcessor
from pydantic import BaseModel


class TestSchema(BaseModel):
    test_field: str


def test_existing_llm_services_still_work():
    """Test that existing LLM services still work correctly after LlamaCPPService implementation"""
    # Create a minimal model dict
    model_dict = create_model_dict()
    
    # Test OpenAI Service
    converter = PdfConverter(
        artifact_dict=model_dict,
        config={
            "use_llm": True,
            "llm_service": "marker.services.openai.OpenAIService",
            "openai_api_key": "test-key",
        },
        llm_service="marker.services.openai.OpenAIService"
    )
    
    assert converter.artifact_dict["llm_service"] is not None
    assert isinstance(converter.llm_service, OpenAIService)
    
    # Test Ollama Service
    converter = PdfConverter(
        artifact_dict=model_dict,
        config={
            "use_llm": True,
            "llm_service": "marker.services.ollama.OllamaService",
        },
        llm_service="marker.services.ollama.OllamaService"
    )
    
    assert converter.artifact_dict["llm_service"] is not None
    assert isinstance(converter.llm_service, OllamaService)
    
    # Test LlamaCPP Service
    converter = PdfConverter(
        artifact_dict=model_dict,
        config={
            "use_llm": True,
            "llm_service": "marker.services.llama_cpp.LlamaCPPService",
            "llama_cpp_base_url": "http://192.168.68.186:8080",
        },
        llm_service="marker.services.llama_cpp.LlamaCPPService"
    )
    
    assert converter.artifact_dict["llm_service"] is not None
    assert isinstance(converter.llm_service, LlamaCPPService)
    
    # Clean up
    del model_dict


def test_llama_cpp_service_drop_in_replacement():
    """Test that LlamaCPPService can be used as a drop-in replacement for existing LLM services"""
    # Create a minimal model dict
    model_dict = create_model_dict()
    
    # Test with LlamaCPPService
    converter = PdfConverter(
        artifact_dict=model_dict,
        config={
            "use_llm": True,
            "llm_service": "marker.services.llama_cpp.LlamaCPPService",
            "llama_cpp_base_url": "http://192.168.68.186:8080",
            "llama_cpp_model": "NuMarkdown",
        },
        llm_service="marker.services.llama_cpp.LlamaCPPService"
    )
    
    assert converter.artifact_dict["llm_service"] is not None
    assert isinstance(converter.llm_service, LlamaCPPService)
    assert converter.llm_service.llama_cpp_base_url == "http://192.168.68.186:8080"
    assert converter.llm_service.llama_cpp_model == "NuMarkdown"
    
    # Clean up
    del model_dict


def test_configuration_options_not_affected():
    """Test that configuration options for existing services are not affected by new implementations"""
    # Create a minimal model dict
    model_dict = create_model_dict()
    
    # Test OpenAI Service configuration
    converter = PdfConverter(
        artifact_dict=model_dict,
        config={
            "use_llm": True,
            "llm_service": "marker.services.openai.OpenAIService",
            "openai_api_key": "test-key",
            "openai_base_url": "https://api.openai.com/v1",
            "openai_model": "gpt-4",
            "timeout": 60,
            "max_retries": 5,
        },
        llm_service="marker.services.openai.OpenAIService"
    )
    
    assert converter.artifact_dict["llm_service"] is not None
    assert isinstance(converter.llm_service, OpenAIService)
    assert converter.llm_service.openai_base_url == "https://api.openai.com/v1"
    assert converter.llm_service.openai_model == "gpt-4"
    assert converter.llm_service.timeout == 60
    assert converter.llm_service.max_retries == 5
    
    # Test LlamaCPP Service configuration
    converter = PdfConverter(
        artifact_dict=model_dict,
        config={
            "use_llm": True,
            "llm_service": "marker.services.llama_cpp.LlamaCPPService",
            "llama_cpp_base_url": "http://192.168.68.186:8080",
            "llama_cpp_model": "NuMarkdown",
            "timeout": 30,
            "max_retries": 2,
        },
        llm_service="marker.services.llama_cpp.LlamaCPPService"
    )
    
    assert converter.artifact_dict["llm_service"] is not None
    assert isinstance(converter.llm_service, LlamaCPPService)
    assert converter.llm_service.llama_cpp_base_url == "http://192.168.68.186:8080"
    assert converter.llm_service.llama_cpp_model == "NuMarkdown"
    assert converter.llm_service.timeout == 30
    assert converter.llm_service.max_retries == 2
    
    # Clean up
    del model_dict


def test_switching_between_llm_services():
    """Test that the system can switch between different LLM services without issues"""
    # Create a minimal model dict
    model_dict = create_model_dict()
    
    # Test switching from OpenAI to LlamaCPP
    converter = PdfConverter(
        artifact_dict=model_dict,
        config={
            "use_llm": True,
            "llm_service": "marker.services.openai.OpenAIService",
            "openai_api_key": "test-key",
        },
        llm_service="marker.services.openai.OpenAIService"
    )
    
    assert isinstance(converter.llm_service, OpenAIService)
    
    # Switch to LlamaCPP
    converter = PdfConverter(
        artifact_dict=model_dict,
        config={
            "use_llm": True,
            "llm_service": "marker.services.llama_cpp.LlamaCPPService",
            "llama_cpp_base_url": "http://192.168.68.186:8080",
        },
        llm_service="marker.services.llama_cpp.LlamaCPPService"
    )
    
    assert isinstance(converter.llm_service, LlamaCPPService)
    
    # Switch to Ollama
    converter = PdfConverter(
        artifact_dict=model_dict,
        config={
            "use_llm": True,
            "llm_service": "marker.services.ollama.OllamaService",
        },
        llm_service="marker.services.ollama.OllamaService"
    )
    
    assert isinstance(converter.llm_service, OllamaService)
    
    # Clean up
    del model_dict


@patch('requests.post')
def test_error_handling_all_services(mock_post):
    """Test that error handling works correctly for all LLM services"""
    # Mock a timeout error
    mock_post.side_effect = Exception("Connection failed")
    
    # Test OpenAI Service error handling
    openai_service = OpenAIService({
        "openai_api_key": "test-key",
        "max_retries": 1,
        "retry_wait_time": 0.1
    })
    
    result = openai_service("Test prompt", None, None, TestSchema)
    assert result == {}  # Should return empty dict on failure
    
    # Test LlamaCPP Service error handling
    llama_service = LlamaCPPService({
        "llama_cpp_base_url": "http://192.168.68.186:8080",
        "max_retries": 1,
        "retry_wait_time": 0.1
    })
    
    result = llama_service("Test prompt", None, None, TestSchema)
    assert result == {}  # Should return empty dict on failure


@patch('requests.post')
def test_successful_responses_all_services(mock_post):
    """Test that successful responses work correctly for all LLM services"""
    # Mock a successful response for LlamaCPP
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "response": '{"test_field": "test_value"}',
        "prompt_eval_count": 10,
        "eval_count": 20
    }
    mock_post.return_value = mock_response
    
    # Test LlamaCPP Service successful response
    llama_service = LlamaCPPService({
        "llama_cpp_base_url": "http://192.168.68.186:8080",
        "max_retries": 1
    })
    
    result = llama_service("Test prompt", None, None, TestSchema)
    assert result == {"test_field": "test_value"}
    
    # Mock a successful response for OpenAI
    with patch.object(OpenAIService, 'get_client') as mock_client:
        mock_completion = Mock()
        mock_choice = Mock()
        mock_choice.message.content = '{"test_field": "test_value"}'
        mock_completion.choices = [mock_choice]
        mock_completion.usage.total_tokens = 30
        mock_client.return_value.beta.chat.completions.parse.return_value = mock_completion
        
        openai_service = OpenAIService({
            "openai_api_key": "test-key",
            "max_retries": 1
        })
        
        result = openai_service("Test prompt", None, None, TestSchema)
        assert result == {"test_field": "test_value"}