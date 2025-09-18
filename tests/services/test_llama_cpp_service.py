import pytest
from unittest.mock import Mock, patch
import requests
import json

from marker.converters.pdf import PdfConverter
from marker.services.llama_cpp import LlamaCPPService
from marker.models import create_model_dict
from marker.processors.llm.llm_complex import LLMComplexRegionProcessor
from marker.processors.llm.llm_meta import LLMSimpleBlockMetaProcessor
from pydantic import BaseModel


def test_llm_llama_cpp_injection():
    """Test that LlamaCPPService can be injected into PdfConverter"""
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
    
    # Clean up
    del model_dict


def test_llm_llama_cpp_config_parameters():
    """Test that LlamaCPPService handles configuration parameters correctly"""
    # Test with custom configuration
    service = LlamaCPPService({
        "llama_cpp_base_url": "http://192.168.68.186:8080",
        "llama_cpp_model": "NuMarkdown",
        "timeout": 45,
        "max_retries": 3,
        "retry_wait_time": 5,
    })
    
    # Check that configuration parameters are set correctly
    assert service.llama_cpp_base_url == "http://192.168.68.186:8080"
    assert service.llama_cpp_model == "NuMarkdown"
    assert service.timeout == 45
    assert service.max_retries == 3
    assert service.retry_wait_time == 5
    
    # Test with default configuration
    service_default = LlamaCPPService()
    
    # Check that default configuration parameters are set correctly
    assert service_default.llama_cpp_base_url == "http://192.168.68.186:8080"
    assert service_default.llama_cpp_model == "NuMarkdown"
    assert service_default.timeout == 30
    assert service_default.max_retries == 2
    assert service_default.retry_wait_time == 3


def test_llm_llama_cpp_with_processors():
    """Test that LlamaCPPService works with LLM processors"""
    # Create a mock LLM service
    mock_service = Mock()
    mock_service.return_value = {"corrected_markdown": "Test markdown"}
    
    # Test with LLMComplexRegionProcessor (BaseLLMSimpleBlockProcessor)
    processor = LLMComplexRegionProcessor({"use_llm": True})
    assert processor.use_llm == True
    
    # Test with LLMSimpleBlockMetaProcessor
    meta_processor = LLMSimpleBlockMetaProcessor([processor], mock_service, {"use_llm": True})
    assert meta_processor.llm_service == mock_service
    assert len(meta_processor.processors) == 1
    assert meta_processor.processors[0] == processor


class TestSchema(BaseModel):
    test_field: str


def test_llm_llama_cpp_error_handling():
    """Test that LlamaCPPService handles errors correctly"""
    # Create a service with custom retry configuration
    service = LlamaCPPService({
        "llama_cpp_base_url": "http://192.168.68.186:8080",
        "llama_cpp_model": "NuMarkdown",
        "max_retries": 2,
        "retry_wait_time": 1
    })
    
    # Test timeout error handling
    with patch('requests.post') as mock_post:
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
        result = service("Test prompt", None, None, TestSchema)
        assert result == {}  # Should return empty dict on failure
    
    # Test connection error handling
    with patch('requests.post') as mock_post:
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")
        result = service("Test prompt", None, None, TestSchema)
        assert result == {}  # Should return empty dict on failure
    
    # Test HTTP error handling
    with patch('requests.post') as mock_post:
        mock_post.side_effect = requests.exceptions.HTTPError("HTTP error")
        result = service("Test prompt", None, None, TestSchema)
        assert result == {}  # Should return empty dict on failure
    
    # Test JSON decode error handling
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"invalid": "json"}
        mock_post.return_value = mock_response
        
        result = service("Test prompt", None, None, TestSchema)
        assert result == {}  # Should return empty dict on failure
    
    # Test successful response
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "response": '{"test_field": "test_value"}',
            "prompt_eval_count": 10,
            "eval_count": 20
        }
        mock_post.return_value = mock_response
        
        result = service("Test prompt", None, None, TestSchema)
        assert result == {"test_field": "test_value"}