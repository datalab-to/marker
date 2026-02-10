"""Tests for OpenAI service fallback mechanism.

Tests the fallback behavior when OpenAI-compatible APIs don't support
Structured Outputs (response_format parameter).
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from pydantic import BaseModel

from marker.services.openai import OpenAIService


class MockResponseSchema(BaseModel):
    """Simple schema for testing JSON parsing."""
    title: str
    content: str


class TestValidateResponse:
    """Tests for validate_response() JSON parsing."""

    def test_parses_clean_json(self):
        """Test parsing clean JSON response."""
        service = OpenAIService(config={"openai_api_key": "test"})
        response = '{"title": "Test", "content": "Hello"}'
        result = service.validate_response(response, MockResponseSchema)
        assert result == {"title": "Test", "content": "Hello"}

    def test_parses_json_with_markdown_fence(self):
        """Test parsing JSON wrapped in markdown code fence."""
        service = OpenAIService(config={"openai_api_key": "test"})
        response = '```json\n{"title": "Test", "content": "Hello"}\n```'
        result = service.validate_response(response, MockResponseSchema)
        assert result == {"title": "Test", "content": "Hello"}

    def test_parses_json_with_generic_fence(self):
        """Test parsing JSON wrapped in generic code fence."""
        service = OpenAIService(config={"openai_api_key": "test"})
        response = '```\n{"title": "Test", "content": "Hello"}\n```'
        result = service.validate_response(response, MockResponseSchema)
        assert result == {"title": "Test", "content": "Hello"}

    def test_parses_json_with_whitespace(self):
        """Test parsing JSON with surrounding whitespace."""
        service = OpenAIService(config={"openai_api_key": "test"})
        response = '  \n{"title": "Test", "content": "Hello"}\n  '
        result = service.validate_response(response, MockResponseSchema)
        assert result == {"title": "Test", "content": "Hello"}

    def test_returns_none_for_invalid_json(self):
        """Test that invalid JSON returns None."""
        service = OpenAIService(config={"openai_api_key": "test"})
        response = 'This is not JSON at all'
        result = service.validate_response(response, MockResponseSchema)
        assert result is None

    def test_returns_none_for_schema_mismatch(self):
        """Test that valid JSON not matching schema returns None."""
        service = OpenAIService(config={"openai_api_key": "test"})
        response = '{"wrong_field": "value"}'
        result = service.validate_response(response, MockResponseSchema)
        assert result is None


class TestOpenAIServiceConfig:
    """Tests for OpenAI service configuration."""

    def test_disable_structured_output_default_false(self):
        """Test that structured output is enabled by default."""
        service = OpenAIService(config={"openai_api_key": "test"})
        assert service.openai_disable_structured_output is False

    def test_disable_structured_output_can_be_enabled(self):
        """Test that structured output can be disabled via config."""
        service = OpenAIService(config={
            "openai_api_key": "test",
            "openai_disable_structured_output": True
        })
        assert service.openai_disable_structured_output is True


class TestOpenAIServiceFallback:
    """Tests for the fallback mechanism when Structured Outputs fails."""

    @patch('marker.services.openai.openai.OpenAI')
    def test_uses_structured_output_by_default(self, mock_openai_class):
        """Test that Structured Outputs is used when not disabled."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"title": "Test", "content": "Hello"}'))]
        mock_response.usage.total_tokens = 100
        mock_client.beta.chat.completions.parse.return_value = mock_response
        
        service = OpenAIService(config={"openai_api_key": "test"})
        result = service("Test prompt", None, None, MockResponseSchema)
        
        # Should have called the structured output method
        mock_client.beta.chat.completions.parse.assert_called_once()
        assert result == {"title": "Test", "content": "Hello"}

    @patch('marker.services.openai.openai.OpenAI')
    def test_uses_fallback_when_disabled(self, mock_openai_class):
        """Test that fallback is used when structured output is disabled."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"title": "Test", "content": "Hello"}'))]
        mock_response.usage.total_tokens = 100
        mock_client.chat.completions.create.return_value = mock_response
        
        service = OpenAIService(config={
            "openai_api_key": "test",
            "openai_disable_structured_output": True
        })
        result = service("Test prompt", None, None, MockResponseSchema)
        
        # Should have called the plain completions method, not structured
        mock_client.chat.completions.create.assert_called_once()
        mock_client.beta.chat.completions.parse.assert_not_called()
        assert result == {"title": "Test", "content": "Hello"}

    @patch('marker.services.openai.openai.OpenAI')
    def test_fallback_on_bad_request_error(self, mock_openai_class):
        """Test automatic fallback when BadRequestError indicates unsupported response_format."""
        from openai import BadRequestError
        
        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # First call raises BadRequestError
        mock_client.beta.chat.completions.parse.side_effect = BadRequestError(
            message="This response_format type is unavailable now",
            response=MagicMock(status_code=400),
            body={"error": {"message": "This response_format type is unavailable now"}}
        )
        
        # Fallback call succeeds
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"title": "Fallback", "content": "Works"}'))]
        mock_response.usage.total_tokens = 100
        mock_client.chat.completions.create.return_value = mock_response
        
        service = OpenAIService(config={"openai_api_key": "test"})
        result = service("Test prompt", None, None, MockResponseSchema)
        
        # Should have tried structured output first, then fallen back
        mock_client.beta.chat.completions.parse.assert_called_once()
        mock_client.chat.completions.create.assert_called_once()
        assert result == {"title": "Fallback", "content": "Works"}
