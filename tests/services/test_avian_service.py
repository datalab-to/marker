import json
from unittest.mock import MagicMock, Mock, patch

import pytest
from PIL import Image
from pydantic import BaseModel

from marker.services.avian import AvianService


class SampleSchema(BaseModel):
    text: str
    confidence: float


@pytest.fixture
def avian_service():
    return AvianService(config={"avian_api_key": "test-key"})


def test_default_config(avian_service):
    assert avian_service.avian_api_key == "test-key"
    assert avian_service.avian_model == "deepseek-v3.2"
    assert avian_service.avian_image_format == "png"


def test_custom_model():
    service = AvianService(
        config={"avian_api_key": "test-key", "avian_model": "kimi-k2.5"}
    )
    assert service.avian_model == "kimi-k2.5"


def test_get_client(avian_service):
    client = avian_service.get_client()
    assert client.api_key == "test-key"
    assert client.base_url.host == "api.avian.io"
    assert str(client.base_url).rstrip("/").endswith("/v1")


def test_process_images_single(avian_service):
    img = Image.new("RGB", (10, 10), color="red")
    result = avian_service.process_images(img)
    assert len(result) == 1
    assert result[0]["type"] == "image_url"
    assert result[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_process_images_list(avian_service):
    images = [
        Image.new("RGB", (10, 10), color="red"),
        Image.new("RGB", (10, 10), color="blue"),
    ]
    result = avian_service.process_images(images)
    assert len(result) == 2
    for item in result:
        assert item["type"] == "image_url"
        assert item["image_url"]["url"].startswith("data:image/png;base64,")


def test_call_success(avian_service):
    expected = {"text": "hello", "confidence": 0.95}
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(expected)
    mock_response.usage.total_tokens = 100

    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse.return_value = mock_response

    with patch.object(avian_service, "get_client", return_value=mock_client):
        result = avian_service(
            prompt="Extract text",
            image=None,
            block=None,
            response_schema=SampleSchema,
        )

    assert result == expected
    mock_client.beta.chat.completions.parse.assert_called_once()
    call_kwargs = mock_client.beta.chat.completions.parse.call_args
    assert call_kwargs.kwargs["model"] == "deepseek-v3.2"


def test_call_updates_block_metadata(avian_service):
    expected = {"text": "hello", "confidence": 0.95}
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(expected)
    mock_response.usage.total_tokens = 42

    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse.return_value = mock_response

    mock_block = MagicMock()

    with patch.object(avian_service, "get_client", return_value=mock_client):
        result = avian_service(
            prompt="Extract text",
            image=None,
            block=mock_block,
            response_schema=SampleSchema,
        )

    assert result == expected
    mock_block.update_metadata.assert_called_once_with(
        llm_tokens_used=42, llm_request_count=1
    )


def test_call_rate_limit_retries(avian_service):
    from openai import RateLimitError

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.headers = {}
    mock_client.beta.chat.completions.parse.side_effect = RateLimitError(
        message="rate limited",
        response=mock_resp,
        body=None,
    )

    with patch.object(avian_service, "get_client", return_value=mock_client):
        with patch("marker.services.avian.time.sleep"):
            result = avian_service(
                prompt="Extract text",
                image=None,
                block=None,
                response_schema=SampleSchema,
                max_retries=2,
            )

    assert result == {}
    assert mock_client.beta.chat.completions.parse.call_count == 3


def test_call_generic_exception_no_retry(avian_service):
    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse.side_effect = ValueError("bad input")

    with patch.object(avian_service, "get_client", return_value=mock_client):
        result = avian_service(
            prompt="Extract text",
            image=None,
            block=None,
            response_schema=SampleSchema,
        )

    assert result == {}
    assert mock_client.beta.chat.completions.parse.call_count == 1


def test_call_with_image(avian_service):
    expected = {"text": "hello", "confidence": 0.95}
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(expected)
    mock_response.usage.total_tokens = 50

    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse.return_value = mock_response

    img = Image.new("RGB", (10, 10), color="red")

    with patch.object(avian_service, "get_client", return_value=mock_client):
        result = avian_service(
            prompt="Describe image",
            image=img,
            block=None,
            response_schema=SampleSchema,
        )

    assert result == expected
    call_kwargs = mock_client.beta.chat.completions.parse.call_args
    messages = call_kwargs.kwargs["messages"]
    content = messages[0]["content"]
    # Should have image data + text prompt
    assert len(content) == 2
    assert content[0]["type"] == "image_url"
    assert content[1]["type"] == "text"


def test_missing_api_key_raises():
    with pytest.raises(AssertionError):
        AvianService(config={})


def test_conditional_import_error():
    """Verify helpful error when openai is not installed."""
    import importlib
    import sys

    import marker.services.avian as avian_mod

    # Temporarily hide the openai module
    real_openai = sys.modules.get("openai")
    sys.modules["openai"] = None
    try:
        importlib.reload(avian_mod)
        with pytest.raises(ImportError, match="openai"):
            avian_mod._import_openai()
    finally:
        if real_openai is not None:
            sys.modules["openai"] = real_openai
        else:
            sys.modules.pop("openai", None)
        importlib.reload(avian_mod)
