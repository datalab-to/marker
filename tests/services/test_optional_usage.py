import json

import httpx
import pytest
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel

from marker.schema.blocks.base import Block, BlockMetadata
from marker.schema.polygon import PolygonBox
from marker.services.azure_openai import AzureOpenAIService
from marker.services.openai import OpenAIService


class Reply(BaseModel):
    answer: str


@pytest.mark.parametrize("provider", ["openai", "azure"])
@pytest.mark.parametrize("usage_kind", ["omitted", "null", "counts"])
@pytest.mark.parametrize("with_block", [False, True])
def test_response_without_usage_keeps_parsed_content(
    monkeypatch, provider, usage_kind, with_block
):
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"answer": "ok"}),
                    "refusal": None,
                },
                "finish_reason": "stop",
            }
        ],
    }
    if usage_kind != "omitted":
        payload["usage"] = (
            {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
            if usage_kind == "counts"
            else None
        )
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=payload)

    http_client = httpx.Client(transport=httpx.MockTransport(respond))
    if provider == "openai":
        service = OpenAIService({"openai_api_key": "test-key"})
        client = OpenAI(
            api_key="test-key",
            base_url="https://example.test/v1",
            http_client=http_client,
        )
    else:
        service = AzureOpenAIService(
            {
                "azure_api_key": "test-key",
                "azure_endpoint": "https://example.test",
                "azure_api_version": "2024-10-21",
                "deployment_name": "test-deployment",
            }
        )
        client = AzureOpenAI(
            api_key="test-key",
            azure_endpoint="https://example.test",
            api_version="2024-10-21",
            http_client=http_client,
        )

    block = (
        Block(
            polygon=PolygonBox.from_bbox([0, 0, 10, 10]),
            block_description="test",
            metadata=BlockMetadata(llm_tokens_used=10, llm_request_count=2),
        )
        if with_block
        else None
    )
    with client:
        monkeypatch.setattr(service, "get_client", lambda: client)
        assert service("test", None, block, Reply) == {"answer": "ok"}

    assert len(requests) == 1
    if block is not None:
        assert block.metadata.llm_request_count == 3
        assert block.metadata.llm_tokens_used == (15 if usage_kind == "counts" else 10)
