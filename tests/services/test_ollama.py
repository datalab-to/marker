from unittest.mock import Mock

import pytest
import requests
from pydantic import BaseModel

from marker.schema.blocks import Text
from marker.schema.polygon import PolygonBox
from marker.services.ollama import OllamaService


class ResponseSchema(BaseModel):
    value: str


@pytest.mark.cpu
@pytest.mark.parametrize(
    "counts, expected_tokens",
    [
        ({}, 0),
        ({"prompt_eval_count": 2}, 2),
        ({"eval_count": 3}, 3),
        ({"prompt_eval_count": 2, "eval_count": 3}, 5),
    ],
)
@pytest.mark.parametrize("with_block", [False, True])
def test_optional_token_counts(monkeypatch, counts, expected_tokens, with_block):
    response = Mock(spec=requests.Response)
    response.json.return_value = {"response": '{"value": "ok"}', **counts}
    monkeypatch.setattr(
        "marker.services.ollama.requests.post", Mock(return_value=response)
    )
    block = Text(polygon=PolygonBox.from_bbox([0, 0, 10, 10])) if with_block else None

    result = OllamaService()("prompt", None, block, ResponseSchema)

    assert result == {"value": "ok"}
    if block is not None:
        assert block.metadata.llm_request_count == 1
        assert block.metadata.llm_tokens_used == expected_tokens
