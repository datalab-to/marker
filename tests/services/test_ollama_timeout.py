import json
from unittest.mock import Mock

import pytest
import requests
from pydantic import BaseModel

from marker.services.ollama import OllamaService


class Reply(BaseModel):
    answer: str


@pytest.fixture
def post(monkeypatch):
    response = Mock()
    response.json.return_value = {
        "response": json.dumps({"answer": "ok"}),
        "prompt_eval_count": 2,
        "eval_count": 3,
    }
    post = Mock(return_value=response)
    monkeypatch.setattr("marker.services.ollama.requests.post", post)
    return post


@pytest.mark.parametrize(
    "configured_timeout,call_timeout,expected",
    [(None, None, 30), (7, None, 7), (7, 11, 11), (None, 11, 11)],
)
def test_ollama_request_honors_timeout(
    post, configured_timeout, call_timeout, expected
):
    config = {} if configured_timeout is None else {"timeout": configured_timeout}
    service = OllamaService(config)

    assert service("test", None, None, Reply, timeout=call_timeout) == {"answer": "ok"}
    assert post.call_args.kwargs.get("timeout") == expected


def test_ollama_timeout_override_does_not_change_default(post):
    service = OllamaService({"timeout": 7})

    service("test", None, None, Reply, timeout=11)
    service("test", None, None, Reply)

    assert [call.kwargs.get("timeout") for call in post.call_args_list] == [11, 7]
    assert service.timeout == 7


def test_ollama_request_timeout_preserves_failure_handling(post):
    post.side_effect = requests.Timeout("synthetic timeout")
    service = OllamaService()

    assert service("test", None, None, Reply, timeout=1) == {}
    post.assert_called_once()
