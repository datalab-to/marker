import json
from types import SimpleNamespace

from pydantic import BaseModel

from marker.services.litellm import LiteLLMService


class _Schema(BaseModel):
    answer: str


def _fake_response(content: str, total_tokens: int = 7):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(total_tokens=total_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_litellm_dispatch_and_drop_params(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _fake_response(json.dumps({"answer": "4"}))

    monkeypatch.setattr("litellm.completion", fake_completion)

    service = LiteLLMService(
        {"litellm_model": "gemini/gemini-2.5-flash", "litellm_api_key": "secret"}
    )
    result = service("What is 2+2?", None, None, _Schema)

    assert result == {"answer": "4"}
    assert captured["model"] == "gemini/gemini-2.5-flash"
    assert captured["drop_params"] is True
    assert captured["response_format"] is _Schema
    assert captured["api_key"] == "secret"
    # Message carries the text prompt.
    assert captured["messages"][0]["content"][-1]["text"] == "What is 2+2?"


def test_litellm_omits_credentials_when_unset(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _fake_response(json.dumps({"answer": "ok"}))

    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.setattr("litellm.completion", fake_completion)

    service = LiteLLMService({"litellm_model": "gpt-5-mini"})
    service("hi", None, None, _Schema)

    # No creds passed -> LiteLLM falls back to the provider's own env vars.
    assert "api_key" not in captured
    assert "api_base" not in captured


def test_litellm_drop_params_opt_out(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _fake_response(json.dumps({"answer": "ok"}))

    monkeypatch.setattr("litellm.completion", fake_completion)

    service = LiteLLMService(
        {"litellm_model": "gpt-5-mini", "litellm_drop_params": False}
    )
    service("hi", None, None, _Schema)

    assert captured["drop_params"] is False


def test_litellm_base_url_forwarded(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _fake_response(json.dumps({"answer": "ok"}))

    monkeypatch.setattr("litellm.completion", fake_completion)

    service = LiteLLMService(
        {
            "litellm_model": "litellm_proxy/gpt-4o-mini",
            "litellm_base_url": "http://localhost:4000",
        }
    )
    service("hi", None, None, _Schema)

    assert captured["api_base"] == "http://localhost:4000"


def test_litellm_retries_on_transient_error(monkeypatch):
    from litellm.exceptions import InternalServerError

    calls = {"n": 0}

    def flaky_completion(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise InternalServerError(
                message="boom", llm_provider="gemini", model="gemini/gemini-2.5-flash"
            )
        return _fake_response(json.dumps({"answer": "recovered"}))

    monkeypatch.setattr("litellm.completion", flaky_completion)

    service = LiteLLMService(
        {"litellm_model": "gemini/gemini-2.5-flash", "litellm_api_key": "x"}
    )
    service.retry_wait_time = 0
    result = service("hi", None, None, _Schema)

    assert calls["n"] == 2
    assert result == {"answer": "recovered"}


def test_litellm_gives_up_and_returns_empty(monkeypatch):
    calls = {"n": 0}

    def bad_completion(**kwargs):
        calls["n"] += 1
        raise ValueError("permanent failure")

    monkeypatch.setattr("litellm.completion", bad_completion)

    service = LiteLLMService({"litellm_model": "gpt-5-mini", "litellm_api_key": "x"})
    result = service("hi", None, None, _Schema)

    # Non-retryable error -> single attempt, empty dict (matches sibling services).
    assert calls["n"] == 1
    assert result == {}
