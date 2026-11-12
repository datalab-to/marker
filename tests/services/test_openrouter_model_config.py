import pytest
from pydantic import BaseModel

from marker.services.openrouter import OpenRouterService


class ServiceConfig(BaseModel):
    openrouter_model: str = "test-model"
    max_retries: int = 0


class ServiceConfigWithKey(ServiceConfig):
    openrouter_api_key: str | None = None


@pytest.mark.parametrize(
    "config",
    [ServiceConfig(), ServiceConfigWithKey()],
    ids=["key-not-declared", "key-unset"],
)
def test_openrouter_environment_key_with_model_config(monkeypatch, config):
    monkeypatch.setenv("OPENROUTER_API_KEY", "environment-test-key")
    before = config.model_dump()

    service = OpenRouterService(config)

    assert service.openrouter_api_key == "environment-test-key"
    assert service.openrouter_model == "test-model"
    assert service.max_retries == 0
    assert config.model_dump() == before


@pytest.mark.parametrize("environment_key", [None, "environment-test-key"])
def test_openrouter_model_key_takes_precedence(monkeypatch, environment_key):
    if environment_key is None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENROUTER_API_KEY", environment_key)
    config = ServiceConfigWithKey(openrouter_api_key="configured-test-key")

    service = OpenRouterService(config)

    assert service.openrouter_api_key == "configured-test-key"
    assert config.openrouter_api_key == "configured-test-key"


@pytest.mark.parametrize("config", [None, {}, {"openrouter_api_key": None}])
def test_openrouter_environment_key_without_model_config(monkeypatch, config):
    monkeypatch.setenv("OPENROUTER_API_KEY", "environment-test-key")

    assert OpenRouterService(config).openrouter_api_key == "environment-test-key"


def test_openrouter_model_config_still_requires_a_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(AssertionError, match="openrouter_api_key"):
        OpenRouterService(ServiceConfig())
