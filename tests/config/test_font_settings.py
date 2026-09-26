import os

import pytest

from marker.settings import Settings

# CPU-only: settings construction, no models or inference server.
pytestmark = pytest.mark.cpu


def test_font_path_follows_font_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("FONT_PATH", raising=False)
    monkeypatch.setenv("FONT_DIR", str(tmp_path))
    settings = Settings()
    assert settings.FONT_PATH == os.path.join(str(tmp_path), settings.FONT_NAME)


def test_explicit_font_path_is_honoured(tmp_path, monkeypatch):
    monkeypatch.setenv("FONT_PATH", str(tmp_path / "custom.ttf"))
    assert Settings().FONT_PATH == str(tmp_path / "custom.ttf")


def test_default_font_dir_is_outside_the_package(monkeypatch):
    monkeypatch.delenv("FONT_DIR", raising=False)
    monkeypatch.delenv("FONT_PATH", raising=False)
    settings = Settings()
    assert not settings.FONT_PATH.startswith(settings.BASE_DIR)
