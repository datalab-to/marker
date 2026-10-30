import asyncio
import importlib
import os
import sys
import types
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

# CPU-only: POST /marker filepath sandbox, no models or inference server.
pytestmark = pytest.mark.cpu


def load_server_module(monkeypatch):
    fake_config_parser = types.ModuleType("marker.config.parser")
    fake_config_parser.ConfigParser = object

    fake_output = types.ModuleType("marker.output")
    fake_output.text_from_rendered = lambda rendered: ("", None, {})

    fake_pdf_converter = types.ModuleType("marker.converters.pdf")
    fake_pdf_converter.PdfConverter = object

    fake_models = types.ModuleType("marker.models")
    fake_models.create_model_dict = lambda: {}

    fake_settings = types.ModuleType("marker.settings")
    fake_settings.settings = SimpleNamespace(
        OUTPUT_IMAGE_FORMAT="PNG",
        OUTPUT_ENCODING="utf-8",
    )

    monkeypatch.setitem(sys.modules, "marker.config.parser", fake_config_parser)
    monkeypatch.setitem(sys.modules, "marker.output", fake_output)
    monkeypatch.setitem(sys.modules, "marker.converters.pdf", fake_pdf_converter)
    monkeypatch.setitem(sys.modules, "marker.models", fake_models)
    monkeypatch.setitem(sys.modules, "marker.settings", fake_settings)
    monkeypatch.delitem(sys.modules, "marker.scripts.server", raising=False)

    return importlib.import_module("marker.scripts.server")


def _stub_convert(server, monkeypatch):
    called = []

    async def fake_convert(params):
        called.append(params.filepath)
        return {"success": True, "output": "EXFIL"}

    monkeypatch.setattr(server, "_convert_pdf", fake_convert)
    return called


def test_convert_pdf_rejects_absolute_filepath_outside_upload_dir(
    monkeypatch, tmp_path
):
    server = load_server_module(monkeypatch)
    monkeypatch.setattr(server, "UPLOAD_DIRECTORY", str(tmp_path))
    called = _stub_convert(server, monkeypatch)

    secret = tmp_path.parent / "secret.pdf"
    secret.write_bytes(b"%PDF-1.4 secret")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(server.convert_pdf(server.CommonParams(filepath=str(secret))))

    assert exc_info.value.status_code in (400, 403)
    assert called == []


def test_convert_pdf_rejects_path_traversal_outside_upload_dir(monkeypatch, tmp_path):
    server = load_server_module(monkeypatch)
    monkeypatch.setattr(server, "UPLOAD_DIRECTORY", str(tmp_path))
    called = _stub_convert(server, monkeypatch)

    secret = tmp_path.parent / "secret.pdf"
    secret.write_bytes(b"%PDF-1.4 secret")
    traversal = os.path.join(str(tmp_path), os.pardir, "secret.pdf")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(server.convert_pdf(server.CommonParams(filepath=traversal)))

    assert exc_info.value.status_code in (400, 403)
    assert called == []


def test_convert_pdf_allows_filepath_inside_upload_dir(monkeypatch, tmp_path):
    server = load_server_module(monkeypatch)
    monkeypatch.setattr(server, "UPLOAD_DIRECTORY", str(tmp_path))
    called = _stub_convert(server, monkeypatch)

    allowed = tmp_path / "doc.pdf"
    allowed.write_bytes(b"%PDF-1.4 ok")

    result = asyncio.run(server.convert_pdf(server.CommonParams(filepath=str(allowed))))

    assert result == {"success": True, "output": "EXFIL"}
    assert called == [str(allowed)]
