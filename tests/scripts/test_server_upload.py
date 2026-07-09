import asyncio
import importlib
import io
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

from starlette.datastructures import UploadFile


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


def make_upload_dir(name: str) -> Path:
    upload_dir = Path(__file__).with_name(name)
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir()
    return upload_dir


def test_convert_pdf_upload_sanitizes_filename_and_keeps_path_inside_upload_dir(
    monkeypatch,
):
    server = load_server_module(monkeypatch)
    upload_dir = make_upload_dir("_tmp_uploads_sanitized")
    monkeypatch.setattr(server, "UPLOAD_DIRECTORY", str(upload_dir))
    monkeypatch.setattr(
        server.uuid, "uuid4", lambda: UUID("11111111-1111-1111-1111-111111111111")
    )

    captured = {}

    async def fake_convert(params):
        captured["filepath"] = params.filepath
        return {"success": True}

    monkeypatch.setattr(server, "_convert_pdf", fake_convert)

    try:
        upload = UploadFile(filename="../../etc/passwd.pdf", file=io.BytesIO(b"pdf"))
        result = asyncio.run(
            server.convert_pdf_upload(
                page_range=None,
                force_ocr=False,
                paginate_output=False,
                output_format="markdown",
                file=upload,
            )
        )

        assert result == {"success": True}
        assert Path(captured["filepath"]).parent == upload_dir
        assert (
            Path(captured["filepath"]).name
            == "11111111111111111111111111111111_passwd.pdf"
        )
        assert not Path(captured["filepath"]).exists()
    finally:
        shutil.rmtree(upload_dir)


def test_convert_pdf_upload_uses_unique_paths_for_same_filename(monkeypatch):
    server = load_server_module(monkeypatch)
    upload_dir = make_upload_dir("_tmp_uploads_unique")
    monkeypatch.setattr(server, "UPLOAD_DIRECTORY", str(upload_dir))

    ids = iter(
        [
            UUID("11111111-1111-1111-1111-111111111111"),
            UUID("22222222-2222-2222-2222-222222222222"),
        ]
    )
    monkeypatch.setattr(server.uuid, "uuid4", lambda: next(ids))

    seen_paths = []

    async def fake_convert(params):
        seen_paths.append(params.filepath)
        return {"success": True}

    monkeypatch.setattr(server, "_convert_pdf", fake_convert)

    try:
        first = UploadFile(filename="report.pdf", file=io.BytesIO(b"first"))
        second = UploadFile(filename="report.pdf", file=io.BytesIO(b"second"))

        asyncio.run(
            server.convert_pdf_upload(
                page_range=None,
                force_ocr=False,
                paginate_output=False,
                output_format="markdown",
                file=first,
            )
        )
        asyncio.run(
            server.convert_pdf_upload(
                page_range=None,
                force_ocr=False,
                paginate_output=False,
                output_format="markdown",
                file=second,
            )
        )

        assert len(seen_paths) == 2
        assert len(set(seen_paths)) == 2
        assert all(Path(path).parent == upload_dir for path in seen_paths)
    finally:
        shutil.rmtree(upload_dir)
