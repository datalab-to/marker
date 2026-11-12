import zipfile
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.cpu

import marker.scripts.server as server_mod
from marker.providers.spreadsheet import SpreadSheetProvider
from marker.providers.utils import safe_weasyprint_url_fetcher, validate_zip_archive
from marker.schema.text.span import sanitize_url
from marker.scripts.common import img_to_html, sanitize_html_for_display
from marker.util import strings_to_classes


def test_server_path_traversal_blocked(monkeypatch, tmp_path):
    monkeypatch.setenv("MARKER_ALLOWED_DIR", str(tmp_path))
    client = TestClient(server_mod.app)

    # Attempt path traversal outside MARKER_ALLOWED_DIR
    resp = client.post("/marker", json={"filepath": "/etc/passwd"})
    assert resp.status_code == 403
    assert "not allowed" in resp.json()["detail"]

    resp2 = client.post(
        "/marker", json={"filepath": str(tmp_path / ".." / "pyproject.toml")}
    )
    assert resp2.status_code == 403
    assert "not allowed" in resp2.json()["detail"]


def test_server_disallowed_extension_blocked(monkeypatch, tmp_path):
    monkeypatch.setenv("MARKER_ALLOWED_DIR", str(tmp_path))
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("top secret")

    client = TestClient(server_mod.app)
    resp = client.post("/marker", json={"filepath": str(secret_file)})
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


def test_server_invalid_params_rejected():
    client = TestClient(server_mod.app)
    resp = client.post(
        "/marker",
        json={"filepath": "test.pdf", "output_format": "invalid_format"},
    )
    assert resp.status_code == 422

    resp2 = client.post(
        "/marker",
        json={"filepath": "test.pdf", "mode": "invalid_mode"},
    )
    assert resp2.status_code == 422


def test_server_upload_size_limit_and_cleanup(monkeypatch, tmp_path):
    monkeypatch.setattr(server_mod, "UPLOAD_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("MARKER_ALLOWED_DIR", str(tmp_path))
    monkeypatch.setattr(server_mod, "MAX_UPLOAD_SIZE", 1024)  # 1 KB limit for test

    client = TestClient(server_mod.app)
    oversized_payload = b"%PDF-1.4\n" + (b"A" * 2048)
    resp = client.post(
        "/marker/upload",
        files={"file": ("../../oversized.pdf", oversized_payload, "application/pdf")},
    )
    assert resp.status_code == 413
    # Ensure temp file was cleaned up and nothing written outside tmp_path
    assert list(tmp_path.iterdir()) == []


def test_safe_weasyprint_url_fetcher():
    # External URLs must be blocked
    with pytest.raises(ValueError, match="Blocked external or local URL"):
        safe_weasyprint_url_fetcher("http://169.254.169.254/latest/meta-data/")

    with pytest.raises(ValueError, match="Blocked external or local URL"):
        safe_weasyprint_url_fetcher("https://evil.example.com/payload.css")

    # Arbitrary local file:// URLs must be blocked
    with pytest.raises(ValueError, match="Blocked external or local URL"):
        safe_weasyprint_url_fetcher("file:///etc/passwd")

    # data: URIs are permitted
    res = safe_weasyprint_url_fetcher("data:text/plain;base64,SGVsbG8=")
    payload = res.get("string") or res["file_obj"].read()
    assert payload == b"Hello"


def test_validate_zip_archive(tmp_path):
    zip_path = tmp_path / "bomb.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("huge.xml", b"0" * (2 * 1024 * 1024))

    # Should pass under default limits
    validate_zip_archive(str(zip_path))

    # Should fail when max_uncompressed_bytes is exceeded
    with pytest.raises(ValueError, match="exceeds limit"):
        validate_zip_archive(str(zip_path), max_uncompressed_bytes=1024)


def test_spreadsheet_html_escaping():
    class DummyCell:
        def __init__(self, value, row=1, column=1):
            self.value = value
            self.row = row
            self.column = column

    class DummySheet:
        title = '<script>alert("sheet")</script>'
        merged_cells = type("MC", (), {"ranges": []})()
        rows = [[DummyCell('<img src="http://evil/x" onerror="alert(1)">', 1, 1), DummyCell("a & b < c", 1, 2)]]

    provider = SpreadSheetProvider.__new__(SpreadSheetProvider)
    html_out = provider._excel_to_html_table(DummySheet())
    assert "<script>" not in html_out
    assert '<img src="http://evil/x"' not in html_out
    assert "&lt;img src=" in html_out
    assert "a &amp; b &lt; c" in html_out


def test_strings_to_classes_restricts_arbitrary_imports():
    with pytest.raises(ValueError, match="is not a class"):
        strings_to_classes(["os.system"])

    with pytest.raises(ValueError, match="must be a Marker component subclass"):
        strings_to_classes(["subprocess.Popen"])

    # Valid marker class succeeds
    from marker.processors.table import TableProcessor

    assert strings_to_classes(["marker.processors.table.TableProcessor"]) == [TableProcessor]


def test_sanitize_url_and_html_helpers():
    assert sanitize_url("javascript:alert(1)") is None
    assert sanitize_url("data:text/html,<script>alert(1)</script>") is None
    assert sanitize_url("vbscript:msgbox(1)") is None
    assert sanitize_url("https://example.com/path?a=1&b='2'") == "https://example.com/path?a=1&amp;b=&#x27;2&#x27;"
    assert sanitize_url("mailto:user@example.com") == "mailto:user@example.com"
    assert sanitize_url("#section-1") == "#section-1"

    cleaned = sanitize_html_for_display(
        '<div>Hello<script>evil()</script><a href="javascript:alert(1)" onclick="steal()">link</a><iframe src="x"></iframe></div>'
    )
    assert "<script" not in cleaned
    assert "<iframe" not in cleaned
    assert "onclick" not in cleaned
    assert "javascript:" not in cleaned


def test_viewer_endpoints_and_ui():
    import io
    from PIL import Image, ImageDraw

    client = TestClient(server_mod.app)

    # 1. GET / serves the interactive Marker Viewer HTML
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    html = root_resp.text
    assert "Marker Viewer · Datalab" in html
    for mode in ("auto", "no_ocr", "fast", "balanced", "vllm"):
        assert f'data-v="{mode}"' in html
    for tab in ("Preview", "Markdown", "Blocks", "Lines", "HTML", "Chunks", "JSON"):
        assert tab in html

    # 2. GET /info returns version, modes, and defaults
    info_resp = client.get("/info")
    assert info_resp.status_code == 200
    info = info_resp.json()
    assert info["version"] == "2.1.0"
    assert info["modes"] == ["auto", "no_ocr", "fast", "balanced", "vllm"]
    assert "defaults" in info

    # 3. POST /upload rejects unsupported file extension
    bad_up = client.post(
        "/upload",
        content=b"bad payload",
        headers={"X-Filename": "exploit.exe"},
    )
    assert bad_up.status_code == 400
    assert "Unsupported file type" in bad_up.json()["error"]

    # 4. POST /upload accepts a valid image and returns document id + page count
    img = Image.new("RGB", (600, 400), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((40, 40), "Marker Viewer Test Page", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    up_resp = client.post(
        "/upload",
        content=buf.getvalue(),
        headers={"X-Filename": "sample.png"},
    )
    assert up_resp.status_code == 200
    up_data = up_resp.json()
    assert "id" in up_data
    assert up_data["pages"] == 1
    assert up_data["filename"] == "sample.png"

    # 5. POST /page converts the uploaded page in no_ocr mode
    page_resp = client.post(
        "/page",
        json={
            "id": up_data["id"],
            "n": 0,
            "opts": {"mode": "no_ocr", "dpi": 150},
        },
    )
    assert page_resp.status_code == 200
    p = page_resp.json()
    assert p["page"] == 1
    assert p["mode_used"] == "no_ocr"
    assert p["img"].startswith("data:image/jpeg;base64,")
    assert isinstance(p["blocks"], list)
    assert isinstance(p["lines"], list)
    assert "timings" in p and "total_ms" in p["timings"]
    assert "html" in p and "json" in p and "chunks" in p

    # 6. Also test uploading and converting a multi-format PDF in fast mode
    pdf_buf = io.BytesIO()
    img.save(pdf_buf, format="PDF")
    pdf_up = client.post(
        "/upload",
        content=pdf_buf.getvalue(),
        headers={"X-Filename": "sample.pdf"},
    )
    assert pdf_up.status_code == 200
    pdf_id = pdf_up.json()["id"]
    pdf_page_resp = client.post(
        "/page",
        json={
            "id": pdf_id,
            "n": 0,
            "opts": {"mode": "fast", "dpi": 150, "keep_headers_footers": True},
        },
    )
    assert pdf_page_resp.status_code == 200
    pdf_p = pdf_page_resp.json()
    assert pdf_p["page"] == 1
    assert pdf_p["mode_used"] == "fast"
    assert "timings" in pdf_p


def test_json_renderer_model_dump_with_extracted_images(monkeypatch):
    from marker.renderers.chunk import json_to_chunks
    from marker.renderers.json import JSONRenderer
    from marker.schema import BlockTypes
    from marker.schema.blocks.base import BlockId, BlockOutput
    from marker.schema.polygon import PolygonBox

    renderer = JSONRenderer()
    monkeypatch.setattr(renderer, "extract_image", lambda doc, bid, to_base64=False: "dGVzdA==")
    pic_id = BlockId(page_id=0, block_id=1, block_type=BlockTypes.Picture)
    page_id = BlockId(page_id=0, block_type=BlockTypes.Page)
    pic_out = BlockOutput(
        html="",
        polygon=PolygonBox.from_bbox([0, 0, 100, 100]),
        id=pic_id,
        children=[],
        section_hierarchy={},
    )
    page_out = BlockOutput(
        html=f"<content-ref src='{pic_id}'></content-ref>",
        polygon=PolygonBox.from_bbox([0, 0, 200, 200]),
        id=page_id,
        children=[pic_out],
        section_hierarchy={},
    )
    json_model = renderer.extract_json(None, page_out)
    dumped = json_model.model_dump()
    assert dumped["children"][0]["images"] == {str(pic_id): "dGVzdA=="}
    chunks = json_to_chunks(json_model, {"Picture", "Figure", "Diagram"})
    assert [c.model_dump() for c in chunks][0]["images"] == {str(pic_id): "dGVzdA=="}
