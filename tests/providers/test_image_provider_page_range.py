import importlib.util
import sys
import types
from pathlib import Path

import pytest
from PIL import Image


class DummyBaseProvider:
    def __init__(self, filepath: str, config=None):
        self.filepath = filepath
        if isinstance(config, dict):
            for key, value in config.items():
                setattr(self, key, value)


class DummyPolygonBox:
    @classmethod
    def from_bbox(cls, bbox):
        return bbox


def load_image_provider_module(monkeypatch):
    fake_marker_providers = types.ModuleType("marker.providers")
    fake_marker_providers.ProviderPageLines = dict
    fake_marker_providers.BaseProvider = DummyBaseProvider

    fake_polygon = types.ModuleType("marker.schema.polygon")
    fake_polygon.PolygonBox = DummyPolygonBox

    fake_text = types.ModuleType("marker.schema.text")
    fake_text.Line = object

    fake_pdftext = types.ModuleType("pdftext.schema")
    fake_pdftext.Reference = object

    monkeypatch.setitem(sys.modules, "marker.providers", fake_marker_providers)
    monkeypatch.setitem(sys.modules, "marker.schema.polygon", fake_polygon)
    monkeypatch.setitem(sys.modules, "marker.schema.text", fake_text)
    monkeypatch.setitem(sys.modules, "pdftext.schema", fake_pdftext)
    monkeypatch.delitem(sys.modules, "marker.providers.image", raising=False)

    module_path = (
        Path(__file__).resolve().parents[2] / "marker" / "providers" / "image.py"
    )
    spec = importlib.util.spec_from_file_location("marker.providers.image", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules["marker.providers.image"] = module
    spec.loader.exec_module(module)
    return module


def test_image_provider_invalid_page_range_raises_value_error(monkeypatch):
    image_module = load_image_provider_module(monkeypatch)
    image_path = Path(__file__).with_name("_tmp_test_image.png")
    try:
        Image.new("RGB", (32, 32), color="white").save(image_path)
        with pytest.raises(
            ValueError, match=r"Invalid page range, values must be between 0 and 0"
        ):
            image_module.ImageProvider(str(image_path), {"page_range": [1]})
    finally:
        if image_path.exists():
            image_path.unlink()
