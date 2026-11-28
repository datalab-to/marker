from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pypdfium2 as pdfium
import pytest
from PIL import Image

from marker.providers.pdf import PdfProvider

# CPU-only: pdftext text/image extraction, no models or inference server.
pytestmark = pytest.mark.cpu


@pytest.mark.config({"page_range": [0]})
def test_pdf_provider(doc_provider):
    assert len(doc_provider) == 12
    assert doc_provider.get_images([0], 72)[0].size == (612, 792)
    assert doc_provider.get_images([0], 96)[0].size == (816, 1056)

    page_lines = doc_provider.get_page_lines(0)
    assert len(page_lines) == 87

    spans = page_lines[0].spans
    assert len(spans) == 2
    assert spans[0].text == "Subspace Adversarial Training"
    assert spans[0].font == "NimbusRomNo9L-Medi"
    assert spans[0].formats == ["plain"]


def test_pdf_providers_serialize_pdfium_rendering(tmp_path, monkeypatch):
    first_path = str(tmp_path / "first.pdf")
    second_path = str(tmp_path / "second.pdf")
    doc = pdfium.PdfDocument.new()
    for _ in range(2):
        doc.new_page(72, 72)
    doc.save(first_path)
    doc.save(second_path)
    doc.close()

    first_provider = PdfProvider(first_path, {"force_ocr": True})
    second_provider = PdfProvider(second_path, {"force_ocr": True})
    events = []
    first_rendering = Event()
    second_started = Event()
    second_opened = Event()
    release_first = Event()

    class StubDocument:
        def __init__(self, filepath):
            self.filepath = filepath
            events.append(("open", filepath))
            if filepath == second_path:
                second_opened.set()

        def init_forms(self):
            pass

        def close(self):
            events.append(("close", self.filepath))

    def render(doc, _idx, _dpi, _flatten_page):
        if doc.filepath == first_path:
            first_rendering.set()
            assert release_first.wait(timeout=5)
        return Image.new("RGB", (1, 1))

    def render_second():
        second_started.set()
        return second_provider.get_images([1], 72)

    monkeypatch.setattr(pdfium, "PdfDocument", StubDocument)
    monkeypatch.setattr(PdfProvider, "_render_image", staticmethod(render))

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(first_provider.get_images, [0], 72)
        try:
            assert first_rendering.wait(timeout=5)
            second = executor.submit(render_second)
            assert second_started.wait(timeout=5)
            assert not second_opened.wait(timeout=1)
        finally:
            release_first.set()
        first.result(timeout=5)
        second.result(timeout=5)

    assert events == [
        ("open", first_path),
        ("close", first_path),
        ("open", second_path),
        ("close", second_path),
    ]
