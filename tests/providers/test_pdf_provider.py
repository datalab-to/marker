import pytest


@pytest.mark.config({"page_range": [0]})
def test_pdf_provider(doc_provider):
    assert len(doc_provider) == 12
    assert doc_provider.get_images([0], 72)[0].size == (612, 792)
    assert doc_provider.get_images([0], 96)[0].size == (816, 1056)

    page_lines = doc_provider.get_page_lines(0)
    assert len(page_lines) == 85

    spans = page_lines[0].spans
    assert len(spans) == 2
    assert spans[0].text == "Subspace Adversarial Training"
    assert spans[0].font == "NimbusRomNo9L-Medi"
    assert spans[0].formats == ["plain"]
