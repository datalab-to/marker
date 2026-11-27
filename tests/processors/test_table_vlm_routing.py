from types import SimpleNamespace

from marker.processors.table import TableProcessor


class FakeSuryaRecognition:  # stands in for the Surya VLM predictor (not a mock-module class)
    pass


def _run(config, vlm_html="<table><tr><td>vlm</td></tr></table>"):
    proc = TableProcessor(FakeSuryaRecognition(), config={"mode": "balanced", **config})
    block = SimpleNamespace(html=None, text_extraction_method=None, block_type="Table", id="b")
    page = SimpleNamespace(page_id=0, pdftext_page=None)
    calls = {"heuristic": 0, "vlm": 0}

    def heuristic(page_, block_):
        calls["heuristic"] += 1
        return "<table><tr><td>heuristic</td></tr></table>"

    def vlm(document, fallback):
        calls["vlm"] += len(fallback)
        for _page, b in fallback:
            b.html = vlm_html
            b.text_extraction_method = "surya"

    proc.collect_tables = lambda document: {0: [block]}
    proc.reconstruct_digital_table = heuristic
    proc.run_ocr_fallback = vlm
    proc.cleanup_contained_blocks = lambda *a, **k: None
    proc._retire_child_lines = lambda *a, **k: None
    proc(SimpleNamespace(pages=[page]))
    return block, calls


def test_balanced_prefers_vlm_for_digital_tables():
    block, calls = _run({})
    assert calls == {"heuristic": 0, "vlm": 1}
    assert "vlm" in block.html


def test_explicit_min_recon_score_keeps_heuristic_first():
    block, calls = _run({"min_recon_score": 0.5})
    assert calls["heuristic"] == 1 and calls["vlm"] == 0
    assert "heuristic" in block.html


def test_heuristic_used_when_vlm_returns_nothing():
    block, calls = _run({}, vlm_html=None)
    assert calls == {"heuristic": 1, "vlm": 1}
    assert "heuristic" in block.html
