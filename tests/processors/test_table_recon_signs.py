import pytest

from marker.processors.table_recon import (
    _NUM_TOK,
    _NUMISH,
    _cell_class,
    _find_header_band,
    reconstruct_table_html,
)


@pytest.mark.parametrize("cell", ["+1.55", "-3.02", "−3.02", "−2.1%", "+10"])
def test_signed_numbers_are_numeric(cell):
    assert _NUMISH.match(cell)
    assert _cell_class(cell) == "num"


@pytest.mark.parametrize("cell", ["+", "−", "-", "+abc", "1.5−"])
def test_non_numbers_stay_text(cell):
    assert not _NUMISH.match(cell)


def test_num_tok_counts_signed_tokens():
    assert len(_NUM_TOK.findall("+7.57 −1.69")) == 2


def _row(y, cells):
    return ([(t, x, x + 30) for t, x in cells], y, y + 8)


def test_header_band_found_above_signed_numeric_rows():
    header = _row(10, [("Gene", 10), ("Log2FC", 100), ("Adj", 200)])
    rows = [
        _row(30 + 12 * i, [(g, 10), (a, 100), (b, 200)])
        for i, (g, a, b) in enumerate(
            [("A", "+1.55", "−1.10"), ("B", "+2.65", "−1.48"),
             ("C", "−1.23", "+1.53"), ("D", "+2.35", "+2.03")]
        )
    ]
    assert _find_header_band([header] + rows) == 30


def test_signed_columns_reconstruct_with_header():
    header = _row(10, [("Gene", 10), ("Log2FC", 100), ("Adj", 200)])
    rows = [
        _row(30 + 12 * i, [(g, 10), (a, 100), (b, 200)])
        for i, (g, a, b) in enumerate(
            [("A", "+1.55", "−1.10"), ("B", "+2.65", "−1.48"),
             ("C", "−1.23", "+1.53"), ("D", "+2.35", "+2.03")]
        )
    ]
    html, _ = reconstruct_table_html([header] + rows)
    assert "<th>Gene</th>" in html and "<th>Log2FC</th>" in html
    assert html.count("<tr>") == 5
    assert "<td>−1.10</td>" in html and "<td>+2.65</td>" in html
