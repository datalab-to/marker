from marker.processors.table_recon import reconstruct_table_html, table_lines_from_pdftext


def _line(y, words):
    """One pdftext line whose cells share a single text run. Spaces are
    zero-width chars at the start of the next word, as pdftext emits them."""
    chars = []
    for text, x0 in words:
        if chars:
            chars.append({"char": " ", "bbox": [x0, y, x0, y + 8]})
        for i, ch in enumerate(text):
            chars.append({"char": ch, "bbox": [x0 + 5 * i, y, x0 + 5 * i + 5, y + 8]})
    x1 = chars[-1]["bbox"][2]
    span = {"text": "".join(c["char"] for c in chars), "bbox": [chars[0]["bbox"][0], y, x1, y + 8], "chars": chars}
    return {"bbox": span["bbox"], "spans": [span]}


def test_single_run_cells_split_into_columns():
    rows = [
        [("ID", 10), ("Log2FC", 100), ("Adj", 200)],
        [("Name", 10), ("+1.55", 100), ("+1.25", 200)],
        [("Gene", 10), ("+2.65", 100), ("+1.48", 200)],
        [("Prot", 10), ("+1.23", 100), ("+1.53", 200)],
        [("Rna", 10), ("+2.35", 100), ("+2.03", 200)],
    ]
    page = {"blocks": [{"lines": [_line(20 + 12 * i, r) for i, r in enumerate(rows)]}]}
    lines = table_lines_from_pdftext(page, (0, 0, 400, 200))
    assert [len(spans) for spans, _, _ in lines] == [3] * 5
    result = reconstruct_table_html(lines)
    assert result is not None
    html, _ = result
    assert html.count("<tr>") == 5
    for v in ("+1.55", "+2.65", "+1.23", "+2.35"):
        assert f"<td>{v}</td>" in html
