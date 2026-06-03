#!/usr/bin/env python3
"""Render a marker JSON result over the original PDF pages as an HTML viewer."""
import argparse, json, html, base64, io, subprocess
from pathlib import Path
import pypdfium2 as pdfium

LABEL_COLORS = {
    "Text": "#3b82f6", "SectionHeader": "#ef4444", "PageHeader": "#a855f7",
    "PageFooter": "#a855f7", "Picture": "#f59e0b", "PictureGroup": "#f59e0b",
    "Figure": "#f59e0b", "FigureGroup": "#f59e0b",
    "Table": "#10b981", "TableGroup": "#10b981",
    "ListGroup": "#06b6d4", "ListItem": "#06b6d4",
    "Form": "#22d3ee",
    "Equation": "#ec4899", "Caption": "#f97316", "Footnote": "#94a3b8",
    "Code": "#facc15",
}
DEFAULT_COLOR = "#6b7280"

def render_pages(pdf, dpi=150):
    doc = pdfium.PdfDocument(str(pdf))
    scale = dpi / 72
    pages = []
    for i in range(len(doc)):
        page = doc[i]
        pil = page.render(scale=scale).to_pil()
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        pages.append({
            "b64": base64.b64encode(buf.getvalue()).decode(),
            "w_px": pil.width, "h_px": pil.height,
            "w_pt": page.get_width(), "h_pt": page.get_height(),
        })
    return pages

def collect_blocks(page_json):
    """Top-level blocks on a marker page (ListItems remain individual after patch)."""
    out = []
    for child in (page_json.get("children") or []):
        out.append({
            "label": child.get("block_type", "?"),
            "bbox": child.get("bbox"),
        })
    return out

def build_html(pdf_name, pages, marker_pages):
    parts = [f"""<!doctype html><meta charset=utf-8>
<title>Marker (patched): {html.escape(pdf_name)}</title>
<style>
  body{{font-family:system-ui;margin:0;background:#111;color:#eee}}
  header{{padding:12px 20px;background:#1f2937;position:sticky;top:0;z-index:10;display:flex;gap:16px;align-items:center;flex-wrap:wrap}}
  h1{{font-size:16px;margin:0}}
  .legend{{display:flex;gap:10px;font-size:12px;flex-wrap:wrap}}
  .swatch{{display:inline-block;width:12px;height:12px;margin-right:4px;vertical-align:middle;border:1px solid #fff3}}
  .page{{margin:20px auto;max-width:fit-content;background:#000;padding:8px 8px 8px;border-radius:6px}}
  .page-title{{font-size:13px;margin:0 0 6px;color:#9ca3af}}
  .canvas{{position:relative;line-height:0;margin-top:14px}}
  .canvas img{{display:block;max-width:90vw;height:auto}}
  .bbox{{position:absolute;border:1.5px solid;box-sizing:border-box;cursor:pointer}}
  .bbox span{{position:absolute;top:-12px;left:-1px;font-size:9px;line-height:11px;padding:0 3px;color:#fff;white-space:nowrap;border-radius:2px 2px 0 0;font-family:ui-monospace,monospace;z-index:2;pointer-events:none;opacity:.85}}
  .bbox:hover{{background:#ffffff22;z-index:5}}
  .bbox:hover span{{opacity:1;z-index:6}}
  .controls label{{font-size:12px;margin-right:8px}}
</style>
<header>
  <h1>{html.escape(pdf_name)} — patched marker (no ListGroup merging)</h1>
  <div class=controls>
    <label><input type=checkbox id=lbls checked> labels</label>
    <label><input type=checkbox id=boxes checked> boxes</label>
    <label><input type=checkbox id=only_li> ListItems only</label>
  </div>
  <div class=legend id=legend></div>
</header>"""]

    labels_seen = set()
    for idx, (img, mp) in enumerate(zip(pages, marker_pages), 1):
        # marker bbox is in PDF points; image is rendered at dpi → multiply by (px / pt)
        sx = img["w_px"] / img["w_pt"]
        sy = img["h_px"] / img["h_pt"]
        blocks = collect_blocks(mp)
        li_count = sum(1 for b in blocks if b["label"] == "ListItem")
        parts.append(f'<div class=page><div class=page-title>page {idx} — {len(blocks)} blocks ({li_count} ListItem)</div>')
        parts.append(f'<div class=canvas style="width:{img["w_px"]}px;height:{img["h_px"]}px">')
        parts.append(f'<img src="data:image/png;base64,{img["b64"]}" width="{img["w_px"]}" height="{img["h_px"]}">')
        for b in blocks:
            if not b["bbox"]:
                continue
            x0, y0, x1, y1 = b["bbox"]
            x0, y0, x1, y1 = x0*sx, y0*sy, x1*sx, y1*sy
            label = b["label"]
            color = LABEL_COLORS.get(label, DEFAULT_COLOR)
            labels_seen.add((label, color))
            parts.append(
                f'<div class="bbox lbl-{html.escape(label)}" style="left:{x0:.1f}px;top:{y0:.1f}px;'
                f'width:{x1-x0:.1f}px;height:{y1-y0:.1f}px;border-color:{color}" '
                f'title="{html.escape(label)}">'
                f'<span style="background:{color}">{html.escape(label)}</span></div>'
            )
        parts.append("</div></div>")

    legend = "".join(
        f'<span><i class=swatch style="background:{c}"></i>{html.escape(l)}</span>'
        for l, c in sorted(labels_seen)
    )
    parts.append(f"<script>document.getElementById('legend').innerHTML={json.dumps(legend)};"
                 "const $=s=>document.querySelectorAll(s);"
                 "lbls.onchange=e=>$('.bbox span').forEach(x=>x.style.display=e.target.checked?'':'none');"
                 "boxes.onchange=e=>$('.bbox').forEach(x=>x.style.borderWidth=e.target.checked?'2px':'0');"
                 "only_li.onchange=e=>$('.bbox').forEach(x=>{x.style.display=(!e.target.checked||x.classList.contains('lbl-ListItem'))?'':'none';});"
                 "</script>")
    return "".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("marker_json")
    ap.add_argument("--out", default="/tmp/marker_view")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--tag", default="", help="suffix added to the output html name")
    args = ap.parse_args()

    pdf = Path(args.pdf).resolve()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(args.marker_json).read_text())
    marker_pages = data["children"]
    print(f"[1/2] rendering {len(marker_pages)} page(s) ...", flush=True)
    pages = render_pages(pdf)
    print(f"[2/2] writing HTML ...", flush=True)
    suffix = f".{args.tag}" if args.tag else ""
    html_path = out / f"{pdf.stem}.marker{suffix}.html"
    html_path.write_text(build_html(pdf.name, pages, marker_pages))
    print(f"-> {html_path}")
    if not args.no_open:
        subprocess.Popen(["firefox", str(html_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    main()
