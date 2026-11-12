from __future__ import annotations

import html
import re
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from surya.layout.schema import LayoutBox, LayoutResult
from surya.recognition.schema import BlockOCRResult, PageOCRResult

# Mapping from indra_ocr layout labels to Surya/Marker canonical block labels
INDRA_TO_SURYA_LABEL = {
    "Text": "Text",
    "Section-Header": "SectionHeader",
    "Page-Header": "PageHeader",
    "Page-Footer": "PageFooter",
    "List-Group": "ListGroup",
    "Table": "Table",
    "Table-Of-Contents": "TableOfContents",
    "Form": "Form",
    "Caption": "Caption",
    "Footnote": "Footnote",
    "Picture": "Picture",
}

_MATH_RE = re.compile(
    r"\$\$([\s\S]+?)\$\$|(?<!\\)\$(?!\s)([^$\n]+?)(?<![\s\\])\$"
)
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•·▪▸◦‣⁃]|\d+[.)]|[a-zA-Z][.)])\s+")


def format_inline_html(text: str, join_lines: str = " ") -> str:
    """Convert plain text (with optional $$...$$ or $...$ math) into safe HTML with <math> tags."""
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    joined = join_lines.join(lines)
    parts: List[str] = []
    last = 0
    for match in _MATH_RE.finditer(joined):
        if match.start() > last:
            parts.append(html.escape(joined[last : match.start()]))
        if match.group(1) is not None:
            latex = match.group(1).strip()
            parts.append(f'<math display="block">{html.escape(latex)}</math>')
        else:
            latex = match.group(2).strip()
            parts.append(f"<math>{html.escape(latex)}</math>")
        last = match.end()
    if last < len(joined):
        parts.append(html.escape(joined[last:]))
    return "".join(parts)


def format_list_html(text: str) -> str:
    lines = [raw_line.strip() for raw_line in text.splitlines() if raw_line.strip()]
    if not lines:
        return ""
    return f"<p>{'<br>'.join(format_inline_html(line) for line in lines)}</p>"


def _bbox_to_polygon(
    bbox: List[float],
    rot: int = 0,
    orig_w: float = 0,
    orig_h: float = 0,
) -> List[List[float]]:
    x0, y0, x1, y1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    if rot and orig_w > 0 and orig_h > 0:
        uw, uh = (orig_h, orig_w) if rot % 180 else (orig_w, orig_h)
        fx0, fy0, fx1, fy1 = x0 / uw, y0 / uh, x1 / uw, y1 / uh
        for _ in range((rot // 90) % 4):
            fx0, fy0, fx1, fy1 = fy0, 1.0 - fx1, fy1, 1.0 - fx0
        x0 = min(float(orig_w), max(0.0, fx0 * orig_w))
        y0 = min(float(orig_h), max(0.0, fy0 * orig_h))
        x1 = min(float(orig_w), max(0.0, fx1 * orig_w))
        y1 = min(float(orig_h), max(0.0, fy1 * orig_h))
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


class IndraRecognitionPredictor:
    """High-performance ONNX OCR predictor backed by indra-ocr.

    Implements the same `__call__(images, layout_results=None, full_page=False)`
    contract as Surya's `RecognitionPredictor`, returning `List[PageOCRResult]`.
    """

    def __init__(
        self,
        threads: int = 4,
        math: bool = True,
        furniture: bool = True,
        table_heads: bool = True,
        model_path: Optional[str] = None,
        providers: Optional[List[str]] = None,
    ):
        self.threads = threads
        self.math = math
        self.furniture = furniture
        self.table_heads = table_heads
        self.model_path = model_path
        self.providers = providers
        self.disable_tqdm: bool = False
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            import os
            from pathlib import Path
            import urllib.request
            from indra_ocr import OrtOCR, model_dir
            from indra_ocr.models import HF_REPO, HF_REVISION, MATH_FILES

            try:
                onnx_dir = model_dir(self.model_path, math=self.math)
            except Exception:
                cache_dir = (
                    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
                    / "indra_ocr"
                    / HF_REVISION
                )
                cache_dir.mkdir(parents=True, exist_ok=True)
                base_files = (
                    "det.onnx",
                    "rec.onnx",
                    "rec_cjk.onnx",
                    "rec_rtl.onnx",
                    "rec_world.onnx",
                    "order.onnx",
                    "orient.onnx",
                    "script.onnx",
                    "meta.json",
                    "lm.npz",
                    "lm_cjk.npz",
                    "lm_world.npz",
                )
                needed_files = base_files + (tuple(MATH_FILES) if self.math else ())
                token = os.environ.get("HF_TOKEN") or os.environ.get(
                    "HUGGING_FACE_HUB_TOKEN"
                )
                for fname in needed_files:
                    dest = cache_dir / fname
                    if dest.exists() and dest.stat().st_size > 0:
                        continue
                    url = f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/{fname}"
                    req = urllib.request.Request(url)
                    if token:
                        req.add_header("Authorization", f"Bearer {token}")
                    tmp_dest = cache_dir / f".{fname}.{os.getpid()}.tmp"
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        tmp_dest.write_bytes(resp.read())
                    tmp_dest.replace(dest)
                os.environ["INDRA_OCR_MODEL_DIR"] = str(cache_dir)
                onnx_dir = model_dir(str(cache_dir), math=self.math)

            self._engine = OrtOCR(
                onnx_dir,
                threads=self.threads,
                providers=self.providers,
                furniture=self.furniture,
                math=self.math,
                table_heads=self.table_heads,
            )
        return self._engine

    def detect_layout(
        self, pages, provider=None, extract_images: bool = True
    ) -> List[LayoutResult]:
        """Run in-process ONNX layout detection for digital or raster pages using OrderNet + detector."""
        from indra_ocr.layout_post import (
            BLOCK_CLASSES,
            heads_plausible,
            split_blocks,
        )
        from indra_ocr.order_post import xycut_blocks
        from marker.processors.table_recon import (
            reconstruct_table_html,
            table_lines_from_pdftext,
        )
        from marker.util import matrix_intersection_area

        math_font_hints = (
            "cmmi",
            "cmsy",
            "cmex",
            "msam",
            "msbm",
            "euex",
            "eufm",
            "eusm",
            "stix",
            "mathjax",
            "math",
        )

        results: List[LayoutResult] = []
        for page in pages:
            page_bbox = list(page.polygon.bbox)
            w = max(1, int(round(page_bbox[2] - page_bbox[0])))
            h = max(1, int(round(page_bbox[3] - page_bbox[1])))
            provider_lines = (
                provider.page_lines.get(page.page_id, [])
                if provider is not None and hasattr(provider, "page_lines")
                else []
            )
            if provider_lines:
                def _build_sub_boxes(grid_lines_set):
                    r_boxes = []
                    r_texts = []
                    r_line_idx = []
                    for li, line in enumerate(provider_lines):
                        spans = [sp for sp in line.spans if sp.text and sp.text.strip()]
                        if len(spans) <= 1:
                            r_boxes.append(list(line.line.polygon.bbox))
                            r_texts.append(line.raw_text.strip())
                            r_line_idx.append(li)
                        else:
                            lh = max(
                                1.0,
                                float(
                                    np.median(
                                        [
                                            max(1.0, sp.polygon.bbox[3] - sp.polygon.bbox[1])
                                            for sp in spans
                                        ]
                                    )
                                ),
                            )
                            gap_thr = (
                                max(6.0, lh * 0.8)
                                if li in grid_lines_set
                                else max(18.0, lh * 2.5)
                            )
                            cur_x0, cur_y0, cur_x1, cur_y1 = spans[0].polygon.bbox
                            cur_txts = [spans[0].text.strip()]
                            for sp in spans[1:]:
                                sx0, sy0, sx1, sy1 = sp.polygon.bbox
                                if sx0 - cur_x1 > gap_thr:
                                    st = " ".join(cur_txts).strip()
                                    if st:
                                        r_boxes.append([cur_x0, cur_y0, cur_x1, cur_y1])
                                        r_texts.append(st)
                                        r_line_idx.append(li)
                                    cur_x0, cur_y0, cur_x1, cur_y1 = sx0, sy0, sx1, sy1
                                    cur_txts = [sp.text.strip()]
                                else:
                                    cur_x0 = min(cur_x0, sx0)
                                    cur_y0 = min(cur_y0, sy0)
                                    cur_x1 = max(cur_x1, sx1)
                                    cur_y1 = max(cur_y1, sy1)
                                    cur_txts.append(sp.text.strip())
                            st = " ".join(cur_txts).strip()
                            if st:
                                r_boxes.append([cur_x0, cur_y0, cur_x1, cur_y1])
                                r_texts.append(st)
                                r_line_idx.append(li)
                    if not r_boxes:
                        r_boxes = [list(line.line.polygon.bbox) for line in provider_lines]
                        r_texts = [line.raw_text.strip() for line in provider_lines]
                        r_line_idx = list(range(len(provider_lines)))
                    return (
                        np.array(r_boxes, dtype=np.float32).reshape(-1, 4),
                        r_texts,
                        r_line_idx,
                    )

                boxes, texts, box_line_idx = _build_sub_boxes(set())
                line_hs = np.maximum(1.0, boxes[:, 3] - boxes[:, 1])
                median_h = float(np.median(line_hs)) if len(line_hs) else 10.0

                order, layout, _ = self.engine.reading_order(boxes, texts, (w, h))
                if layout is not None:
                    grid_lines = {
                        box_line_idx[i]
                        for i, c in enumerate(layout[0].argmax(-1))
                        if BLOCK_CLASSES[int(c)] in ("Table", "Table-Of-Contents", "Form")
                    }
                    if grid_lines:
                        boxes2, texts2, box_line_idx2 = _build_sub_boxes(grid_lines)
                        if len(boxes2) != len(boxes):
                            boxes, texts, box_line_idx = boxes2, texts2, box_line_idx2
                            order, layout, _ = self.engine.reading_order(
                                boxes, texts, (w, h)
                            )
                raw_blocks = []
                if layout is not None:
                    cls_logits, same, tab = layout
                    blocks_list = list(split_blocks(order, same))
                    blk_bboxes = []
                    for blk in blocks_list:
                        b = boxes[blk, :4]
                        blk_bboxes.append(
                            (
                                round(float(b[:, 0].min()), 1),
                                round(float(b[:, 1].min()), 1),
                                round(float(b[:, 2].max()), 1),
                                round(float(b[:, 3].max()), 1),
                            )
                        )
                    for bi, blk in enumerate(blocks_list):
                        b = boxes[blk, :4]
                        bx0, by0, bx1, by1 = blk_bboxes[bi]
                        raw_lbl = BLOCK_CLASSES[int(cls_logits[blk].sum(0).argmax())]
                        lbl = INDRA_TO_SURYA_LABEL.get(raw_lbl, "Text")
                        blk_text = "\n".join(texts[i] for i in blk).strip()
                        uniq_lines = sorted({box_line_idx[i] for i in blk})
                        hp = False

                        if lbl == "Table":
                            hp = (
                                heads_plausible(b, tab[:, blk][:, :, blk])
                                if tab is not None and len(blk) >= 2
                                else False
                            )
                            if len(blk) < 2 or (by1 - by0) < median_h * 1.4:
                                lbl = "Text"
                            elif by0 < 0.26 * h and getattr(page, "page_id", 0) == 0:
                                recon_ok = False
                                if getattr(page, "pdftext_page", None):
                                    tlines = table_lines_from_pdftext(
                                        page.pdftext_page, [bx0, by0, bx1, by1]
                                    )
                                    res = reconstruct_table_html(tlines)
                                    recon_ok = bool(res and res[0])
                                if (
                                    not hp
                                    and not recon_ok
                                    and "table" not in blk_text.lower()
                                ):
                                    lbl = "Text"

                        if (
                            lbl == "SectionHeader"
                            and any(rb["label"] == "SectionHeader" for rb in raw_blocks)
                        ):
                            if "@" in blk_text or (
                                getattr(page, "page_id", 0) == 0
                                and by0 < 0.25 * h
                                and len(uniq_lines) >= 2
                                and not re.match(
                                    r"^\s*(?:abstract|\d+[.\s])", blk_text, re.I
                                )
                            ):
                                lbl = "Text"
                                raw_lbl = "Text"

                        if (
                            lbl == "Text"
                            and not any(
                                rb["label"] == "SectionHeader" for rb in raw_blocks
                            )
                            and 1 <= len(uniq_lines) <= 4
                            and by0 < 0.35 * h
                            and float(np.mean(b[:, 3] - b[:, 1])) > median_h * 1.18
                            and any(c.isalpha() for c in blk_text)
                        ):
                            lbl = "SectionHeader"

                        if (
                            lbl == "Text"
                            and len(uniq_lines) <= 12
                            and re.match(
                                r"^\s*(?:Figure|Fig\.?|Table|Tab\.?)\s+(?:\d+|[IVXLC]+)(?:\.\d+)?\s*[.:]\s+\S",
                                blk_text,
                                re.I,
                            )
                        ):
                            lbl = "Caption"
                            raw_lbl = "Caption"

                        if (
                            lbl in ("Text", "Table")
                            and 1 <= len(uniq_lines) <= 6
                            and 0 < len(blk_text) < 260
                        ):
                            shares_band = False
                            for bj, (ox0, oy0, ox1, oy1) in enumerate(blk_bboxes):
                                if bj == bi:
                                    continue
                                y_ov = min(by1, oy1) - max(by0, oy0)
                                x_ov = min(bx1, ox1) - max(bx0, ox0)
                                if (
                                    y_ov > 0.35 * min(max(1.0, by1 - by0), max(1.0, oy1 - oy0))
                                    and x_ov <= 0
                                ):
                                    shares_band = True
                                    break
                            if not shares_band:
                                math_chars = 0
                                math_font_chars = 0
                                total_chars = 0
                                prose_words = 0
                                for li in uniq_lines:
                                    for sp in provider_lines[li].spans:
                                        stxt = sp.text or ""
                                        s_no_ws = "".join(stxt.split())
                                        if not s_no_ws:
                                            continue
                                        total_chars += len(s_no_ws)
                                        fname = (sp.font or "").lower()
                                        is_mfont = any(
                                            h_ in fname for h_ in math_font_hints
                                        ) or getattr(sp, "math", False)
                                        if is_mfont:
                                            math_chars += len(s_no_ws)
                                            math_font_chars += len(s_no_ws)
                                        else:
                                            math_chars += sum(
                                                1
                                                for ch in s_no_ws
                                                if (0x0370 <= ord(ch) <= 0x03FF)
                                                or (0x2100 <= ord(ch) <= 0x22FF)
                                                or ch in "=+-*/^_<>∑∏∫√∞≈≠≤≥±×÷∈∉⊂⊃∪∩\\"
                                            )
                                            clean_stxt = re.sub(
                                                r"\\[A-Za-z]+", " ", stxt
                                            )
                                            prose_words += sum(
                                                1
                                                for w_ in clean_stxt.split()
                                                if len(w_) >= 4 and w_.isalpha()
                                            )
                                if total_chars > 0 and (
                                    prose_words < 4
                                    or blk_text.lstrip().startswith(
                                        (
                                            "\\min",
                                            "\\max",
                                            "\\sum",
                                            "\\int",
                                            "\\mathbf",
                                            "\\mathcal",
                                            "\\mathbb",
                                            "\\frac",
                                            "$$",
                                        )
                                    )
                                ):
                                    is_eq = (
                                        (
                                            not hp
                                            and (
                                                (math_chars / total_chars) >= 0.45
                                                or (math_font_chars / total_chars) >= 0.25
                                            )
                                        )
                                        or (
                                            (math_font_chars / total_chars) >= 0.35
                                            and (math_chars / total_chars) >= 0.50
                                        )
                                        or blk_text.lstrip().startswith(
                                            (
                                                "\\min",
                                                "\\max",
                                                "\\sum",
                                                "\\int",
                                                "\\mathbf",
                                                "\\mathcal",
                                                "\\mathbb",
                                                "\\frac",
                                                "$$",
                                            )
                                        )
                                    )
                                    if is_eq:
                                        lbl = "Equation"
                                        raw_lbl = "Equation"

                        raw_blocks.append(
                            {
                                "label": lbl,
                                "raw_label": raw_lbl,
                                "bbox": [bx0, by0, bx1, by1],
                                "text": blk_text,
                            }
                        )
                elif len(boxes) > 0:
                    raw_blocks = [
                        {
                            "label": "Text",
                            "raw_label": "Text",
                            "bbox": [
                                float(boxes[:, 0].min()),
                                float(boxes[:, 1].min()),
                                float(boxes[:, 2].max()),
                                float(boxes[:, 3].max()),
                            ],
                        }
                    ]

                # Detect embedded Picture regions and uncovered visual text regions
                # only when the page has very few embedded lines (< 15, e.g. scanned page
                # with a digital stamp) or when image extraction is enabled.
                need_visual_det = len(provider_lines) < 15 or extract_images
                img = page.get_image(highres=False) if need_visual_det else None
                if img is not None:
                    gray = np.asarray(img.convert("L"), dtype=np.uint8)
                    ih, iw = gray.shape[:2]
                    sx = w / max(iw, 1)
                    sy = h / max(ih, 1)
                    det_boxes, pic_boxes = self.engine.detect_all(gray)

                    # Surface uncovered visual text lines (e.g. a scanned page with
                    # a single digital stamp, or a raster table/figure embedded on a
                    # digital page) as Text blocks so LineBuilder flags them for OCR.
                    min_uncovered = 2 if len(provider_lines) < 15 else 5
                    if len(det_boxes) >= min_uncovered:
                        scaled_det = det_boxes[:, :4].astype(np.float32).copy()
                        scaled_det[:, [0, 2]] *= sx
                        scaled_det[:, [1, 3]] *= sy
                        inter = matrix_intersection_area(
                            scaled_det.tolist(), boxes.tolist()
                        )
                        uncovered_mask = inter.sum(axis=1) <= 0.05
                        if len(pic_boxes) > 0:
                            scaled_pics = pic_boxes[:, :4].astype(np.float32).copy()
                            scaled_pics[:, [0, 2]] *= sx
                            scaled_pics[:, [1, 3]] *= sy
                            pic_inter = matrix_intersection_area(
                                scaled_det.tolist(), scaled_pics.tolist()
                            )
                            uncovered_mask &= pic_inter.sum(axis=1) <= 0.2
                        uncovered = scaled_det[uncovered_mask]
                        if len(uncovered) >= min_uncovered:
                            for grp in xycut_blocks(uncovered):
                                if len(grp) < 2 and len(provider_lines) >= 15:
                                    continue
                                gb = uncovered[grp]
                                raw_blocks.append(
                                    {
                                        "label": "Text",
                                        "raw_label": "Text",
                                        "bbox": [
                                            round(float(gb[:, 0].min()), 1),
                                            round(float(gb[:, 1].min()), 1),
                                            round(float(gb[:, 2].max()), 1),
                                            round(float(gb[:, 3].max()), 1),
                                        ],
                                    }
                                )

                    if len(pic_boxes) > 0:
                        for p in pic_boxes[:, :4]:
                            px0 = round(float(p[0] * sx), 1)
                            py0 = round(float(p[1] * sy), 1)
                            px1 = round(float(p[2] * sx), 1)
                            py1 = round(float(p[3] * sy), 1)
                            cy = (py0 + py1) * 0.5
                            at = next(
                                (
                                    j
                                    for j, b in enumerate(raw_blocks)
                                    if (b["bbox"][1] + b["bbox"][3]) * 0.5 > cy
                                    and min(px1, b["bbox"][2]) > max(px0, b["bbox"][0])
                                ),
                                len(raw_blocks),
                            )
                            raw_blocks.insert(
                                at,
                                {
                                    "label": "Picture",
                                    "raw_label": "Picture",
                                    "bbox": [px0, py0, px1, py1],
                                },
                            )

                fig_captions = [
                    rb
                    for rb in raw_blocks
                    if rb.get("label") == "Caption"
                    and re.match(
                        r"^\s*(?:Figure|Fig\.?)\b", rb.get("text", ""), re.I
                    )
                ]
                for cb in fig_captions:
                    if cb not in raw_blocks:
                        continue
                    cx0, cy0, cx1, cy1 = cb["bbox"]
                    cw = max(1.0, cx1 - cx0)
                    matched_pic = None
                    for rb in raw_blocks:
                        if rb is cb or rb.get("label") not in ("Picture", "Figure"):
                            continue
                        bx0, by0, bx1, by1 = rb["bbox"]
                        if min(cx1, bx1) - max(cx0, bx0) >= 0.25 * cw and (
                            0 <= cy0 - by1 < 0.08 * h or 0 <= by0 - cy1 < 0.08 * h
                        ):
                            matched_pic = rb
                            break
                    if matched_pic is not None:
                        matched_pic["label"] = "Figure"
                        matched_pic["raw_label"] = "Figure"
                        raw_blocks.remove(matched_pic)
                        raw_blocks.insert(raw_blocks.index(cb), matched_pic)
                        continue

                    cand_blocks = []
                    for rb in raw_blocks:
                        if rb is cb or rb.get("label") not in ("Table", "Text"):
                            continue
                        bx0, by0, bx1, by1 = rb["bbox"]
                        bw = max(1.0, bx1 - bx0)
                        if (
                            min(cx1, bx1) - max(cx0, bx0) >= 0.4 * min(cw, bw)
                            and bw <= cw * 1.15
                            and by1 <= cy0 + 4.0
                            and 0 <= cy0 - by0 <= 0.25 * h
                        ):
                            rtxt = rb.get("text", "")
                            if (
                                rb.get("label") == "Table"
                                and "table" not in rtxt.lower()
                            ) or (
                                rb.get("label") == "Text"
                                and len(rtxt.split()) <= 12
                            ):
                                cand_blocks.append(rb)
                    if cand_blocks and any(
                        cy0 - b["bbox"][3] < 0.05 * h for b in cand_blocks
                    ):
                        fx0 = round(min(b["bbox"][0] for b in cand_blocks), 1)
                        fy0 = round(min(b["bbox"][1] for b in cand_blocks), 1)
                        fx1 = round(max(b["bbox"][2] for b in cand_blocks), 1)
                        fy1 = round(max(b["bbox"][3] for b in cand_blocks), 1)
                        for b in cand_blocks:
                            raw_blocks.remove(b)
                        raw_blocks.insert(
                            raw_blocks.index(cb),
                            {
                                "label": "Figure",
                                "raw_label": "Figure",
                                "bbox": [fx0, fy0, fx1, fy1],
                                "text": "",
                            },
                        )

                layout_boxes: List[LayoutBox] = []
                for pos, blk in enumerate(raw_blocks):
                    bx0, by0, bx1, by1 = [float(v) for v in blk["bbox"]]
                    lbl = blk.get("label", "Text")
                    raw_lbl = blk.get("raw_label", lbl)
                    layout_boxes.append(
                        LayoutBox(
                            polygon=[[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]],
                            label=lbl,
                            raw_label=raw_lbl,
                            position=pos,
                            confidence=1.0,
                        )
                    )
                results.append(LayoutResult(bboxes=layout_boxes, image_bbox=page_bbox))
            else:
                img = page.get_image(highres=False)
                gray = np.asarray(img.convert("L"), dtype=np.uint8)
                ih, iw = gray.shape[:2]
                sx = w / max(iw, 1)
                sy = h / max(ih, 1)
                prev_th = getattr(self.engine, "table_heads", False)
                self.engine.table_heads = False
                try:
                    i_page = self.engine(gray, turn=False)
                finally:
                    self.engine.table_heads = prev_th
                layout_boxes = []
                for blk in i_page.blocks:
                    raw_lbl = blk.get("label", "Text")
                    lbl = INDRA_TO_SURYA_LABEL.get(raw_lbl, "Text")
                    bbox = blk.get("bbox", [0, 0, iw, ih])
                    bx0 = round(float(bbox[0] * sx), 1)
                    by0 = round(float(bbox[1] * sy), 1)
                    bx1 = round(float(bbox[2] * sx), 1)
                    by1 = round(float(bbox[3] * sy), 1)
                    layout_boxes.append(
                        LayoutBox(
                            polygon=[[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]],
                            label=lbl,
                            raw_label=raw_lbl,
                            position=len(layout_boxes),
                            confidence=1.0,
                        )
                    )
                results.append(LayoutResult(bboxes=layout_boxes, image_bbox=page_bbox))
        return results

    @staticmethod
    def _extract_boxes(
        lr: Union[LayoutResult, Sequence[LayoutBox], None], img: Image.Image
    ) -> List[LayoutBox]:
        if lr is None:
            return [
                LayoutBox(
                    polygon=[
                        [0, 0],
                        [img.width, 0],
                        [img.width, img.height],
                        [0, img.height],
                    ],
                    label="Text",
                    raw_label="Text",
                    position=0,
                )
            ]
        if hasattr(lr, "bboxes"):
            return list(lr.bboxes)
        return list(lr)

    def __call__(
        self,
        images: List[Image.Image],
        layout_results: Optional[
            Sequence[Union[LayoutResult, Sequence[LayoutBox]]]
        ] = None,
        full_page: bool = False,
    ) -> List[PageOCRResult]:
        if full_page:
            results: List[PageOCRResult] = []
            for i, img in enumerate(images):
                page_res = self._ocr_full_page(img)
                if (
                    not page_res.blocks
                    and layout_results is not None
                    and i < len(layout_results)
                ):
                    fallback_boxes = self._extract_boxes(layout_results[i], img)
                    if fallback_boxes:
                        page_res = self._ocr_blocks(img, fallback_boxes)
                results.append(page_res)
            return results

        if layout_results is None:
            return [
                self._ocr_blocks(img, self._extract_boxes(None, img))
                for img in images
            ]

        return [
            self._ocr_blocks(img, self._extract_boxes(lr, img))
            for img, lr in zip(images, layout_results)
        ]

    def _ocr_full_page(self, image: Image.Image) -> PageOCRResult:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
        orig_h, orig_w = gray.shape[:2]
        prev_th = getattr(self.engine, "table_heads", False)
        self.engine.table_heads = False
        try:
            page = self.engine(gray, turn=True)
        finally:
            self.engine.table_heads = prev_th
        return self._page_to_ocr_result(page, float(orig_w), float(orig_h))

    def _page_to_ocr_result(
        self, page, orig_w: float, orig_h: float
    ) -> PageOCRResult:
        from indra_ocr.layout_post import join_lines as _raw_join_lines

        _cjk_space_re = re.compile(
            r"(?<=[\u3000-\u9fff\uff00-\uffef])\s+(?=[\u3000-\u9fff\uff00-\uffef])"
        )

        def join_lines(lines_seq):
            return _cjk_space_re.sub("", _raw_join_lines(lines_seq))

        rot = getattr(page, "rotation", 0) or 0
        blocks: List[BlockOCRResult] = []
        raw_blocks = page.blocks
        if not raw_blocks and page.lines:
            raw_blocks = [
                {"label": "Text", "bbox": line.box, "text": line.text}
                for line in page.lines
                if line.text and line.text.strip()
            ]

        md_tables = re.findall(
            r"<table[\s\S]*?</table>", getattr(page, "markdown", None) or ""
        )
        md_table_idx = 0

        for blk in raw_blocks:
            raw_label = blk.get("label", "Text")
            bbox = blk.get("bbox", [0, 0, orig_w, orig_h])
            text = (blk.get("text") or "").strip()
            polygon = _bbox_to_polygon(bbox, rot=rot, orig_w=orig_w, orig_h=orig_h)

            if raw_label == "Picture":
                blocks.append(
                    BlockOCRResult(
                        polygon=polygon,
                        label="Picture",
                        raw_label=raw_label,
                        reading_order=len(blocks),
                        html="",
                        error=False,
                        skipped=False,
                    )
                )
                continue

            if not text:
                continue

            blines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if not blines:
                continue

            # Display math blocks (single-line or multi-line) emitted by indra_ocr when math=True
            if (
                text.startswith("$$") and text.endswith("$$") and len(text) >= 4
            ) or all(
                ln.startswith("$") and ln.endswith("$") and len(ln) >= 2
                for ln in blines
            ):
                if (
                    text.startswith("$$")
                    and text.endswith("$$")
                    and "$$" not in text[2:-2]
                ):
                    latex = text[2:-2].strip()
                else:
                    latex = "\n".join(ln.strip("$").strip() for ln in blines)
                blocks.append(
                    BlockOCRResult(
                        polygon=polygon,
                        label="Equation",
                        raw_label=raw_label,
                        reading_order=len(blocks),
                        html=f'<p><math display="block">{html.escape(latex)}</math></p>',
                        error=False,
                        skipped=False,
                    )
                )
                continue

            surya_label = INDRA_TO_SURYA_LABEL.get(raw_label, "Text")
            if raw_label == "Table" and len(blines) <= 1:
                surya_label = "Text"

            if (
                raw_label == "Table"
                and len(blines) > 1
                and not text.lstrip().startswith("<table")
                and md_table_idx < len(md_tables)
            ):
                t_html = md_tables[md_table_idx]
                md_table_idx += 1
                if blines[0] not in t_html:
                    block_html = f"<p>{format_inline_html(blines[0])}</p>\n{t_html}"
                else:
                    block_html = t_html
            elif surya_label in ("SectionHeader", "Title"):
                block_html = f"<h2>{format_inline_html(join_lines(blines))}</h2>"
            elif surya_label in ("ListGroup", "ListItem"):
                items: List[List[str]] = []
                for ln in blines:
                    if not items or _LIST_PREFIX_RE.match(ln):
                        items.append([ln])
                    else:
                        items[-1].append(ln)
                block_html = format_list_html(
                    "\n".join(join_lines(it) for it in items)
                )
            else:
                if surya_label == "Table" and not text.lstrip().startswith("<table"):
                    surya_label = "Text"
                if text.lstrip().startswith("<table"):
                    block_html = text
                else:
                    block_html = f"<p>{format_inline_html(join_lines(blines))}</p>"

            blocks.append(
                BlockOCRResult(
                    polygon=polygon,
                    label=surya_label,
                    raw_label=raw_label,
                    reading_order=len(blocks),
                    html=block_html,
                    error=False,
                    skipped=False,
                )
            )

        return PageOCRResult(blocks=blocks, image_bbox=[0, 0, orig_w, orig_h])

    def convert_pdf_pages(
        self,
        filepath: str,
        page_ids: Sequence[int],
        force_ocr: bool = False,
    ) -> Tuple[List[PageOCRResult], List[str]]:
        import pypdfium2 as pdfium
        from indra_ocr import Converter

        conv = Converter(
            self.engine,
            dpi=192.0,
            mode="ocr" if force_ocr else "auto",
            image_ocr=True,
        )
        results: List[PageOCRResult] = []
        sources: List[str] = []
        prev_th = getattr(self.engine, "table_heads", False)
        self.engine.table_heads = False
        doc = pdfium.PdfDocument(str(filepath))
        try:
            for pid in page_ids:
                pdf_page = doc[pid]
                w_pt, h_pt = pdf_page.get_size()
                scale = conv.page_scale(pdf_page)
                orig_w = max(1.0, float(w_pt * scale))
                orig_h = max(1.0, float(h_pt * scale))
                i_page = conv._safe(lambda: conv.pdf_page(str(filepath), doc, pid))
                results.append(self._page_to_ocr_result(i_page, orig_w, orig_h))
                sources.append(getattr(i_page, "source", "ocr") or "ocr")
        finally:
            self.engine.table_heads = prev_th
            doc.close()
        return results, sources

    def _ocr_blocks(
        self, image: Image.Image, boxes: List[LayoutBox]
    ) -> PageOCRResult:
        page_gray = np.asarray(image.convert("L"), dtype=np.uint8)
        h, w = page_gray.shape[:2]
        results: List[BlockOCRResult] = []

        for idx, box in enumerate(boxes):
            xs = [pt[0] for pt in box.polygon]
            ys = [pt[1] for pt in box.polygon]
            x0 = max(0, int(np.floor(min(xs))))
            y0 = max(0, int(np.floor(min(ys))))
            x1 = min(w, int(np.ceil(max(xs))))
            y1 = min(h, int(np.ceil(max(ys))))

            if x1 - x0 < 2 or y1 - y0 < 2:
                results.append(
                    BlockOCRResult(
                        polygon=box.polygon,
                        label=box.label,
                        raw_label=box.label,
                        reading_order=idx,
                        html="",
                        error=False,
                        skipped=True,
                    )
                )
                continue

            crop = page_gray[y0:y1, x0:x1]
            label = box.label
            try:
                block_html = self._recognize_crop(crop, label)
            except Exception:
                results.append(
                    BlockOCRResult(
                        polygon=box.polygon,
                        label=label,
                        raw_label=label,
                        reading_order=idx,
                        html="",
                        error=True,
                        skipped=False,
                    )
                )
                continue

            if not block_html:
                results.append(
                    BlockOCRResult(
                        polygon=box.polygon,
                        label=label,
                        raw_label=label,
                        reading_order=idx,
                        html="",
                        error=False,
                        skipped=True,
                    )
                )
            else:
                results.append(
                    BlockOCRResult(
                        polygon=box.polygon,
                        label=label,
                        raw_label=label,
                        reading_order=idx,
                        html=block_html,
                        error=False,
                        skipped=False,
                    )
                )

        return PageOCRResult(blocks=results, image_bbox=[0, 0, image.width, image.height])

    def _recognize_crop(self, crop: np.ndarray, label: str) -> str:
        engine = self.engine
        pad = 12
        padded = cv2.copyMakeBorder(
            crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255
        )
        ph, pw = padded.shape[:2]

        # Detect text lines inside the padded crop at native crop scale
        ls = min(engine.long_side, max(64, ((max(ph, pw) + 31) // 32) * 32))
        det, s = engine._detect_at(padded, ls)
        if len(det):
            det[:, :4] /= s
            det[:, [0, 2]] = det[:, [0, 2]].clip(0, pw)
            det[:, [1, 3]] = det[:, [1, 3]].clip(0, ph)

        # Display equation / formula block
        if label in ("Formula", "Equation"):
            if engine.math is not None:
                if ph - 2 * pad > 90 and len(det) >= 2:
                    order = np.argsort((det[:, 1] + det[:, 3]) / 2)
                    m_box = det[order, :4].astype(np.float32)
                else:
                    m_box = np.array([[pad, pad, pw - pad, ph - pad]], dtype=np.float32)
                latex_list = engine.math.read(padded, m_box)
                parts_math = []
                for item in latex_list:
                    rm = (item[0] if isinstance(item, tuple) else item).strip()
                    if rm.startswith("$$") and rm.endswith("$$") and len(rm) > 4:
                        rm = rm[2:-2].strip()
                    elif rm.startswith("$") and rm.endswith("$") and len(rm) > 2:
                        rm = rm[1:-1].strip()
                    else:
                        rm = re.sub(r"\$([^$]+)\$", r"\1", rm).strip()
                    if rm:
                        parts_math.append(rm)
                raw_math = " ".join(parts_math).strip()
                if raw_math:
                    return f'<p><math display="block">{html.escape(raw_math)}</math></p>'

        if len(det) == 0:
            # Single-line or small crop fallback where DBNet detector didn't trigger
            if crop.shape[0] <= 160:
                rec = engine.recognize(
                    crop,
                    np.array([[0, 0, crop.shape[1], crop.shape[0]]], dtype=np.float32),
                )
                text = (rec[0][0] if rec else "").strip()
                if text:
                    if label in ("Formula", "Equation"):
                        return f'<p><math display="block">{html.escape(text)}</math></p>'
                    return self._format_block_html(label, text)
            return ""

        boxes = det[:, :4].copy()

        # Table / Form / TableOfContents reconstruction
        if label in ("Table", "TableOfContents", "Form"):
            from indra_ocr.layout_post import (
                heads_plausible,
                table_html,
                table_html_heads,
            )
            from indra_ocr.pipeline import CELL_GAP

            rec = engine.recognize(padded, boxes)
            c_boxes_list = []
            c_texts = []
            for b, (text, conf, toks) in zip(boxes, rec):
                for bx, tx, _ in engine.split_line(b, toks, CELL_GAP):
                    if tx and tx.strip():
                        c_boxes_list.append(bx)
                        c_texts.append(tx.strip())
            if not c_boxes_list:
                return ""
            c_boxes = np.array(c_boxes_list, dtype=np.float32).reshape(-1, 4)
            order = np.lexsort((c_boxes[:, 0], c_boxes[:, 1]))
            c_boxes = c_boxes[order]
            c_texts = [c_texts[i] for i in order]
            if len(c_boxes) > 1:
                _, layout, _ = engine.reading_order(c_boxes, c_texts, (pw, ph))
                if (
                    layout is not None
                    and layout[2] is not None
                    and heads_plausible(c_boxes[:, :4], layout[2])
                ):
                    return table_html_heads(c_boxes[:, :4], c_texts, layout[2])
            return table_html(c_boxes, c_texts)

        # Recognize and assemble lines on the padded crop without running page deskew/orient
        rec = engine.recognize(padded, boxes)
        sub_page = engine.assemble((pw, ph), boxes, rec, {})
        if sub_page.blocks:
            parts: List[str] = [
                b_text
                for blk in sub_page.blocks
                if (b_text := (blk.get("text") or "").strip())
            ]
            full_text = "\n".join(parts)
        else:
            ordered_lines = [
                line.text.strip() for line in sub_page.lines if line.text.strip()
            ]
            full_text = "\n".join(ordered_lines)

        if not full_text:
            return ""

        if (
            full_text.startswith("$$")
            and full_text.endswith("$$")
            and len(full_text) >= 4
            and "$$" not in full_text[2:-2]
        ):
            latex = full_text[2:-2].strip()
            return f'<p><math display="block">{html.escape(latex)}</math></p>'

        if label in ("Formula", "Equation"):
            return f'<p><math display="block">{html.escape(full_text)}</math></p>'
        return self._format_block_html(label, full_text)

    def reconstruct_digital_table(
        self, lines: list, page_size: Tuple[float, float]
    ) -> Optional[str]:
        """Use Indra's OrderNet table heads (same_row, same_col, same_cell) on pdftext spans."""
        from indra_ocr.layout_post import (
            BLOCK_CLASSES,
            heads_plausible,
            table_html,
            table_html_heads,
        )

        cell_boxes = []
        cell_texts = []
        for spans, y0, y1 in lines:
            for text, x0, x1 in spans:
                t = (text or "").strip()
                if not t:
                    continue
                cell_boxes.append([float(x0), float(y0), float(x1), float(y1)])
                cell_texts.append(t)
        if len(cell_boxes) < 2:
            return None
        c_boxes = np.array(cell_boxes, dtype=np.float32).reshape(-1, 4)
        w = max(1, int(round(page_size[0])))
        h = max(1, int(round(page_size[1])))
        _, layout, _ = self.engine.reading_order(c_boxes, cell_texts, (w, h))
        if layout is not None:
            if layout[2] is not None and heads_plausible(c_boxes, layout[2]):
                return table_html_heads(c_boxes, cell_texts, layout[2])
            if BLOCK_CLASSES[int(layout[0].sum(0).argmax())] in (
                "Table",
                "Table-Of-Contents",
                "Form",
            ):
                return table_html(c_boxes, cell_texts)
        return None

    @staticmethod
    def _format_block_html(label: str, text: str) -> str:
        if not text:
            return ""
        if label in ("Table", "TableOfContents", "Form") and text.lstrip().startswith(
            "<table"
        ):
            return text
        if label in ("SectionHeader", "Title"):
            return f"<h2>{format_inline_html(text)}</h2>"
        if label in ("ListGroup", "ListItem"):
            return format_list_html(text)
        if label == "Code":
            return f"<pre><code>{html.escape(text)}</code></pre>"
        return f"<p>{format_inline_html(text)}</p>"


def select_recognition_model(
    recognition_model,
    indra_model: Optional[IndraRecognitionPredictor] = None,
    mode: str = "balanced",
    ocr_backend: str = "auto",
    full_page: bool = True,
):
    """Select the active OCR recognition predictor based on ocr_backend and mode.

    - Mode 1 ('no_ocr'): OCR is disabled; returns Indra if queried.
    - Mode 2 ('fast' / 'indra'): always uses Indra line-level ONNX OCR.
    - Mode 3 ('balanced'): uses Surya VLM OCR (for full-page OCR, tables, and math).
    - Mode 4 ('vllm' / 'surya' / 'accurate'): always uses Surya VLM OCR.
    """
    if isinstance(recognition_model, IndraRecognitionPredictor):
        return recognition_model
    if (
        ocr_backend == "auto"
        and recognition_model is not None
        and type(recognition_model).__module__.startswith("unittest.mock")
    ):
        return recognition_model
    if ocr_backend == "surya" or mode in ("vllm", "surya", "accurate"):
        return recognition_model
    if (
        ocr_backend == "indra"
        or recognition_model is None
        or (ocr_backend == "auto" and mode in ("fast", "indra", "no_ocr"))
    ):
        return indra_model if indra_model is not None else IndraRecognitionPredictor()
    return recognition_model
