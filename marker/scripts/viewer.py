"""Interactive web viewer for Marker (modelled on indra-ocr-viewer).

Run standalone or via `marker_gui` / `marker_server`:
    marker_gui --port 8000
    marker_server --port 8000
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional

import pypdfium2
from PIL import Image

from marker.builders.document import DocumentBuilder
from marker.builders.line import LineBuilder
from marker.builders.ocr import OcrBuilder
from marker.builders.structure import StructureBuilder
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.logger import get_logger
from marker.models import create_model_dict
from marker.providers.registry import provider_from_filepath
from marker.renderers.chunk import ChunkRenderer, json_to_chunks
from marker.renderers.json import JSONRenderer
from marker.renderers.markdown import MarkdownRenderer, cleanup_text
from marker.schema import BlockTypes
from marker.scripts.common import sanitize_html_for_display
from marker.settings import settings

logger = get_logger()

try:
    from indra_ocr.viewer import LOGO
except Exception:
    LOGO = ""

MARKER_VERSION = "2.1.0"
MAX_SIDE = 1600
MODES = ("auto", "no_ocr", "fast", "indra", "balanced", "vllm", "surya", "accurate")
TIMING_KEYS = ("layout_ms", "ocr_ms", "postprocess_ms", "render_ms", "total_ms")

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".tiff",
    ".tif",
    ".bmp",
    ".pptx",
    ".docx",
    ".xlsx",
    ".html",
    ".htm",
    ".epub",
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".tiff",
    ".tif",
    ".bmp",
}

GROUP_CONTAINER_TYPES = {
    BlockTypes.ListGroup,
    BlockTypes.FigureGroup,
    BlockTypes.PictureGroup,
    BlockTypes.TableGroup,
}


def _default_device_mode() -> str:
    return "balanced" if settings.TORCH_DEVICE_MODEL == "cuda" else "fast"


def _parse_page_range_spec(spec: str, total_pages: int, max_pages: int) -> List[int]:
    """Parse a comma-separated page range string like '0,2-5' (0-indexed)."""
    spec = (spec or "").strip()
    if not spec:
        return list(range(min(total_pages, max(1, max_pages))))
    pages: List[int] = []
    seen = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a_str, b_str = part.split("-", 1)
            a = max(0, int(a_str.strip()))
            b = min(total_pages - 1, int(b_str.strip()))
            for p in range(a, b + 1):
                if 0 <= p < total_pages and p not in seen:
                    seen.add(p)
                    pages.append(p)
        else:
            p = int(part)
            if 0 <= p < total_pages and p not in seen:
                seen.add(p)
                pages.append(p)
    return pages[: max(1, max_pages)]


def _inline_images_in_html(html_str: str, encoded_images: Dict[str, str]) -> str:
    """Replace relative image src attributes with base64 data URLs for preview."""
    if not html_str or not encoded_images:
        return html_str

    def _repl(match: re.Match) -> str:
        quote = match.group(1)
        src = match.group(2)
        if src in encoded_images:
            return f"src={quote}{encoded_images[src]}{quote}"
        return match.group(0)

    return re.sub(r"""src=(['"])([^'"]+)\1""", _repl, html_str)


def _extract_body_html(full_html: str) -> str:
    """Extract inner content of <body>...</body> from HTMLRenderer output."""
    m = re.search(r"<body[^>]*>(.*)</body>", full_html or "", flags=re.S | re.I)
    return m.group(1).strip() if m else (full_html or "").strip()


class ViewerEngine:
    """Holds uploaded documents and converts one page per request with the
    selected pipeline options so the browser displays each page as it finishes."""

    def __init__(
        self,
        model_supplier: Optional[Callable[[], Dict[str, Any]]] = None,
        defaults: Optional[Dict[str, Any]] = None,
        max_upload_bytes: int = 100 * 1024 * 1024,
    ):
        self._model_supplier = model_supplier
        self._models: Optional[Dict[str, Any]] = None
        self.max_upload_bytes = max_upload_bytes
        self.defaults = defaults or {
            "mode": "auto",
            "force_ocr": False,
            "use_llm": False,
            "strip_existing_ocr": False,
            "keep_headers_footers": False,
            "dpi": 150,
            "max_pages": 20,
        }
        self.tmp = tempfile.TemporaryDirectory(prefix="marker_viewer_")
        self.docs_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.docs: Dict[str, Dict[str, Any]] = {}

    def get_models(self) -> Dict[str, Any]:
        if self._model_supplier is not None:
            models = self._model_supplier()
            if models is not None:
                return models
        if self._models is None:
            with self.model_lock:
                if self._models is None:
                    self._models = create_model_dict()
        return self._models

    def info(self, has_api_docs: bool = True) -> Dict[str, Any]:
        return {
            "version": MARKER_VERSION,
            "device": str(settings.TORCH_DEVICE_MODEL),
            "default_device_mode": _default_device_mode(),
            "modes": ["auto", "no_ocr", "fast", "balanced", "vllm"],
            "defaults": self.defaults,
            "api_docs": has_api_docs,
        }

    def upload(self, data: bytes, filename: str = "document.pdf") -> Dict[str, Any]:
        if not data:
            raise ValueError("Uploaded file is empty.")
        if len(data) > self.max_upload_bytes:
            raise ValueError("Uploaded file exceeds maximum allowed size.")

        raw_name = os.path.basename(filename or "document.pdf")
        ext = os.path.splitext(raw_name)[1].lower()
        if not ext:
            if data[:5] == b"%PDF-":
                ext = ".pdf"
            else:
                ext = ".png"
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")

        did = uuid.uuid4().hex[:12]
        filepath = os.path.join(self.tmp.name, f"{did}{ext}")
        with open(filepath, "wb") as f:
            f.write(data)

        try:
            if ext == ".pdf" or data[:5] == b"%PDF-":
                pdf_doc = pypdfium2.PdfDocument(filepath)
                try:
                    num_pages = len(pdf_doc)
                finally:
                    pdf_doc.close()
            elif ext in IMAGE_EXTENSIONS:
                with Image.open(filepath) as im:
                    num_pages = max(1, int(getattr(im, "n_frames", 1)))
            else:
                provider_cls = provider_from_filepath(filepath)
                with provider_cls(filepath) as provider:
                    num_pages = len(provider)
        except Exception as exc:
            try:
                os.remove(filepath)
            except OSError:
                pass
            raise ValueError(f"Could not read document: {exc}") from exc

        with self.docs_lock:
            self.docs[did] = {
                "id": did,
                "path": filepath,
                "filename": raw_name,
                "ext": ext,
                "pages": max(1, num_pages),
            }
            while len(self.docs) > 24:
                old_id = next(iter(self.docs))
                old_info = self.docs.pop(old_id, None)
                if old_info and os.path.exists(old_info["path"]):
                    try:
                        os.remove(old_info["path"])
                    except OSError:
                        pass

        return {
            "id": did,
            "pages": max(1, num_pages),
            "filename": raw_name,
        }

    def page(self, did: str, n: int, opts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        opts = opts or {}
        with self.docs_lock:
            doc_info = self.docs.get(did)
        if doc_info is None:
            raise ValueError("Document not found — please drop or select the file again.")
        filepath = doc_info["path"]
        total_pages = doc_info["pages"]
        if n < 0 or n >= total_pages:
            raise ValueError(f"Page index {n} out of range (0..{total_pages - 1}).")

        mode = str(opts.get("mode") or "auto").lower()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")

        force_ocr = bool(opts.get("force_ocr", False))
        use_llm = bool(opts.get("use_llm", False))
        strip_existing_ocr = bool(opts.get("strip_existing_ocr", False))
        keep_headers_footers = bool(
            opts.get("keep_headers_footers", opts.get("furniture", False))
        )
        dpi = int(max(72, min(300, float(opts.get("dpi") or 150))))

        cli_options = {
            "page_range": str(n),
            "force_ocr": force_ocr,
            "disable_ocr": mode == "no_ocr",
            "use_llm": use_llm,
            "strip_existing_ocr": strip_existing_ocr,
            "keep_pageheader_in_output": keep_headers_footers,
            "keep_pagefooter_in_output": keep_headers_footers,
            "mode": None if mode == "auto" else mode,
        }
        config_parser = ConfigParser(cli_options)
        config_dict = config_parser.generate_config_dict()
        config_dict["pdftext_workers"] = 1
        config_dict["disable_tqdm"] = True

        if True:
            t_start = time.perf_counter()
            models = self.get_models()
            converter = PdfConverter(
                config=config_dict,
                artifact_dict=models,
                processor_list=config_parser.get_processors(),
                renderer=None,
                llm_service=config_parser.get_llm_service(),
            )
            resolved_mode = converter.mode

            provider_cls = provider_from_filepath(filepath)
            layout_builder = converter.resolve_dependencies(
                converter.layout_builder_class
            )
            line_builder = converter.resolve_dependencies(LineBuilder)
            ocr_builder = converter.resolve_dependencies(OcrBuilder)
            doc_builder = DocumentBuilder(converter.config)

            with provider_cls(filepath, converter.config) as provider:
                t_l0 = time.perf_counter()
                document = doc_builder.build_document(provider)
                if not doc_builder.disable_ocr and ocr_builder.can_convert_pdf_with_indra(
                    provider, layout_builder
                ):
                    layout_ms = (time.perf_counter() - t_l0) * 1000.0
                    t_ocr0 = time.perf_counter()
                    ocr_builder.convert_pdf_with_indra(document, provider)
                    ocr_ms = (time.perf_counter() - t_ocr0) * 1000.0
                else:
                    layout_builder(document, provider)
                    line_builder(document, provider)
                    doc_builder.render_highres(document, provider)
                    layout_ms = (time.perf_counter() - t_l0) * 1000.0
                    t_ocr0 = time.perf_counter()
                    if not doc_builder.disable_ocr:
                        ocr_builder(document, provider)
                    ocr_ms = (time.perf_counter() - t_ocr0) * 1000.0

                provider_lines = list(
                    (getattr(provider, "page_lines", {}) or {}).get(n, [])
                )
                display_img = provider.get_images([n], dpi)[0].convert("RGB")

            t_pp0 = time.perf_counter()
            structure_builder_cls = converter.resolve_dependencies(StructureBuilder)
            structure_builder_cls(document)
            for processor in converter.processor_list:
                processor(document)
            postprocess_ms = (time.perf_counter() - t_pp0) * 1000.0

            t_r0 = time.perf_counter()
            md_renderer = converter.resolve_dependencies(MarkdownRenderer)
            json_renderer = converter.resolve_dependencies(JSONRenderer)
            chunk_renderer = converter.resolve_dependencies(ChunkRenderer)

            doc_output = document.render(md_renderer.block_config)
            full_html, extracted_images = md_renderer.extract_html(document, doc_output)
            markdown_str = cleanup_text(md_renderer.md_cls.convert(full_html))

            img_fmt = settings.OUTPUT_IMAGE_FORMAT.lower()
            encoded_images: Dict[str, str] = {}
            for img_name, pil_im in (extracted_images or {}).items():
                buf = io.BytesIO()
                pil_im.convert("RGB").save(buf, format=settings.OUTPUT_IMAGE_FORMAT)
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                encoded_images[img_name] = f"data:image/{img_fmt};base64,{b64}"

            preview_html = sanitize_html_for_display(
                _inline_images_in_html(_extract_body_html(full_html), encoded_images)
            )
            clean_full_html = sanitize_html_for_display(full_html)

            page_json_dict: Dict[str, Any] = {}
            page_chunks_list: List[Dict[str, Any]] = []
            if doc_output.children:
                page_json_model = json_renderer.extract_json(
                    document, doc_output.children[0]
                )
                page_json_dict = page_json_model.model_dump(mode="json")
                raw_chunks = json_to_chunks(
                    page_json_model,
                    set(str(b) for b in chunk_renderer.image_blocks),
                )
                if isinstance(raw_chunks, list):
                    page_chunks_list = [c.model_dump(mode="json") for c in raw_chunks]
                elif raw_chunks is not None:
                    page_chunks_list = [raw_chunks.model_dump(mode="json")]

            # Prepare display image (downscaled to MAX_SIDE if needed)
            orig_w, orig_h = display_img.size
            scale = min(1.0, MAX_SIDE / max(orig_w, orig_h, 1))
            if scale < 1.0:
                disp_w = max(1, int(round(orig_w * scale)))
                disp_h = max(1, int(round(orig_h * scale)))
                display_img = display_img.resize(
                    (disp_w, disp_h), Image.Resampling.LANCZOS
                )
            else:
                disp_w, disp_h = orig_w, orig_h

            jpg_buf = io.BytesIO()
            display_img.save(jpg_buf, format="JPEG", quality=84)
            img_data_url = "data:image/jpeg;base64," + base64.b64encode(
                jpg_buf.getvalue()
            ).decode("ascii")

            page_obj = document.pages[0]
            page_size = (
                max(1.0, float(page_obj.polygon.width)),
                max(1.0, float(page_obj.polygon.height)),
            )

            def _scale_bbox(poly) -> List[float]:
                rescaled = poly.rescale(page_size, (disp_w, disp_h)).bbox
                return [round(float(v), 1) for v in rescaled[:4]]

            # Collect layout blocks in reading order
            ordered_blocks = []
            seen_block_ids = set()

            def _visit_block(blk):
                if blk is None or blk.removed or blk.id in seen_block_ids:
                    return
                if blk.block_type in GROUP_CONTAINER_TYPES and blk.structure:
                    for child_id in blk.structure:
                        _visit_block(page_obj.get_block(child_id))
                    return
                if blk.block_type in (BlockTypes.Line, BlockTypes.Span, BlockTypes.Char):
                    return
                seen_block_ids.add(blk.id)
                ordered_blocks.append(blk)

            for bid in page_obj.structure or []:
                _visit_block(page_obj.get_block(bid))

            blocks_out = []
            block_methods = set()
            for blk in ordered_blocks:
                btype = (
                    blk.block_type.name
                    if hasattr(blk.block_type, "name")
                    else str(blk.block_type)
                )
                b_method = (
                    blk.text_extraction_method
                    or page_obj.text_extraction_method
                    or "pdftext"
                )
                block_methods.add(b_method)
                b_text = (blk.raw_text(document) or "").strip()
                b_html = sanitize_html_for_display(getattr(blk, "html", "") or "")
                blocks_out.append(
                    {
                        "id": str(blk.id),
                        "label": btype,
                        "box": _scale_bbox(blk.polygon),
                        "text": b_text,
                        "html": b_html,
                        "source": b_method,
                        "ignored": bool(blk.ignore_for_output),
                    }
                )

            # Collect text lines on the page
            lines_out = []
            doc_lines = page_obj.contained_blocks(document, (BlockTypes.Line,))
            if doc_lines:
                for ln in doc_lines:
                    if ln.removed:
                        continue
                    l_txt = (ln.raw_text(document) or "").strip()
                    if not l_txt:
                        continue
                    lines_out.append(
                        {
                            "box": _scale_bbox(ln.polygon),
                            "text": l_txt,
                        }
                    )
            elif provider_lines:
                for po in provider_lines:
                    l_txt = (po.raw_text or "").strip()
                    if not l_txt:
                        continue
                    lines_out.append(
                        {
                            "box": _scale_bbox(po.line.polygon),
                            "text": l_txt,
                        }
                    )

            # Determine page extraction source chip
            page_method = page_obj.text_extraction_method or "pdftext"
            if resolved_mode == "no_ocr":
                source_chip = "no_ocr"
            elif resolved_mode in ("vllm", "surya", "accurate"):
                source_chip = "surya"
            elif resolved_mode in ("fast", "indra"):
                if page_method == "surya":
                    source_chip = "indra"
                elif "surya" in block_methods:
                    source_chip = "hybrid"
                else:
                    source_chip = "text"
            else:
                if page_method == "surya":
                    source_chip = "surya"
                elif "surya" in block_methods:
                    source_chip = "hybrid"
                else:
                    source_chip = "text"

            render_ms = (time.perf_counter() - t_r0) * 1000.0
            total_ms = (time.perf_counter() - t_start) * 1000.0

            timings = {
                "layout_ms": round(layout_ms, 1),
                "ocr_ms": round(ocr_ms, 1),
                "postprocess_ms": round(postprocess_ms, 1),
                "render_ms": round(render_ms, 1),
                "total_ms": round(total_ms, 1),
            }

            return {
                "page": n + 1,
                "page_index": n,
                "w": disp_w,
                "h": disp_h,
                "img": img_data_url,
                "source": source_chip,
                "mode_used": resolved_mode,
                "blocks": blocks_out,
                "lines": lines_out,
                "markdown": markdown_str,
                "preview_html": preview_html,
                "html": clean_full_html,
                "chunks": page_chunks_list,
                "json": page_json_dict,
                "images_count": len(encoded_images),
                "timings": timings,
            }


def render_viewer_html() -> str:
    return PAGE.replace("__LOGO__", LOGO)


def make_handler(engine: ViewerEngine):
    """Standard library HTTP handler for running the viewer without FastAPI/uvicorn."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(
                    200,
                    render_viewer_html().encode("utf-8"),
                    "text/html; charset=utf-8",
                )
            elif self.path == "/info":
                self._send(200, json.dumps(engine.info(has_api_docs=False)).encode("utf-8"))
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            try:
                if self.path == "/upload":
                    fname = self.headers.get("X-Filename", "document.pdf")
                    res = engine.upload(body, filename=fname)
                elif self.path == "/page":
                    q = json.loads(body)
                    res = engine.page(q["id"], int(q["n"]), q.get("opts", {}))
                else:
                    return self._send(404, b"not found", "text/plain")
                self._send(200, json.dumps(res).encode("utf-8"))
            except Exception as exc:
                logger.warning(f"Viewer request error: {exc}")
                self._send(400, json.dumps({"error": str(exc)}).encode("utf-8"))

    return Handler


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Marker Viewer · Datalab</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
 --font-body:"Geist",ui-sans-serif,system-ui,sans-serif;--font-mono:"Geist Mono",ui-monospace,"SF Mono",Menlo,monospace;
 --page:oklch(98.5% .004 263);--surface:oklch(96.5% .006 263);--surface-2:oklch(94% .007 263);
 --rule:oklch(86% .008 263);--rule-strong:oklch(78% .01 263);
 --ink:oklch(18% .012 263);--ink-2:oklch(36% .012 263);--ink-3:oklch(48% .014 263);--ink-4:oklch(70% .014 263);
 --accent:oklch(38% .24 263);--accent-soft:oklch(96% .04 263);--accent-strong:oklch(30% .26 263);--accent-ink:oklch(99% 0 0);
 --ok:oklch(58% .15 145);--warn:oklch(72% .16 75);--err:oklch(56% .2 25);--focus:oklch(38% .24 263 / .35);
 --hairline:1px;--ease:cubic-bezier(.25,1,.5,1)}
@media (min-resolution:2dppx){:root{--hairline:.5px}}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 var(--font-body);-webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.mono{font-family:var(--font-mono)}
.label{font:500 10.5px/1 var(--font-mono);letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3)}
/* top bar */
.top{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:14px;height:56px;padding:0 24px;background:color-mix(in oklch,var(--page) 88%,transparent);backdrop-filter:blur(8px);border-bottom:var(--hairline) solid var(--rule)}
.brand{display:flex;align-items:center;gap:10px;font-weight:600;letter-spacing:-.01em}
.mark{width:30px;height:30px;margin:0 -4px}
.brand .sep{color:var(--ink-4);font-weight:400}.brand .prod{font-weight:500}
.ver{font:11px var(--font-mono);color:var(--ink-3);border:var(--hairline) solid var(--rule);border-radius:999px;padding:2px 8px}
.top .sp{flex:1}.top nav{display:flex;align-items:center;gap:18px;font-size:13px}.top nav a{color:var(--ink-2)}
.cta{background:var(--accent);color:var(--accent-ink)!important;padding:7px 12px;border-radius:6px;font-weight:500}.cta:hover{background:var(--accent-strong);text-decoration:none}
/* layout */
.wrap{display:grid;grid-template-columns:316px minmax(0,1fr);min-height:calc(100vh - 56px)}
aside{border-right:var(--hairline) solid var(--rule);padding:20px;display:flex;flex-direction:column;gap:20px;position:sticky;top:56px;height:calc(100vh - 56px);overflow:auto;background:var(--page)}
main{padding:24px;display:flex;flex-direction:column;gap:24px;min-width:0}
@media(max-width:920px){.wrap{grid-template-columns:1fr}aside{position:static;height:auto;border-right:0;border-bottom:var(--hairline) solid var(--rule)}}
/* drop zone */
#drop{border:1px dashed var(--rule-strong);border-radius:10px;padding:18px 14px;text-align:center;cursor:pointer;background:var(--surface);transition:border-color var(--ease) .12s,background .12s}
#drop:hover,#drop.over{border-color:var(--accent);background:var(--accent-soft)}
#drop b{display:block;font-weight:500;margin-bottom:2px}#drop span{color:var(--ink-3);font-size:12px}
#fname{margin-top:8px;font:12px var(--font-mono);color:var(--ink-2);word-break:break-all}
/* controls */
.group{display:flex;flex-direction:column;gap:10px}
.seg{display:grid;grid-template-columns:repeat(5,1fr);border:var(--hairline) solid var(--rule-strong);border-radius:7px;overflow:hidden}
.seg button{border:0;background:transparent;padding:7px 2px;font:500 11.5px var(--font-body);color:var(--ink-2);cursor:pointer;white-space:nowrap}
.seg button+button{border-left:var(--hairline) solid var(--rule)}.seg button.on{background:var(--ink);color:var(--page)}
.hint{font-size:12px;color:var(--ink-3);line-height:1.4}
.sw{display:flex;align-items:center;justify-content:space-between;gap:10px;font-size:13px;color:var(--ink-2);cursor:pointer;user-select:none}
.sw input{appearance:none;width:30px;height:18px;border-radius:999px;background:var(--rule-strong);position:relative;cursor:pointer;transition:background .15s;flex:none}
.sw input::after{content:"";position:absolute;top:2px;left:2px;width:14px;height:14px;border-radius:50%;background:#fff;transition:transform .15s var(--ease)}
.sw input:checked{background:var(--accent)}.sw input:checked::after{transform:translateX(12px)}
.sw input:focus-visible{outline:3px solid var(--focus)}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.field{display:flex;flex-direction:column;gap:5px}.field select,.field input{font:13px var(--font-body);padding:6px 8px;border:var(--hairline) solid var(--rule-strong);border-radius:6px;background:#fff;color:var(--ink);width:100%}
.btn{border:0;background:var(--accent);color:var(--accent-ink);font:500 13px var(--font-body);padding:9px 12px;border-radius:7px;cursor:pointer;text-align:center}
.btn:hover{background:var(--accent-strong)}.btn:disabled{background:var(--rule-strong);cursor:default}
.btn-sec{border:var(--hairline) solid var(--rule-strong);background:#fff;color:var(--ink-2);font:500 12px var(--font-body);padding:6px 10px;border-radius:6px;cursor:pointer}
.btn-sec:hover{background:var(--surface)}
.dot{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;vertical-align:0}
#stats{font:12px/1.7 var(--font-mono);color:var(--ink-2)}#stats b{color:var(--ink);font-weight:500}
#status{font:500 12px var(--font-mono);color:var(--ink-2);min-height:18px}
/* progress banner */
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}
.pbanner{position:sticky;top:68px;z-index:4;background:#fff;border:1.5px solid var(--accent);border-radius:12px;padding:14px 18px;box-shadow:0 6px 20px oklch(18% .012 263 / .10);display:flex;flex-direction:column;gap:10px;transition:border-color .2s,background .2s}
.pbanner[hidden]{display:none!important}
.pbanner.done{border-color:var(--ok);background:oklch(98.5% .015 145)}
.pbanner.err{border-color:var(--err);background:oklch(98% .02 25)}
.prow{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.spin{width:18px;height:18px;border:2.5px solid var(--accent-soft);border-top-color:var(--accent);border-radius:50%;animation:spin .75s linear infinite;flex:none}
.pbanner.done .spin,.pbanner.err .spin{display:none}
.pbadge{display:inline-flex;align-items:center;justify-content:center;width:20px;height:20px;border-radius:50%;font:700 12px var(--font-mono);color:#fff;flex:none}
.pbanner.done .pbadge{background:var(--ok)}
.pbanner.err .pbadge{background:var(--err)}
.ptitle{font:600 14px var(--font-body);color:var(--ink)}
.pbanner.err .ptitle{color:var(--err)}
.psub{font:500 12.5px var(--font-mono);color:var(--ink-2);margin-left:auto}
.ptrack{height:8px;border-radius:999px;background:var(--surface-2);overflow:hidden;position:relative}
.pfill{height:100%;border-radius:999px;background:var(--accent);transition:width .25s var(--ease)}
.pfill.indet{width:100%!important;background:linear-gradient(90deg,var(--accent-soft) 0%,var(--accent) 50%,var(--accent-soft) 100%);background-size:200% 100%;animation:shimmer 1.4s ease-in-out infinite}
.pbanner.done .pfill{background:var(--ok)}
.pbanner.err .pfill{background:var(--err)}
.inflight{border:1.5px dashed var(--accent);border-radius:12px;padding:36px 24px;text-align:center;background:color-mix(in oklch,var(--accent-soft) 45%,#fff);color:var(--ink-2);display:flex;flex-direction:column;align-items:center;gap:10px}
.inflight b{font-size:15px;color:var(--ink)}
.callout{border:var(--hairline) solid var(--rule);border-radius:10px;padding:12px;background:var(--surface);font-size:12.5px;color:var(--ink-2)}
/* empty state */
.empty{border:1px dashed var(--rule-strong);border-radius:14px;padding:64px 24px;text-align:center;color:var(--ink-3)}
.empty h2{font-weight:500;color:var(--ink);letter-spacing:-.015em;margin:0 0 6px;font-size:22px}
/* page card */
.card{background:#fff;border:var(--hairline) solid var(--rule);border-radius:12px;overflow:hidden}
.ch{display:flex;align-items:center;flex-wrap:wrap;gap:10px 14px;padding:12px 16px;border-bottom:var(--hairline) solid var(--rule)}
.ch h3{margin:0;font-size:15px;font-weight:600;letter-spacing:-.01em}
.chip{font:500 10.5px/1 var(--font-mono);letter-spacing:.06em;text-transform:uppercase;padding:5px 7px;border-radius:5px;background:var(--surface-2);color:var(--ink-2)}
.chip.text,.chip.no_ocr{background:oklch(95% .04 145);color:oklch(38% .12 145)}
.chip.indra{background:oklch(95% .04 200);color:oklch(38% .12 200)}
.chip.surya{background:var(--accent-soft);color:var(--accent)}
.chip.hybrid{background:oklch(95% .05 75);color:oklch(42% .12 75)}
.chip.error{background:oklch(95% .05 25);color:var(--err)}
.meta{font:12px var(--font-mono);color:var(--ink-3)}.ch .sp{flex:1}
.tbar{display:flex;height:6px;border-radius:3px;overflow:hidden;background:var(--surface-2);width:170px}.tbar i{display:block;height:100%}
.body{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,1fr);min-height:440px}
@media(max-width:1200px){.body{grid-template-columns:1fr}}
.pageview{padding:16px;background:var(--surface);border-right:var(--hairline) solid var(--rule)}
svg.pg{width:100%;height:auto;display:block;background:#fff;box-shadow:0 1px 2px oklch(18% .012 263 / .08),0 0 0 var(--hairline) var(--rule)}
.right{display:flex;flex-direction:column;min-width:0}
.tabs{display:flex;align-items:center;gap:2px;padding:8px 12px 0;border-bottom:var(--hairline) solid var(--rule);overflow-x:auto}
.tabs button{border:0;background:transparent;padding:8px 10px;font:500 12.5px var(--font-body);color:var(--ink-3);cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px;white-space:nowrap}
.tabs button.on{color:var(--ink);border-bottom-color:var(--accent)}
.tabs .sp{flex:1}
.tabs .copy-btn{font:500 11px var(--font-mono);padding:4px 8px;margin-bottom:4px;border:var(--hairline) solid var(--rule);border-radius:5px;background:var(--surface);color:var(--ink-2)}
.tabs .copy-btn:hover{background:var(--surface-2)}
.pane{flex:1;overflow:auto;max-height:78vh;padding:14px 16px}
.pane.src{white-space:pre-wrap;font:12.5px/1.6 var(--font-mono);color:var(--ink-2)}
.preview{font-size:14px;line-height:1.6}
.preview h1,.preview h2,.preview h3,.preview h4{letter-spacing:-.015em;line-height:1.25;margin:1em 0 .4em}
.preview h1{font-size:20px}.preview h2{font-size:17px}.preview h3{font-size:15px}
.preview p{margin:.5em 0}.preview ul,.preview ol{padding-left:1.4em}
.preview table{border-collapse:collapse;margin:10px 0;font-size:12.5px;width:auto;max-width:100%}
.preview td,.preview th{border:var(--hairline) solid var(--rule-strong);padding:4px 7px;vertical-align:top}.preview th{background:var(--surface)}
.preview math{font:12.5px var(--font-mono);background:var(--accent-soft);color:var(--accent-strong);padding:1px 5px;border-radius:4px}
.preview math[display="block"]{display:block;padding:8px 10px;margin:8px 0;overflow-x:auto}
.preview pre{background:var(--surface);padding:10px 12px;border-radius:6px;overflow-x:auto;font:12px/1.5 var(--font-mono)}
.preview code{font:12px var(--font-mono);background:var(--surface);padding:1px 4px;border-radius:4px}
.preview img{max-width:100%;height:auto;border-radius:6px;border:var(--hairline) solid var(--rule);margin:6px 0}
.ln-row,.bk-row{display:flex;align-items:baseline;gap:10px;padding:4px 6px;border-radius:5px;font-size:13px;cursor:default}
.ln-row:hover,.ln-row.on,.bk-row:hover,.bk-row.on{background:var(--accent-soft)}
.ln-row .k,.bk-row .k{font:11px var(--font-mono);color:var(--ink-4);min-width:26px;text-align:right}
.ln-row .t,.bk-row .t{flex:1;min-width:0;word-break:break-word}
.bk-row .btag{font:500 10.5px var(--font-mono);padding:2px 6px;border-radius:4px;background:var(--surface-2);color:var(--ink-2);white-space:nowrap}
.bk-row.ignored{opacity:.55}
/* overlays */
rect.ln{fill:transparent;stroke:oklch(55% .15 263);stroke-width:1;vector-effect:non-scaling-stroke;opacity:.5}
rect.ln.on{fill:oklch(55% .2 263 / .18);stroke-width:2.5;opacity:1}
rect.bk{fill:transparent;stroke-width:2;vector-effect:non-scaling-stroke;cursor:pointer}
rect.bk.on{fill:oklch(55% .2 263 / .12);stroke-width:3.5}
text.bl{font-family:var(--font-mono);font-weight:600;paint-order:stroke;stroke:#fff;pointer-events:none}
path.ord{fill:none;stroke:oklch(60% .2 30);stroke-width:1.5;vector-effect:non-scaling-stroke;opacity:.82;pointer-events:none}
circle.on-n{fill:oklch(60% .2 30);pointer-events:none}
text.n{font-family:var(--font-mono);font-weight:600;fill:#fff;text-anchor:middle;dominant-baseline:central;pointer-events:none}
.hide-lines rect.ln,.hide-blocks rect.bk,.hide-blocks text.bl,.hide-labels text.bl,.hide-order .ord,.hide-order circle.on-n,.hide-order text.n{display:none}
footer{padding:18px 24px;border-top:var(--hairline) solid var(--rule);display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px;font-size:12px;color:var(--ink-3)}
</style></head><body>
<div class="top">
 <div class="brand"><img class="mark" src="data:image/png;base64,__LOGO__" alt="Datalab">Datalab<span class="sep">/</span><span class="prod">Marker</span></div>
 <span class="ver" id="ver"></span><span class="ver" id="dev" hidden></span>
 <div class="sp"></div>
 <nav><a href="/docs" id="apidocs" target="_blank" rel="noopener" hidden>API Docs</a><a href="https://github.com/datalab-to/marker" target="_blank" rel="noopener">GitHub</a><a href="https://documentation.datalab.to" target="_blank" rel="noopener">Docs</a><a class="cta" href="https://www.datalab.to/?utm_source=marker-viewer" target="_blank" rel="noopener">Try Chandra on Datalab</a></nav>
</div>
<div class="wrap">
<aside>
 <div>
  <div id="drop"><b>Drop a PDF, image, or document</b><span>PDF, PNG, JPG, DOCX, PPTX, XLSX, HTML, EPUB</span><input id="file" type="file" accept=".pdf,.png,.jpg,.jpeg,.gif,.webp,.tiff,.tif,.bmp,.pptx,.docx,.xlsx,.html,.htm,.epub" hidden></div>
  <div id="fname"></div>
 </div>
 <div class="group"><div class="label">Mode</div>
  <div class="seg" id="mode">
   <button data-v="auto">Auto</button>
   <button data-v="no_ocr">No OCR</button>
   <button data-v="fast">Fast</button>
   <button data-v="balanced">Balanced</button>
   <button data-v="vllm">vLLM</button>
  </div>
  <div class="hint" id="modehint"></div>
 </div>
 <div class="group"><div class="label">Pipeline</div>
  <label class="sw">Force OCR<input type="checkbox" id="force_ocr"></label>
  <label class="sw">Use LLM (hybrid)<input type="checkbox" id="use_llm"></label>
  <label class="sw">Strip existing OCR<input type="checkbox" id="strip_ocr"></label>
  <label class="sw">Keep headers and footers<input type="checkbox" id="furniture"></label>
  <div class="row2">
   <div class="field"><span class="label">DPI</span><select id="dpi"><option selected>150</option><option>200</option><option>300</option></select></div>
   <div class="field"><span class="label">Max pages</span><input id="maxp" type="number" min="1" value="20"></div>
  </div>
  <div class="field"><span class="label">Page range (0-indexed)</span><input id="prange" type="text" placeholder="All (or e.g. 0,2-5)"></div>
  <button class="btn" id="rerun" disabled>Convert again with these options</button>
 </div>
 <div class="group"><div class="label">Show</div>
  <label class="sw">Layout blocks<input type="checkbox" class="disp" data-c="blocks" checked></label>
  <label class="sw">Block labels<input type="checkbox" class="disp" data-c="labels" checked></label>
  <label class="sw">Reading order<input type="checkbox" class="disp" data-c="order" checked></label>
  <label class="sw">Lines<input type="checkbox" class="disp" data-c="lines"></label>
  <div id="legend" class="hint"></div>
 </div>
 <div class="group"><div class="label">Run</div>
  <div id="status"></div>
  <div id="stats"></div>
  <button class="btn-sec" id="dl-md" hidden>Download full Markdown (.md)</button>
 </div>
 <div class="callout">Need handwriting, complex forms, and highest-accuracy VLM parsing at scale? <a href="https://www.datalab.to/?utm_source=marker-viewer" target="_blank" rel="noopener">Chandra on Datalab</a> includes 5,000 free pages/month with a work email.</div>
</aside>
<main>
 <div class="pbanner" id="pbanner" hidden>
  <div class="prow">
   <div class="spin" id="pspin"></div>
   <div class="pbadge" id="pbadge" hidden>✓</div>
   <div class="ptitle" id="ptitle">Converting document…</div>
   <div class="psub" id="psub">0 / 0 pages · 0.0s</div>
   <button class="btn-sec" id="pbanner-dl" hidden>Download .md</button>
  </div>
  <div class="ptrack"><div class="pfill indet" id="pfill" style="width:0%"></div></div>
 </div>
 <div id="out" style="display:flex;flex-direction:column;gap:24px"><div class="empty"><h2>Convert a document</h2>Drop a PDF, image, or Office document on the left. Pages appear as they finish with interactive layout blocks, reading order, lines, Markdown, HTML, Chunks, and JSON.</div></div>
</main>
</div>
<footer><span>Marker · high-speed document conversion by <a href="https://www.datalab.to" target="_blank" rel="noopener">Datalab</a></span><span class="mono">Runs locally: your files stay on this machine.</span></footer>
<script>
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'})[c]);
const BLOCK={
 "Text":"oklch(55% .02 263)",
 "TextInlineMath":"oklch(48% .16 250)",
 "SectionHeader":"oklch(45% .22 263)",
 "ListItem":"oklch(58% .15 145)",
 "Table":"oklch(55% .18 320)",
 "Form":"oklch(50% .16 25)",
 "TableOfContents":"oklch(55% .12 180)",
 "Equation":"oklch(52% .22 300)",
 "Picture":"oklch(60% .15 160)",
 "Figure":"oklch(60% .15 160)",
 "Diagram":"oklch(60% .15 160)",
 "Caption":"oklch(62% .14 200)",
 "Footnote":"oklch(60% .12 60)",
 "Code":"oklch(48% .18 230)",
 "Blockquote":"oklch(55% .1 230)",
 "PageHeader":"oklch(68% .03 263)",
 "PageFooter":"oklch(68% .03 263)"
};
const LEGEND_ITEMS=[
 ["Text","oklch(55% .02 263)"],
 ["SectionHeader","oklch(45% .22 263)"],
 ["ListItem","oklch(58% .15 145)"],
 ["Table","oklch(55% .18 320)"],
 ["Equation","oklch(52% .22 300)"],
 ["Picture/Figure","oklch(60% .15 160)"],
 ["Caption/Footnote","oklch(62% .14 200)"],
 ["Header/Footer","oklch(68% .03 263)"]
];
const STAGES=[
 ["layout_ms","Layout","oklch(45% .22 263)"],
 ["ocr_ms","OCR","oklch(58% .15 145)"],
 ["postprocess_ms","Processors","oklch(68% .15 75)"],
 ["render_ms","Render","oklch(55% .18 320)"]
];
const HINT={
 auto:"Picks by device: Balanced on GPU (CUDA), Fast on CPU / Apple Silicon.",
 no_ocr:"Pure PDF text layer + Indra ONNX layout on CPU. Zero OCR calls (~24–33 pg/s).",
 fast:"In-process Indra ONNX layout, reading order, and ONNX OCR on CPU (~7–33 pg/s).",
 balanced:"Smart routing: pdftext + Indra on clean prose; Surya 2.5 VLM on tables, math, CJK & scans.",
 vllm:"Full-page Surya 2.5 VLM OCR on every page, skipping pdftext and Indra entirely."
};
let mode="auto",doc=null,stats=[],pagesMd=[],runId=0,timerId=null,runAbort=null;
function setMode(v){mode=v;$$('#mode button').forEach(b=>b.classList.toggle('on',b.dataset.v===v));$('#modehint').textContent=HINT[v]||''}
$$('#mode button').forEach(b=>b.onclick=()=>{setMode(b.dataset.v);if(doc)run();});
$('#legend').innerHTML=LEGEND_ITEMS.map(([k,c])=>`<span style="white-space:nowrap;margin-right:8px"><span class="dot" style="background:${c}"></span>${k}</span>`).join(' ');
function applyDisplay(){const o=$('#out');$$('.disp').forEach(i=>o.classList.toggle('hide-'+i.dataset.c,!i.checked))}
$$('.disp').forEach(i=>i.onchange=applyDisplay);applyDisplay();

const drop=$('#drop'),file=$('#file');
drop.onclick=()=>file.click();file.onchange=()=>file.files[0]&&start(file.files[0]);
drop.ondragover=e=>{e.preventDefault();drop.classList.add('over')};drop.ondragleave=()=>drop.classList.remove('over');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('over');e.dataTransfer.files[0]&&start(e.dataTransfer.files[0])};
$('#rerun').onclick=()=>doc&&run();
function dlMarkdown(){
 if(!pagesMd.length)return;
 const blob=new Blob([pagesMd.join('\n\n---\n\n')],{type:'text/markdown;charset=utf-8'});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);
 a.download=((doc&&doc.name)||'document').replace(/\.[^.]+$/,'')+'.md';a.click();
}
$('#dl-md').onclick=dlMarkdown;
$('#pbanner-dl').onclick=dlMarkdown;

function setBanner(state,title,sub,pct,indet){
 const b=$('#pbanner'),f=$('#pfill'),bd=$('#pbadge');
 b.hidden=false;
 b.classList.toggle('done',state==='done');
 b.classList.toggle('err',state==='err');
 bd.hidden=state==='run';
 bd.textContent=state==='err'?'!':'✓';
 $('#ptitle').textContent=title;
 $('#psub').textContent=sub||'';
 f.classList.toggle('indet',!!indet);
 f.style.width=Math.max(0,Math.min(100,pct||0))+'%';
 $('#pbanner-dl').hidden=!(state==='done'&&pagesMd.length);
 $('#status').textContent=title+(sub?' ('+sub+')':'');
}

function stopTimer(){if(timerId){clearInterval(timerId);timerId=null;}}

fetch('/info').then(r=>r.json()).then(i=>{
 $('#ver').textContent='v'+(i.version||'2.1.0');
 if(i.device){$('#dev').textContent=String(i.device).toUpperCase();$('#dev').hidden=false;}
 if(i.api_docs)$('#apidocs').hidden=false;
 const d=i.defaults||{};
 setMode(d.mode||'auto');
 $('#force_ocr').checked=!!d.force_ocr;
 $('#use_llm').checked=!!d.use_llm;
 $('#strip_ocr').checked=!!d.strip_existing_ocr;
 $('#furniture').checked=!!d.keep_headers_footers;
 if(d.dpi)$('#dpi').value=String(Math.round(d.dpi));
 if(d.max_pages)$('#maxp').value=d.max_pages;
}).catch(()=>setMode('auto'));
setMode('auto');

function parsePageIndices(spec,total,maxP){
 spec=(spec||'').trim();
 if(!spec){const N=Math.min(total,Math.max(1,maxP));return Array.from({length:N},(_,i)=>i);}
 const out=[],seen=new Set();
 for(const part of spec.split(',')){
  const s=part.trim();if(!s)continue;
  if(s.includes('-')){
   const [aStr,bStr]=s.split('-',2);
   const a=Math.max(0,parseInt(aStr,10)||0),b=Math.min(total-1,parseInt(bStr,10)||0);
   for(let p=a;p<=b;p++){if(p>=0&&p<total&&!seen.has(p)){seen.add(p);out.push(p);}}
  }else{
   const p=parseInt(s,10);
   if(!Number.isNaN(p)&&p>=0&&p<total&&!seen.has(p)){seen.add(p);out.push(p);}
  }
 }
 return out.slice(0,Math.max(1,maxP));
}

function abortRun(){if(runAbort){try{runAbort.abort();}catch(_){}runAbort=null;}}

async function start(f){
 stopTimer();abortRun();
 $('#fname').textContent=f.name+' · '+(f.size/1024).toFixed(0)+' KB';
 setBanner('run',`Uploading ${f.name}…`,(f.size/1024).toFixed(0)+' KB',100,true);
 try{
  const r=await fetch('/upload',{method:'POST',headers:{'X-Filename':encodeURIComponent(f.name)},body:await f.arrayBuffer()});
  const u=await r.json();
  if(!r.ok||u.error||u.detail){setBanner('err','Upload error: '+(u.error||u.detail||r.statusText),'',100,false);return}
  doc={...u,name:f.name};$('#rerun').disabled=false;run();
 }catch(err){
  setBanner('err','Upload failed: '+(err&&err.message?err.message:String(err)),'',100,false);
 }
}

function opts(){
 return{
  mode,
  force_ocr:$('#force_ocr').checked,
  use_llm:$('#use_llm').checked,
  strip_existing_ocr:$('#strip_ocr').checked,
  keep_headers_footers:$('#furniture').checked,
  dpi:+$('#dpi').value
 };
}

async function run(){
 stopTimer();abortRun();
 const id=++runId,ctrl=new AbortController(),o=opts(),maxP=Math.max(1,+$('#maxp').value||20);
 runAbort=ctrl;
 const pageList=parsePageIndices($('#prange').value,doc.pages,maxP);
 if(!pageList.length){setBanner('err','No valid pages in range (0..'+(doc.pages-1)+')','',100,false);return;}
 $('#out').innerHTML='';stats=[];pagesMd=[];$('#dl-md').hidden=true;summary();
 const inflight=document.createElement('div');
 inflight.className='inflight';
 const t0=performance.now();
 let curIdx=0,curPage=pageList[0]+1;
 const updateTick=()=>{
  const elapsed=((performance.now()-t0)/1000).toFixed(1)+'s';
  const pct=Math.round((curIdx/pageList.length)*100);
  const sub=`${curIdx} / ${pageList.length} pages (${pct}%) · ${elapsed}`;
  setBanner('run',`Converting Page ${curPage} (${curIdx+1} of ${pageList.length}) · mode: ${o.mode}`,sub,pct,curIdx===0);
  inflight.innerHTML=`<div class="spin"></div><b>Converting Page ${curPage} (${curIdx+1} of ${pageList.length})…</b><span class="mono" style="font-size:12px">${elapsed} elapsed · mode: ${esc(o.mode)}</span>`;
 };
 $('#out').appendChild(inflight);
 updateTick();
 timerId=setInterval(updateTick,100);
 for(let idx=0;idx<pageList.length;idx++){
  if(id!==runId)return;
  curIdx=idx;curPage=pageList[idx]+1;
  updateTick();
  const n=pageList[idx];
  let p;
  try{
   const resp=await fetch('/page',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:doc.id,n,opts:o}),signal:ctrl.signal});
   p=await resp.json();
  }catch(err){
   if(id!==runId||(err&&err.name==='AbortError'))return;
   stopTimer();inflight.remove();
   setBanner('err',`Error on page ${n+1}: ${err&&err.message?err.message:String(err)}`,`${idx} / ${pageList.length} pages completed`,100,false);
   return;
  }
  if(id!==runId)return;
  if(p.error||p.detail){
   stopTimer();
   inflight.remove();
   setBanner('err',`Error on page ${n+1}: ${p.error||p.detail}`,`${idx} / ${pageList.length} pages completed`,100,false);
   return;
  }
  inflight.remove();
  render(p);
  if(idx+1<pageList.length)$('#out').appendChild(inflight);
  stats.push(p.timings);if(p.markdown)pagesMd.push(p.markdown);$('#dl-md').hidden=!pagesMd.length;summary();
 }
 stopTimer();
 inflight.remove();
 const totMs=performance.now()-t0;
 const avgMs=(totMs/pageList.length).toFixed(0);
 const pps=(1000*pageList.length/Math.max(1,totMs)).toFixed(1);
 setBanner('done',`Converted ${pageList.length} page${pageList.length>1?'s':''} in ${(totMs/1000).toFixed(2)}s`,`${avgMs} ms/page · ${pps} pages/s`,100,false);
}

function summary(){
 const el=$('#stats');if(!stats.length){el.innerHTML='';return}
 const n=stats.length,s=k=>stats.reduce((a,t)=>a+(t[k]||0),0),tot=Math.max(1,s('total_ms'));
 el.innerHTML=`<b>${(tot/n).toFixed(0)} ms</b> per page · <b>${(1000*n/tot).toFixed(1)}</b> pages/s<br>`+
  STAGES.filter(([k])=>s(k)>0).map(([k,l])=>`${l} ${(s(k)/n).toFixed(0)} ms`).join(' · ');
}

function render(p){
 const c=document.createElement('div');c.className='card';const t=p.timings||{},tot=t.total_ms||1;
 const u=Math.max(1,p.w/640);
 const srcMap={text:'Text layer + Indra',no_ocr:'Text layer (No OCR)',indra:'Indra ONNX OCR',hybrid:'Hybrid (Text + VLM)',surya:'Surya VLM',error:'Error'};
 const src=srcMap[p.source]||p.source;
 const bar=STAGES.map(([k,l,col])=>`<i style="width:${100*(t[k]||0)/tot}%;background:${col}" title="${l}: ${t[k]||0} ms"></i>`).join('');
 const extra=(p.images_count?` · ${p.images_count} image${p.images_count>1?'s':''}`:'')+(p.mode_used?` · ${p.mode_used}`:'');
 const blocks=p.blocks.map((b,k)=>{
  const col=BLOCK[b.label]||'oklch(55% .02 263)',[x0,y0,x1,y1]=b.box;
  return `<rect class="bk" data-b="${k}" x="${x0-2}" y="${y0-2}" width="${Math.max(2,x1-x0+4)}" height="${Math.max(2,y1-y0+4)}" stroke="${col}"/><text class="bl" data-b="${k}" x="${x0}" y="${Math.max(11*u,y0-4*u)}" fill="${col}" font-size="${10.5*u}" stroke-width="${3*u}">${esc(b.label)}</text>`;
 }).join('');
 const lines=p.lines.map((l,k)=>`<rect class="ln" data-k="${k}" x="${l.box[0]}" y="${l.box[1]}" width="${Math.max(1,l.box[2]-l.box[0])}" height="${Math.max(1,l.box[3]-l.box[1])}"/>`).join('');
 const activeBlocks=p.blocks.filter(b=>!b.ignored);
 const ordItems=activeBlocks.length?activeBlocks:p.lines;
 const cen=ordItems.map(it=>[(it.box[0]+it.box[2])/2,(it.box[1]+it.box[3])/2]);
 const ord=cen.length>1?`<path class="ord" d="M${cen.map(pt=>pt.map(v=>v.toFixed(1)).join(',')).join(' L')}"/>`:'';
 const nums=cen.map((pt,k)=>`<circle class="on-n" cx="${pt[0]}" cy="${pt[1]}" r="${7.5*u}"/><text class="n" x="${pt[0]}" y="${pt[1]}" font-size="${8.5*u}">${k+1}</text>`).join('');
 const blockRows=p.blocks.map((b,k)=>{
  const col=BLOCK[b.label]||'oklch(55% .02 263)';
  const snippet=(b.text||'').slice(0,220)||'(image / non-text block)';
  return `<div class="bk-row ${b.ignored?'ignored':''}" data-b="${k}"><span class="k">${k+1}</span><span class="btag" style="border-left:3px solid ${col}">${esc(b.label)}</span><span class="t">${esc(snippet)}</span><span class="meta">${esc(b.source||'')}</span></div>`;
 }).join('');
 const lineRows=p.lines.map((l,k)=>`<div class="ln-row" data-k="${k}"><span class="k">${k+1}</span><span class="t">${esc(l.text)||'<span style="color:var(--ink-4)">(no text)</span>'}</span></div>`).join('');
 const tabs=['Preview','Markdown','Blocks','Lines','HTML','Chunks','JSON'];
 c.innerHTML=`<div class="ch"><h3>Page ${p.page}</h3><span class="chip ${p.source}">${esc(src)}</span><span class="meta">${p.blocks.length} blocks · ${p.lines.length} lines${esc(extra)}</span><span class="sp"></span><div class="tbar">${bar}</div><span class="meta">${t.total_ms} ms</span></div>
 <div class="body">
  <div class="pageview"><svg class="pg" viewBox="0 0 ${p.w} ${p.h}"><image href="${p.img}" width="${p.w}" height="${p.h}"/>${blocks}${lines}${ord}${nums}</svg></div>
  <div class="right">
   <div class="tabs">${tabs.map((n,i)=>`<button data-tab="${n}" class="${i?'':'on'}">${n}</button>`).join('')}<span class="sp"></span><button class="copy-btn" data-copy="1">Copy Markdown</button></div>
   <div class="pane preview" data-pane="Preview">${p.preview_html||'<span class="meta">(Empty page)</span>'}</div>
   <div class="pane src" data-pane="Markdown" hidden>${esc(p.markdown||'')}</div>
   <div class="pane" data-pane="Blocks" hidden>${blockRows||'<span class="meta">No blocks detected.</span>'}</div>
   <div class="pane" data-pane="Lines" hidden>${lineRows||'<span class="meta">No line boxes (full-page VLM blocks).</span>'}</div>
   <div class="pane src" data-pane="HTML" hidden>${esc(p.html||'')}</div>
   <div class="pane src" data-pane="Chunks" hidden>${esc(JSON.stringify(p.chunks||[],null,2))}</div>
   <div class="pane src" data-pane="JSON" hidden>${esc(JSON.stringify(p.json||{},null,2))}</div>
  </div>
 </div>`;
 c.querySelectorAll('.tabs button[data-tab]').forEach(b=>b.onclick=()=>{
  c.querySelectorAll('.tabs button[data-tab]').forEach(x=>x.classList.toggle('on',x===b));
  c.querySelectorAll('.pane').forEach(x=>x.hidden=x.dataset.pane!==b.dataset.tab);
 });
 const copyBtn=c.querySelector('.copy-btn');
 if(copyBtn)copyBtn.onclick=()=>{navigator.clipboard.writeText(p.markdown||'');copyBtn.textContent='Copied!';setTimeout(()=>copyBtn.textContent='Copy Markdown',1200);};
 const hlLine=k=>c.querySelectorAll('[data-k]').forEach(el=>el.classList.toggle('on',el.dataset.k==k));
 const hlBlock=b=>c.querySelectorAll('[data-b]').forEach(el=>el.classList.toggle('on',el.dataset.b==b));
 c.querySelectorAll('.ln-row').forEach(el=>{el.onmouseenter=()=>hlLine(el.dataset.k);el.onmouseleave=()=>hlLine(-1)});
 c.querySelectorAll('rect.ln').forEach(el=>{el.onmouseenter=()=>{hlLine(el.dataset.k);const r=c.querySelector(`.ln-row[data-k="${el.dataset.k}"]`);r&&r.scrollIntoView({block:'nearest'})};el.onmouseleave=()=>hlLine(-1)});
 c.querySelectorAll('.bk-row').forEach(el=>{el.onmouseenter=()=>hlBlock(el.dataset.b);el.onmouseleave=()=>hlBlock(-1)});
 c.querySelectorAll('rect.bk').forEach(el=>{
  el.onmouseenter=()=>{hlBlock(el.dataset.b);const r=c.querySelector(`.bk-row[data-b="${el.dataset.b}"]`);r&&r.scrollIntoView({block:'nearest'})};
  el.onmouseleave=()=>hlBlock(-1);
 });
 $('#out').appendChild(c);
}
</script></body></html>"""


def viewer_cli(port: int = 8000, host: str = "127.0.0.1"):
    engine = ViewerEngine()
    srv = ThreadingHTTPServer((host, port), make_handler(engine))
    print(f"Marker Viewer running on http://{host}:{port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    viewer_cli()
