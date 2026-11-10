#!/usr/bin/env python
"""Run marker over the olmOCR-bench PDFs and report throughput.

This is the reproducible harness behind the README benchmark numbers. It does
two jobs at once:

  * writes one markdown file per page in olmOCR-bench's expected layout
    (``<out>/<category>/<pdf-stem>_pg<N>_repeat1.md``) so the outputs can be
    scored directly by olmOCR-bench's own checker (see benchmarks/README.md);
  * times the whole run at a chosen worker concurrency and prints the
    deployment-relevant **throughput** (pages/sec), not just single-stream
    latency.

Marker's balanced/fast modes call a shared surya inference server for the VLM /
layout work; each worker here holds only the small CPU models. Point workers at
a running server with ``--surya-url`` (or the ``SURYA_INFERENCE_URL`` env var)
and scale ``--workers`` to drive real concurrency against it -- a single stream
badly understates a server-based system.

Example (balanced, 48-way concurrency against a local surya server):

    python benchmarks/inference.py \
        --bench-dir /path/to/olmOCR-bench/bench_data \
        --out /tmp/marker_balanced --mode balanced --workers 48 \
        --surya-url http://localhost:8000/v1
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

# olmOCR-bench output normalization (same dir); robust to how the script is run.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from postprocess import postprocess as _postprocess

# Per-worker globals (set once in _init, reused for every page the worker does).
_CONV = None
_OUT = None
_PDFDIR = None
_NORMALIZE = True


def _output_stem(out_dir, rel_pdf, base, page_num):
    d = os.path.join(out_dir, os.path.dirname(rel_pdf))
    return os.path.join(d, f"{base}_pg{page_num}_repeat1")


def _init(
    mode,
    disable_ocr,
    force_ocr,
    surya_url,
    out,
    pdfdir,
    use_llm,
    llm_service,
    normalize,
):
    global _CONV, _OUT, _PDFDIR, _NORMALIZE
    _NORMALIZE = normalize
    if surya_url:
        os.environ["SURYA_INFERENCE_URL"] = surya_url
        # Attach to the already-running server; never autostart a local one.
        os.environ["SURYA_INFERENCE_AUTOSTART"] = "false"

    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    # Serialize model init across workers: many simultaneous CUDA-context inits
    # on a shared GPU can wedge the driver. One worker initializes at a time;
    # inference then runs fully concurrent.
    import fcntl

    lock = open("/tmp/marker_bench_init.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        no_server = mode in ("fast", "indra", "no_ocr") and not surya_url
        load_indra = mode not in ("vllm", "surya", "accurate")
        models = create_model_dict(load_indra=load_indra, no_server=no_server)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()

    config = {
        "mode": mode,
        "output_format": "markdown",
        "html_tables_in_markdown": True,
        "disable_image_extraction": True,
        "disable_tqdm": True,
        # pdftext is not thread-safe (pypdfium2); one instance per process.
        "pdftext_workers": 1,
    }
    if disable_ocr:
        config["disable_ocr"] = True
    if force_ocr:
        config["force_ocr"] = True
    if use_llm:
        config["use_llm"] = True

    kwargs = {"artifact_dict": models, "config": config}
    if use_llm and llm_service:
        kwargs["llm_service"] = llm_service
    # Build the converter ONCE per worker; olmOCR-bench PDFs are single-page.
    _CONV = PdfConverter(**kwargs)
    _OUT, _PDFDIR = out, pdfdir


def _work(pdf):
    import pypdfium2 as pdfium

    from marker.output import text_from_rendered

    rel = os.path.relpath(pdf, _PDFDIR)
    base = os.path.splitext(os.path.basename(pdf))[0]
    doc = pdfium.PdfDocument(pdf)
    n_pages = len(doc)
    doc.close()

    records = []
    for page in range(1, n_pages + 1):
        stem = _output_stem(_OUT, rel, base, page)
        md_path = stem + ".md"
        if os.path.exists(md_path) and os.path.getsize(md_path) > 10:
            continue
        os.makedirs(os.path.dirname(stem), exist_ok=True)
        start = time.monotonic()
        err = ""
        try:
            if n_pages > 1:
                _CONV.config["page_range"] = [page - 1]
            else:
                _CONV.config.pop("page_range", None)
            md, _, _ = text_from_rendered(_CONV(pdf))
            if _NORMALIZE:
                md = _postprocess(md)
        except Exception as e:  # keep going; record the failure per page
            md, err = "", f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            _CONV.config.pop("page_range", None)
        elapsed_ms = (time.monotonic() - start) * 1000
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md or "")
        records.append(
            {
                "pdf": rel,
                "page": page,
                "latency_ms": round(elapsed_ms, 1),
                "chars": len(md or ""),
                "error": err,
            }
        )
    return records


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--bench-dir",
        required=True,
        help="olmOCR-bench bench_data dir (contains the PDFs subdir)",
    )
    ap.add_argument("--out", required=True, help="output dir for per-page markdown")
    ap.add_argument(
        "--mode",
        choices=["fast", "balanced", "indra", "no_ocr", "vllm", "surya", "accurate"],
        default="balanced",
    )
    ap.add_argument(
        "--disable-ocr",
        action="store_true",
        help="fast mode with no VLM at all (pure CPU text layer)",
    )
    ap.add_argument(
        "--force-ocr",
        action="store_true",
        help="full-page VLM OCR every page (balanced = surya-ocr-2 config)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="concurrent worker processes (drive server concurrency)",
    )
    ap.add_argument(
        "--surya-url",
        default=os.environ.get("SURYA_INFERENCE_URL"),
        help="URL of the running surya inference server",
    )
    ap.add_argument(
        "--source-subdir", default="pdfs", help="subdir of --bench-dir holding the PDFs"
    )
    ap.add_argument("--limit", type=int, default=0, help="cap number of PDFs (debug)")
    ap.add_argument(
        "--use-llm",
        action="store_true",
        help="enable LLM refinement processors (accurate mode)",
    )
    ap.add_argument(
        "--llm-service",
        default="marker.services.openrouter.OpenRouterService",
        help="marker LLM service import path (with --use-llm)",
    )
    ap.add_argument(
        "--raw",
        action="store_true",
        help="write raw marker markdown without olmOCR-bench "
        "output normalization (see postprocess.py)",
    )
    args = ap.parse_args()

    pdf_dir = os.path.join(args.bench_dir, args.source_subdir)
    pdfs = sorted(glob.glob(os.path.join(pdf_dir, "**", "*.pdf"), recursive=True))
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        raise SystemExit(f"no PDFs under {pdf_dir}")
    os.makedirs(args.out, exist_ok=True)

    label = f"{args.mode}{' no-ocr' if args.disable_ocr else ''}"
    print(f"marker {label}: {len(pdfs)} pdfs, workers={args.workers}", flush=True)

    init_args = (
        args.mode,
        args.disable_ocr,
        args.force_ocr,
        args.surya_url,
        args.out,
        pdf_dir,
        args.use_llm,
        args.llm_service,
        not args.raw,
    )
    latf = open(os.path.join(args.out, "latency.jsonl"), "w")
    n_pages = 0
    t0 = time.monotonic()
    if args.workers <= 1:
        _init(*init_args)
        for pdf in pdfs:
            for r in _work(pdf):
                latf.write(json.dumps(r) + "\n")
                n_pages += 1
            latf.flush()
    else:
        # spawn: forking after torch/CUDA init deadlocks.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init,
            initargs=init_args,
            mp_context=ctx,
        ) as ex:
            for recs in ex.map(_work, pdfs, chunksize=1):
                for r in recs:
                    latf.write(json.dumps(r) + "\n")
                    n_pages += 1
                latf.flush()
    wall = time.monotonic() - t0
    latf.close()

    pgs = n_pages / wall if wall else 0
    print(
        f"DONE marker {label}: {n_pages} pages in {wall:.1f}s | "
        f"throughput {pgs:.2f} pages/s "
        f"({wall / max(n_pages, 1) * 1000:.0f} ms/page effective, "
        f"workers={args.workers})",
        flush=True,
    )


if __name__ == "__main__":
    main()
