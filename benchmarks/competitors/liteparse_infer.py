#!/usr/bin/env python
"""liteparse throughput + scoreable output (run in liteparse's own venv).

liteparse exposes no batch API (only ``parse(single_file)``), so the native way
to parallelize a stateless CPU parser is multiprocessing: one LiteParse per
worker (``num_workers=1`` so the process pool is the only source of
parallelism), files sharded across the pool.

``--no-ocr`` sets ``ocr_enabled=False`` -> pure text-layer extraction, directly
comparable to marker fast-no-OCR (no OCR / no VLM, CPU text layer only). With
OCR on, scanned pages fall back to Tesseract, which dominates wall time.

Output is liteparse's markdown (``output_format="markdown"``, read from
``page.markdown``): headings, tables and reading order, the same kind of output
marker is scored on. ``page.text`` is liteparse's spatial plain-text layout,
which has no table or heading markup.

Writes ``<out>/<category>/<stem>_pg1_repeat1.md`` for olmOCR-bench scoring and
prints sustained pages/sec.
"""

import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor

_LP = None
_OUT = None
_PDFDIR = None


def _init(ocr_enabled, out, pdfdir):
    global _LP, _OUT, _PDFDIR
    from liteparse import LiteParse

    _LP = LiteParse(
        ocr_enabled=ocr_enabled,
        num_workers=1,
        quiet=True,
        output_format="markdown",
    )
    _OUT, _PDFDIR = out, pdfdir


def _one(pdf):
    try:
        result = _LP.parse(pdf)
        pages = getattr(result, "pages", None)
        text = (
            "\n\n".join(getattr(p, "markdown", "") or "" for p in pages)
            if pages
            else getattr(result, "text", "") or ""
        )
        if _OUT:
            rel = os.path.relpath(pdf, _PDFDIR)
            base = os.path.splitext(os.path.basename(pdf))[0]
            d = os.path.join(_OUT, os.path.dirname(rel))
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{base}_pg1_repeat1.md"), "w") as f:
                f.write(text)
        return len(text)
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--out", default="", help="write markdown here for scoring")
    ap.add_argument("--source-subdir", default="pdfs")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument(
        "--no-ocr",
        action="store_true",
        help="disable Tesseract fallback (pure text layer)",
    )
    args = ap.parse_args()

    pdf_dir = os.path.join(args.bench_dir, args.source_subdir)
    pdfs = sorted(glob.glob(os.path.join(pdf_dir, "**", "*.pdf"), recursive=True))
    ocr_enabled = not args.no_ocr
    print(
        f"liteparse: {len(pdfs)} pdfs, workers={args.workers}, "
        f"ocr_enabled={ocr_enabled}",
        flush=True,
    )

    t0 = time.monotonic()
    ok = 0
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init,
        initargs=(ocr_enabled, args.out, pdf_dir),
    ) as ex:
        for n in ex.map(_one, pdfs, chunksize=8):
            if n >= 0:
                ok += 1
    dt = time.monotonic() - t0
    print(
        f"DONE liteparse{' no-ocr' if args.no_ocr else ''}: {ok}/{len(pdfs)} "
        f"in {dt:.1f}s | throughput {ok / dt:.2f} pages/s "
        f"({dt / len(pdfs) * 1000:.1f} ms/page effective)",
        flush=True,
    )


if __name__ == "__main__":
    main()
