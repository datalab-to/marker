# Benchmarks

Reproducible harness for the quality ([OmniParseBench](https://github.com/datalab-to/omni_parse_bench) and [olmOCR-bench](https://github.com/allenai/olmocr)) and throughput numbers in the top-level README.

- **OmniParseBench** ([dataset](https://huggingface.co/datasets/datalab-to/omni_parse_bench) · [repo](https://github.com/datalab-to/omni_parse_bench)): 16,288 pass/fail unit tests across 2,937 pages from 2,343 documents in 92 languages. Reports a **Headline** score (mean of **Text**, **Tables**, and **Layout** family pass rates) plus per-suite scores across 22 suites.
- **olmOCR-bench** ([repo](https://github.com/allenai/olmocr)): 1,403 single-page PDFs and ~8,400 pass/fail unit tests across 8 categories (**Overall** = macro-average over all 8 categories; **Digital-only** = macro-average over the 6 non-scanned categories).
- **Throughput** is sustained pages/sec at real worker concurrency on one host, reported directly by `inference.py`. This is the deployment-relevant number; single-stream latency badly understates a server-based system.

## Layout

```
benchmarks/
  inference.py            # run marker over olmOCR-bench: scoreable output + throughput
  postprocess.py          # olmOCR-bench output normalization (applied by inference.py)
  summarize.py            # per-category pass rates -> Overall (macro) + Digital-only
  plot_benchmarks.py      # regenerate all benchmark charts in data/images/
  competitors/
    liteparse_infer.py    # liteparse (own venv): --no-ocr toggle, multiprocessing
    docling_infer.py      # docling (own venv): native convert_all + concurrency
    mineru_infer.sh       # MinerU (own venv): native pipeline batch
```

## 0. Running OmniParseBench

Install the official [OmniParseBench](https://github.com/datalab-to/omni_parse_bench) package (which downloads the 2,937-page dataset from Hugging Face `datalab-to/omni_parse_bench` and runs the unit-test scorer):

```bash
pip install "omni-parse-bench[benchmark]"
```

Convert the benchmark pages with Marker in your chosen mode (`--mode no_ocr|fast|balanced|vllm`) and score the generated Markdown/JSON blocks with `omni_parse_bench` (passing block polygons/labels for the 2,403 layout tests). To regenerate the benchmark charts in `data/images/`:

```bash
python benchmarks/plot_benchmarks.py
```

## 1. Get olmOCR-bench

```bash
git clone https://github.com/allenai/olmocr
cd olmocr && pip install -e .
# download the bench PDFs + test jsonls (see the olmocr repo for the current
# command; it populates olmocr/bench/bench_data/)
python -m olmocr.bench.download
```

`bench_data/` then contains `pdfs/<category>/*.pdf` and the per-category
`*.jsonl` test files. Pass that `bench_data` path as `--bench-dir` below.

## 2. Start a surya inference server (marker balanced / vllm modes)

Marker's `balanced` and `vllm` modes offload VLM OCR work to a shared Surya
server; the benchmark workers hold only small CPU/ONNX models and talk to it over
HTTP. Start one (see the surya docs), note its URL, and pass it with
`--surya-url` (or export `SURYA_INFERENCE_URL`). Give the server the GPU; run
the marker workers as thin CPU processes. `fast` and `no_ocr` modes run
in-process Indra ONNX layout/OCR on CPU and need no external server.

## 3. Run marker + measure throughput

```bash
BENCH=/path/to/olmocr/olmocr/bench/bench_data
SURYA=http://localhost:8000/v1

# balanced  (smart routing: pdftext + Indra ONNX on clean prose, Surya VLM on tables/math/CJK/scans)
python benchmarks/inference.py --bench-dir $BENCH --out ./out_marker_balanced \
    --mode balanced --workers 48 --surya-url $SURYA

# vllm  (full-page Surya VLM OCR on every page)
python benchmarks/inference.py --bench-dir $BENCH --out ./out_marker_vllm \
    --mode vllm --workers 48 --surya-url $SURYA

# fast  (in-process Indra ONNX layout + ONNX OCR on CPU, no VLM server needed)
python benchmarks/inference.py --bench-dir $BENCH --out ./out_marker_fast \
    --mode fast --workers 32

# no_ocr  (pure CPU text layer + Indra ONNX layout, zero OCR calls)
python benchmarks/inference.py --bench-dir $BENCH --out ./out_marker_noocr \
    --mode no_ocr --workers 32

# balanced + LLM  (balanced + LLM refinement; needs an LLM service configured)
python benchmarks/inference.py --bench-dir $BENCH --out ./out_marker_llm \
    --mode balanced --use-llm --workers 32 --surya-url $SURYA
```

Each run prints `throughput <N> pages/s` and writes per-page markdown plus a
`latency.jsonl` (per-request latency). Tune `--workers` to your server's
capacity; more workers past the server's saturation point only grow the queue.

By default the written markdown is normalized for the olmOCR-bench checker
(`postprocess.py`: HTML `<sub>`/`<sup>` → Unicode, strip stray markdown escapes,
unescape math, drop image alt-text — the same normalization Chandra's official
run applies). This is output-only and lifts the score slightly (balanced ~+0.3,
mostly arxiv_math + tables). Pass `--raw` to write unmodified marker markdown.

## 4. Score with olmOCR-bench

Point the checker at the output dir (it reads the `*.jsonl` tests from
`bench_data`):

```bash
python -m olmocr.bench.benchmark --dir /tmp/marker_balanced
```

It prints a pass rate per category. Feed those into `summarize.py` to get the
Overall (macro) and Digital-only numbers we report:

```bash
python benchmarks/summarize.py \
    arxiv_math=82.0 old_scans_math=63.8 tables=72.8 old_scans=43.3 \
    long_tiny_text=71.3 multi_column=77.0 headers_footers=95.8 baseline=99.7
# -> Overall 75.7   Digital-only 83.1
```

(or `--json scores.json` with a `{category: pass_rate}` map.)

## 5. Competitors

Each competitor installs into its **own** venv (their dependencies conflict with
marker's and with each other). Run its script with that venv's interpreter; each
writes scoreable markdown to `--out` and prints throughput, then score the
output dir exactly as in step 4.

```bash
# liteparse (CPU). --no-ocr = pure text layer, comparable to marker fast-no-OCR.
liteparse-venv/bin/python benchmarks/competitors/liteparse_infer.py \
    --bench-dir $BENCH --out /tmp/liteparse --no-ocr

# docling. Default install is CPU-only torch; install CUDA torch for --device cuda.
docling-venv/bin/python benchmarks/competitors/docling_infer.py \
    --bench-dir $BENCH --out /tmp/docling --device auto

# MinerU (native pipeline batch on GPU).
BENCH_DIR=$BENCH OUT=/tmp/mineru MINERU_BIN=mineru-venv/bin/mineru \
    bash benchmarks/competitors/mineru_infer.sh
```

## Fairness notes

- **One GPU, one system at a time.** Throughput is "best sustained pages/sec on
  a single GPU." When measuring a GPU competitor, give it the whole GPU (tear
  down the marker/surya server first) so the comparison is like-for-like.
- **Native parallelism per tool.** There is no single parallelization scheme
  that fits every system: marker uses thin CPU workers against a shared server;
  liteparse uses multiprocessing (no batch API); docling uses its in-process
  `convert_all` concurrency; MinerU uses its pipeline directory batch. Each is
  run at the concurrency it was designed for.
- **Latency vs throughput.** `latency.jsonl` records single-stream-style
  per-request latency; the headline `pages/s` is concurrent throughput. They are
  different metrics — don't compare one system's latency to another's throughput.
```
