---
name: marker
description: Convert documents (PDF, image, PPTX, DOCX, XLSX, HTML, EPUB) to Markdown, JSON, HTML, or RAG-ready chunks using the locally installed marker-pdf tool. Use whenever the user wants to convert, parse, OCR, or extract the contents (text, tables, forms, equations, images) of a document into Markdown or structured output — especially when they point at a .pdf/.docx/.pptx/.xlsx/.epub/.html file or a folder of them and want clean Markdown/JSON out. Also use for table-only extraction, OCR-only passes, and optional LLM-boosted (--use_llm) accuracy.
---

# Marker — document → Markdown/JSON/HTML/chunks

Marker (`marker-pdf`) converts PDF, image, PPTX, DOCX, XLSX, HTML, and EPUB files
to Markdown, JSON, HTML, or chunks — with tables, forms, equations, inline math,
links, code blocks, and extracted images. This repo is installed locally in
editable mode into `.venv`, exposing the `marker_single`, `marker`,
`marker_server`, and `marker_gui` commands.

## Environment

Always run marker inside this repo's virtual environment:

```bash
source .venv/bin/activate      # then use marker_single / marker directly
# or, without activating:
.venv/bin/marker_single ...
```

The first VLM-using run auto-spawns a local surya inference server. That needs
`vllm` (NVIDIA GPU) or the `llama-server` binary from llama.cpp (CPU / Apple
Silicon). If no inference backend is available, or you just want fast text-layer
extraction, pass `--disable_ocr` — it never calls the VLM (equations and scanned
pages are skipped). Point at an already-running server with
`SURYA_INFERENCE_URL=http://host:port/v1`.

## Convert a single file

```bash
marker_single /path/to/file.pdf --output_dir ./output
```

The output goes to a subfolder named after the input file, containing the
Markdown (or chosen format), a `_meta.json`, and any extracted images.

Common options:

- `--output_format [markdown|json|html|chunks]` — default `markdown`. Use
  `chunks` for RAG-ready flattened blocks with full per-block HTML; `json` for
  the tree structure (block types listed in `marker/schema/__init__.py`).
- `--output_dir PATH` — where results are saved.
- `--mode [balanced|fast]` — `balanced` (VLM layout, highest quality, best on
  GPU) vs `fast` (lightweight CPU layout, minimal VLM use). Defaults by device:
  `balanced` on GPU, `fast` on CPU/MPS.
- `--disable_ocr` — pure text-layer extraction, never calls the VLM. Best when
  there is no inference backend or the input is clean born-digital PDF.
- `--page_range "0,5-10,20"` — restrict pages.
- `--force_ocr` — re-OCR every page (fixes garbled embedded text).
- `--use_llm` — boost accuracy with an LLM (merges cross-page tables, handles
  inline math, formats tables, extracts form values). Requires an LLM service
  configured (see below).
- `--paginate_output` — separate pages with a numbered horizontal rule.
- `--disable_image_extraction` — don't extract images (with `--use_llm`, images
  are replaced by descriptions).
- `--keep_pageheader_in_output` / `--keep_pagefooter_in_output` — keep running
  headers/footers (stripped by default).
- `--debug` — save per-page layout/text debug images + bbox JSON.
- `marker_single --help` — full flag list. `marker_single config --help` lists
  every builder/processor/converter option for a `--config_json` file.

## Convert a folder (batch)

```bash
marker /path/to/input/folder --output_dir ./output
```

Supports all `marker_single` options, plus `--workers N`, `--skip_existing`
(resume a run), `--max_files N`, and `--disable_multiprocessing`. All workers
share one inference server that the parent spawns.

## Table-only extraction

```bash
marker_single FILE --converter_cls marker.converters.table.TableConverter --output_format json
```

Add `--use_llm` for hard tables (multi-page merges, complex spans). Tables are
emitted as HTML `<table>` blocks; `--output_format json` gives table blocks with
page bounding boxes.

## OCR only

```bash
marker_single FILE --converter_cls marker.converters.ocr.OCRConverter
```

Add `--keep_chars` to keep per-character boxes (digital PDFs only).

## LLM-boosted mode

Pass `--use_llm` plus a service. Default is Gemini
(`--gemini_api_key`, or `GOOGLE_API_KEY`). Others via `--llm_service=...`:

- Claude: `marker.services.claude.ClaudeService` (`--claude_api_key`, `--claude_model_name`)
- OpenAI / compatible: `marker.services.openai.OpenAIService` (`--openai_api_key`, `--openai_model`, `--openai_base_url`)
- Vertex: `marker.services.vertex.GoogleVertexService` (`--vertex_project_id`)
- Ollama (local): `marker.services.ollama.OllamaService` (`--ollama_base_url`, `--ollama_model`)
- Azure OpenAI: `marker.services.azure_openai.AzureOpenAIService`
- OpenRouter: `marker.services.openrouter.OpenRouterService`

Highest-quality inline math: `--use_llm --redo_inline_math`.
Never hardcode API keys into commands committed to the repo — pass them via the
matching environment variable and confirm with the user which service/key to use.

## Use from Python

```python
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

converter = PdfConverter(artifact_dict=create_model_dict())
rendered = converter("FILEPATH")
text, _, images = text_from_rendered(rendered)   # markdown, metadata, images
```

For custom config, build a `marker.config.parser.ConfigParser(config_dict)` and
pass `config=...`, `processor_list=...`, `renderer=...`, `llm_service=...` to the
converter (see repo README "Custom configuration").

## API server

```bash
marker_server --port 8001    # docs at localhost:8001/docs
```

POST to `/marker` with `{"filepath": "...", ...}` or `/marker/upload` (multipart).
The server exposes only `page_range`, `mode`, `force_ocr`, `paginate_output`,
`output_format` — use the CLI/Python for `--use_llm` or `--disable_ocr`.

## Workflow guidance

1. Confirm the input path(s) and desired `--output_format`.
2. Pick a mode: on CPU / no inference backend, start with
   `--mode fast --disable_ocr` for born-digital PDFs; use `--mode balanced` (or
   `--force_ocr`) for scans, math, or garbled text.
3. Run the command, then read the produced Markdown/JSON from `--output_dir` and
   report where it landed. If text is garbled, retry with `--force_ocr`; for
   difficult tables/forms, add `--use_llm`.
