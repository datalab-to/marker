---
name: marker
description: >-
  Convert PDFs, images, and Office documents (DOCX, PPTX, XLSX, HTML, EPUB) to
  Markdown, JSON, RAG chunks, or HTML using Marker (marker-pdf). Use this skill
  whenever the user or task needs to read, parse, OCR, extract tables/equations/images
  from documents, run batch folder conversions, or launch the Marker viewer/API server.
---

# Marker Document Conversion Skill (`marker-pdf`)

Marker converts PDF, image (`PNG`, `JPG`, `WEBP`, `TIFF`, `GIF`, `BMP`), `DOCX`, `PPTX`, `XLSX`, `HTML`, and `EPUB` files into clean **Markdown**, **JSON**, **RAG Chunks**, or **HTML**.

---

## 1. Choosing the Right Conversion Mode (`--mode`)

Marker provides four conversion modes (`--mode no_ocr|fast|balanced|vllm`). When `--mode` is omitted (or `--mode auto`), Marker selects `balanced` on CUDA GPUs and `fast` on CPU / Apple Silicon MPS:

| Mode | Speed | Hardware | How It Works & When to Use |
| :--- | :--- | :--- | :--- |
| **`no_ocr`** | `~24–33 pg/s` | CPU | Pure embedded PDF text layer + in-process Indra ONNX layout & reading order. Zero OCR calls. Use on known born-digital PDFs when maximum CPU speed is needed. |
| **`fast`** | `~7–33 pg/s` | CPU / MPS / GPU | In-process Indra ONNX layout, reading order, and ONNX line/page OCR. Needs **no external VLM server**. Default on CPU and Apple Silicon. |
| **`balanced`** | `~2.9+ pg/s` | GPU (or CPU w/ llama.cpp) | Smart routing: `pdftext` + Indra ONNX layout on clean Latin prose pages; routes tables, math/equations, CJK/non-Latin scripts, and scanned/garbled pages to the Surya 2.5 VLM. Default on CUDA GPUs. |
| **`vllm`** | VLM speed | GPU | Runs full-page Surya 2.5 VLM OCR on every page, skipping `pdftext` and Indra entirely. Best for degraded scans, dense equations, and complex multi-script layouts. |

---

## 2. CLI Quick Reference

### Convert a Single File (`marker_single`)

```bash
# Print Markdown directly to stdout (best for AI agents & shell pipelines)
marker_single document.pdf --stdout --mode fast

# Convert specific pages (0-indexed) to stdout
marker_single document.pdf --page_range "0,2-5" --stdout

# Save output (.md, _meta.json, and extracted images) to a directory
marker_single document.pdf --output_dir ./output --mode balanced

# Output formats: markdown (default), json, chunks (for RAG), html
marker_single document.pdf --output_format chunks --output_dir ./output

# Force OCR on all pages & inline math, or strip bad existing OCR text
marker_single document.pdf --force_ocr --output_dir ./output
marker_single document.pdf --strip_existing_ocr --output_dir ./output

# Keep running page headers and footers (stripped by default)
marker_single document.pdf --keep_headers_footers --output_dir ./output

# Hybrid LLM mode (merges multi-page tables, formats complex forms & inline math)
# Requires GOOGLE_API_KEY (or --llm_service for OpenAI/Claude/Ollama/OpenRouter/Vertex)
marker_single document.pdf --use_llm --output_dir ./output

# Keep the Surya VLM inference server alive across multiple CLI invocations
marker_single document.pdf --mode balanced --keep_server --output_dir ./output
```

### Convert a Folder or Batch of Files (`marker`)

```bash
# Convert all supported documents in a folder (auto-sizes worker pool)
marker ./input_folder --output_dir ./output

# Recursive folder conversion, skipping already-converted files (resume support)
marker ./input_folder --output_dir ./output --recursive --skip_existing

# Fast CPU-only batch conversion with explicit worker count
marker ./input_folder --output_dir ./output --mode fast --workers 8

# Multi-node / multi-shard batch run (shard 0 of 4)
marker ./input_folder --output_dir ./output --num_chunks 4 --chunk_idx 0
```

### Interactive Web Viewer & REST API Server

```bash
# Launch the interactive web viewer (inspect layout boxes, reading order, lines, Markdown/JSON)
marker_gui --port 8000

# Launch the FastAPI server + interactive viewer (OpenAPI docs at /docs)
marker_server --port 8000
```

---

## 3. Python API Recipes

### Standard Conversion (`PdfConverter`)

```python
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered

config = {
    "output_format": "markdown",  # "markdown" | "json" | "html" | "chunks"
    "mode": "fast",               # "no_ocr" | "fast" | "balanced" | "vllm"
    # "page_range": "0,2-4",
    # "force_ocr": False,
    # "use_llm": False,
}
config_parser = ConfigParser(config)

converter = PdfConverter(
    config=config_parser.generate_config_dict(),
    artifact_dict=create_model_dict(
        load_indra=config["mode"] != "vllm",
        no_server=config["mode"] in ("no_ocr", "fast"),
    ),
    processor_list=config_parser.get_processors(),
    renderer=config_parser.get_renderer(),
    llm_service=config_parser.get_llm_service(),
)

rendered = converter("document.pdf")
text, ext, images = text_from_rendered(rendered)
```

### Extract Only Tables (`TableConverter`)

```python
from marker.converters.table import TableConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

converter = TableConverter(artifact_dict=create_model_dict())
rendered = converter("document.pdf")
markdown_tables, _, _ = text_from_rendered(rendered)
```

### Inspect Document Blocks Programmatically

```python
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.schema import BlockTypes

converter = PdfConverter(artifact_dict=create_model_dict(no_server=True), config={"mode": "fast"})
document = converter.build_document("document.pdf")
tables = document.contained_blocks((BlockTypes.Table,))
equations = document.contained_blocks((BlockTypes.Equation,))
```
