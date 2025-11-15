# Marker PDF - Docker Deployment Guide

This guide explains how to deploy Marker PDF using Docker with both GPU and CPU support.

## Table of Contents

- [Quick Start](#quick-start)
- [Image Variants](#image-variants)
- [Building Images](#building-images)
- [Running Containers](#running-containers)
- [Configuration](#configuration)
- [OpenWebUI Integration](#openwebui-integration)
- [Volume Mounts](#volume-mounts)
- [Troubleshooting](#troubleshooting)

## Quick Start

### GPU Version (Recommended)

```bash
# Build the GPU image
docker build -f Dockerfile.gpu -t marker-pdf:gpu .

# Run with docker-compose
docker compose up marker-gpu -d

# Or run directly
docker run -d \
  --name marker-pdf-gpu \
  --gpus all \
  -p 8001:8001 \
  -v marker-models:/root/.cache/huggingface \
  -v marker-torch:/root/.cache/torch \
  marker-pdf:gpu
```

### CPU Version

```bash
# Build the CPU image
docker build -f Dockerfile.cpu -t marker-pdf:cpu .

# Run with docker-compose
docker compose --profile cpu up marker-cpu -d

# Or run directly
docker run -d \
  --name marker-pdf-cpu \
  -p 8001:8001 \
  -v marker-models:/root/.cache/huggingface \
  -v marker-torch:/root/.cache/torch \
  marker-pdf:cpu
```

## Image Variants

### `marker-pdf:gpu` (Big Image)

- **Base**: NVIDIA CUDA 12.1.0 with cuDNN 8
- **Size**: ~8-10 GB
- **GPU Support**: NVIDIA GPUs with CUDA
- **Performance**: ~0.18s per page on GPU
- **VRAM**: 5GB peak per worker, 3.5GB average
- **Use Case**: Production deployments, high-volume processing

### `marker-pdf:cpu` (Small Image)

- **Base**: Python 3.11 Slim
- **Size**: ~3-4 GB
- **GPU Support**: None (CPU only)
- **Performance**: Slower than GPU (varies by CPU)
- **Memory**: Lower memory footprint
- **Use Case**: Development, testing, low-volume processing

## Building Images

### Build GPU Image

```bash
docker build -f Dockerfile.gpu -t marker-pdf:gpu .
```

### Build CPU Image

```bash
docker build -f Dockerfile.cpu -t marker-pdf:cpu .
```

### Build with Docker Compose

```bash
# Build both images
docker compose build

# Build specific service
docker compose build marker-gpu
docker compose build marker-cpu
```

## Running Containers

### Using Docker Compose (Recommended)

**GPU Version:**
```bash
# Start GPU version
docker compose up marker-gpu -d

# View logs
docker compose logs -f marker-gpu

# Stop
docker compose down
```

**CPU Version:**
```bash
# Start CPU version
docker compose --profile cpu up marker-cpu -d

# View logs
docker compose --profile cpu logs -f marker-cpu

# Stop
docker compose --profile cpu down
```

### Using Docker Run

**GPU Version:**
```bash
docker run -d \
  --name marker-pdf-gpu \
  --gpus all \
  -p 8001:8001 \
  -v marker-models:/root/.cache/huggingface \
  -v marker-torch:/root/.cache/torch \
  -v ./uploads:/app/uploads \
  -v ./output:/app/output \
  -e USE_LLM=true \
  -e OPENAI_API_KEY=your-api-key \
  marker-pdf:gpu
```

**CPU Version:**
```bash
docker run -d \
  --name marker-pdf-cpu \
  -p 8001:8001 \
  -v marker-models:/root/.cache/huggingface \
  -v marker-torch:/root/.cache/torch \
  -v ./uploads:/app/uploads \
  -v ./output:/app/output \
  marker-pdf:cpu
```

## Configuration

### Environment Variables

The following environment variables can be configured:

#### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8001` | Server port |
| `HOST` | `0.0.0.0` | Server host |
| `TORCH_DEVICE` | `cuda`/`cpu` | Device to use (auto-set by image) |

#### LLM Service Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_LLM` | `false` | Enable LLM mode |
| `LLM_SERVICE` | - | Full path to LLM service class |

#### OpenAI Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model name |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API base URL |
| `OPENAI_IMAGE_FORMAT` | `webp` | Image format (webp/png) |

#### Gemini Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | - | Google Gemini API key |
| `GOOGLE_API_KEY` | - | Alias for GEMINI_API_KEY |
| `GEMINI_MODEL_NAME` | `gemini-2.0-flash` | Model name |
| `THINKING_BUDGET` | - | Thinking token budget |

#### Claude Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_API_KEY` | - | Anthropic API key |
| `ANTHROPIC_API_KEY` | - | Alias for CLAUDE_API_KEY |
| `CLAUDE_MODEL_NAME` | `claude-3-7-sonnet-20250219` | Model name |
| `MAX_CLAUDE_TOKENS` | `8192` | Max output tokens |

#### Azure OpenAI Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AZURE_ENDPOINT` | - | Azure endpoint URL |
| `AZURE_API_KEY` | - | Azure API key |
| `AZURE_API_VERSION` | - | API version |
| `DEPLOYMENT_NAME` | - | Deployment name |

#### Ollama Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2-vision` | Model name |

#### Processing Options (OpenWebUI Compatible)

| Variable | Default | Description |
|----------|---------|-------------|
| `SKIP_CACHE` | `false` | Skip using cached results |
| `FORCE_OCR` | `false` | Force OCR on all pages |
| `PAGINATE` | `false` | Add page numbers to output |
| `DISABLE_IMAGE_EXTRACTION` | `false` | Don't extract images |
| `OUTPUT_FORMAT` | `markdown` | Output format (markdown/json/html) |

#### Advanced Options

| Variable | Default | Description |
|----------|---------|-------------|
| `LANGS` | `en` | Languages to support |
| `BATCH_MULTIPLIER` | `1` | Batch size multiplier |
| `MAX_PAGES` | - | Max pages to process |
| `TIMEOUT` | `30` | LLM request timeout (seconds) |
| `MAX_RETRIES` | `2` | Max retry attempts |
| `RETRY_WAIT_TIME` | `3` | Wait between retries (seconds) |
| `MAX_OUTPUT_TOKENS` | - | Max tokens to generate |

### Example Configurations

#### OpenAI with Custom Settings

```yaml
environment:
  - USE_LLM=true
  - LLM_SERVICE=marker.services.openai.OpenAIService
  - OPENAI_API_KEY=sk-...
  - OPENAI_MODEL=gpt-4o
  - FORCE_OCR=true
  - OUTPUT_FORMAT=json
  - PAGINATE=true
```

#### Ollama with Local Model

```yaml
environment:
  - USE_LLM=true
  - LLM_SERVICE=marker.services.ollama.OllamaService
  - OLLAMA_BASE_URL=http://host.docker.internal:11434
  - OLLAMA_MODEL=llama3.2-vision
  - OUTPUT_FORMAT=markdown
```

#### Claude with Azure

```yaml
environment:
  - USE_LLM=true
  - LLM_SERVICE=marker.services.claude.ClaudeService
  - CLAUDE_API_KEY=sk-ant-...
  - CLAUDE_MODEL_NAME=claude-3-7-sonnet-20250219
  - MAX_CLAUDE_TOKENS=16384
```

## OpenWebUI Integration

Marker PDF is designed to work seamlessly with OpenWebUI. Here's how to configure it:

### 1. Add as a Tool in OpenWebUI

In OpenWebUI, add Marker PDF as an external tool with the following configuration:

**Endpoint URL:**
```
http://marker-pdf:8001/marker/upload
```

**Method:** POST

**Content Type:** multipart/form-data

### 2. Configure OpenWebUI Settings

Use the following JSON config in OpenWebUI's "Additional Config" section:

```json
{
  "use_llm": true,
  "skip_cache": false,
  "force_ocr": false,
  "paginate": false,
  "strip_existing_ocr": false,
  "disable_image_extraction": false,
  "format_lines": true,
  "output_format": "markdown"
}
```

### 3. Available Toggles in OpenWebUI

- **Use LLM**: Enable LLM for better accuracy
- **Skip Cache**: Don't use cached results
- **Force OCR**: Force OCR on all pages
- **Paginate**: Add page numbers to output
- **Strip Existing OCR**: Remove existing OCR text
- **Disable Image Extraction**: Don't extract images
- **Format Lines**: Format output lines
- **Output Format**: Choose markdown/json/html

### 4. Docker Compose Setup for OpenWebUI

```yaml
services:
  marker-pdf:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    networks:
      - openwebui
    environment:
      - USE_LLM=true
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - marker-models:/root/.cache/huggingface
      - marker-torch:/root/.cache/torch

  openwebui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    networks:
      - openwebui
    environment:
      - MARKER_API_URL=http://marker-pdf:8001

networks:
  openwebui:
    driver: bridge
```

## Volume Mounts

### Required Volumes

**Model Cache** (Highly Recommended):
```bash
-v marker-models:/root/.cache/huggingface
-v marker-torch:/root/.cache/torch
```
These volumes persist downloaded models across container restarts. Without them, models (~2-3GB) will be re-downloaded each time.

### Optional Volumes

**Uploads Directory:**
```bash
-v ./uploads:/app/uploads
```
Persist uploaded files.

**Output Directory:**
```bash
-v ./output:/app/output
```
Access generated output files from the host.

**Custom Configuration:**
```bash
-v ./config:/app/config
```
Mount custom configuration files.

## Accessing the Server

Once the container is running:

- **API Endpoint**: http://localhost:8001
- **API Documentation**: http://localhost:8001/docs
- **Health Check**: http://localhost:8001/

### Example API Request

```bash
# Upload and convert a PDF
curl -X POST "http://localhost:8001/marker/upload" \
  -F "file=@document.pdf" \
  -F "output_format=markdown"

# Convert from filepath (if mounted)
curl -X POST "http://localhost:8001/marker" \
  -H "Content-Type: application/json" \
  -d '{
    "filepath": "/app/uploads/document.pdf",
    "output_format": "markdown",
    "use_llm": true
  }'
```

## Troubleshooting

### Models Keep Re-downloading

**Problem**: Models are downloaded every time the container starts.

**Solution**: Ensure you're using volume mounts for model cache:
```bash
-v marker-models:/root/.cache/huggingface
-v marker-torch:/root/.cache/torch
```

### GPU Not Detected

**Problem**: Container can't access GPU.

**Solution**:
1. Install NVIDIA Container Toolkit
2. Use `--gpus all` flag or docker-compose GPU configuration
3. Verify with: `docker run --rm --gpus all marker-pdf:gpu nvidia-smi`

### Out of Memory

**Problem**: Container runs out of VRAM/RAM.

**Solution**:
- GPU: Ensure at least 5GB VRAM available
- CPU: Reduce batch size with `BATCH_MULTIPLIER=0.5`
- Limit max pages: `MAX_PAGES=100`

### Port Already in Use

**Problem**: Port 8001 is already in use.

**Solution**: Change the port mapping:
```bash
-p 8002:8001  # Use port 8002 on host
```
Or set `PORT` environment variable:
```bash
-e PORT=8002 -p 8002:8002
```

### Slow Performance on CPU

**Problem**: Processing is very slow on CPU.

**Solution**:
- Use GPU version if available
- Reduce batch size: `BATCH_MULTIPLIER=0.5`
- Disable LLM: `USE_LLM=false`
- Skip image extraction: `DISABLE_IMAGE_EXTRACTION=true`

### Connection Refused in OpenWebUI

**Problem**: OpenWebUI can't connect to Marker PDF.

**Solution**:
1. Ensure both containers are on the same network
2. Use container name as hostname: `http://marker-pdf:8001`
3. Check firewall rules
4. Verify with: `docker compose logs marker-gpu`

## Advanced Usage

### Multi-GPU Setup

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 2  # Use 2 GPUs
          capabilities: [gpu]
```

### Custom Entrypoint

```bash
docker run -it marker-pdf:gpu /bin/bash
# Run custom commands inside container
```

### Building from Specific Commit

```bash
docker build \
  --build-arg GIT_COMMIT=$(git rev-parse HEAD) \
  -f Dockerfile.gpu \
  -t marker-pdf:gpu-$(git rev-parse --short HEAD) \
  .
```

## Performance Benchmarks

### GPU Version (NVIDIA A100)
- Speed: ~0.18s per page
- Throughput: ~5.5 pages/second
- VRAM: 3.5GB average, 5GB peak

### CPU Version (16-core)
- Speed: ~2-5s per page (varies by CPU)
- Throughput: ~0.2-0.5 pages/second
- RAM: 2-4GB

## Support

For issues and questions:
- GitHub Issues: https://github.com/datalab-to/marker
- Documentation: https://github.com/datalab-to/marker/blob/main/README.md

## License

This Docker setup follows the same license as the Marker project.
