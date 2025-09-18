# Marker Project Brief

## Project Overview

Marker is a high-performance document conversion tool that converts various document formats (PDF, images, PPTX, DOCX, XLSX, HTML, EPUB) into structured formats including markdown, JSON, HTML, and chunks. The tool is designed for accuracy and speed, utilizing deep learning models where necessary to extract and format content properly.

## Core Requirements

1. Convert documents accurately to multiple output formats
2. Handle complex document elements like tables, equations, forms, and images
3. Support multiple languages through OCR capabilities
4. Provide optional LLM-enhanced accuracy for complex documents
5. Work efficiently on both GPU and CPU platforms
6. Be extensible with custom formatting and processing logic
7. Support batch processing for high throughput

## Project Goals

1. Achieve state-of-the-art accuracy in document conversion
2. Maintain high performance with minimal resource usage
3. Provide a simple interface for integration into other systems
4. Enable customization for specific document processing needs
5. Support both local and cloud-based deployment options

## Key Features

- Multi-format input support (PDF, images, Office documents, etc.)
- Multi-format output (markdown, JSON, HTML, chunks)
- Advanced table formatting and extraction
- Equation detection and LaTeX formatting
- Image extraction and embedding
- Artifact removal (headers, footers, etc.)
- Optional LLM-enhanced accuracy
- GPU and CPU support
- Extensible processing pipeline
- Batch processing capabilities
- Structured extraction with JSON schemas (beta)