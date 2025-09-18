# Marker System Architecture

## Overview

Marker follows a modular pipeline architecture that processes documents through a series of specialized components. The system is designed to be extensible, allowing users to customize processing steps and add their own logic.

## Core Components

### 1. Providers
Located at `marker/providers/`
- Responsible for providing information from source files (PDF, images, Office documents, etc.)
- Different providers exist for different file types
- Handle the initial extraction of raw data from documents

### 2. Builders
Located at `marker/builders/`
- Generate initial document blocks and fill in text using information from providers
- Create the foundational structure of the document representation
- Handle layout detection and text extraction

### 3. Processors
Located at `marker/processors/`
- Process specific blocks to enhance formatting and structure
- Examples include table formatters, equation handlers, and text cleaners
- Applied in a specific order to transform raw blocks into well-formatted content

### 4. Renderers
Located at `marker/renderers/`
- Convert processed blocks into final output formats (markdown, JSON, HTML, chunks)
- Each renderer implements a specific output format
- Handle the final transformation to user-facing formats

### 5. Schema
Located at `marker/schema/`
- Define the data structures for all block types
- Provide a consistent representation of document elements throughout the pipeline
- Enable structured processing and manipulation of document components

### 6. Converters
Located at `marker/converters/`
- Orchestrate the end-to-end pipeline execution
- Coordinate between providers, builders, processors, and renderers
- Handle different conversion modes (PDF, table-only, OCR-only, etc.)

## Data Flow

1. **Input**: Document file is provided to a Converter
2. **Provider**: Extracts raw data from the document
3. **Builder**: Creates initial block structure with text content
4. **Processors**: Sequentially enhance and format blocks
5. **Renderer**: Transforms processed blocks into final output format
6. **Output**: Returns formatted content and metadata

## Key Technical Decisions

1. **Modular Pipeline**: Each stage of processing is isolated, allowing for easy customization and extension
2. **Block-Based Representation**: Documents are represented as hierarchical blocks, enabling granular processing
3. **Configuration-Driven**: Processing behavior can be customized through configuration files
4. **Multiple Output Formats**: Single processing pipeline can produce multiple output formats
5. **Optional LLM Integration**: LLM processing can be added to enhance accuracy where needed

## Design Patterns

1. **Strategy Pattern**: Different providers, processors, and renderers can be swapped based on configuration
2. **Factory Pattern**: Converters are instantiated based on document type and configuration
3. **Pipeline Pattern**: Sequential processing through well-defined stages
4. **Observer Pattern**: Processors can react to specific block types

## Component Relationships

```
[Converters] --> [Providers] --> [Builders] --> [Processors] --> [Renderers]
     |              |              |
     |              |             |              |              |
     +----------> [Schema] <-----+--------------+--------------+
```

## Critical Implementation Paths

1. **PDF Processing Path**: 
   - PdfConverter → PdfProvider → DocumentBuilder → [Processor Chain] → [Renderer]
   - Handles the majority of document types and use cases

2. **Table Extraction Path**:
   - TableConverter → PdfProvider → DocumentBuilder → TableProcessor → TableRenderer
   - Specialized path for table-focused extraction

3. **OCR Path**:
   - OCRConverter → ImageProvider → OCRBuilder → [Processor Chain] → [Renderer]
   - Handles image-based documents requiring OCR

4. **Structured Extraction Path**:
   - ExtractionConverter → [Provider] → [Builder] → LLMProcessors → JSONRenderer
   - Uses LLMs for structured data extraction based on schemas

## Extensibility Points

1. **Custom Processors**: Users can implement and register their own processors
2. **Custom Renderers**: Additional output formats can be implemented as new renderers
3. **Custom Providers**: Support for new input formats through custom providers
4. **Configuration**: Behavior can be modified through configuration files