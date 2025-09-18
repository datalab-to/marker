# Marker Technology Stack

## Core Technologies

### Programming Language
- **Python 3.10+**: Primary programming language for the entire project

### Deep Learning Framework
- **PyTorch**: Used for all deep learning models and inference

### Document Processing Libraries
- **surya**: Custom OCR and layout detection library
- **texify**: LaTeX equation detection and formatting library

### Web Framework
- **FastAPI**: Used for the API server implementation
- **Streamlit**: Used for the interactive GUI application

### Package Management
- **Poetry**: Dependency management and packaging

### Testing Framework
- **pytest**: Unit and integration testing

## Key Dependencies

### Core Dependencies
- `torch`: PyTorch for deep learning
- `numpy`: Numerical computing
- `opencv-python`: Image processing
- `Pillow`: Image handling
- `pdfplumber`: PDF text extraction
- `pdfminer.six`: PDF parsing
- `beautifulsoup4`: HTML parsing
- `lxml`: XML/HTML processing
- `python-multipart`: File upload handling for FastAPI

### Optional Dependencies
- `llama-cpp-python`: For local LLM inference
- `google-generativeai`: For Gemini API integration
- `openai`: For OpenAI API integration
- `anthropic`: For Claude API integration
- `python-docx`: For DOCX file processing
- `python-pptx`: For PPTX file processing
- `openpyxl`: For XLSX file processing
- `epublib`: For EPUB file processing

## Development Setup

### Prerequisites
- Python 3.10 or higher
- PyTorch (with CUDA support for GPU acceleration)
- Poetry for dependency management

### Installation Steps
1. Clone the repository
2. Install dependencies with `poetry install`
3. For full functionality, install optional dependencies with `pip install marker-pdf[full]`

### Development Workflow
- Use `pytest` for running tests
- Follow the modular pipeline architecture for extensions
- Use the provided converters, providers, builders, processors, and renderers as examples for new components

## Technical Constraints

### Hardware Requirements
- **GPU**: Recommended for optimal performance (CUDA-compatible NVIDIA GPU)
- **CPU**: Minimum 4 cores for batch processing
- **Memory**: Minimum 8GB RAM, 16GB+ recommended
- **Storage**: SSD storage recommended for model files and temporary processing

### Performance Considerations
- VRAM usage: ~3.5GB average per worker, up to 5GB peak
- Supports multi-GPU setups for increased throughput
- Batch processing significantly improves performance

### Model Dependencies
- Custom trained models from surya and texify projects
- Optional LLM integration (Gemini, OpenAI, Claude, Ollama, etc.)

## Tool Usage Patterns

### CLI Tools
- `marker_single`: Convert a single document
- `marker`: Convert multiple documents
- `marker_chunk_convert`: Convert documents across multiple GPUs
- `marker_gui`: Launch the Streamlit GUI
- `marker_server`: Start the FastAPI server

### Configuration
- Environment variables for device selection (`TORCH_DEVICE`)
- Command-line flags for customization
- JSON configuration files for complex setups
- ConfigParser for programmatic configuration

### Extension Points
- Custom processors by implementing the processor interface
- Custom renderers by implementing the renderer interface
- Custom providers by implementing the provider interface
- Configuration through the ConfigParser system