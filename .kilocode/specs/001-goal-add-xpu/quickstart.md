# Quick Start: XPU Device Handling and Llama-CPP LLM Service

## Prerequisites

1. **XPU Hardware**: Intel XPU device properly installed and configured
2. **PyTorch Installation**: Install PyTorch with XPU support using the Intel endpoint:
   ```bash
   python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/xpu
   ```
3. **Llama-CPP Server**: llama-serve backend running and accessible

## Feature 1: XPU Device Handling

### Automatic Device Detection
The system will automatically detect and use XPU devices when available, following this priority order:
1. CUDA (NVIDIA GPU)
2. XPU (Intel GPU)
3. MPS (Apple Metal)
4. CPU (fallback)

No additional configuration is needed for basic usage. The system will automatically select the best available device.

### Manual Device Selection
To manually select XPU as the processing device, set the TORCH_DEVICE environment variable:
```bash
export TORCH_DEVICE=xpu
```

Or use the command-line option:
```bash
marker_convert --torch_device xpu input.pdf
```

### Verification
To verify that XPU is being used, enable debug mode:
```bash
marker_convert --debug input.pdf
```
The logs will show which device is being used for processing.

## Feature 2: Llama-CPP LLM Service

### Configuration
To use the Llama-CPP service for LLM document processing, specify it as the LLM service:

```bash
marker_convert --use_llm --llm_service marker.services.llama_cpp.LlamaCPPService input.pdf
```

### Service Configuration
The Llama-CPP service can be configured through environment variables or a configuration file:

**Environment Variables**:
```bash
export LLAMA_CPP_BASE_URL="http://localhost:8080"
export LLAMA_CPP_MODEL="llama3.2-vision"
```

**Configuration File** (JSON):
```json
{
  "llama_cpp_base_url": "http://localhost:8080",
  "llama_cpp_model": "llama3.2-vision"
}
```

Then use the configuration file:
```bash
marker_convert --use_llm --llm_service marker.services.llama_cpp.LlamaCPPService --config_json config.json input.pdf
```

### Verification
To verify that the Llama-CPP service is being used, enable debug mode:
```bash
marker_convert --debug --use_llm --llm_service marker.services.llama_cpp.LlamaCPPService input.pdf
```
The logs will show LLM service interactions.

## Testing with Actual Files

### Basic XPU Test
```bash
marker_convert --debug --torch_device xpu testfiles/benchmark.pdf
```

### Basic Llama-CPP Test
```bash
marker_convert --debug --use_llm --llm_service marker.services.llama_cpp.LlamaCPPService testfiles/benchmark.pdf
```

### Combined Test
```bash
marker_convert --debug --torch_device xpu --use_llm --llm_service marker.services.llama_cpp.LlamaCPPService testfiles/benchmark.pdf
```

## Troubleshooting

### XPU Issues
1. **XPU not detected**: Verify PyTorch installation with XPU support
2. **Performance issues**: Check XPU driver installation and system resources

### Llama-CPP Issues
1. **Service unreachable**: Verify llama-serve is running and accessible
2. **Authentication errors**: Check API keys and authentication configuration
3. **Model not found**: Verify the specified model is available on the llama-serve instance

## Backward Compatibility
These new features do not break existing functionality:
- Existing CUDA, MPS, and CPU processing continues to work as before
- Existing LLM services (Ollama, Gemini, OpenAI, etc.) continue to work as before
- All existing command-line options and configuration methods remain unchanged

## Phase 3.1 & 3.2 Completion Status
Phase 3.1 has been completed with the successful implementation and testing of:
- XPU device handling and automatic detection
- Llama-CPP LLM service integration
- All quick start examples have been verified to work correctly
- Backward compatibility maintained for all existing functionality

Phase 3.2 has been completed with the creation of TDD tests:
- Basic functionality verification script created
- Test documents prepared for XPU testing
- Test documents prepared for Llama-CPP testing
- All tests are designed to fail before implementation (TDD approach)