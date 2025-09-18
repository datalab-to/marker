# Research Findings: XPU Device Handling and Llama-CPP LLM Service

## Decision: XPU Device Integration Approach
**Decision**: Implement XPU support by extending existing device detection and handling mechanisms in the Marker project, following the same patterns used for CUDA support.

**Rationale**: 
- The existing codebase already has well-defined patterns for device detection and handling through the `GPUManager` class in `marker/utils/gpu.py`
- XPU support can be added by extending these existing patterns rather than creating new ones
- This approach maintains consistency with the existing codebase architecture
- Minimal changes required to achieve the goal

**Alternatives considered**:
- Creating a completely new device management system - Rejected because it would add unnecessary complexity
- Modifying the existing CUDA-specific code directly - Rejected because it would break existing CUDA support

## Decision: Llama-CPP Service Integration Approach
**Decision**: Create a new LLM service implementation that follows the existing service pattern used by other LLM providers (Ollama, Gemini, etc.)

**Rationale**:
- The existing LLM service architecture is well-defined with a clear `BaseService` class
- New services can be added by implementing this interface
- This approach maintains consistency with existing LLM service implementations
- Allows for easy configuration and selection through the existing CLI options

**Alternatives considered**:
- Creating a completely new LLM service architecture - Rejected because it would add unnecessary complexity
- Modifying existing service implementations directly - Rejected because it could break existing functionality

## Decision: PyTorch XPU Installation
**Decision**: Use the Intel endpoint for PyTorch 2.8.0 as specified in the project documentation

**Rationale**:
- The specific PyTorch version and Intel endpoint are already documented
- This version is known to work with XPU hardware
- Using the documented version ensures compatibility

**Command for installation**:
```
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/xpu
```

## Decision: Device Selection Logic
**Decision**: Extend the existing device selection logic in `settings.py` to include XPU detection

**Rationale**:
- The existing `TORCH_DEVICE_MODEL` property already handles CUDA and MPS detection
- Adding XPU detection follows the same pattern
- Priority order will be: CUDA > XPU > MPS > CPU (based on performance characteristics)

## Key Technical Findings

### XPU Device Detection
- XPU devices can be detected using Intel's PyTorch implementation
- Similar to CUDA detection, we need to check for hardware availability
- The MPS server functionality may need to be extended or adapted for XPU

### Llama-CPP Service Integration
- The service should integrate with llama-serve backend
- Follow the same pattern as existing services like OllamaService
- Use the llama-cpp-python library for implementation

### Integration Points
1. `marker/utils/gpu.py` - Extend GPUManager for XPU support
2. `marker/settings.py` - Extend device detection logic
3. `marker/services/` - Add new LlamaCPPService implementation
4. `pyproject.toml` - Add Intel PyTorch endpoint dependency

## Implementation Approach
Based on the research, the implementation will focus on:
1. Adding XPU detection to the settings module
2. Extending the GPUManager class to handle XPU devices
3. Creating a new LlamaCPPService class that follows the existing service pattern
4. Updating project dependencies to include Intel PyTorch endpoint

## Phase 3.1 & 3.2 Completion Status
Phase 3.1 has been completed with the successful implementation of:
- XPU device handling and detection
- Llama-CPP LLM service integration
- All functional requirements met as specified in the spec.md file

Phase 3.2 has been completed with the creation of TDD tests:
- Basic functionality verification script created
- Test documents prepared for XPU testing
- Test documents prepared for Llama-CPP testing
- All tests are designed to fail before implementation (TDD approach)