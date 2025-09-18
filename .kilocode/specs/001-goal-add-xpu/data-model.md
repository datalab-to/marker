# Data Model: XPU Device Handling and Llama-CPP LLM Service

## Entity: Device

### Description
Represents a processing unit available on the system for document processing tasks.

### Fields
- **type**: String - The type of device (XPU, CUDA, MPS, CPU)
- **available**: Boolean - Whether the device is available for use
- **index**: Integer - Device index for multi-device systems (e.g., multiple GPUs)
- **status**: String - Current status of the device (available, busy, error)
- **performance_characteristics**: Object - Performance metrics and capabilities

### Validation Rules
- Device type must be one of: XPU, CUDA, MPS, CPU
- Available must be a boolean value
- Index must be a non-negative integer
- Status must be one of: available, busy, error

### State Transitions
- available → busy (when device is in use)
- busy → available (when device is free)
- any → error (when device encounters an error)
- error → available (when device recovers)

## Entity: LLM Service Configuration

### Description
Represents the configuration for selecting and using different LLM services for document processing.

### Fields
- **service_type**: String - The type of LLM service (llama-cpp, ollama, gemini, openai, claude, etc.)
- **enabled**: Boolean - Whether the service is enabled for use
- **connection_details**: Object - Connection parameters specific to the service type
- **availability_status**: String - Current availability status of the service
- **model_name**: String - The specific model to use with this service

### Validation Rules
- Service type must be one of the supported services
- Enabled must be a boolean value
- Connection details must contain required fields for the service type
- Availability status must be one of: available, unavailable, error
- Model name must be a non-empty string

### State Transitions
- enabled=true → available (when service is configured and reachable)
- enabled=true → unavailable (when service is configured but unreachable)
- any → error (when service encounters an error)
- error → available (when service recovers)

## Entity: Device Selection Policy

### Description
Represents the rules and preferences for selecting processing devices based on availability and performance.

### Fields
- **priority_order**: Array[String] - Ordered list of device types by preference
- **fallback_behavior**: Object - Rules for fallback when preferred devices are unavailable
- **load_balancing**: Boolean - Whether to distribute load across multiple devices

### Validation Rules
- Priority order must contain valid device types
- Fallback behavior must define valid fallback rules
- Load balancing must be a boolean value

### Relationships
- Device Selection Policy references multiple Device entities
- Device Selection Policy is used by the device selection logic in settings.py

## Integration with Existing Model
The new entities extend the existing Marker architecture without breaking changes:

1. **Device** entity integrates with:
   - `marker/settings.py` - Device detection logic
   - `marker/utils/gpu.py` - GPUManager class for device handling
   - Existing converter and processor classes that use device selection

2. **LLM Service Configuration** entity integrates with:
   - `marker/services/` - New LlamaCPPService implementation
   - `marker/config/parser.py` - LLM service selection logic
   - Existing LLM processor classes that use service selection

3. **Device Selection Policy** entity integrates with:
   - `marker/settings.py` - Automatic device selection logic
   - Configuration options for overriding default selection

## Phase 3.1 & 3.2 Completion Status
Phase 3.1 has been completed with the successful implementation of all data model entities:
- Device entity now supports XPU device type
- LLM Service Configuration entity now supports llama-cpp service type
- Device Selection Policy entity works with XPU devices
- All integration points have been implemented and tested

Phase 3.2 has been completed with the creation of TDD tests:
- Basic functionality verification script created for testing Device and LLM Service entities
- Test documents prepared for XPU testing
- Test documents prepared for Llama-CPP testing
- All tests are designed to fail before implementation (TDD approach)