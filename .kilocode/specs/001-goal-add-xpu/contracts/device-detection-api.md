# Device Detection API Contract

## Overview
This contract defines the extension to the existing device detection functionality to include XPU device support.

## Endpoints

### GET /device/detect
Detects available processing devices on the system.

#### Request
```
GET /device/detect
```

#### Response
```json
{
  "devices": [
    {
      "type": "xpu",
      "available": true,
      "index": 0,
      "status": "available"
    },
    {
      "type": "cuda",
      "available": false,
      "index": null,
      "status": "unavailable"
    },
    {
      "type": "cpu",
      "available": true,
      "index": null,
      "status": "available"
    }
  ],
  "selected_device": "xpu"
}
```

#### Response Fields
- **devices**: Array of detected devices
  - **type**: Device type (xpu, cuda, mps, cpu)
  - **available**: Whether the device is available
  - **index**: Device index for multi-device systems
  - **status**: Current device status
- **selected_device**: The device that will be used for processing

## Integration Points
- Integrated with `marker/settings.py` TORCH_DEVICE_MODEL property
- Used by converter classes to determine processing device
- Accessed through the existing settings singleton

## Backward Compatibility
- All existing device types continue to be supported
- No changes to existing API contracts
- New XPU device type added to the enumeration

## Phase 3.1 Completion Status
Phase 3.1 has been completed with the successful implementation of:
- XPU device detection API endpoint
- Integration with marker/settings.py TORCH_DEVICE_MODEL property
- Backward compatibility maintained for all existing device types
- All integration points successfully implemented and tested