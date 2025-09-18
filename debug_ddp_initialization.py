import os
import sys
import torch

# Add the project root to the path
sys.path.insert(0, '.')

from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.settings import settings

def test_model_creation():
    """Test model creation functionality"""
    
    print("=== Model Creation Debug Script ===")
    
    # Check device availability
    device = settings.TORCH_DEVICE_MODEL
    print(f"Using device: {device}")
    
    if "xpu" in device:
        print("XPU device detected")
        if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
            print("Warning: XPU not available despite being selected")
    elif device == "cpu":
        print("Using CPU device")
    
    # Test 1: Model creation
    print("\n=== Test 1: Model creation ===")
    try:
        models = create_model_dict(device=device)
        print(f"Successfully created models: {list(models.keys())}")
        print("All models created successfully")
    except Exception as e:
        print(f"Error in model creation: {e}")
    
    print("\n=== Debug script completed ===")

if __name__ == "__main__":
    test_model_creation()