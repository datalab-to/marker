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
    
    print("=== Model Creation Test ===")
    
    # Check device availability
    device = settings.TORCH_DEVICE_MODEL
    print(f"Using device: {device}")
    
    # Test 1: Show current XPU status
    print("\n=== Test 1: XPU Status ===")
    if hasattr(torch, 'xpu'):
        print(f"XPU available: {torch.xpu.is_available()}")
        if torch.xpu.is_available():
            print(f"XPU device count: {torch.xpu.device_count()}")
            for i in range(torch.xpu.device_count()):
                print(f"  XPU {i}: {torch.xpu.get_device_name(i)}")
    else:
        print("XPU not available in PyTorch")
    
    # Test 2: Test model creation
    print("\n=== Test 2: Model Creation ===")
    
    # Configuration: Simple model creation
    print("  Creating models:")
    models = create_model_dict(device=device)
    print(f"    Models created: {len(models)}")
    print(f"    Model names: {list(models.keys())}")
    
    print("\n=== Model Creation Test Completed ===")

if __name__ == "__main__":
    test_model_creation()