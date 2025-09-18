#!/usr/bin/env python3
"""
Test script to verify XPU support in Marker
"""

import torch
import intel_extension_for_pytorch as ipex
from marker.converters.pdf import PdfConverter
from marker.settings import settings

def test_xpu_support():
    """Test XPU support in the environment"""
    print("=== XPU Support Test ===")
    
    # Check PyTorch and XPU availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"XPU available: {torch.xpu.is_available()}")
    print(f"XPU device count: {torch.xpu.device_count()}")
    
    if torch.xpu.is_available():
        print(f"XPU device name: {torch.xpu.get_device_name(0)}")
        
        # Test tensor operations on XPU
        x = torch.randn(1000, 1000).to("xpu")
        y = torch.randn(1000, 1000).to("xpu")
        z = torch.matmul(x, y)
        print("Basic XPU tensor operations: SUCCESS")
        
        # Test Intel Extension for PyTorch
        print(f"Intel Extension for PyTorch version: {ipex.__version__}")
        print("Intel Extension for PyTorch: AVAILABLE")
    else:
        print("XPU is not available!")
        return False
    
    print("\n=== Marker Configuration Test ===")
    
    # Check Marker settings
    print(f"TORCH_DEVICE setting: {settings.TORCH_DEVICE}")
    print(f"TORCH_DEVICE_MODEL setting: {settings.TORCH_DEVICE_MODEL}")
    print(f"MODEL_DTYPE setting: {settings.MODEL_DTYPE}")
    
    # Try to initialize a converter (this will load models)
    try:
        converter = PdfConverter(artifact_dict={})
        print("Marker PDF Converter initialization: SUCCESS")
        del converter
    except Exception as e:
        print(f"Marker PDF Converter initialization: FAILED - {e}")
        return False
    
    print("\n=== XPU Support Verification: PASSED ===")
    return True

if __name__ == "__main__":
    success = test_xpu_support()
    if success:
        print("\nAll tests passed! XPU support is properly configured for Marker.")
    else:
        print("\nSome tests failed. Please check the configuration.")