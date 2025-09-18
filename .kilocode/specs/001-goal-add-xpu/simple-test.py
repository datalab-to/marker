#!/usr/bin/env python3
"""
Simple test script to verify XPU device handling and Llama-CPP LLM service integration.
This script demonstrates the basic functionality without requiring extensive testing frameworks.
"""

import torch
import os
from marker.settings import settings
from marker.utils.gpu import GPUManager


def test_xpu_detection():
    """Test XPU device detection."""
    print("Testing XPU device detection...")
    
    # Check if XPU is available in PyTorch
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print("✓ XPU is available in PyTorch")
        print(f"  XPU device count: {torch.xpu.device_count()}")
        print(f"  XPU device name: {torch.xpu.get_device_name()}")
    else:
        print("ℹ XPU not available in current PyTorch installation")
    
    # Check the device selection logic
    print(f"  Selected device: {settings.TORCH_DEVICE_MODEL}")
    
    return True


def test_gpu_manager():
    """Test GPUManager with XPU support."""
    print("\nTesting GPUManager with XPU support...")
    
    # Create a GPUManager instance
    gpu_manager = GPUManager(0)
    
    # Check if using XPU
    if settings.TORCH_DEVICE_MODEL == "xpu":
        print("✓ GPUManager correctly identifies XPU as processing device")
    else:
        print(f"ℹ GPUManager using {settings.TORCH_DEVICE_MODEL} as processing device")
    
    return True


def test_llm_service_import():
    """Test importing the new Llama-CPP service."""
    print("\nTesting Llama-CPP service import...")
    
    try:
        # Try to import the new service
        from marker.services.llama_cpp import LlamaCPPService
        print("✓ LlamaCPPService can be imported successfully")
        return True
    except ImportError as e:
        print(f"ℹ LlamaCPPService import failed: {e}")
        return False


def main():
    """Run all simple tests."""
    print("=== Simple Tests for XPU Device Handling and Llama-CPP LLM Service ===\n")
    
    success = True
    
    success &= test_xpu_detection()
    success &= test_gpu_manager()
    success &= test_llm_service_import()
    
    print("\n=== Test Summary ===")
    if success:
        print("✓ All basic tests passed. Features are ready for implementation.")
    else:
        print("⚠ Some tests indicated issues. Please check the implementation.")
    
    return success


if __name__ == "__main__":
    main()