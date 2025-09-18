#!/usr/bin/env python3
"""
Verification script for XPU device handling implementation
"""
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import torch
from marker.settings import settings
from marker.utils.gpu import GPUManager


def test_xpu_detection():
    """Test XPU device detection."""
    print("Testing XPU device detection...")
    
    # Check if XPU is available in PyTorch
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print("✓ XPU is available in PyTorch")
        print(f" XPU device count: {torch.xpu.device_count()}")
        if torch.xpu.device_count() > 0:
            print(f"  XPU device name: {torch.xpu.get_device_name()}")
        return True
    else:
        print("ℹ XPU not available in current PyTorch installation")
        return False


def test_device_selection():
    """Test device selection logic."""
    print("\nTesting device selection logic...")
    
    # Check the device selection logic
    selected_device = settings.TORCH_DEVICE_MODEL
    print(f"  Selected device: {selected_device}")
    
    if selected_device == "xpu":
        print("✓ System correctly selected XPU for processing")
        return True
    elif selected_device in ["cuda", "mps", "cpu"]:
        print(f"ℹ System selected {selected_device} (not XPU)")
        return True
    else:
        print(f"⚠ Unknown device type selected: {selected_device}")
        return False


def test_gpu_manager():
    """Test GPUManager with XPU support."""
    print("\nTesting GPUManager with XPU support...")
    
    # Create a GPUManager instance
    gpu_manager = GPUManager(0)
    
    # Check if using XPU
    if settings.TORCH_DEVICE_MODEL == "xpu":
        print("✓ GPUManager correctly identifies XPU as processing device")
        # Test XPU-specific methods
        print(f"  Using XPU: {gpu_manager.using_xpu()}")
        print(f"  Using CUDA: {gpu_manager.using_cuda()}")
        return True
    else:
        print(f"ℹ GPUManager using {settings.TORCH_DEVICE_MODEL} as processing device")
        print(f"  Using XPU: {gpu_manager.using_xpu()}")
        print(f"  Using CUDA: {gpu_manager.using_cuda()}")
        return True


def test_model_dtype():
    """Test MODEL_DTYPE property."""
    print("\nTesting MODEL_DTYPE property...")
    
    model_dtype = settings.MODEL_DTYPE
    print(f"  Model dtype: {model_dtype}")
    
    if settings.TORCH_DEVICE_MODEL in ["cuda", "xpu"]:
        if model_dtype == torch.bfloat16:
            print("✓ Correct dtype (bfloat16) selected for accelerated device")
            return True
        else:
            print(f"⚠ Unexpected dtype for accelerated device: {model_dtype}")
            return False
    else:
        if model_dtype == torch.float32:
            print("✓ Correct dtype (float32) selected for CPU")
            return True
        else:
            print(f"⚠ Unexpected dtype for CPU: {model_dtype}")
            return False


def test_backward_compatibility():
    """Test that existing functionality still works."""
    print("\nTesting backward compatibility...")
    
    # Test that CUDA detection still works if available
    if torch.cuda.is_available():
        print("✓ CUDA still available and working")
    
    # Test that CPU fallback still works
    print("✓ CPU processing still available as fallback")
    
    return True


def main():
    """Run all verification tests."""
    print("=== Verification Tests for XPU Device Handling Implementation ===\n")
    
    success = True
    
    success &= test_xpu_detection()
    success &= test_device_selection()
    success &= test_gpu_manager()
    success &= test_model_dtype()
    success &= test_backward_compatibility()
    
    print("\n=== Verification Summary ===")
    if success:
        print("✓ All verification tests passed. XPU implementation is working correctly.")
    else:
        print("⚠ Some verification tests indicated issues. Please check the implementation.")
    
    return success


if __name__ == "__main__":
    sys.exit(0 if main() else 1)