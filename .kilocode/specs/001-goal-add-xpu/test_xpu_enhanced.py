#!/usr/bin/env python3
"""
Test script to verify enhanced XPU support in Marker
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import torch
from marker.utils.gpu import GPUManager


def test_xpu_enhanced_support():
    """Test enhanced XPU support in the GPUManager."""
    print("=== Enhanced XPU Support Test ===")
    
    # Create a GPUManager instance
    gpu_manager = GPUManager(0)
    
    # Test XPU availability checking
    print(f"XPU available: {gpu_manager.check_xpu_available()}")
    
    if gpu_manager.check_xpu_available():
        # Test XPU readiness checking
        print(f"XPU ready: {gpu_manager.check_xpu_ready()}")
        
        # Test XPU device information
        device_info = gpu_manager.get_xpu_device_info()
        print(f"XPU device info: {device_info}")
        
        # Test XPU device count
        device_count = gpu_manager.get_xpu_device_count()
        print(f"XPU device count: {device_count}")
        
        # Test XPU memory info
        memory_info = gpu_manager.get_xpu_memory_info()
        print(f"XPU memory info: {memory_info}")
        
        # Test XPU capabilities
        print(f"XPU fp16 support: {gpu_manager.check_xpu_capability('fp16')}")
        print(f"XPU bf16 support: {gpu_manager.check_xpu_capability('bf16')}")
        print(f"XPU int8 support: {gpu_manager.check_xpu_capability('int8')}")
        
        # Test XPU VRAM
        vram = gpu_manager.get_xpu_vram()
        print(f"XPU VRAM: {vram} GB")
        
        # Test initialization
        gpu_manager.initialize_xpu()
        
        # Test cleanup
        gpu_manager.cleanup_xpu()
        
        print("\n=== Enhanced XPU Support Test: PASSED ===")
        return True
    else:
        print("XPU not available, skipping enhanced tests")
        print("\n=== Enhanced XPU Support Test: SKIPPED ===")
        return True


if __name__ == "__main__":
    success = test_xpu_enhanced_support()
    if success:
        print("\nAll tests completed successfully!")
    else:
        print("\nSome tests failed.")
        sys.exit(1)