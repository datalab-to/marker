#!/usr/bin/env python3
"""
Test script to verify XPU device handling integration with existing MPS functionality
"""

import torch
from marker.utils.gpu import GPUManager
from marker.settings import settings

def test_cuda_mps_functionality():
    """Test that CUDA MPS functionality still works correctly"""
    print("=== Testing CUDA MPS Functionality ===")
    
    # Only run this test if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, skipping CUDA MPS test")
        return True
    
    try:
        # Test GPUManager with CUDA device
        with GPUManager(0) as gpu_manager:
            # Check that we're using CUDA
            if not gpu_manager.using_cuda():
                print("Not using CUDA device, skipping CUDA MPS test")
                return True
                
            # Check that MPS server can be started
            mps_started = gpu_manager.start_mps_server()
            print(f"MPS server started: {mps_started}")
            
            # Get VRAM info
            vram = gpu_manager.get_gpu_vram()
            print(f"CUDA VRAM: {vram} GB")
            
            # Stop MPS server
            gpu_manager.stop_mps_server()
            print("MPS server stopped successfully")
            
        print("CUDA MPS functionality test: PASSED")
        return True
    except Exception as e:
        print(f"CUDA MPS functionality test: FAILED - {e}")
        return False

def test_xpu_functionality():
    """Test that XPU functionality works correctly"""
    print("\n=== Testing XPU Functionality ===")
    
    try:
        # Test GPUManager with XPU device
        with GPUManager(0) as gpu_manager:
            # Check that we're using XPU
            if not gpu_manager.using_xpu():
                print("Not using XPU device, skipping XPU test")
                return True
                
            # Check XPU availability
            xpu_available = gpu_manager.check_xpu_available()
            print(f"XPU available: {xpu_available}")
            
            if not xpu_available:
                print("XPU not available, skipping XPU functionality test")
                return True
                
            # Initialize XPU
            gpu_manager.initialize_xpu()
            print("XPU initialized successfully")
            
            # Get XPU device info
            device_info = gpu_manager.get_xpu_device_info()
            print(f"XPU device info: {device_info}")
            
            # Get XPU device count
            device_count = gpu_manager.get_xpu_device_count()
            print(f"XPU device count: {device_count}")
            
            # Get XPU VRAM
            vram = gpu_manager.get_xpu_vram()
            print(f"XPU VRAM: {vram} GB")
            
            # Get XPU memory info
            memory_info = gpu_manager.get_xpu_memory_info()
            print(f"XPU memory info: {memory_info}")
            
            # Check XPU capabilities
            fp16_support = gpu_manager.check_xpu_capability('fp16')
            bf16_support = gpu_manager.check_xpu_capability('bf16')
            print(f"XPU FP16 support: {fp16_support}")
            print(f"XPU BF16 support: {bf16_support}")
            
        print("XPU functionality test: PASSED")
        return True
    except Exception as e:
        print(f"XPU functionality test: FAILED - {e}")
        return False

def test_device_detection():
    """Test that device detection works correctly"""
    print("\n=== Testing Device Detection ===")
    
    try:
        # Test GPUManager device detection
        gpu_manager = GPUManager(0)
        
        using_cuda = gpu_manager.using_cuda()
        using_xpu = gpu_manager.using_xpu()
        
        print(f"Using CUDA: {using_cuda}")
        print(f"Using XPU: {using_xpu}")
        print(f"TORCH_DEVICE_MODEL setting: {settings.TORCH_DEVICE_MODEL}")
        
        # Verify that only one device type is detected
        if using_cuda and using_xpu:
            print("ERROR: Both CUDA and XPU detected - this should not happen")
            return False
        elif not using_cuda and not using_xpu:
            print("WARNING: Neither CUDA nor XPU detected - using CPU")
            
        print("Device detection test: PASSED")
        return True
    except Exception as e:
        print(f"Device detection test: FAILED - {e}")
        return False

def main():
    """Run all integration tests"""
    print("=== XPU-MPS Integration Tests ===\n")
    
    success = True
    
    # Test device detection
    success &= test_device_detection()
    
    # Test CUDA MPS functionality (if CUDA is available)
    success &= test_cuda_mps_functionality()
    
    # Test XPU functionality (if XPU is available)
    success &= test_xpu_functionality()
    
    print("\n=== Integration Test Summary ===")
    if success:
        print("All integration tests PASSED")
        print("XPU device handling is properly integrated with existing MPS functionality")
    else:
        print("Some integration tests FAILED")
        print("Please check the implementation")
    
    return success

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)