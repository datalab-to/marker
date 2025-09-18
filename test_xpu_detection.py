#!/usr/bin/env python3

import torch
from marker.settings import settings
from marker.utils.gpu import GPUManager
from marker.logger import configure_logging, get_logger

configure_logging()
logger = get_logger()

def test_xpu_detection():
    logger.info("Testing XPU device detection and initialization...")
    
    # Check if Intel Extension for PyTorch is available
    try:
        import intel_extension_for_pytorch as ipex
        logger.info("Intel Extension for PyTorch (IPEX) is available")
    except ImportError:
        logger.warning("Intel Extension for PyTorch (IPEX) is not available")
        return
    
    # Check if XPU is available in PyTorch
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        logger.info("XPU devices are available in PyTorch")
        
        # Get device count
        device_count = torch.xpu.device_count()
        logger.info(f"Number of XPU devices: {device_count}")
        
        # Test each device
        for i in range(device_count):
            try:
                # Get device name
                device_name = torch.xpu.get_device_name(i)
                logger.info(f"XPU device {i}: {device_name}")
                
                # Test basic functionality
                x = torch.randn(100, 100, device=f"xpu:{i}")
                logger.info(f"Basic tensor operations work on XPU device {i}")
                del x
                
                # Get memory info
                if hasattr(torch.xpu, 'mem_get_info'):
                    free, total = torch.xpu.mem_get_info(i)
                    logger.info(f"XPU device {i} memory - Free: {free / (1024**3):.2f} GB, Total: {total / (1024**3):.2f} GB")
                
            except Exception as e:
                logger.error(f"Error testing XPU device {i}: {e}")
    else:
        logger.info("XPU devices are not available in PyTorch")
    
    # Check settings
    logger.info(f"TORCH_DEVICE_MODEL setting: {settings.TORCH_DEVICE_MODEL}")
    
    # Test GPUManager
    logger.info("Testing GPUManager with XPU...")
    try:
        gpu_manager = GPUManager(0)
        if gpu_manager.using_xpu():
            logger.info("GPUManager correctly detected XPU usage")
            
            # Test XPU availability check
            if gpu_manager.check_xpu_available():
                logger.info("GPUManager confirmed XPU is available")
                
                # Test XPU readiness
                if gpu_manager.check_xpu_ready():
                    logger.info("XPU device is ready for use")
                else:
                    logger.warning("XPU device is available but not ready for use")
            else:
                logger.warning("GPUManager could not confirm XPU availability")
        else:
            logger.info("GPUManager is not configured to use XPU")
            
    except Exception as e:
        logger.error(f"Error testing GPUManager: {e}")

if __name__ == "__main__":
    test_xpu_detection()