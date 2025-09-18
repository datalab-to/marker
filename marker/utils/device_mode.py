import torch
from marker.logger import get_logger
from marker.settings import settings

logger = get_logger()


class DeviceMode:
    """Enumeration of device modes"""
    CUDA = "cuda"
    XPU = "xpu"
    MPS = "mps"
    CPU = "cpu"


def detect_device_mode() -> str:
    """
    Detect the appropriate device mode based on hardware availability and configuration.
    
    Returns:
        str: The detected device mode (cuda, xpu, mps, or cpu)
    """
    # Check if TORCH_DEVICE is manually set
    if settings.TORCH_DEVICE is not None:
        device = settings.TORCH_DEVICE.lower()
        if "cuda" in device:
            return DeviceMode.CUDA
        elif "xpu" in device:
            return DeviceMode.XPU
        elif "mps" in device:
            return DeviceMode.MPS
        else:
            return DeviceMode.CPU
    
    # Auto-detect based on hardware availability
    if torch.cuda.is_available():
        logger.info("CUDA device detected, using NVIDIA path with MPS")
        return DeviceMode.CUDA
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        logger.info("XPU device detected, using Intel path with continuous batching")
        return DeviceMode.XPU
    elif torch.backends.mps.is_available():
        logger.info("MPS device detected, using Apple Silicon path")
        return DeviceMode.MPS
    else:
        logger.info("No specialized hardware detected, using CPU")
        return DeviceMode.CPU


def is_nvidia_path(device_mode: str) -> bool:
    """
    Check if the device mode should use the NVIDIA path with MPS.
    
    Args:
        device_mode (str): The device mode to check
        
    Returns:
        bool: True if using NVIDIA path, False otherwise
    """
    return device_mode == DeviceMode.CUDA


def is_intel_path(device_mode: str) -> bool:
    """
    Check if the device mode should use the Intel path with continuous batching.
    
    Args:
        device_mode (str): The device mode to check
        
    Returns:
        bool: True if using Intel path, False otherwise
    """
    return device_mode == DeviceMode.XPU