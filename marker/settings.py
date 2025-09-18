from typing import Optional

from dotenv import find_dotenv
from pydantic import computed_field
from pydantic_settings import BaseSettings
import torch
import os


class Settings(BaseSettings):
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "conversion_results")
    FONT_DIR: str = os.path.join(BASE_DIR, "static", "fonts")
    DEBUG_DATA_FOLDER: str = os.path.join(BASE_DIR, "debug_data")
    ARTIFACT_URL: str = "https://models.datalab.to/artifacts"
    FONT_NAME: str = "GoNotoCurrent-Regular.ttf"
    FONT_PATH: str = os.path.join(FONT_DIR, FONT_NAME)
    LOGLEVEL: str = "INFO"

    # General
    OUTPUT_ENCODING: str = "utf-8"
    OUTPUT_IMAGE_FORMAT: str = "JPEG"

    # LLM
    GOOGLE_API_KEY: Optional[str] = ""

    # General models
    TORCH_DEVICE: Optional[str] = (
        None  # Device to use for PyTorch operations. Supports cuda, xpu, mps, and cpu.
              # When None, automatically detects the best available device.
              # Note: MPS device does not work for text detection, and will default to CPU.
              # For XPU devices, specify "xpu" or "xpu:<index>" for multi-device systems.
              # When set to a specific device, that device will be used directly without
              # automatic detection. Ensure the specified device is available and properly
              # configured in your system.
    )

    @computed_field
    @property
    def TORCH_DEVICE_MODEL(self) -> str:
        """Determine the best available device for PyTorch operations.
        
        Device selection priority:
        1. CUDA (NVIDIA GPUs)
        2. XPU (Intel GPUs)
        3. MPS (Apple Silicon GPUs)
        4. CPU (fallback)
        
        Users can override automatic detection by setting TORCH_DEVICE.
        For XPU devices, specify "xpu" or "xpu:<index>" for multi-device systems.
        
        When TORCH_DEVICE is manually set to a specific device, that device will be used
        directly without any automatic detection or validation. It's the user's responsibility
        to ensure the specified device is available and properly configured.
        
        XPU support requires Intel's PyTorch Extension (IPEX) to be installed and properly
        configured. For multi-XPU systems, device indexing starts at 0.
        """
        if self.TORCH_DEVICE is not None:
            return self.TORCH_DEVICE

        if torch.cuda.is_available():
            return "cuda"

        # Check for XPU availability (Intel GPUs)
        # XPU support requires Intel's PyTorch extension (IPEX)
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return "xpu"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    @computed_field
    @property
    def MODEL_DTYPE(self) -> torch.dtype:
        """Determine the appropriate data type for model operations.
        
        Uses bfloat16 for GPU devices (CUDA/XPU) for better performance,
        and float32 for CPU/MPS devices for compatibility.
        """
        if self.TORCH_DEVICE_MODEL in ["cuda", "xpu"]:
            return torch.bfloat16
        else:
            return torch.float32

    # XPU settings
    # DDP is no longer supported for XPU devices

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


settings = Settings()
