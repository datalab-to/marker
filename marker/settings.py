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
        None  # Note: MPS device does not work for text detection, and will default to CPU
    )

    @computed_field
    @property
    def TORCH_DEVICE_MODEL(self) -> str:
        if self.TORCH_DEVICE is not None:
            return self.TORCH_DEVICE

        if torch.cuda.is_available():
            return "cuda"

        # guard for older torch builds without .mps
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "cpu"

    @computed_field
    @property
    def MODEL_DTYPE(self) -> torch.dtype:

        # Prefer bfloat16 on CUDA, float16 on MPS, float32 on CPU
        if self.TORCH_DEVICE_MODEL == "cuda":
            return torch.bfloat16
        if self.TORCH_DEVICE_MODEL == "mps":
            return torch.bfloat16
        return torch.float32
    # Convenience helper for cleaner branching elsewhere

    @property
    def USING_CUDA(self) -> bool:
        return self.TORCH_DEVICE_MODEL == "cuda"

    @property
    def USING_MPS(self) -> bool:
        return self.TORCH_DEVICE_MODEL == "mps"

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


settings = Settings()
