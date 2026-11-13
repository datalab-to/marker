from typing import Optional
import os
import platform
import shutil
import subprocess

from dotenv import find_dotenv
from pydantic import computed_field, model_validator
from pydantic_settings import BaseSettings


def _has_nvidia_gpu() -> bool:
    if os.path.exists("/dev/nvidia0") or os.path.exists("/proc/driver/nvidia/version"):
        return True
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    try:
        result = subprocess.run(
            [nvidia_smi, "-L"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0 and "GPU" in result.stdout
    except Exception:
        return False


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


class Settings(BaseSettings):
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "conversion_results")
    # The rendering font is a runtime download; keep it out of the package tree, which may
    # be read-only (site-packages) or a git checkout (editable install).
    FONT_DIR: str = os.path.join(
        os.environ.get("XDG_CACHE_HOME")
        or os.path.join(os.path.expanduser("~"), ".cache"),
        "datalab",
        "marker",
        "fonts",
    )
    DEBUG_DATA_FOLDER: str = os.path.join(BASE_DIR, "debug_data")
    ARTIFACT_URL: str = "https://models.datalab.to/artifacts"
    FONT_NAME: str = "GoNotoCurrent-Regular.ttf"
    FONT_PATH: Optional[str] = (
        None  # derived from FONT_DIR and FONT_NAME unless set explicitly
    )
    LOGLEVEL: str = "INFO"

    # General
    OUTPUT_ENCODING: str = "utf-8"
    OUTPUT_IMAGE_FORMAT: str = "JPEG"

    # LLM
    GOOGLE_API_KEY: Optional[str] = ""

    # General models
    TORCH_DEVICE: Optional[str] = (
        None  # Device preference override ("cuda", "mps", or "cpu")
    )

    @computed_field
    @property
    def TORCH_DEVICE_MODEL(self) -> str:
        if self.TORCH_DEVICE is not None:
            return self.TORCH_DEVICE

        if _has_nvidia_gpu():
            return "cuda"

        if _is_apple_silicon():
            return "mps"

        return "cpu"

    @model_validator(mode="after")
    def derive_font_path(self):
        # Derived after validation so an overridden FONT_DIR is honoured; a class-level
        # default would bind the default FONT_DIR at import time.
        if not self.FONT_PATH:
            self.FONT_PATH = os.path.join(self.FONT_DIR, self.FONT_NAME)
        return self

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


settings = Settings()
