from typing import Optional

from dotenv import find_dotenv
from pydantic import computed_field, model_validator
from pydantic_settings import BaseSettings
import torch
import os


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
        None  # Device for the local torch models (ocr error, fast layout); the VLM runs on the inference server
    )

    @computed_field
    @property
    def TORCH_DEVICE_MODEL(self) -> str:
        if self.TORCH_DEVICE is not None:
            return self.TORCH_DEVICE

        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
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
