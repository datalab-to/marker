import os
from typing import Annotated

import openai

from marker.services.openai import OpenAIService


class AtlasCloudService(OpenAIService):
    """OpenAI-compatible LLM service backed by Atlas Cloud."""

    atlascloud_base_url: Annotated[
        str, "The Atlas Cloud OpenAI-compatible API base url. No trailing slash."
    ] = "https://api.atlascloud.ai/v1"
    atlascloud_model: Annotated[str, "The Atlas Cloud multimodal model id to use."] = (
        "google/gemini-3.5-flash"
    )
    atlascloud_api_key: Annotated[
        str, "The Atlas Cloud API key (falls back to $ATLASCLOUD_API_KEY or $ATLAS_CLOUD_API_KEY)."
    ] = None
    atlascloud_image_format: Annotated[
        str, "Image format sent to the model. Use 'png' for broad compatibility."
    ] = "png"

    def __init__(self, config=None):
        if config is None:
            config = {}
        if isinstance(config, dict):
            config = self._with_env_defaults(config)
        super().__init__(config)
        self.atlascloud_base_url = self.atlascloud_base_url.rstrip("/")
        self.openai_base_url = self.atlascloud_base_url
        self.openai_model = self.atlascloud_model
        self.openai_api_key = self.atlascloud_api_key
        self.openai_image_format = self.atlascloud_image_format

    def _with_env_defaults(self, config: dict) -> dict:
        atlascloud_api_key = os.environ.get("ATLASCLOUD_API_KEY") or os.environ.get(
            "ATLAS_CLOUD_API_KEY"
        )
        atlascloud_base_url = (
            os.environ.get("ATLASCLOUD_API_BASE")
            or os.environ.get("ATLASCLOUD_BASE_URL")
            or os.environ.get("ATLAS_CLOUD_API_BASE")
            or os.environ.get("ATLAS_CLOUD_BASE_URL")
        )
        updates = dict(config)
        if atlascloud_api_key and not updates.get("atlascloud_api_key"):
            updates["atlascloud_api_key"] = atlascloud_api_key
        if atlascloud_base_url and not updates.get("atlascloud_base_url"):
            updates["atlascloud_base_url"] = atlascloud_base_url.rstrip("/")
        return updates

    def get_client(self) -> openai.OpenAI:
        return openai.OpenAI(
            api_key=self.atlascloud_api_key,
            base_url=self.openai_base_url,
        )
