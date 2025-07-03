from typing import Optional, List, Annotated
from io import BytesIO

import PIL
from pydantic import BaseModel

from marker.schema.blocks import Block
from marker.util import assign_config, verify_config_keys
import base64


class BaseService:
    timeout: Annotated[int, "The timeout to use for the service."] = 30
    max_retries: Annotated[
        int, "The maximum number of retries to use for the service."
    ] = 2
    retry_wait_time: Annotated[int, "The wait time between retries."] = 3

    def img_to_base64(self, img: PIL.Image.Image):
        image_bytes = BytesIO()
        img.save(image_bytes, format="WEBP")
        return base64.b64encode(image_bytes.getvalue()).decode("utf-8")

    def process_images(self, images: List[PIL.Image.Image]) -> list:
        raise NotImplementedError

    def format_image_for_llm(self, image):
        if not image:
            return []

        if not isinstance(image, list):
            image = [image]

        image_parts = self.process_images(image)
        return image_parts

    def __init__(self, config: Optional[BaseModel | dict] = None):
        assign_config(self, config)

        # Ensure we have all necessary fields filled out (API keys, etc.)
        verify_config_keys(self)

    def __call__(
        self,
        prompt: str,
        image: PIL.Image.Image | List[PIL.Image.Image] | None,
        block: Block | None,
        response_schema: type[BaseModel],
        max_retries: int | None = None,
        timeout: int | None = None,
    ):
        raise NotImplementedError
