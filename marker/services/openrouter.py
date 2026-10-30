import json
import os
import time
from typing import Annotated, List

import openai
import PIL
from openai import APITimeoutError, RateLimitError
from PIL import Image
from pydantic import BaseModel

from marker.logger import get_logger
from marker.schema.blocks import Block
from marker.services import BaseService

logger = get_logger()


class OpenRouterService(BaseService):
    """LLM service backed by OpenRouter (https://openrouter.ai).

    OpenRouter exposes an OpenAI-compatible API, so this mirrors the OpenAI
    service but routes through openrouter.ai and defaults to Gemini Flash 3.5.
    Used with ``--use_llm`` for accurate mode.
    """

    openrouter_base_url: Annotated[
        str, "The OpenRouter API base url.  No trailing slash."
    ] = "https://openrouter.ai/api/v1"
    openrouter_model: Annotated[str, "The OpenRouter model id to use."] = (
        "google/gemini-3.5-flash"
    )
    openrouter_api_key: Annotated[
        str, "The OpenRouter API key (falls back to $OPENROUTER_API_KEY)."
    ] = None
    openrouter_image_format: Annotated[
        str, "Image format sent to the model. Use 'png' for better compatibility."
    ] = "webp"
    openrouter_allow_data_collection: Annotated[
        bool,
        "Allow non-zero-data-retention providers.  Some models (e.g. Gemini) "
        "have no ZDR endpoint, so requests fail unless this is allowed.",
    ] = True

    def __init__(self, config=None):
        if isinstance(config, BaseModel):
            config = config.model_dump()

        # Fall back to $OPENROUTER_API_KEY so the key need not be passed
        # explicitly (must be set before BaseService verifies required keys).
        env_key = os.environ.get("OPENROUTER_API_KEY")
        if env_key:
            if config is None:
                config = {"openrouter_api_key": env_key}
            elif isinstance(config, dict) and not config.get("openrouter_api_key"):
                config = {**config, "openrouter_api_key": env_key}
        super().__init__(config)

    def process_images(self, images: List[Image.Image]) -> List[dict]:
        if isinstance(images, Image.Image):
            images = [images]
        img_fmt = self.openrouter_image_format
        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/{};base64,{}".format(
                        img_fmt, self.img_to_base64(img, format=img_fmt)
                    ),
                },
            }
            for img in images
        ]

    def __call__(
        self,
        prompt: str,
        image: PIL.Image.Image | List[PIL.Image.Image] | None,
        block: Block | None,
        response_schema: type[BaseModel],
        max_retries: int | None = None,
        timeout: int | None = None,
    ):
        if max_retries is None:
            max_retries = self.max_retries
        if timeout is None:
            timeout = self.timeout

        client = self.get_client()
        image_data = self.format_image_for_llm(image)
        messages = [
            {
                "role": "user",
                "content": [*image_data, {"type": "text", "text": prompt}],
            }
        ]

        extra_headers = {
            "X-Title": "Marker",
            "HTTP-Referer": "https://github.com/datalab-to/marker",
        }
        extra_body = {}
        if self.openrouter_allow_data_collection:
            # allow routing to non-ZDR providers (Gemini has no ZDR endpoint)
            extra_body["provider"] = {"data_collection": "allow"}

        total_tries = max_retries + 1
        for tries in range(1, total_tries + 1):
            try:
                response = client.chat.completions.parse(
                    extra_headers=extra_headers,
                    extra_body=extra_body,
                    model=self.openrouter_model,
                    messages=messages,
                    timeout=timeout,
                    response_format=response_schema,
                )
                response_text = response.choices[0].message.content
                usage = getattr(response, "usage", None)
                total_tokens = getattr(usage, "total_tokens", 0) if usage else 0
                if block:
                    block.update_metadata(
                        llm_tokens_used=total_tokens, llm_request_count=1
                    )
                return json.loads(response_text)
            except (APITimeoutError, RateLimitError) as e:
                if tries == total_tries:
                    logger.error(
                        f"Rate limit error: {e}. Max retries reached. Giving up. "
                        f"(Attempt {tries}/{total_tries})",
                    )
                    break
                wait_time = tries * self.retry_wait_time
                logger.warning(
                    f"Rate limit error: {e}. Retrying in {wait_time}s... "
                    f"(Attempt {tries}/{total_tries})",
                )
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"OpenRouter inference failed: {e}")
                break

        return {}

    def get_client(self) -> openai.OpenAI:
        api_key = self.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        return openai.OpenAI(api_key=api_key, base_url=self.openrouter_base_url)
