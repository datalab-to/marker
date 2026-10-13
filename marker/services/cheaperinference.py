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


class CheaperInferenceService(BaseService):
    """LLM service backed by Cheaper Inference (https://cheaperinference.com).

    Cheaper Inference exposes an OpenAI-compatible API, so this mirrors the
    OpenAI service but routes through api.cheaperinference.com and defaults
    to gpt-5.4-mini.  Used with ``--use_llm`` for accurate mode.
    """

    cheaperinference_base_url: Annotated[
        str, "The Cheaper Inference API base url.  No trailing slash."
    ] = "https://api.cheaperinference.com/v1"
    cheaperinference_model: Annotated[str, "The Cheaper Inference model id to use."] = (
        "gpt-5.4-mini"
    )
    cheaperinference_api_key: Annotated[
        str,
        "The Cheaper Inference API key (falls back to $CHEAPER_INFERENCE_API_KEY).",
    ] = None
    cheaperinference_image_format: Annotated[
        str, "Image format sent to the model. Use 'png' for better compatibility."
    ] = "webp"

    def __init__(self, config=None):
        # Fall back to $CHEAPER_INFERENCE_API_KEY so the key need not be passed
        # explicitly (must be set before BaseService verifies required keys).
        env_key = os.environ.get("CHEAPER_INFERENCE_API_KEY")
        if env_key:
            if config is None:
                config = {"cheaperinference_api_key": env_key}
            elif isinstance(config, dict) and not config.get(
                "cheaperinference_api_key"
            ):
                config = {**config, "cheaperinference_api_key": env_key}
        super().__init__(config)

    def process_images(self, images: List[Image.Image]) -> List[dict]:
        if isinstance(images, Image.Image):
            images = [images]
        img_fmt = self.cheaperinference_image_format
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

        total_tries = max_retries + 1
        for tries in range(1, total_tries + 1):
            try:
                response = client.chat.completions.parse(
                    model=self.cheaperinference_model,
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
                logger.error(f"Cheaper Inference request failed: {e}")
                break

        return {}

    def get_client(self) -> openai.OpenAI:
        api_key = self.cheaperinference_api_key or os.environ.get(
            "CHEAPER_INFERENCE_API_KEY"
        )
        return openai.OpenAI(api_key=api_key, base_url=self.cheaperinference_base_url)
