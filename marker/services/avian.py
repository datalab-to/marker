import json
import time
from typing import Annotated, List

import openai
import PIL
from marker.logger import get_logger
from openai import APITimeoutError, RateLimitError
from PIL import Image
from pydantic import BaseModel

from marker.schema.blocks import Block
from marker.services import BaseService

logger = get_logger()


class AvianService(BaseService):
    """LLM service for the Avian inference API (OpenAI-compatible).

    Available models:
        - deepseek-v3.2     (DeepSeek V3.2, 164K context)
        - kimi-k2.5         (Kimi K2.5, 128K context)
        - glm-5             (GLM-5, 128K context)
        - minimax-m2.5      (MiniMax M2.5, 1M context)

    Usage:
        --llm_service marker.services.avian.AvianService
        --avian_api_key <your-key>
        --avian_model deepseek-v3.2
    """

    avian_api_key: Annotated[
        str, "The API key to use for the Avian service."
    ] = None
    avian_model: Annotated[
        str,
        "The model name to use for the Avian service. "
        "Options: deepseek-v3.2, kimi-k2.5, glm-5, minimax-m2.5",
    ] = "deepseek-v3.2"
    avian_image_format: Annotated[
        str,
        "The image format to use for the Avian service. "
        "Use 'png' for better compatibility.",
    ] = "png"

    def process_images(self, images: List[Image.Image]) -> List[dict]:
        if isinstance(images, Image.Image):
            images = [images]

        img_fmt = self.avian_image_format
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
                "content": [
                    *image_data,
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        total_tries = max_retries + 1
        for tries in range(1, total_tries + 1):
            try:
                response = client.beta.chat.completions.parse(
                    extra_headers={
                        "X-Title": "Marker",
                        "HTTP-Referer": "https://github.com/datalab-to/marker",
                    },
                    model=self.avian_model,
                    messages=messages,
                    timeout=timeout,
                    response_format=response_schema,
                )
                response_text = response.choices[0].message.content
                total_tokens = response.usage.total_tokens
                if block:
                    block.update_metadata(
                        llm_tokens_used=total_tokens, llm_request_count=1
                    )
                return json.loads(response_text)
            except (APITimeoutError, RateLimitError) as e:
                if tries == total_tries:
                    logger.error(
                        f"Rate limit error: {e}. Max retries reached. Giving up. (Attempt {tries}/{total_tries})",
                    )
                    break
                else:
                    wait_time = tries * self.retry_wait_time
                    logger.warning(
                        f"Rate limit error: {e}. Retrying in {wait_time} seconds... (Attempt {tries}/{total_tries})",
                    )
                    time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Avian inference failed: {e}")
                break

        return {}

    def get_client(self) -> openai.OpenAI:
        return openai.OpenAI(
            api_key=self.avian_api_key,
            base_url="https://api.avian.io/v1",
        )
