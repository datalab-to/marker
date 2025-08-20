import json
import time
from typing import Annotated, List

import PIL
from marker.logger import get_logger
from openai import AzureOpenAI, APITimeoutError, RateLimitError
from PIL import Image
from pydantic import BaseModel

from marker.schema.blocks import Block
from marker.services import BaseService

logger = get_logger()


class AzureOpenAIService(BaseService):
    azure_endpoint: Annotated[
        str, "The Azure OpenAI endpoint URL. No trailing slash."
    ] = None
    azure_api_key: Annotated[
        str, "The API key to use for the Azure OpenAI service."
    ] = None
    azure_api_version: Annotated[str, "The Azure OpenAI API version to use."] = None
    deployment_name: Annotated[
        str, "The deployment name for the Azure OpenAI model."
    ] = None

    def process_images(self, images: List[PIL.Image.Image]) -> list:
        if isinstance(images, Image.Image):
            images = [images]

        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/webp;base64,{}".format(self.img_to_base64(img)),
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
                    model=self.deployment_name,
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
                # Rate limit exceeded
                if tries == total_tries:
                    # Last attempt failed. Give up
                    logger.error(
                        f"Rate limit error: {e}. Max retries reached. Giving up. (Attempt {tries}/{total_tries})"
                    )
                    break
                else:
                    wait_time = tries * self.retry_wait_time
                    logger.warning(
                        f"Rate limit error: {e}. Retrying in {wait_time} seconds... (Attempt {tries}/{total_tries})"
                    )
                    time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Azure OpenAI inference failed: {e}")
                break

        return {}

    def get_client(self) -> AzureOpenAI:
        return AzureOpenAI(
            api_version=self.azure_api_version,
            azure_endpoint=self.azure_endpoint,
            api_key=self.azure_api_key,
        )
