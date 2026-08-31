import json
import os
import time
from typing import Annotated, List

import litellm
import PIL
from litellm.exceptions import (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from PIL import Image
from pydantic import BaseModel

from marker.logger import get_logger
from marker.schema.blocks import Block
from marker.services import BaseService

logger = get_logger()

# Transient errors worth retrying. Auth/validation errors are NOT here on
# purpose - retrying a bad key or an invalid request just wastes time.
RETRYABLE_ERRORS = (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)


class LiteLLMService(BaseService):
    """LLM service backed by LiteLLM (https://github.com/BerriAI/litellm).

    LiteLLM exposes a single OpenAI-style ``completion`` call that routes to
    100+ providers (OpenAI, Anthropic, Gemini, Bedrock, Vertex, Azure, Groq,
    a self-hosted LiteLLM proxy, ...) from one ``provider/model`` string. Unlike
    pointing ``OpenAIService`` at a base url, this reaches providers whose auth
    is not OpenAI-compatible (Bedrock SigV4, Vertex service accounts, Azure AD),
    since LiteLLM handles each provider's native signing. Used with ``--use_llm``.
    """

    litellm_model: Annotated[
        str,
        "The litellm model string, e.g. 'gpt-5-mini', 'anthropic/claude-opus-4-8', "
        "'gemini/gemini-2.5-flash', or 'litellm_proxy/<model>' for a proxy.",
    ] = "gemini/gemini-2.5-flash"
    litellm_api_key: Annotated[
        str,
        "API key for the target provider. Leave blank to let LiteLLM read the "
        "provider's own env var (e.g. GEMINI_API_KEY, ANTHROPIC_API_KEY).",
    ] = ""
    litellm_base_url: Annotated[
        str,
        "Optional API base url, e.g. a LiteLLM proxy at http://localhost:4000. "
        "No trailing slash. Leave blank for the provider's default endpoint.",
    ] = ""
    litellm_image_format: Annotated[
        str,
        "Image format sent to the model. Use 'png' for better compatibility.",
    ] = "webp"
    litellm_drop_params: Annotated[
        bool,
        "Silently drop request params a given provider does not support "
        "(e.g. response_format on models without structured output). Keep on "
        "unless you need a provider to hard-error on unsupported params.",
    ] = True

    def process_images(self, images: List[Image.Image]) -> List[dict]:
        """Encode images as OpenAI-style multimodal parts.

        LiteLLM normalizes this OpenAI content format to each provider's native
        schema, so the same payload works across providers.
        """
        if isinstance(images, Image.Image):
            images = [images]

        img_fmt = self.litellm_image_format
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

        completion_kwargs = self.completion_kwargs(timeout, response_schema)

        total_tries = max_retries + 1
        for tries in range(1, total_tries + 1):
            try:
                response = litellm.completion(messages=messages, **completion_kwargs)
                response_text = response.choices[0].message.content
                usage = getattr(response, "usage", None)
                total_tokens = getattr(usage, "total_tokens", 0) if usage else 0
                if block:
                    block.update_metadata(
                        llm_tokens_used=total_tokens, llm_request_count=1
                    )
                return json.loads(response_text)
            except RETRYABLE_ERRORS as e:
                if tries == total_tries:
                    logger.error(
                        f"Rate limit/transient error: {e}. Max retries reached. "
                        f"Giving up. (Attempt {tries}/{total_tries})",
                    )
                    break
                wait_time = tries * self.retry_wait_time
                logger.warning(
                    f"Rate limit/transient error: {e}. Retrying in {wait_time}s... "
                    f"(Attempt {tries}/{total_tries})",
                )
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"LiteLLM inference failed: {e}")
                break

        return {}

    def completion_kwargs(self, timeout: int, response_schema: type[BaseModel]) -> dict:
        """Assemble litellm.completion kwargs.

        Credentials are omitted when unset so LiteLLM falls back to the target
        provider's own environment variables.
        """
        kwargs = {
            "model": self.litellm_model,
            "timeout": timeout,
            "response_format": response_schema,
            "drop_params": self.litellm_drop_params,
        }
        if self.max_output_tokens:
            kwargs["max_tokens"] = self.max_output_tokens

        api_key = self.litellm_api_key or os.environ.get("LITELLM_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key

        api_base = self.litellm_base_url or os.environ.get("LITELLM_BASE_URL")
        if api_base:
            kwargs["api_base"] = api_base

        return kwargs
