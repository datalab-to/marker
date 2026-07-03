import json
import time
from typing import Annotated, List, TypeVar

import openai
import PIL
from marker.logger import get_logger
from openai import APITimeoutError, RateLimitError, BadRequestError
from PIL import Image
from pydantic import BaseModel

from marker.schema.blocks import Block
from marker.services import BaseService

logger = get_logger()

T = TypeVar("T")


class OpenAIService(BaseService):
    openai_base_url: Annotated[
        str, "The base url to use for OpenAI-like models.  No trailing slash."
    ] = "https://api.openai.com/v1"
    openai_model: Annotated[str, "The model name to use for OpenAI-like model."] = (
        "gpt-4o-mini"
    )
    openai_api_key: Annotated[
        str, "The API key to use for the OpenAI-like service."
    ] = None
    openai_image_format: Annotated[
        str,
        "The image format to use for the OpenAI-like service. Use 'png' for better compatability",
    ] = "webp"
    openai_disable_structured_output: Annotated[
        bool,
        "Disable Structured Outputs (response_format) for APIs that don't support it (e.g., DeepSeek).",
    ] = False

    def process_images(self, images: List[Image.Image]) -> List[dict]:
        """
        Generate the base-64 encoded message to send to an
        openAI-compatabile multimodal model.

        Args:
            images: Image or list of PIL images to include
            format: Format to use for the image; use "png" for better compatability.

        Returns:
            A list of OpenAI-compatbile multimodal messages containing the base64-encoded images.
        """
        if isinstance(images, Image.Image):
            images = [images]

        img_fmt = self.openai_image_format
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

    def validate_response(self, response_text: str, schema: type[T]) -> dict | None:
        """
        Parse JSON from plain text response.
        Used as fallback for APIs that don't support Structured Outputs.
        """
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        try:
            # Try to parse as JSON first
            out_schema = schema.model_validate_json(response_text)
            return out_schema.model_dump()
        except Exception:
            try:
                # Re-parse with fixed escapes
                escaped_str = response_text.replace("\\", "\\\\")
                out_schema = schema.model_validate_json(escaped_str)
                return out_schema.model_dump()
            except Exception:
                return None

    def _call_with_structured_output(
        self,
        client: openai.OpenAI,
        messages: list,
        response_schema: type[BaseModel],
        timeout: int,
    ) -> dict | None:
        """Call OpenAI API using Structured Outputs (response_format)."""
        response = client.beta.chat.completions.parse(
            extra_headers={
                "X-Title": "Marker",
                "HTTP-Referer": "https://github.com/datalab-to/marker",
            },
            model=self.openai_model,
            messages=messages,
            timeout=timeout,
            response_format=response_schema,
        )
        response_text = response.choices[0].message.content
        return json.loads(response_text), response.usage.total_tokens

    def _call_with_fallback(
        self,
        client: openai.OpenAI,
        messages: list,
        response_schema: type[BaseModel],
        timeout: int,
    ) -> dict | None:
        """
        Call OpenAI API using plain chat completions with schema in prompt.
        Fallback for APIs that don't support Structured Outputs.
        """
        schema_example = response_schema.model_json_schema()
        system_prompt = f"""Follow the instructions given by the user prompt. You must provide your response in JSON format matching this schema:

{json.dumps(schema_example, indent=2)}

Respond only with the JSON, nothing else. Do not include ```json, ```, or any other formatting."""

        # Prepend system message
        messages_with_system = [
            {"role": "system", "content": system_prompt},
            *messages,
        ]

        response = client.chat.completions.create(
            extra_headers={
                "X-Title": "Marker",
                "HTTP-Referer": "https://github.com/datalab-to/marker",
            },
            model=self.openai_model,
            messages=messages_with_system,
            timeout=timeout,
        )
        response_text = response.choices[0].message.content
        parsed = self.validate_response(response_text, response_schema)
        return parsed, response.usage.total_tokens

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

        # Track if we should use fallback mode
        use_fallback = self.openai_disable_structured_output

        total_tries = max_retries + 1
        for tries in range(1, total_tries + 1):
            try:
                if use_fallback:
                    result, total_tokens = self._call_with_fallback(
                        client, messages, response_schema, timeout
                    )
                else:
                    result, total_tokens = self._call_with_structured_output(
                        client, messages, response_schema, timeout
                    )

                if result is None:
                    logger.warning("LLM did not return a valid response")
                    return {}

                if block:
                    block.update_metadata(
                        llm_tokens_used=total_tokens, llm_request_count=1
                    )
                return result

            except BadRequestError as e:
                # Check if this is a "response_format not supported" error
                error_msg = str(e).lower()
                if (
                    not use_fallback
                    and "response_format" in error_msg
                    or "unavailable" in error_msg
                ):
                    logger.warning(
                        f"Structured Outputs not supported by this API, falling back to plain completions: {e}"
                    )
                    use_fallback = True
                    # Retry immediately with fallback
                    continue
                else:
                    logger.error(f"OpenAI inference failed: {e}")
                    break

            except (APITimeoutError, RateLimitError) as e:
                # Rate limit exceeded
                if tries == total_tries:
                    # Last attempt failed. Give up
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
                logger.error(f"OpenAI inference failed: {e}")
                break

        return {}

    def get_client(self) -> openai.OpenAI:
        return openai.OpenAI(api_key=self.openai_api_key, base_url=self.openai_base_url)
