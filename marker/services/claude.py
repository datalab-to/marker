import json
import time
from typing import List, Annotated, T

import PIL
from PIL import Image
import anthropic
from anthropic import RateLimitError, APITimeoutError
from marker.logger import get_logger
from pydantic import BaseModel

from marker.schema.blocks import Block
from marker.services import BaseService

logger = get_logger()


class ClaudeService(BaseService):
    claude_model_name: Annotated[
        str, "The name of the Google model to use for the service."
    ] = "claude-3-7-sonnet-20250219"
    claude_api_key: Annotated[str, "The Claude API key to use for the service."] = None
    max_claude_tokens: Annotated[
        int, "The maximum number of tokens to use for a single Claude request."
    ] = 8192

    def process_images(self, images: List[Image.Image]) -> List[dict]:
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/webp",
                    "data": self.img_to_base64(img),
                },
            }
            for img in images
        ]

    def validate_response(self, response_text: str, schema: type[T]) -> T:
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        try:
            # Try to parse as JSON first
            out_schema = schema.model_validate_json(response_text)
            out_json = out_schema.model_dump()
            return out_json
        except Exception:
            try:
                # Re-parse with fixed escapes
                escaped_str = response_text.replace("\\", "\\\\")
                out_schema = schema.model_validate_json(escaped_str)
                return out_schema.model_dump()
            except Exception:
                return

    def get_client(self):
        return anthropic.Anthropic(
            api_key=self.claude_api_key,
        )

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

        schema_example = response_schema.model_json_schema()
        system_prompt = f"""
Follow the instructions given by the user prompt.  You must provide your response in JSON format matching this schema:

{json.dumps(schema_example, indent=2)}

Respond only with the JSON schema, nothing else.  Do not include ```json, ```,  or any other formatting.
""".strip()

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
                response = client.messages.create(
                    system=system_prompt,
                    model=self.claude_model_name,
                    max_tokens=self.max_claude_tokens,
                    messages=messages,
                    timeout=timeout,
                )
                # Extract and validate response
                response_text = response.content[0].text
                return self.validate_response(response_text, response_schema)
            except (RateLimitError, APITimeoutError) as e:
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
                logger.error(f"Error during Claude API call: {e}")
                break

        return {}
