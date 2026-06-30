import json
import time
import traceback
from io import BytesIO
from typing import List, Annotated, Mapping

from PIL import Image
from google import genai
from google.genai import types
from google.genai.errors import APIError
from marker.logger import get_logger
from pydantic import BaseModel

from marker.schema.blocks import Block
from marker.services import BaseService

logger = get_logger()


class BaseGeminiService(BaseService):
    gemini_model_name: Annotated[
        str, "The name of the Google model to use for the service."
    ] = "gemini-2.0-flash"
    thinking_budget: Annotated[
        int, "The thinking token budget to use for the service."
    ] = None

    def img_to_bytes(self, img: Image.Image):
        image_bytes = BytesIO()
        img.save(image_bytes, format="WEBP")
        return image_bytes.getvalue()

    def get_google_client(self, timeout: int):
        raise NotImplementedError

    def process_images(self, images):
        image_parts = [
            types.Part.from_bytes(data=self.img_to_bytes(img), mime_type="image/webp")
            for img in images
        ]
        return image_parts

    def __call__(
        self,
        prompt: str,
        image: Image.Image | List[Image.Image] | None,
        block: Block | None,
        response_schema: type[BaseModel],
        max_retries: int | None = None,
        timeout: int | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ):
        if max_retries is None:
            max_retries = self.max_retries

        if timeout is None:
            timeout = self.timeout

        client = self.get_google_client(timeout=timeout)
        image_parts = self.format_image_for_llm(image)

        total_tries = max_retries + 1
        temperature = 0
        for tries in range(1, total_tries + 1):
            config = {
                "temperature": temperature,
                "response_schema": response_schema,
                "response_mime_type": "application/json",
            }
            if self.max_output_tokens:
                config["max_output_tokens"] = self.max_output_tokens

            if self.thinking_budget is not None:
                # For gemini models, we can optionally set a thinking budget in the config
                config["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=self.thinking_budget
                )

            try:
                responses = client.models.generate_content(
                    model=self.gemini_model_name,
                    contents=image_parts
                    + [
                        prompt
                    ],  # According to gemini docs, it performs better if the image is the first element
                    config=config,
                )
                output = responses.candidates[0].content.parts[0].text
                total_tokens = responses.usage_metadata.total_token_count
                if block:
                    block.update_metadata(
                        llm_tokens_used=total_tokens, llm_request_count=1
                    )
                return json.loads(output)
            except APIError as e:
                if e.code in [429, 443, 503]:
                    # Rate limit exceeded
                    if tries == total_tries:
                        # Last attempt failed. Give up
                        logger.error(
                            f"APIError: {e}. Max retries reached. Giving up. (Attempt {tries}/{total_tries})",
                        )
                        break
                    else:
                        wait_time = tries * self.retry_wait_time
                        logger.warning(
                            f"APIError: {e}. Retrying in {wait_time} seconds... (Attempt {tries}/{total_tries})",
                        )
                        time.sleep(wait_time)
                else:
                    logger.error(f"APIError: {e}")
                    break
            except json.JSONDecodeError as e:
                temperature = 0.2  # Increase temperature slightly to try and get a different respons

                # The response was not valid JSON
                if tries == total_tries:
                    # Last attempt failed. Give up
                    logger.error(
                        f"JSONDecodeError: {e}. Max retries reached. Giving up. (Attempt {tries}/{total_tries})",
                    )
                    break
                else:
                    logger.warning(
                        f"JSONDecodeError: {e}. Retrying... (Attempt {tries}/{total_tries})",
                    )
            except Exception as e:
                logger.error(f"Exception: {e}")
                traceback.print_exc()
                break

        return {}


class GoogleGeminiService(BaseGeminiService):
    gemini_api_key: Annotated[str, "The Google API key to use for the service."] = None

    def get_google_client(self, timeout: int):
        return genai.Client(
            api_key=self.gemini_api_key,
            http_options={"timeout": timeout * 1000},  # Convert to milliseconds
        )
