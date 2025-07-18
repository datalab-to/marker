import json
import time
import traceback
from collections import deque
from enum import Enum
from io import BytesIO
from threading import Lock
from typing import List, Annotated

import PIL
from google import genai
from google.genai import types
from google.genai.errors import APIError
from marker.logger import get_logger
from pydantic import BaseModel

from marker.schema.blocks import Block
from marker.services import BaseService

logger = get_logger()


class GeminiModel(str, Enum):
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    DEFAULT = "gemini-2.5-flash"


# Rate limiting settings
MODEL_LIMITS = {
    GeminiModel.GEMINI_2_5_PRO: {"rpm": 5},
    GeminiModel.GEMINI_2_5_FLASH: {"rpm": 10},
    GeminiModel.DEFAULT: {"rpm": 10},  # Corresponds to gemini-2.5-flash
}

# Global request tracker and lock
REQUEST_TIMESTAMPS = {model: deque() for model in GeminiModel}
RATE_LIMIT_LOCK = Lock()


class BaseGeminiService(BaseService):
    gemini_model_name: Annotated[
        GeminiModel, "The name of the Google model to use for the service."
    ] = GeminiModel.DEFAULT

    def img_to_bytes(self, img: PIL.Image.Image):
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

        # Proactive rate limiting
        with RATE_LIMIT_LOCK:
            model_name = self.gemini_model_name
            rpm_limit = MODEL_LIMITS.get(model_name, {"rpm": 10})["rpm"]
            request_history = REQUEST_TIMESTAMPS[model_name]

            current_time = time.time()
            # Remove timestamps older than 60 seconds
            while request_history and current_time - request_history[0] > 60:
                request_history.popleft()

            if len(request_history) >= rpm_limit:
                wait_time = 60 - (current_time - request_history[0])
                if wait_time > 0:
                    logger.warning(
                        f"RPM limit for {model_name} reached. Waiting for {wait_time:.2f} seconds."
                    )
                    time.sleep(wait_time)

            # Record the new request timestamp
            request_history.append(time.time())

        client = self.get_google_client(timeout=timeout)
        image_parts = self.format_image_for_llm(image)

        total_tries = max_retries + 1
        for tries in range(1, total_tries + 1):
            try:
                responses = client.models.generate_content(
                    model=self.gemini_model_name,
                    contents=image_parts
                    + [
                        prompt
                    ],  # According to gemini docs, it performs better if the image is the first element
                    config={
                        "temperature": 0,
                        "response_schema": response_schema,
                        "response_mime_type": "application/json",
                    },
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
