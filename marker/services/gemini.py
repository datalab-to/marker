import json
import time
from io import BytesIO
from typing import List, Annotated

import PIL
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel

import httpx
from marker.schema.blocks import Block
from marker.services import BaseService

class BaseGeminiService(BaseService):
    gemini_model_name: Annotated[
        str,
        "The name of the Google model to use for the service."
    ] = "gemini-2.0-flash"

    def img_to_bytes(self, img: PIL.Image.Image):
        image_bytes = BytesIO()
        img.save(image_bytes, format="WEBP")
        return image_bytes.getvalue()

    def get_google_client(self, timeout: int):
        raise NotImplementedError

    def __call__(
            self,
            prompt: str,
            image: PIL.Image.Image | List[PIL.Image.Image],
            block: Block,
            response_schema: type[BaseModel],
            max_retries: int | None = None,
            timeout: int | None = None
    ):
        if max_retries is None:
            max_retries = self.max_retries

        if timeout is None:
            timeout = self.timeout

        if not isinstance(image, list):
            image = [image]
        print('ssssss', flush=True)
        client = self.get_google_client(timeout=timeout)
        image_parts = [types.Part.from_bytes(data=self.img_to_bytes(img), mime_type="image/webp") for img in image]

        tries = 0
        while tries < max_retries:
            try:
                responses = client.models.generate_content(
                    model=self.gemini_model_name,
                    contents=image_parts + [prompt], # According to gemini docs, it performs better if the image is the first element
                    config={
                        "temperature": 0,
                        "response_schema": response_schema,
                        "response_mime_type": "application/json",
                    },
                )
                output = responses.candidates[0].content.parts[0].text
                total_tokens = responses.usage_metadata.total_token_count
                block.update_metadata(llm_tokens_used=total_tokens, llm_request_count=1)
                return json.loads(output)
            except APIError as e:
                if e.code in [429, 443, 503]:
                    # Rate limit exceeded
                    tries += 1
                    wait_time = tries * 3
                    print(f"APIError: {e}. Retrying in {wait_time} seconds... (Attempt {tries}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(e)
                    break
            except Exception as e:
                print(e)
                break

        return {}


class GoogleGeminiService(BaseGeminiService):
    gemini_api_key: Annotated[
        str,
        "The Google API key to use for the service."
    ] = 'AIzaSyCguwG1QrgHvrbJwT5g5S7IIs3yZTUOaEg'

    # import os
    # os.environ["HTTP_PROXY"] = "http://172.21.0.16:7890"
    # os.environ["HTTPS_PROXY"] = "http://172.21.0.16:7890"

    def get_google_client(self, timeout: int):
        return genai.Client(
            api_key=self.gemini_api_key,
            http_options={"timeout": timeout * 1000}, # Convert to milliseconds
        )
