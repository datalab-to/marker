import json
from typing import Annotated, List

import PIL
import requests
from marker.logger import get_logger
from pydantic import BaseModel

from marker.schema.blocks import Block
from marker.services import BaseService

logger = get_logger()


class OllamaService(BaseService):
    ollama_base_url: Annotated[
        str, "The base url to use for ollama.  No trailing slash."
    ] = "http://localhost:11434"
    ollama_model: Annotated[str, "The model name to use for ollama."] = (
        "llama3.2-vision"
    )

    def process_images(self, images):
        image_bytes = [self.img_to_base64(img) for img in images]
        return image_bytes

    def __call__(
        self,
        prompt: str,
        image: PIL.Image.Image | List[PIL.Image.Image] | None,
        block: Block | None,
        response_schema: type[BaseModel],
        max_retries: int | None = None,
        timeout: int | None = None,
    ):
        url = f"{self.ollama_base_url}/api/generate"
        headers = {"Content-Type": "application/json"}

        schema = response_schema.model_json_schema()
        format_schema = {
            "type": "object",
            "properties": schema["properties"],
            "required": schema["required"],
        }

        image_bytes = self.format_image_for_llm(image)

        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "format": format_schema,
            "images": image_bytes,
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()

            total_tokens = (
                response_data["prompt_eval_count"] + response_data["eval_count"]
            )

            if block:
                block.update_metadata(llm_request_count=1, llm_tokens_used=total_tokens)

            data = response_data["response"]
            return json.loads(data)
        except Exception as e:
            logger.warning(f"Ollama inference failed: {e}")

        return {}
