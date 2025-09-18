import json
import os
from typing import Annotated, List

import PIL
from PIL import Image
import requests
from marker.logger import get_logger
from pydantic import BaseModel

from marker.schema.blocks import Block
from marker.services import BaseService

logger = get_logger()


class LlamaCPPService(BaseService):
    """
    LLM service implementation for llama.cpp server.
    
    This service connects to a remote llama.cpp server to process prompts with optional image inputs.
    The service uses the /v1/chat/completions endpoint for OpenAI compatibility and supports structured output via JSON schema.
    
    Environment Variables:
        LLAMA_CPP_BASE_URL: Base URL for the llama.cpp server (default: "http://192.168.68.186:8080")
        LLAMA_CPP_MODEL: Model name to use for inference (default: "NuMarkdown")
    """
    llama_cpp_base_url: Annotated[
        str, "The base url to use for llama-cpp. No trailing slash."
    ] = os.getenv("LLAMA_CPP_BASE_URL", "http://192.168.68.186:8080")
    llama_cpp_model: Annotated[str, "The model name to use for llama-cpp."] = (
        os.getenv("LLAMA_CPP_MODEL", "MiMo7b")
    )

    def process_images(self, images):
        # Use JPEG format for llama.cpp server compatibility
        image_bytes = [self.img_to_base64(img, "JPEG") for img in images]
        return image_bytes

    def __call__(
        self,
        prompt: str,
        image: Image.Image | List[Image.Image] | None,
        block: Block | None,
        response_schema: type[BaseModel],
        max_retries: int | None = None,
        timeout: int | None = None,
    ):
        url = f"{self.llama_cpp_base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        # Format images if provided
        image_bytes = self.format_image_for_llm(image)

        # Create messages for chat completion API
        if image_bytes:
            # For multimodal requests, use the content array format
            content = [{"type": "text", "text": prompt}]
            for image_data in image_bytes:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                })  # type: ignore
            messages = [{"role": "user", "content": content}]
        else:
            # For text-only requests, use simple string content
            messages = [{"role": "user", "content": prompt}]

        # Create the payload for the chat completion API
        payload = {
            "model": self.llama_cpp_model,
            "messages": messages,
            "stream": False,
            "response_format": {"type": "json_object"}
        }

        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout or self.timeout
            )
            response.raise_for_status()
            response_data = response.json()

            # Extract token usage information
            usage = response_data.get("usage", {})
            total_tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

            if block:
                block.update_metadata(llm_request_count=1, llm_tokens_used=total_tokens)

            # Extract the response content
            choices = response_data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "{}")
                # Try to parse as JSON, fallback to raw content if it's not valid JSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    logger.warning(f"Response content is not valid JSON: {content}")
                    return {}
            else:
                logger.warning("No choices in response")
                return {}
        except Exception as e:
            logger.warning(f"LlamaCPP inference failed: {e}")

        return {}
