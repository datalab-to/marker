"""Groq LLM service for marker.

Groq provides fast inference with an OpenAI-compatible API.
See https://console.groq.com/docs/quickstart for available models.
"""

from typing import Annotated

from marker.services.openai import OpenAIService


class GroqService(OpenAIService):
    """
    Groq LLM service using OpenAI-compatible API.
    
    Groq offers extremely fast inference for various LLMs including
    Llama, Mixtral, and Gemma models.
    
    Usage:
        marker_single document.pdf --use_llm \\
            --llm_service marker.services.groq.GroqService \\
            --groq_api_key YOUR_API_KEY \\
            --groq_model llama-3.3-70b-versatile
    """
    
    groq_api_key: Annotated[
        str, "The API key for Groq. Get one at https://console.groq.com"
    ] = None
    groq_model: Annotated[
        str, 
        "The Groq model to use. Options include: llama-3.3-70b-versatile, "
        "llama-3.1-8b-instant, mixtral-8x7b-32768, gemma2-9b-it"
    ] = "llama-3.3-70b-versatile"
    groq_image_format: Annotated[
        str,
        "The image format to use for Groq. Use 'png' for better compatibility.",
    ] = "png"
    
    # Groq uses OpenAI-compatible API endpoint
    openai_base_url: str = "https://api.groq.com/openai/v1"
    
    # Groq doesn't support Structured Outputs, so always use fallback
    openai_disable_structured_output: bool = True
    
    def __init__(self, config=None):
        super().__init__(config)
        # Map groq-specific config to openai config
        if self.groq_api_key:
            self.openai_api_key = self.groq_api_key
        if self.groq_model:
            self.openai_model = self.groq_model
        if self.groq_image_format:
            self.openai_image_format = self.groq_image_format
