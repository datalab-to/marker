from typing import Annotated

from google import genai

from marker.services.gemini import BaseGeminiService

class GoogleVertexService(BaseGeminiService):
    vertex_project_id: Annotated[
        str,
        "Google Cloud Project ID for Vertex AI.",
    ] = None
    vertex_location: Annotated[
        str,
        "Google Cloud Location for Vertex AI.",
    ] = "us-central1"
    gemini_model_name: Annotated[
        str,
        "The name of the Google model to use for the service."
    ] = "gemini-2.5-flash"
    vertex_dedicated: Annotated[
        bool,
        "Whether to use a dedicated Vertex AI instance."
    ] = False

    def get_google_client(self, timeout: int):
        http_options = {"timeout": timeout * 1000} # Convert to milliseconds
        if self.vertex_dedicated:
            http_options["headers"] = {"x-vertex-ai-llm-request-type": "dedicated"}
        return genai.Client(
            vertexai=True,
            project=self.vertex_project_id,
            location=self.vertex_location,
            http_options=http_options,
        )