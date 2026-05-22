from typing import Annotated, List

from pydantic import BaseModel

from marker.processors.llm import BaseLLMSimpleBlockProcessor, BlockData, PromptData
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.settings import settings


class LLMImageContextDescriptionProcessor(BaseLLMSimpleBlockProcessor):
    """
    Generate image descriptions that incorporate page context — the page's
    extracted markdown text and the page image are sent alongside the figure
    image, so the LLM can describe the figure with awareness of its
    surrounding content. Produces a short caption (used as alt text) and a
    long description (rendered as a hidden HTML comment).

    Independent of --use_llm's existing LLMImageDescriptionProcessor and
    compatible with extract_images either on or off.
    """
    block_types = (
        BlockTypes.Picture,
        BlockTypes.Figure,
    )
    llm_image_context: Annotated[
        bool,
        "Enable context-aware LLM image descriptions (sends page markdown + page image alongside the figure).",
    ] = False
    image_context_prompt: Annotated[
        str,
        "The prompt used for context-aware image descriptions.",
    ] = """You are a document analysis expert. You are given three things:
1. The extracted text of a page, in markdown form. The figure you must describe is referenced inline by its filename `{filename}`.
2. An image of the full page (first attached image).
3. An image of just the figure to describe (second attached image).

Your task: write a description of the figure in the context of the page.
- Be faithful to what is actually shown. Include all pertinent numeric data, axis labels, legends, titles, and visible text.
- Tie the figure to the surrounding text on the page when relevant (e.g. "supports the section's claim that X").
- Do not invent details that are not present in the image.

Return two fields:
- `short_caption`: a single-sentence alt-text-style caption (≤ 140 characters).
- `long_description`: a complete description suitable for a text-only model that cannot see the image. Include the relevant page context.

**Page markdown:**
```
{page_markdown}
```
"""

    def inference_blocks(self, document: Document) -> List[BlockData]:
        if not self.llm_image_context:
            return []
        return super().inference_blocks(document)

    def _build_page_markdown(self, document: Document, page, target_block) -> str:
        """
        Walk the page's structure in order, emitting raw text of each block.
        Where the target figure sits, emit a markdown image reference using
        the deterministic output filename so the LLM has an explicit anchor.
        """
        image_ext = settings.OUTPUT_IMAGE_FORMAT.lower()
        parts = []
        structure = page.structure or []
        for block_id in structure:
            block = document.get_block(block_id)
            if block is None or block.ignore_for_output:
                continue
            if block.id == target_block.id:
                parts.append(f"![]({block.id.to_path()}.{image_ext})")
                continue
            if block.block_type in (BlockTypes.Picture, BlockTypes.Figure):
                parts.append(f"![]({block.id.to_path()}.{image_ext})")
                continue
            text = block.raw_text(document).strip()
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    def block_prompts(self, document: Document) -> List[PromptData]:
        prompt_data: List[PromptData] = []
        image_ext = settings.OUTPUT_IMAGE_FORMAT.lower()
        for block_data in self.inference_blocks(document):
            block = block_data["block"]
            page = block_data["page"]

            page_markdown = self._build_page_markdown(document, page, block)
            filename = f"{block.id.to_path()}.{image_ext}"
            prompt = self.image_context_prompt.format(
                filename=filename, page_markdown=page_markdown
            )

            page_image = page.get_image(highres=False)
            figure_image = self.extract_image(document, block)

            prompt_data.append(
                {
                    "prompt": prompt,
                    "image": [page_image, figure_image],
                    "block": block,
                    "schema": ImageContextSchema,
                    "page": page,
                }
            )

        return prompt_data

    def rewrite_block(
        self, response: dict, prompt_data: PromptData, document: Document
    ):
        block = prompt_data["block"]

        if not response or "long_description" not in response:
            block.update_metadata(llm_error_count=1)
            return

        long_description = response.get("long_description", "")
        short_caption = response.get("short_caption", "")

        if len(long_description) < 10:
            block.update_metadata(llm_error_count=1)
            return

        block.long_description = long_description
        block.short_caption = short_caption or long_description[:140]


class ImageContextSchema(BaseModel):
    short_caption: str
    long_description: str
