from pydantic import BaseModel
from marker.processors.llm import PromptData, BaseLLMSimpleBlockProcessor, BlockData
from marker.schema import BlockTypes
from marker.schema.document import Document
from typing import Annotated, List

class LLMStrikethroughProcessor(BaseLLMSimpleBlockProcessor):
    block_types = (BlockTypes.Text, BlockTypes.ListGroup)
    min_text_length: Annotated[
        int,
        "Minimum text length to consider for strikethrough detection",
    ] = 3

    strikethrough_detection_prompt = """You're a text correction expert specializing in accurately detecting and reproducing strikethrough text from images.
You will receive an image and an HTML representation of text. Your task is to detect any strikethrough text in the image and correct the HTML to properly represent it.

Strikethrough text appears with a horizontal line through the middle of the text. This formatting indicates that the text has been deleted, cancelled, or is no longer valid.

# Guidelines:
- Carefully examine the image for any text that has a line through it
- Use the <del> HTML tag to mark strikethrough text
- Keep the text content exactly as it appears, only adding strikethrough markup
- If no strikethrough text is detected, respond with "No corrections needed."

# Instructions
1. Carefully examine the provided text block image
2. Compare it with the HTML representation
3. Write an analysis of whether any text appears with strikethrough formatting
4. Set corrections_needed to true if strikethrough text is found, false otherwise
5. If corrections_needed is true, provide the corrected HTML with <del> tags

# Example
Input:
```html
<p>This price is $100 but now it's $75.</p>
```
Output (if $100 has a line through it):
analysis: The text "$100" appears with a strikethrough line in the image, indicating it's the old price that has been replaced.
corrections_needed: true
corrected_html: <p>This price is <del>$100</del> but now it's $75.</p>

# Input
```html
{text_html}
```
"""

    def inference_blocks(self, document: Document) -> List[BlockData]:
        blocks = super().inference_blocks(document)
        out_blocks = []
        for block_data in blocks:
            block = block_data["block"]
            raw_text = block.raw_text(document)

            # Skip blocks that are too short.
            if len(raw_text.strip()) < self.min_text_length:
                continue

            out_blocks.append(block_data)
        return out_blocks

    def block_prompts(self, document: Document) -> List[PromptData]:
        prompt_data = []
        for block_data in self.inference_blocks(document):
            block = block_data["block"]
            text = block.html if block.html else block.raw_text(document)

            prompt = self.strikethrough_detection_prompt.replace("{text_html}", text)
            image = self.extract_image(document, block)

            prompt_data.append({
                "prompt": prompt,
                "image": image,
                "block": block,
                "schema": StrikethroughSchema,
                "page": block_data["page"]
            })
        return prompt_data

    def rewrite_block(self, response: dict, prompt_data: PromptData, document: Document):
        block = prompt_data["block"]

        if not response:
            block.update_metadata(llm_error_count=1)
            return

        # No corrections needed.
        if not response.get("corrections_needed", False):
            return

        # Sanity check.
        corrected_html = response.get("corrected_html")
        if not corrected_html:
            block.update_metadata(llm_error_count=1)
            return

        # Ensure balanced tags.
        balanced_tags = corrected_html.count("<del") == corrected_html.count("</del>")
        if not balanced_tags:
            block.update_metadata(llm_error_count=1)
            return

        # Ensure the new HTML actually contains strikethrough tags.
        if "<del>" not in corrected_html:
            block.update_metadata(llm_error_count=1)
            return

        block.html = corrected_html

class StrikethroughSchema(BaseModel):
    analysis: str
    corrections_needed: bool
    corrected_html: str
