import markdown2
from pydantic import BaseModel
from marker.processors.llm import PromptData, BaseLLMSimpleBlockProcessor, BlockData

from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.util import strip_code_fence

from typing import Annotated, List


class LLMHandwritingProcessor(BaseLLMSimpleBlockProcessor):
    block_types = (BlockTypes.Handwriting, BlockTypes.Text)
    handwriting_generation_prompt: Annotated[
        str,
        "The prompt to use for OCRing handwriting.",
        "Default is a string containing the Gemini prompt."
    ] = """You are an expert editor specializing in accurately reproducing text from images.
You will receive an image of a text block. Your task is to generate markdown to properly represent the content of the image.  Do not omit any text present in the image - make sure everything is included in the markdown representation.  The markdown representation should be as faithful to the original image as possible.

Formatting should be in markdown, with the following rules:
- * for italics, ** for bold, and ` for inline code.
- Headers should be formatted with #, with one # for the largest header, and up to 6 for the smallest.
- Lists should be formatted with either - or 1. for unordered and ordered lists, respectively.
- Links should be formatted with [text](url).
- Use ``` for code blocks.
- Inline math should be formatted with <math>math expression</math>.
- Display math should be formatted with <math display="block">math expression</math>.
- Values and labels should be extracted from forms, and put into markdown tables, with the labels on the left side, and values on the right.  The headers should be "Labels" and "Values".  Other text in the form can appear between the tables.
- Tables should be formatted with markdown tables, with the headers bolded.

**Instructions:**
1. Carefully examine the provided block image.
2. Output the markdown representing the content of the image.
"""

    def inference_blocks(self, document: Document) -> List[BlockData]:
        blocks = super().inference_blocks(document)
        out_blocks = []
        for block_data in blocks:
            raw_text = block_data["block"].raw_text(document)
            block = block_data["block"]

            # Don't process text blocks that contain lines already
            if block.block_type == BlockTypes.Text:
                lines = block.contained_blocks(document, (BlockTypes.Line,))
                if len(lines) > 0 or len(raw_text.strip()) > 0:
                    continue
            out_blocks.append(block_data)
        return out_blocks


    def block_prompts(self, document: Document) -> List[PromptData]:
        prompt_data = []
        for block_data in self.inference_blocks(document):
            block = block_data["block"]
            prompt = self.handwriting_generation_prompt
            image = self.extract_image(document, block)

            prompt_data.append({
                "prompt": prompt,
                "image": image,
                "block": block,
                "schema": HandwritingSchema,
                "page": block_data["page"]
            })
        return prompt_data

    def rewrite_block(self, response: dict, prompt_data: PromptData, document: Document):
        block = prompt_data["block"]
        raw_text = block.raw_text(document)

        if not response or "markdown" not in response:
            block.update_metadata(llm_error_count=1)
            return

        markdown = response["markdown"]
        if len(markdown) < len(raw_text) * .5:
            block.update_metadata(llm_error_count=1)
            return

        markdown = strip_code_fence(markdown, "markdown")
        block.html = markdown2.markdown(markdown, extras=["tables"])

class HandwritingSchema(BaseModel):
    markdown: str
