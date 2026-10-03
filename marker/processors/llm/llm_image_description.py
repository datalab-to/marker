from pydantic import BaseModel

from marker.processors.llm import PromptData, BaseLLMSimpleBlockProcessor, BlockData

from marker.schema import BlockTypes
from marker.schema.document import Document

from typing import Annotated, List


class LLMImageDescriptionProcessor(BaseLLMSimpleBlockProcessor):
    block_types = (
        BlockTypes.Picture,
        BlockTypes.Figure,
        BlockTypes.Diagram,
    )
    extract_images: Annotated[bool, "Extract images from the document."] = True
    image_description_prompt: Annotated[
        str,
        "The prompt to use for generating image descriptions.",
        "Default is a string containing the Gemini prompt.",
    ] = """You are a document analysis expert who specializes in creating text descriptions for images.
You will receive an image of a picture or figure.  Your job will be to create a short description of the image.
**Instructions:**
1. Carefully examine the provided image.
2. Analyze any text that was extracted from within the image.
3. Output a faithful description of the image.  Make sure there is enough specific detail to accurately reconstruct the image.  If the image is a figure or contains numeric data, include the numeric data in the output.
4. Output the description in the same language as the input text. If the input text is in French, write the description in French. If the input text is in German, write the description in German, and so on. Only use English if the input text is in English or if there is no input text.
**Example:**
Input:
```text
"Fruit Preference Survey"
20, 15, 10
Apples, Bananas, Oranges
```
Output:
In this figure, a bar chart titled "Fruit Preference Survey" is showing the number of people who prefer different types of fruits.  The x-axis shows the types of fruits, and the y-axis shows the number of people.  The bar chart shows that most people prefer apples, followed by bananas and oranges.  20 people prefer apples, 15 people prefer bananas, and 10 people prefer oranges.
**Input:**
```text
{raw_text}
```
"""

    def inference_blocks(self, document: Document) -> List[BlockData]:
        blocks = super().inference_blocks(document)
        if self.extract_images:
            return []
        return blocks

    def block_prompts(self, document: Document) -> List[PromptData]:
        prompt_data = []
        for block_data in self.inference_blocks(document):
            block = block_data["block"]
            page = block_data["page"]
            raw_text = block.raw_text(document)
            # Include surrounding page text for language context if image has little text
            if len(raw_text.strip()) < 20:
                page_text = page.raw_text(document)[:200]
                if page_text.strip():
                    raw_text = raw_text + "\n\n[Surrounding document text for language context:]\n" + page_text
            prompt = self.image_description_prompt.replace(
                "{raw_text}", raw_text
            )
            image = self.extract_image(document, block)

            prompt_data.append(
                {
                    "prompt": prompt,
                    "image": image,
                    "block": block,
                    "schema": ImageSchema,
                    "page": block_data["page"],
                }
            )

        return prompt_data

    def rewrite_block(
        self, response: dict, prompt_data: PromptData, document: Document
    ):
        block = prompt_data["block"]

        if not response or "image_description" not in response:
            block.update_metadata(llm_error_count=1)
            return

        image_description = response["image_description"]
        if len(image_description) < 10:
            block.update_metadata(llm_error_count=1)
            return

        block.description = image_description


class ImageSchema(BaseModel):
    image_description: str
