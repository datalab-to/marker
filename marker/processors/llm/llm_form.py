from typing import List

from pydantic import BaseModel

from marker.output import json_to_html
from marker.processors.llm import PromptData, BaseLLMSimpleBlockProcessor, BlockData
from marker.processors.llm.llm_utils import (
    strip_code_fences,
    string_indicates_no_corrections,
)

from marker.schema import BlockTypes
from marker.schema.document import Document


class LLMFormProcessor(BaseLLMSimpleBlockProcessor):
    block_types = (BlockTypes.Form,)
    form_rewriting_prompt = """You are a text correction expert specializing in accurately reproducing text from images.
You will receive an image of a text block and an html representation of the form in the image.
Your task is to correct any errors in the html representation, and format it properly.
Values and labels should appear in html tables, with the labels on the left side, and values on the right.  Other text in the form can appear between the tables.  Only use the tags `table, p, span, i, b, th, td, tr, and div`.  Do not omit any text from the form - make sure everything is included in the html representation.  It should be as faithful to the original form as possible.
**Instructions:**
1. Carefully examine the provided form block image.
2. Analyze the html representation of the form.
3. Compare the html representation to the image.
4. Output a single JSON object (and only JSON) matching this schema:
    - `comparison`: short string
    - `correction_needed`: boolean
    - `corrected_html`: corrected HTML string (empty when `correction_needed` is false)
5. If the html representation is correct, or you cannot read the image properly, set `correction_needed` to false and `corrected_html` to "".
6. If the html representation contains errors, set `correction_needed` to true and provide the corrected HTML in `corrected_html`.
**Example:**
Input:
```html
<table>
    <tr>
        <td>Label 1</td>
        <td>Label 2</td>
        <td>Label 3</td>
    </tr>
    <tr>
        <td>Value 1</td>
        <td>Value 2</td>
        <td>Value 3</td>
    </tr>
</table> 
```
Output:
```json
{
  "comparison": "The html representation has the labels in the first row and the values in the second row. It should be corrected to have the labels on the left side and the values on the right side.",
  "correction_needed": true,
  "corrected_html": "<table><tr><td>Label 1</td><td>Value 1</td></tr><tr><td>Label 2</td><td>Value 2</td></tr><tr><td>Label 3</td><td>Value 3</td></tr></table>"
}
```
**Input:**
```html
{block_html}
```
"""

    def inference_blocks(self, document: Document) -> List[BlockData]:
        blocks = super().inference_blocks(document)
        out_blocks = []
        for block_data in blocks:
            block = block_data["block"]
            children = block.contained_blocks(document, (BlockTypes.TableCell,))
            if not children:
                continue
            out_blocks.append(block_data)
        return out_blocks

    def block_prompts(self, document: Document) -> List[PromptData]:
        prompt_data = []
        for block_data in self.inference_blocks(document):
            block = block_data["block"]
            block_html = json_to_html(block.render(document))
            prompt = self.form_rewriting_prompt.replace("{block_html}", block_html)
            image = self.extract_image(document, block)
            prompt_data.append(
                {
                    "prompt": prompt,
                    "image": image,
                    "block": block,
                    "schema": FormSchema,
                    "page": block_data["page"],
                }
            )
        return prompt_data

    def rewrite_block(
        self, response: dict, prompt_data: PromptData, document: Document
    ):
        block = prompt_data["block"]
        block_html = json_to_html(block.render(document))

        if not response:
            block.update_metadata(llm_error_count=1)
            return

        correction_needed = response.get("correction_needed", None)
        corrected_html = response.get("corrected_html", "") or ""

        # specifically checking for False
        if correction_needed is False:
            return

        # Fallback if LLM did not adhere to schema but used phases like: "No corrections needed."
        if string_indicates_no_corrections(corrected_html):
            return

        if not corrected_html:
            block.update_metadata(llm_error_count=1)
            return

        # The original table is okay
        if string_indicates_no_corrections(corrected_html):
            return

        # Potentially a partial response
        if len(corrected_html) < len(block_html) * 0.33:
            block.update_metadata(llm_error_count=1)
            return

        corrected_html = strip_code_fences(corrected_html)
        block.html = corrected_html


class FormSchema(BaseModel):
    comparison: str = ""
    correction_needed: bool = False
    corrected_html: str = ""
