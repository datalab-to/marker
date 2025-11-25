from pydantic import BaseModel

from marker.processors.llm import BaseLLMSimpleBlockProcessor, PromptData, BlockData
from marker.schema import BlockTypes
from marker.schema.document import Document

from typing import Annotated, List


class LLMEquationProcessor(BaseLLMSimpleBlockProcessor):
    block_types = (BlockTypes.Equation,)
    min_equation_height: Annotated[
        float,
        "The minimum ratio between equation height and page height to consider for processing.",
     ] = 0.06
    image_expansion_ratio: Annotated[
        float,
        "The ratio to expand the image by when cropping.",
    ] = 0.05 # Equations sometimes get bboxes that are too tight
    redo_inline_math: Annotated[
        bool,
        "Whether to redo inline math blocks.",
    ] = False
    equation_latex_prompt: Annotated[
        str,
        "The prompt to use for generating LaTeX from equations.",
        "Default is a string containing the Gemini prompt."
    ] = r"""You're an expert mathematician who is good at writing LaTeX code and html for equations.
You'll receive an image of a math block, along with the text extracted from the block.  It may contain one or more equations. Your job is to write html that represents the content of the image, with the equations in LaTeX format.

Some guidelines:
- Output valid html, where all the equations can render properly.
- Use <math display="block"> as a block equation delimiter and <math> for inline equations.  Do not use $ or $$ as delimiters.
- Keep the LaTeX code inside the math tags simple, concise, and KaTeX compatible.
- Enclose all equations in the correct math tags. Use multiple math tags inside the html to represent multiple equations.
- Only use the html tags math, i, b, p, and br.
- Make sure to include all the equations in the image in the html output.
- Make sure to include other text in the image in the correct positions along with the equations.

**Instructions:**
1. Carefully examine the provided image.
2. Analyze the existing html, which may include LaTeX code.
3. Write a short analysis of how the html should be corrected to represent the image.
4. If the html and LaTeX are correct, write "No corrections needed."
5. If the html and LaTeX are incorrect, generate the corrected html.
6. Output only the analysis, then the corrected html or "No corrections needed."
**Example:**
Input:
```html
The following equation illustrates the Pythagorean theorem:
x2 + y2 = z2

And this equation is a bit more complex:
(ab * x5 + x2 + 2 * x + 123)/t
```
Output:
analysis: The equations are not formatted as LaTeX, or enclosed in math tags.
```html
<p>The following equation illustrates the Pythagorean theorem:</p> 
<math display="block">x^{2} + y^{2} = z^{2}</math>

<p>And this equation is a bit more complex, and contains <math>ab \cdot x^{5}</math>:</p>
<math display="block">\frac{ab \cdot x^{5} + x^{2} + 2 \cdot x + 123}{t}</math>
```
**Input:**
```html
{equation}
```
"""

    def inference_blocks(self, document: Document) -> List[BlockData]:
        if self.llm_tables_only:
            return []
        blocks = super().inference_blocks(document)
        out_blocks = []
        for block_data in blocks:
            block = block_data["block"]
            page = block_data["page"]

            # If we redo inline math, we redo all equations
            if all([
                block.polygon.height / page.polygon.height < self.min_equation_height,
                not self.redo_inline_math
            ]):
                continue
            out_blocks.append(block_data)
        return out_blocks

    def block_prompts(self, document: Document) -> List[PromptData]:
        prompt_data = []
        for block_data in self.inference_blocks(document):
            block = block_data["block"]
            text = block.html if block.html else block.raw_text(document)
            prompt = self.equation_latex_prompt.replace("{equation}", text)
            image = self.extract_image(document, block)

            prompt_data.append({
                "prompt": prompt,
                "image": image,
                "block": block,
                "schema": EquationSchema,
                "page": block_data["page"]
            })

        return prompt_data


    def rewrite_block(self, response: dict, prompt_data: PromptData, document: Document):
        block = prompt_data["block"]
        text = block.html if block.html else block.raw_text(document)

        if not response or "corrected_equation" not in response:
            block.update_metadata(llm_error_count=1)
            return

        html_equation = response["corrected_equation"]

        if "no corrections needed" in html_equation.lower():
            return

        balanced_tags = html_equation.count("<math") == html_equation.count("</math>")
        if not all([
            html_equation,
            balanced_tags,
            len(html_equation) > len(text) * .3,
        ]):
            block.update_metadata(llm_error_count=1)
            return

        block.html = html_equation

class EquationSchema(BaseModel):
    analysis: str
    corrected_equation: str