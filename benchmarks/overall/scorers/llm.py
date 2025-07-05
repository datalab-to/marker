import json
import os
import tempfile
import time
from typing import List

from PIL import Image
from google.genai.errors import APIError
from google import genai
import pypdfium2 as pdfium

from benchmarks.overall.scorers import BaseScorer, BlockScores
from marker.settings import settings

rating_prompt = """
You're a document analysis expert who is comparing some markdown to an image to make sure the markdown is correct. You're rating how effectively the provided markdown represents the full text and formatting in the image provided.
You're given an image, along with the extracted markdown:
- Some parts of the page may have been recognized as images and linked from the markdown, like `![](_page_0_Picture_0.jpeg)`.
- Tables will be formatted as Github flavored markdown.
- Block equations will be in LaTeX.
- The image and markdown may be in any language.
- The markdown is based on the text extracted from the document, and sometimes the document may have had bad OCR applied to it, resulting in gibberish text.

The markdown should fully capture the meaning and formatting of the text in the image. You'll evaluate the markdown based on the image provided.

**Instructions**
Follow this process to evaluate the markdown:
1. Carefully examine the image.
2. Carefully examine the markdown input provided.
3. Compare the image to the markdown representation.  Does the markdown representation properly represent the important text and formatting in the image?
4. Assign component scores, as described below.

These are the primary scores:
- Overall - the overall quality of the markdown as compared to the image.
- Text quality - the quality of the text extraction from the image.
- Formatting quality - the quality of the formatting applied to the markdown, as compared to the image.

Depending on which elements are present in the markdown, you will assign element-specific scores.
- Tables - how effectively the tables have been extracted and formatted.
- Forms - how effectively the forms have extracted and formatted.
- Equations - how effectively block equations have been converted to LaTeX.
- Section headers - if all of the section headers have been detected, and the right levels set.
- Lists - if the lists have been properly extracted and formatted.
- Images - if images are identified and placed correctly.

Notes on scoring:
- To get a 5/5, all of the important text from the image must appear in the markdown, and the formatting should be correct (minor mistakes okay).  It's okay to omit some text that isn't important to the meaning, like page numbers and chapter headings.  If the entire page is an image, it's okay if the markdown is just a link to the image, unless the image would be better represented as text.
- A 3/5 may have small missing text elements from the markdown and/or moderate formatting issues.
- A 1/5 will have major missing text segments from the markdown or completely unreadable formatting.
- Use 0/5 if a field isn't applicable, like if the image doesn't contain a table.

If text that is important to the meaning of the document is missing, do not score higher than 3/5.

Output json, like in the example below.

**Example**
Input
```markdown
# Section 1
This is some *markdown* extracted from a document.  Here is a block equation:
$$\frac{ab \cdot x^5 + x^2 + 2 \cdot x + 123}{t}$$
```
Output
```json
{
    "image_description": "In the image, there is a section header 'Section 1', followed by some text and a block equation.",
    "markdown_description": "In the markdown, there is a section header 'Section 1', followed by some text and a block equation.",
    "comparison": "The text and formatting matches the image.  There are no formatting or text extraction issues.  The equations and section headers are correct.",
    "overall": 5,
    "text": 5,
    "formatting": 5,
    "section_headers": 5,
	"tables": 0,
	"forms": 0,
    "equations": 5,
	"lists": 0,
	"images": 0
}
```
**Input**
```markdown
{{markdown}}
```
**Output**
"""

comparison_keys = ["comparison"]
description_keys = ["image_description", "markdown_description"]
text_keys = comparison_keys + description_keys
score_keys = ["overall", "text", "formatting", "section_headers", "tables", "forms", "equations",
            "lists", "images"]


class LLMScorer(BaseScorer):
    def __call__(self, sample, gt_markdown: List[str], markdown: str) -> BlockScores:
        pdf_bytes = sample["pdf"]
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(pdf_bytes)
            f.flush()
            f.seek(0)
            doc = pdfium.PdfDocument(f.name)
            img = doc[0].render(scale=96/72).to_pil()
            doc.close()

        return self.llm_rater(img, markdown)


    def llm_rater(self, img: Image.Image, markdown: str) -> BlockScores:
        if not markdown:
            null_scores = {k: 1 for k in score_keys}
            text_scores = {k: "" for k in text_keys}
            null_scores.update(text_scores)
            return {
                "score": 1,
                "specific_scores": null_scores
            }
        req_keys = text_keys + score_keys
        properties = {}
        for key in req_keys:
            content_type = "INTEGER" if key in score_keys else "STRING"
            properties[key] = {"type": content_type}

        response_schema = {
            "required": req_keys,
            "properties": properties,
            "type": "OBJECT"
        }
        prompt = rating_prompt.replace("{{markdown}}", markdown)
        response = self.llm_response_wrapper([img, prompt], response_schema)
        assert all([k in response for k in req_keys]), f"Missing keys in response: {response}"
        return {
            "score": response["overall"],
            "specific_scores": response,
        }

    def llm_response_wrapper(self, prompt, response_schema, depth=0):
        client = genai.Client(
            http_options={"timeout": 60000},
            vertexai=True,
            project=os.getenv("VERTEX_PROJECT_ID"),
            location=os.getenv("VERTEX_LOCATION"),
        )
        try:
            responses = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "temperature": 0,
                    "response_schema": response_schema,
                    "response_mime_type": "application/json",
                },
            )
            output = responses.candidates[0].content.parts[0].text
            return json.loads(output)
        except APIError as e:
            print(f"Hit Gemini rate limit, waiting 120 seconds")
            time.sleep(120)
            if depth > 2:
                raise e
            return self.llm_response_wrapper(prompt, response_schema, depth + 1)