import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Annotated

from marker.logger import get_logger
from marker.processors.llm import BaseLLMComplexBlockProcessor
from marker.schema import BlockTypes
from marker.schema.blocks import BlockId
from marker.schema.document import Document
from marker.schema.groups import PageGroup
from pydantic import BaseModel
from tqdm import tqdm

logger = get_logger()

FORMAT_TAGS = ["b", "i", "u", "del", "math", "sub", "sup", "a", "code", "p", "img"]
BLOCK_MAP = {
    "Text": [],
    "TextInlineMath": [],
    "Table": ["table", "tbody", "tr", "td", "th"],
    "ListGroup": ["ul", "li"],
    "SectionHeader": [],
    "Form": ["form", "input", "select", "textarea", "table", "tbody", "tr", "td", "th"],
    "Figure": [],
    "Picture": [],
    "Code": ["pre"],
    "TableOfContents": ["table", "tbody", "tr", "td", "th"],
}
ALL_TAGS = FORMAT_TAGS + [tag for tags in BLOCK_MAP.values() for tag in tags]


class LLMPageCorrectionProcessor(BaseLLMComplexBlockProcessor):
    block_correction_prompt: Annotated[
        str, "The user prompt to guide the block correction process."
    ] = None
    default_user_prompt = """Your goal is to reformat the blocks to be as correct as possible, without changing the underlying meaning of the text within the blocks.  Mostly focus on reformatting the content.  Ignore minor formatting issues like extra <i> tags."""
    page_prompt = """You're a text correction expert specializing in accurately reproducing text from PDF pages. You will be given a JSON list of blocks on a PDF page, along with the image for that page.  The blocks will be formatted like the example below.  The blocks will be presented in reading order.

```json
[
    {
        "bbox": [x1, y1, x2, y2],
        "id": "/page/0/Text/1",
        "block_type": "Text",
        "html": "<p>Some text here</p>",
    }, ...
]
```

You will also be given a prompt from the user that tells you how to correct the blocks.  Your task is to analyze the blocks and the image, then follow the prompt to correct the blocks.

Here are the types of changes you can make in response to the prompt:

- Reorder the blocks to reflect the correct reading order.
- Change the block type to the correct type - the potential types are "SectionHeader", "Form", "Text", "Table", "Figure", "Picture", "ListGroup", "PageFooter", "PageHeader", "Footnote", or "Equation".  In this case, update the html as well to match the new block type.
- Make edits to block content by changing the HTML.

Guidelines:
- Only use the following tags: {{format_tags}}.  Do not use any other tags.  
- The math tag can have the attribute `display="block"` to indicate display math, the a tag can have the attribute `href="..."` to indicate a link, and td and th tags can have the attribute `colspan="..."` and `rowspan="..."` to indicate table cells that span multiple columns or rows.  There can be a "block-type" attribute on p tags.  Do not use any other attributes.
- Keep LaTeX formulas inside <math> tags - these are important for downstream processing.
- Bboxes are normalized 0-1000
- The order of the JSON list is the reading order for the blocks
- Follow the user prompt faithfully, and only make additional changes if there is a significant issue with correctness.
- Stay faithful to the original image, and do not insert any content that is not present in the image or the blocks, unless specifically requested by the user prompt.

**Instructions:**
1. Carefully examine the provided JSON representation of the page, along with the image.
2. Analyze the user prompt.
3. Identify any issues you'll need to fix, and write a short analysis.
4. If everything is fine, output "no_corrections"  Otherwise, output the type of correction needed: ["reorder", "rewrite", "reorder_first"].  Rewrite includes rewriting html and changing the block type.  If you need to do both, then perform only the reordering, and output "reorder_first", so we can do the rewriting later.
5. If corrections are needed, output any blocks that need updates:
    a. If reading order needs to be changed, output the IDs of the blocks in the correct order, and keep block_type and html blank, like this:
    ```json
    [
        {
            "id": "/page/0/Text/1",
            "block_type": "",
            "html": ""
        },
        ...
    ]

    b. If blocks need to be rewritten, output the block ids and new HTML for the blocks, like this:
        ```json
        [
            {
                "id": "/page/0/Text/1",
                "block_type": "Text",
                "html": "<p>New HTML content here</p>"
            },
            ...
        ]
        ```

**Example:**
Input:
Blocks
```json
[
    {
        "bbox": [x1, y1, x2, y2],
        "id": "/page/0/Text/1",
        "block_type": "Text",
        "html": "1.14 Vector Operations",
    },
    {
        "bbox": [x1, y1, x2, y2],
        "id": "/page/0/Text/2",
        "block_type": "Text",
        "html": "<p>You can perform many operations on a vector, including...</p>",
    },
]
```
User Prompt
Ensure that all blocks have the correct labels, and that reading order is correct.
Output:
Analysis: The blocks are in the correct reading order, but the first block should actually be a SectionHeader.
```json
[
    {
        "id": "/page/0/Text/1",
        "block_type": "SectionHeader",
        "html": "<h1>1.14 Vector Operations</h1>"
    }
]
```

**Input:**
Blocks
```json
{{page_json}}
```
User Prompt
{{user_prompt}}
"""

    def get_selected_blocks(
        self,
        document: Document,
        page: PageGroup,
    ) -> List[dict]:
        selected_blocks = page.structure_blocks(document)
        json_blocks = [
            self.normalize_block_json(block, document, page)
            for i, block in enumerate(selected_blocks)
        ]
        return json_blocks

    def process_rewriting(self, document: Document, page1: PageGroup):
        page_blocks = self.get_selected_blocks(document, page1)
        image = page1.get_image(document, highres=False)

        prompt = (
            self.page_prompt.replace("{{page_json}}", json.dumps(page_blocks))
            .replace("{{format_tags}}", json.dumps(ALL_TAGS))
            .replace("{{user_prompt}}", self.block_correction_prompt)
        )
        response = self.llm_service(prompt, image, page1, PageSchema)
        logger.debug(f"Got reponse from LLM: {response}")

        if not response or "correction_type" not in response:
            logger.warning("LLM did not return a valid response")
            return

        correction_type = response["correction_type"]
        if correction_type == "no_corrections":
            return
        elif correction_type in ["reorder", "reorder_first"]:
            self.load_blocks(response)
            self.handle_reorder(response["blocks"], page1)

            # If we needed to reorder first, we will handle the rewriting next
            if correction_type == "reorder_first":
                self.process_rewriting(document, page1)
        elif correction_type == "rewrite":
            self.load_blocks(response)
            self.handle_rewrites(response["blocks"], document)
        else:
            logger.warning(f"Unknown correction type: {correction_type}")
            return

    def load_blocks(self, response):
        if isinstance(response["blocks"], str):
            response["blocks"] = json.loads(response["blocks"])

    def handle_reorder(self, blocks: list, page1: PageGroup):
        unique_page_ids = set()
        document_page_ids = [str(page1.page_id)]
        document_pages = [page1]

        for block_data in blocks:
            try:
                page_id, _, _ = block_data["id"].split("/")
                unique_page_ids.add(page_id)
            except Exception as e:
                logger.debug(f"Error parsing block ID {block_data['id']}: {e}")
                continue

        if set(document_page_ids) != unique_page_ids:
            logger.debug(
                "Some page IDs in the response do not match the document's pages"
            )
            return

        for page_id, document_page in zip(unique_page_ids, document_pages):
            block_ids_for_page = []
            for block_data in blocks:
                try:
                    page_id, block_type, block_id = block_data["id"].split("/")
                    block_id = BlockId(
                        page_id=page_id,
                        block_id=block_id,
                        block_type=getattr(BlockTypes, block_type),
                    )
                    block_ids_for_page.append(block_id)
                except Exception as e:
                    logger.debug(f"Error parsing block ID {block_data['id']}: {e}")
                    continue

                # Both sides should have the same values, just be reordered
                if not all(
                    [
                        block_id in document_page.structure
                        for block_id in block_ids_for_page
                    ]
                ):
                    logger.debug(
                        f"Some blocks for page {page_id} not found in document"
                    )
                    continue

                if not all(
                    [
                        block_id in block_ids_for_page
                        for block_id in document_page.structure
                    ]
                ):
                    logger.debug(
                        f"Some blocks in document page {page_id} not found in response"
                    )
                    continue

                # Swap the order of blocks in the document page
                document_page.structure = block_ids_for_page

    def handle_rewrites(self, blocks: list, document: Document):
        for block_data in blocks:
            try:
                block_id = block_data["id"].strip().lstrip("/")
                _, page_id, block_type, block_id = block_id.split("/")
                block_id = BlockId(
                    page_id=page_id,
                    block_id=block_id,
                    block_type=getattr(BlockTypes, block_type),
                )
                block = document.get_block(block_id)
                if not block:
                    logger.debug(f"Block {block_id} not found in document")
                    continue

                if hasattr(block, "html"):
                    block.html = block_data["html"]
            except Exception as e:
                logger.debug(f"Error parsing block ID {block_data['id']}: {e}")
                continue

    def rewrite_blocks(self, document: Document):
        if not self.block_correction_prompt:
            return

        # Don't show progress if there are no blocks to process
        total_blocks = len(document.pages)
        if total_blocks == 0:
            return

        pbar = tqdm(
            total=max(1, total_blocks - 1),
            desc=f"{self.__class__.__name__} running",
            disable=self.disable_tqdm,
        )

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            for future in as_completed(
                [
                    executor.submit(self.process_rewriting, document, page)
                    for page in document.pages
                ]
            ):
                future.result()  # Raise exceptions if any occurred
                pbar.update(1)

        pbar.close()


class BlockSchema(BaseModel):
    id: str
    html: str
    block_type: str


class PageSchema(BaseModel):
    analysis: str
    correction_type: str
    blocks: List[BlockSchema]
