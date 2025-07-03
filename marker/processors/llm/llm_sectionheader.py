import json
from typing import List, Tuple

from tqdm import tqdm

from marker.logger import get_logger
from marker.processors.llm import BaseLLMComplexBlockProcessor
from marker.schema import BlockTypes
from marker.schema.blocks import Block
from marker.schema.document import Document
from marker.schema.groups import PageGroup
from pydantic import BaseModel

logger = get_logger()


class LLMSectionHeaderProcessor(BaseLLMComplexBlockProcessor):
    page_prompt = """You're a text correction expert specializing in accurately analyzing complex PDF documents. You will be given a list of all of the section headers from a document, along with their page number and approximate dimensions.  The headers will be formatted like below, and will be presented in order.

```json
[
    {
        "bbox": [x1, y1, x2, y2],
        "width": x2 - x1,
        "height": y2 - y1,
        "page": 0,
        "id": "/page/0/SectionHeader/1",
        "html": "<h1>Introduction</h1>",
    }, ...
]
```

Bboxes have been normalized to 0-1000.

Your goal is to make sure that the section headers have the correct levels (h1, h2, h3, h4, h5, or h6).  If a section header does not have the right level, edit the html to fix it.

Guidelines:
- Edit the blocks to ensure that the section headers have the correct levels.
- Only edit the h1, h2, h3, h4, h5, and h6 tags.  Do not change any other tags or content in the headers.
- Only output the headers that changed (if nothing changed, output nothing).
- Every header you output needs to have one and only one level tag (h1, h2, h3, h4, h5, or h6).

**Instructions:**
1. Carefully examine the provided section headers and JSON.
2. Identify any changes you'll need to make, and write a short analysis.
3. Output "no_corrections", or "corrections_needed", depending on whether you need to make changes.
4. If corrections are needed, output any blocks that need updates.  Only output the block ids and html, like this:
        ```json
        [
            {
                "id": "/page/0/SectionHeader/1",
                "html": "<h2>Introduction</h2>"
            },
            ...
        ]
        ```

**Example:**
Input:
Section Headers
```json
[
    {
        "bbox": [x1, y1, x2, y2],
        "id": "/page/0/SectionHeader/1",
        "page": 0,
        "html": "1 Vector Operations",
    },
    {
        "bbox": [x1, y1, x2, y2],
        "id": "/page/0/SectionHeader/2",
        "page": 0,
        "html": "1.1 Vector Addition",
    },
]
```
Output:
Analysis: The first section header is missing the h1 tag, and the second section header is missing the h2 tag.
```json
[
    {
        "id": "/page/0/SectionHeader/1",
        "html": "<h1>1 Vector Operations</h1>"
    },
    {
        "id": "/page/0/SectionHeader/2",
        "html": "<h2>1.1 Vector Addition</h2>"
    }
]
```

**Input:**
Section Headers
```json
{{section_header_json}}
```
"""

    def get_selected_blocks(
        self,
        document: Document,
        page: PageGroup,
    ) -> List[dict]:
        selected_blocks = page.structure_blocks(document)
        json_blocks = [
            self.normalize_block_json(block, document, page, i)
            for i, block in enumerate(selected_blocks)
        ]
        return json_blocks

    def process_rewriting(
        self, document: Document, section_headers: List[Tuple[Block, dict]]
    ):
        section_header_json = [sh[1] for sh in section_headers]
        for item in section_header_json:
            _, _, page_id, block_type, block_id = item["id"].split("/")
            item["page"] = page_id
            item["width"] = item["bbox"][2] - item["bbox"][0]
            item["height"] = item["bbox"][3] - item["bbox"][1]
            del item["block_type"]  # Not needed, since they're all section headers

        prompt = self.page_prompt.replace(
            "{{section_header_json}}", json.dumps(section_header_json)
        )
        response = self.llm_service(
            prompt, None, document.pages[0], SectionHeaderSchema
        )
        logger.debug(f"Got section header reponse from LLM: {response}")

        if not response or "correction_type" not in response:
            logger.warning("LLM did not return a valid response")
            return

        correction_type = response["correction_type"]
        if correction_type == "no_corrections":
            return

        self.load_blocks(response)
        self.handle_rewrites(response["blocks"], document)

    def load_blocks(self, response):
        if isinstance(response["blocks"], str):
            response["blocks"] = json.loads(response["blocks"])

    def rewrite_blocks(self, document: Document):
        # Don't show progress if there are no blocks to process
        section_headers = [
            (block, self.normalize_block_json(block, document, page))
            for page in document.pages
            for block in page.structure_blocks(document)
            if block.block_type == BlockTypes.SectionHeader
        ]
        if len(section_headers) == 0:
            return

        pbar = tqdm(
            total=1,
            desc=f"Running {self.__class__.__name__}",
            disable=self.disable_tqdm,
        )

        self.process_rewriting(document, section_headers)
        pbar.update(1)
        pbar.close()


class BlockSchema(BaseModel):
    id: str
    html: str


class SectionHeaderSchema(BaseModel):
    analysis: str
    correction_type: str
    blocks: List[BlockSchema]
