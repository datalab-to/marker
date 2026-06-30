import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Annotated, Literal, cast

from marker.logger import get_logger
from marker.processors.llm import BaseLLMComplexBlockProcessor
from marker.processors.llm.llm_utils import (
    inject_analysis_prompt,
    strip_code_fences,
    string_indicates_no_corrections,
)
from marker.schema import BlockTypes
from marker.schema.blocks import BlockId
from marker.schema.document import Document
from marker.schema.groups import PageGroup
from marker.telemetry import build_marker_trace_headers
from pydantic import BaseModel, Field
from tqdm import tqdm

logger = get_logger()

CorrectionType = Literal["reorder", "rewrite", "reorder_first"]

def _normalize_correction_type(value: str | None) -> CorrectionType | None:
    if not value:
        return None
    lowered = value.strip().lower()
    lowered = lowered.replace("-", "_").replace(" ", "_")
    if lowered in ("reorder", "rewrite", "reorder_first"):
        return cast(CorrectionType, lowered)
    return None


def _parse_block_id_parts(block_id: str) -> tuple[str, str, str] | None:
    parts = block_id.strip().lstrip("/").split("/")
    # if block_id starts with "/page/", see schema.blocks.base.BlockId class
    if len(parts) == 4 and parts[0] == "page":
        _, page_id, block_type, block_num = parts
        return page_id, block_type, block_num
    if len(parts) == 3:
        page_id, block_type, block_num = parts
        return page_id, block_type, block_num
    return None


@dataclass(frozen=True)
class _NormalizedPageCorrectionResponse:
    correction_needed: bool
    correction_type: CorrectionType | None
    blocks: list


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
    analysis_style: Annotated[
        str,
        "How to structure the LLM analysis field: 'summary', 'deep' or 'auto'.",
    ] = "summary"
    block_correction_prompt: Annotated[
        str | None,
        "The user prompt to guide the block correction process. If None, will use `default_user_prompt`.",
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

- ("reorder") Reorder the blocks to reflect the correct reading order.
- ("rewrite") Change the block type to the correct type - the potential types are "SectionHeader", "Form", "Text", "Table", "Figure", "Picture", "ListGroup", "PageFooter", "PageHeader", "Footnote", or "Equation".  In this case, update the html as well to match the new block type.
- ("rewrite") Make edits to block content by changing the HTML.

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
3. {{analysis_instruction}}
4. Output a single JSON object (and only JSON) matching this schema:
    {{analysis_schema}}
    - `correction_needed`: boolean
    - `correction_type`: one of ["reorder", "rewrite", "reorder_first", None] (omit or set null when `correction_needed` is false). "rewrite" includes rewriting html and changing the block type. If you need to do both "rewrite" and "reorder", then perform only the reordering, and output "reorder_first", so we can do the rewriting later.
    - `blocks`: array of objects with `id`, `block_type`, and `html` (only include the blocks that need updates)
5. If `correction_needed` is false, set `blocks` to an empty array.
6. If `correction_needed` is true, output any blocks that need updates:
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
```json
{
  "analysis": "The blocks are in the correct reading order, but the first block should actually be a SectionHeader.",
  "correction_needed": true,
  "correction_type": "rewrite",
  "blocks": [
    {
      "id": "/page/0/Text/1",
      "block_type": "SectionHeader",
      "html": "<h1>1.14 Vector Operations</h1>"
    }
  ]
}
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
            for block in selected_blocks
        ]
        return json_blocks

    def process_rewriting(self, document: Document, page1: PageGroup):
        if self.llm_service is None:
            raise ValueError("LLM service is not configured")

        page_blocks = self.get_selected_blocks(document, page1)
        image = page1.get_image(document, highres=False)

        prompt_template = inject_analysis_prompt(self.page_prompt, self.analysis_style)
        prompt = (
            prompt_template.replace("{{page_json}}", json.dumps(page_blocks))
            .replace("{{format_tags}}", json.dumps(ALL_TAGS))
            .replace(
                "{{user_prompt}}",
                self.block_correction_prompt or self.default_user_prompt,
            )
        )
        headers = build_marker_trace_headers(
            source_path=document.filepath,
            processor=self.__class__.__name__,
            block_id=str(page1.id),
            page_id=page1.page_id,
        )
        response = self.llm_service(
            prompt, image, page1, PageSchema, extra_headers=headers
        )
        logger.debug(f"Got reponse from LLM: {response}")

        normalized = self._normalize_response(response)
        if normalized is None:
            logger.warning("LLM did not return a valid response")
            return

        if not normalized.correction_needed:
            return

        if not normalized.blocks:
            return

        correction_type = normalized.correction_type
        if correction_type is None:
            correction_type = self._infer_correction_type_from_blocks(normalized.blocks)
            if correction_type is None:
                logger.warning("Unable to determine correction type from LLM response")
                return

        if correction_type in ["reorder", "reorder_first"]:
            self.handle_reorder(normalized.blocks, page1)

            if correction_type == "reorder_first":
                self.process_rewriting(document, page1)
        elif correction_type == "rewrite":
            self.handle_rewrites(normalized.blocks, document)
        else:
            logger.warning(f"Unknown correction type: {correction_type}")
            return

    def _normalize_response(self, response) -> _NormalizedPageCorrectionResponse | None:
        if response is None:
            return None

        if isinstance(response, str):
            text = strip_code_fences(response)
            if string_indicates_no_corrections(text):
                return _NormalizedPageCorrectionResponse(
                    correction_needed=False, correction_type=None, blocks=[]
                )
            try:
                response = json.loads(text)
            except Exception:
                return None

        if isinstance(response, list):
            correction_type = self._infer_correction_type_from_blocks(response)
            return _NormalizedPageCorrectionResponse(
                correction_needed=len(response) > 0,
                correction_type=correction_type,
                blocks=response,
            )

        if not isinstance(response, dict):
            return None

        blocks = response.get("blocks", [])
        if isinstance(blocks, str):
            try:
                blocks = json.loads(strip_code_fences(blocks))
            except Exception:
                blocks = []

        correction_needed = response.get("correction_needed", None)
        correction_type = _normalize_correction_type(
            response.get("correction_type", None)
        )
        if correction_needed is None:
            raw_type = str(response.get("correction_type", "") or "")
            if string_indicates_no_corrections(raw_type):
                correction_needed = False
            else:
                correction_needed = bool(correction_type) or bool(blocks)

        if not bool(correction_needed):
            correction_type = None
            blocks = []

        return _NormalizedPageCorrectionResponse(
            correction_needed=bool(correction_needed),
            correction_type=correction_type,
            blocks=blocks if isinstance(blocks, list) else [],
        )

    def _infer_correction_type_from_blocks(self, blocks: list) -> CorrectionType | None:
        if not isinstance(blocks, list) or not blocks:
            return None

        # Reorder responses are a list of IDs with empty block_type/html.
        is_reorder = True
        for block in blocks:
            if not isinstance(block, dict):
                is_reorder = False
                break
            if (block.get("block_type") or "").strip() or (
                block.get("html") or ""
            ).strip():
                is_reorder = False
                break

        if is_reorder:
            return "reorder"

        return "rewrite"

    def handle_reorder(self, blocks: list, page1: PageGroup):
        expected_page_id = str(page1.page_id)
        parsed_ids: list[BlockId] = []
        response_page_ids: set[str] = set()

        for block_data in blocks:
            try:
                parts = _parse_block_id_parts(block_data["id"])
                if parts is None:
                    continue
                page_id, block_type, block_num = parts
                response_page_ids.add(str(page_id))
                parsed_ids.append(
                    # NOTE: not sure if doing int() may break
                    BlockId(
                        page_id=int(page_id),
                        block_id=int(block_num),
                        block_type=getattr(BlockTypes, block_type),
                    )
                )
            except Exception as e:
                logger.debug(f"Error parsing block ID {block_data.get('id')}: {e}")
                continue

        if response_page_ids != {expected_page_id}:
            logger.debug(
                "Some page IDs in the response do not match the document's page"
            )
            return

        if not parsed_ids:
            return

        if set(parsed_ids) != set(page1.structure or []):
            logger.debug(
                "Reorder response does not contain the same blocks as the page"
            )
            return

        page1.structure = parsed_ids

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
    analysis: str = ""
    correction_needed: bool = False
    correction_type: CorrectionType | None = None
    blocks: List[BlockSchema] = Field(default_factory=list)
