import json
import re
from functools import lru_cache
from html import unescape
from dataclasses import dataclass
from typing import List, Tuple, Annotated

from tqdm import tqdm
from transformers import AutoTokenizer

from marker.logger import get_logger
from marker.processors.llm import BaseLLMComplexBlockProcessor
from marker.processors.llm.llm_utils import (
    inject_analysis_prompt,
    strip_code_fences,
    string_indicates_no_corrections,
)
from marker.schema import BlockTypes
from marker.schema.blocks import Block
from marker.schema.document import Document
from marker.schema.groups import PageGroup
from marker.telemetry import build_marker_trace_headers
from pydantic import BaseModel, Field

logger = get_logger()


@dataclass(frozen=True)
class _NormalizedSectionHeaderResponse:
    correction_needed: bool
    blocks: list
    score: int = 5


class LLMSectionHeaderProcessor(BaseLLMComplexBlockProcessor):
    analysis_style: Annotated[
        str,
        "How to structure the LLM analysis field: 'summary', 'deep' or 'auto'. "
        "'summary' is brief, 'deep' is more detailed, 'auto' lets the LLM decide."
        "'deep' is HIGHLY recommended for best results, at the expense of more tokens and e2e latency.",
    ] = "summary"
    max_rewrite_retries: Annotated[
        int,
        "How many times to retry section header correction when confidence is low. "
        "Increase this to improve results at the cost of more LLM calls. "
        "Defaults to 0 to preserve existing marker behavior.",
    ] = 0
    max_chunk_tokens: Annotated[
        int,
        "Max prompt tokens per chunk (0 disables chunking). "
        "Highly recommended to enable chunking for large documents. "
        "Defaults to 0 to preserve marker's existing behavior.",
    ] = 0
    chunk_tokenizer_hf_model_id: Annotated[
        str,
        "HuggingFace tokenizer model id used for chunking token counts (best-effort).",
    ] = ""
    neighbor_text_max_blocks: Annotated[
        int,
        "Max number of neighboring text blocks to include before/after each header. "
        "Recommended to try 1-2 along with chunking (`max_chunk_tokens`) for better context. "
        "Defaults to 0 to preserve existing marker behavior.",
    ] = 0
    neighbor_text_max_chars: Annotated[
        int,
        "Max chars of neighboring text to include per side (before/after).",
    ] = 250
    recent_headers_max_count: Annotated[
        int,
        "How many recently-processed headers to include as context for later chunks. "
        "If no chunking is used (`max_chunk_tokens == 0`), this has no effect. "
        "Defaults to 0 to preserve existing marker behavior.",
    ] = 0
    page_prompt = """You're a text correction expert specializing in accurately analyzing complex PDF documents. You will be given a list of all of the section headers from a document, along with their page number and approximate dimensions.  The headers will be formatted like below, and will be presented in order.

```json
[
    {
        "bbox": [x1, y1, x2, y2],
        "width": x2 - x1,
        "height": y2 - y1,
        "line_height": 12.5,
        "prev_text": "…",
        "next_text": "…",
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
- Only include the headers that changed in the `blocks` array (if nothing changed, set `correction_needed` to false and `blocks` to []).
- Every header you output needs to have one and only one level tag (h1, h2, h3, h4, h5, or h6).
- Use layout cues to infer hierarchy: larger `height`/`line_height` and smaller left indent (`bbox[0]`) indicate higher-level headings.
- Headings with similar `line_height`/`height` and left indent are peers; assign them the same h-level even if their current tags differ.
- Avoid skipping levels (e.g., h2 -> h4). Keep nesting consistent across the document.

Notes:
- Additional fields such as `line_height`, `line_height_norm`, or `left` may be provided; use them when available.
- `prev_text` and `next_text` will be provided and represent nearby document content; you must pay attention to them to 
    disambiguate header depth. This is especially important as the section headers themselves may come with a lot of errors.

**Instructions:**
1. Carefully examine the provided section headers and JSON.
2. {{analysis_instruction}}
3. Output a single JSON object (and only JSON) matching this schema:
    {{analysis_schema}}
    - `correction_needed`: boolean
    - `blocks`: array of objects with `id` and `html` (only include changed headers)
    - `score`: integer 1-5 indicating confidence in the corrected header levels (5 = fully confident)
        IMPORTANT: you must output `score` last, only after writing ALL the blocks.
4. If `correction_needed` is false, set `blocks` to an empty array.
5. Only correct the headers in the "Section Headers" JSON. Use "Recent Headers" for context only.

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
```json
{
  "analysis": "The first section header is missing the h1 tag, and the second section header is missing the h2 tag.",
  "correction_needed": true,
  "blocks": [
    {
      "id": "/page/0/SectionHeader/1",
      "html": "<h1>1 Vector Operations</h1>"
    },
    {
      "id": "/page/0/SectionHeader/2",
      "html": "<h2>1.1 Vector Addition</h2>"
    }
  ],
  "score": 4
}
```

**Input:**
Recent Headers (already processed; for context only)
```json
{{recent_headers_json}}
```

Section Headers
```json
{{section_header_json}}
```
"""

    @staticmethod
    def _round_1dp(value: float) -> float:
        return round(float(value), 1)

    @staticmethod
    def _extract_h_level(html: str) -> int | None:
        match = re.search(r"<h([1-6])\\b", html or "", flags=re.IGNORECASE)
        if not match:
            return None
        try:
            return int(match.group(1))
        except Exception:  # noqa: BLE001 - best effort
            return None

    @staticmethod
    def _html_to_compact_text(html: str) -> str:
        text = re.sub(r"<[^>]+>", " ", html or "")
        text = unescape(text)
        text = re.sub(r"\\s+", " ", text).strip()
        return text

    @staticmethod
    @lru_cache(maxsize=8)
    def _load_tokenizer(hf_model_id: str):
        return AutoTokenizer.from_pretrained(
            hf_model_id, use_fast=True, trust_remote_code=True
        )

    def _get_tokenizer(self):
        candidates = [self.chunk_tokenizer_hf_model_id.strip()] if self.chunk_tokenizer_hf_model_id else []
        candidates.extend(
            [
                "Qwen/Qwen3-8B",
                "Qwen/Qwen2.5-7B-Instruct",
            ]
        )
        for model_id in [c for c in candidates if c]:
            try:
                return self._load_tokenizer(model_id)
            except Exception as exc:  # noqa: BLE001 - best effort
                logger.warning(
                    "Unable to load tokenizer %s for chunking: %s", model_id, exc
                )
        return None

    def _count_prompt_tokens(self, text: str) -> int:
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return len(text) // 4
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:  # noqa: BLE001 - best effort
            return len(text) // 4

    def _page_structure_index(self, page: PageGroup) -> dict[str, int]:
        structure = page.structure or []
        return {str(block_id): idx for idx, block_id in enumerate(structure)}

    def _get_neighbor_text(
        self,
        document: Document,
        page: PageGroup,
        block: Block,
        *,
        direction: int,
        page_index: dict[str, int],
    ) -> str:
        if self.neighbor_text_max_blocks <= 0 or self.neighbor_text_max_chars <= 0:
            return ""
        structure = page.structure or []
        pos = page_index.get(str(block.id))
        if pos is None:
            return ""

        collected: list[str] = []
        i = pos + direction
        while (
            0 <= i < len(structure)
            and len(collected) < self.neighbor_text_max_blocks
        ):
            neighbor = page.get_block(structure[i])
            if neighbor is None:
                i += direction
                continue
            if neighbor.block_type in (BlockTypes.Text, BlockTypes.ListItem):
                neighbor_json = self.normalize_block_json(neighbor, document, page)
                neighbor_text = self._html_to_compact_text(
                    neighbor_json.get("html", "")
                )
                if neighbor_text:
                    collected.append(neighbor_text)
            i += direction

        joined = " ".join(collected).strip()
        if len(joined) > self.neighbor_text_max_chars:
            return joined[: self.neighbor_text_max_chars - 1].rstrip() + "…"
        return joined

    def _build_section_header_item(
        self,
        document: Document,
        page: PageGroup,
        block: Block,
        raw_item: dict,
        *,
        page_index: dict[str, int],
    ) -> dict:
        item = dict(raw_item)
        _, _, page_id, block_type, block_id = item["id"].split("/")
        item["page"] = page_id
        item["bbox"] = [self._round_1dp(v) for v in item["bbox"]]
        item["width"] = self._round_1dp(item["bbox"][2] - item["bbox"][0])
        item["height"] = self._round_1dp(item["bbox"][3] - item["bbox"][1])
        item["left"] = item["bbox"][0]

        line_height = block.line_height(document)
        item["line_height"] = self._round_1dp(line_height)
        page_height = page.polygon.height if page else 0
        item["line_height_norm"] = self._round_1dp(
            (line_height / page_height) * 1000 if page_height else 0
        )

        item["prev_text"] = self._get_neighbor_text(
            document, page, block, direction=-1, page_index=page_index
        )
        item["next_text"] = self._get_neighbor_text(
            document, page, block, direction=1, page_index=page_index
        )

        item.pop("block_type", None)  # Not needed, since they're all section headers
        return item

    def _build_recent_headers_context(
        self,
        document: Document,
        section_header_blocks: list[Block],
        *,
        max_count: int,
    ) -> list[dict]:
        if max_count <= 0:
            return []
        recent: list[dict] = []
        for block in section_header_blocks[-max_count:]:
            page = document.get_page(block.page_id)
            if page is None:
                continue
            block_json = self.normalize_block_json(block, document, page)
            html = block_json.get("html", "")
            recent.append(
                {
                    "id": str(block.id),
                    "html": html,
                    "level": self._extract_h_level(html),
                }
            )
        return recent

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

    def process_rewriting(
        self, document: Document, section_headers: List[Tuple[Block, dict]]
    ):
        if self.llm_service is None:
            raise ValueError("LLM service is not configured")
        prompt_template = inject_analysis_prompt(self.page_prompt, self.analysis_style)
        max_attempts = max(0, int(self.max_rewrite_retries)) + 1
        max_chunk_tokens = int(self.max_chunk_tokens or 0)

        page_indexes: dict[int, dict[str, int]] = {}
        for page in document.pages:
            if page.page_id is None:
                continue
            page_indexes[page.page_id] = self._page_structure_index(page)

        processed_blocks: list[Block] = []
        cursor = 0
        chunk_idx = 0

        while cursor < len(section_headers):
            chunk_idx += 1
            recent_ctx = self._build_recent_headers_context(
                document, processed_blocks, max_count=int(self.recent_headers_max_count)
            )

            chunk_blocks: list[Block] = []
            chunk_items: list[dict] = []

            while cursor < len(section_headers):
                block, raw_item = section_headers[cursor]
                if block.page_id is None:
                    cursor += 1
                    continue
                page = document.get_page(block.page_id)
                if page is None:
                    cursor += 1
                    continue
                if page.page_id is None:
                    page_index = self._page_structure_index(page)
                else:
                    page_index = page_indexes.get(page.page_id) or self._page_structure_index(page)
                    page_indexes[page.page_id] = page_index

                candidate = self._build_section_header_item(
                    document, page, block, raw_item, page_index=page_index
                )
                candidate_items = [*chunk_items, candidate]
                prompt = (
                    prompt_template.replace(
                        "{{recent_headers_json}}", json.dumps(recent_ctx)
                    )
                    .replace("{{section_header_json}}", json.dumps(candidate_items))
                )
                tokens = self._count_prompt_tokens(prompt)
                if max_chunk_tokens > 0 and tokens > max_chunk_tokens:
                    if not chunk_items:
                        chunk_items.append(candidate)
                        chunk_blocks.append(block)
                        cursor += 1
                    break

                chunk_items.append(candidate)
                chunk_blocks.append(block)
                cursor += 1

            if not chunk_blocks:
                continue

            for attempt in range(1, max_attempts + 1):
                refreshed_items: list[dict] = []
                for block in chunk_blocks:
                    if block.page_id is None:
                        continue
                    page = document.get_page(block.page_id)
                    if page is None:
                        continue
                    if page.page_id is None:
                        page_index = self._page_structure_index(page)
                    else:
                        page_index = page_indexes.get(page.page_id) or self._page_structure_index(page)
                        page_indexes[page.page_id] = page_index
                    raw_item = self.normalize_block_json(block, document, page)
                    refreshed_items.append(
                        self._build_section_header_item(
                            document, page, block, raw_item, page_index=page_index
                        )
                    )

                prompt = (
                    prompt_template.replace(
                        "{{recent_headers_json}}", json.dumps(recent_ctx)
                    )
                    .replace("{{section_header_json}}", json.dumps(refreshed_items))
                )

                headers = build_marker_trace_headers(
                    source_path=document.filepath,
                    processor=self.__class__.__name__,
                    block_id=str(document.pages[0].id),
                    page_id=document.pages[0].page_id,
                    extra={
                        "SectionHeaderCount": len(refreshed_items),
                        "SectionHeaderChunkIndex": chunk_idx,
                        "SectionHeaderAttempt": attempt,
                    },
                )
                response = self.llm_service(
                    prompt,
                    None,
                    document.pages[0],
                    SectionHeaderSchema,
                    extra_headers=headers,
                )
                logger.debug(
                    f"section header prompt sent to LLM: {prompt}\n"
                    f"Got section header reponse from LLM: {response}"
                )

                normalized = self._normalize_response(response)
                if normalized is None:
                    logger.warning("LLM did not return a valid response")
                    return

                if normalized.correction_needed and normalized.blocks:
                    self.handle_rewrites(normalized.blocks, document)

                if normalized.score >= 5 or attempt >= max_attempts:
                    break

                logger.info(
                    "Section header rewriting low score %s on attempt %s/%s; retrying chunk %s.",
                    normalized.score,
                    attempt,
                    max_attempts,
                    chunk_idx,
                )

            processed_blocks.extend(chunk_blocks)

    def _normalize_response(self, response) -> _NormalizedSectionHeaderResponse | None:
        if response is None:
            return None

        if isinstance(response, str):
            text = strip_code_fences(response)
            if string_indicates_no_corrections(text):
                return _NormalizedSectionHeaderResponse(
                    correction_needed=False, blocks=[]
                )
            try:
                response = json.loads(text)
            except Exception:
                return None

        if isinstance(response, list):
            return _NormalizedSectionHeaderResponse(
                correction_needed=len(response) > 0, blocks=response, score=5
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
        if correction_needed is None:
            correction_needed = bool(blocks)

        score = response.get("score", 5)
        try:
            score = int(score)
        except (TypeError, ValueError):
            score = 5
        score = max(1, min(5, score))

        return _NormalizedSectionHeaderResponse(
            correction_needed=bool(correction_needed),
            blocks=blocks if isinstance(blocks, list) else [],
            score=score,
        )

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
    analysis: str = ""
    correction_needed: bool = False
    blocks: List[BlockSchema] = Field(default_factory=list)
    score: int = 5