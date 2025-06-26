from __future__ import annotations

from typing import List, Sequence, Union

from pydantic import BaseModel

from marker.schema import BlockTypes
from marker.schema.blocks import Block, BlockId, BlockOutput
from marker.schema.groups.page import PageGroup


class DocumentOutput(BaseModel):
    children: List[BlockOutput]
    html: str
    block_type: BlockTypes = BlockTypes.Document


class TocItem(BaseModel):
    title: str
    heading_level: int
    page_id: int
    polygon: List[List[float]]


class Document(BaseModel):
    filepath: Union[str | bytes]
    pages: List[PageGroup]
    block_type: BlockTypes = BlockTypes.Document
    table_of_contents: List[TocItem] | None = None
    debug_data_path: str | None = None # Path that debug data was saved to

    def get_block(self, block_id: BlockId):
        page = self.get_page(block_id.page_id)
        block = page.get_block(block_id)
        if block:
            return block
        return None

    def get_page(self, page_id):
        for page in self.pages:
            if page.page_id == page_id:
                return page
        return None

    def get_next_block(self, block: Block, ignored_block_types: List[BlockTypes] = None):
        if ignored_block_types is None:
            ignored_block_types = []
        next_block = None

        # Try to find the next block in the current page
        page = self.get_page(block.page_id)
        next_block = page.get_next_block(block, ignored_block_types)
        if next_block:
            return next_block

        # If no block found, search subsequent pages
        for page in self.pages[self.pages.index(page) + 1:]:
            next_block = page.get_next_block(None, ignored_block_types)
            if next_block:
                return next_block
        return None

    def get_next_page(self, page: PageGroup):
        page_idx = self.pages.index(page)
        if page_idx + 1 < len(self.pages):
            return self.pages[page_idx + 1]
        return None

    def get_prev_block(self, block: Block):
        page = self.get_page(block.page_id)
        prev_block = page.get_prev_block(block)
        if prev_block:
            return prev_block
        prev_page = self.get_prev_page(page)
        if not prev_page:
            return None
        return prev_page.get_block(prev_page.structure[-1])
    
    def get_prev_page(self, page: PageGroup):
        page_idx = self.pages.index(page)
        if page_idx > 0:
            return self.pages[page_idx - 1]
        return None

    def assemble_html(self, child_blocks: List[Block]):
        template = ""
        for c in child_blocks:
            template += f"<content-ref src='{c.id}'></content-ref>"
        return template

    def render(self):
        child_content = []
        section_hierarchy = None
        for page in self.pages:
            rendered = page.render(self, None, section_hierarchy)
            section_hierarchy = rendered.section_hierarchy.copy()
            child_content.append(rendered)

        return DocumentOutput(
            children=child_content,
            html=self.assemble_html(child_content)
        )

    def contained_blocks(self, block_types: Sequence[BlockTypes] = None) -> List[Block]:
        blocks = []
        for page in self.pages:
            blocks += page.contained_blocks(self, block_types)
        return blocks
