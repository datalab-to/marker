import re
from typing import Annotated

from marker.builders import BaseBuilder
from marker.schema import BlockTypes
from marker.schema.blocks import ListItem, Text
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.registry import get_block_class

# Bullets, numbers ("1.", "1)"), letters ("a.", "A)"), roman numerals
LIST_ITEM_START_PATTERN = re.compile(
    r"^\s*(?:[•●○ഠ ം◦■▪▫–—-]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)]|\(?[ivxlcIVXLC]{1,5}[.)])\s"
)


class StructureBuilder(BaseBuilder):
    """
    A builder for grouping blocks together based on their structure.
    """

    gap_threshold: Annotated[
        float,
        "The minimum gap between blocks to consider them part of the same group.",
    ] = 0.05

    def __init__(self, config=None):
        super().__init__(config)

    def __call__(self, document: Document):
        for page in document.pages:
            self.group_caption_blocks(page)
            self.split_list_groups(document, page)

    def split_list_groups(self, document: Document, page: PageGroup):
        """Layout emits whole ListGroup regions. On pdftext pages those contain
        raw Lines - group consecutive lines into ListItems on bullet boundaries
        so the renderer can emit <li> items. OCR'd groups already carry html."""
        for block_id in page.structure.copy():
            block = page.get_block(block_id)
            if block.block_type != BlockTypes.ListGroup or block.html:
                continue
            if not block.structure:
                # Layout emitted a list region that never received children -
                # drop it rather than rendering an empty <ul>.
                block.removed = True
                page.structure.remove(block_id)
                continue

            lines = block.structure_blocks(page)
            if any(b.block_type != BlockTypes.Line for b in lines):
                continue

            item_line_groups = []
            current_item = []
            found_bullets = 0
            for line in lines:
                if LIST_ITEM_START_PATTERN.match(line.raw_text(document)):
                    found_bullets += 1
                    if current_item:
                        item_line_groups.append(current_item)
                    current_item = [line]
                else:
                    current_item.append(line)
            if current_item:
                item_line_groups.append(current_item)

            if found_bullets == 0:
                # Not actually a list - demote to text
                generated_block = Text(
                    polygon=block.polygon,
                    page_id=block.page_id,
                    structure=block.structure,
                )
                page.replace_block(block, generated_block)
                continue

            item_ids = []
            for item_lines in item_line_groups:
                polygon = item_lines[0].polygon.merge(
                    [line.polygon for line in item_lines]
                )
                list_item = page.add_block(ListItem, polygon)
                list_item.structure = [line.id for line in item_lines]
                item_ids.append(list_item.id)
            block.structure = item_ids

    def group_caption_blocks(self, page: PageGroup):
        gap_threshold_px = self.gap_threshold * page.polygon.height
        static_page_structure = page.structure.copy()
        remove_ids = list()

        for i, block_id in enumerate(static_page_structure):
            block = page.get_block(block_id)
            if block.block_type not in [
                BlockTypes.Table,
                BlockTypes.Figure,
                BlockTypes.Picture,
            ]:
                continue

            if block.id in remove_ids:
                continue

            block_structure = [block_id]
            selected_polygons = [block.polygon]
            caption_types = [BlockTypes.Caption, BlockTypes.Footnote]

            prev_block = page.get_prev_block(block)
            next_block = page.get_next_block(block)

            if (
                prev_block
                and prev_block.block_type in caption_types
                and prev_block.polygon.minimum_gap(block.polygon) < gap_threshold_px
                and prev_block.id not in remove_ids
            ):
                block_structure.insert(0, prev_block.id)
                selected_polygons.append(prev_block.polygon)

            if (
                next_block
                and next_block.block_type in caption_types
                and next_block.polygon.minimum_gap(block.polygon) < gap_threshold_px
            ):
                block_structure.append(next_block.id)
                selected_polygons.append(next_block.polygon)

            if len(block_structure) > 1:
                # Create a merged block
                new_block_cls = get_block_class(
                    BlockTypes[block.block_type.name + "Group"]
                )
                new_polygon = block.polygon.merge(selected_polygons)
                group_block = page.add_block(new_block_cls, new_polygon)
                group_block.structure = block_structure

                # Update the structure of the page to reflect the new block
                page.update_structure_item(block_id, group_block.id)
                remove_ids.extend(block_structure)
        page.remove_structure_items(remove_ids)
