from typing import Annotated

from marker.builders import BaseBuilder
from marker.schema import BlockTypes
from marker.schema.blocks import Text
from marker.schema.document import Document
from marker.schema.groups import ListGroup
from marker.schema.groups.page import PageGroup
from marker.schema.registry import get_block_class


class StructureBuilder(BaseBuilder):
    """
    A builder for grouping blocks together based on their structure.
    """
    gap_threshold: Annotated[
        float,
        "The minimum gap between blocks to consider them part of the same group.",
    ] = 0.05
    list_gap_threshold: Annotated[
        float,
        "The minimum gap between list items to consider them part of the same group.",
    ] = 0.1

    def __init__(self, config=None):
        super().__init__(config)

    def __call__(self, document: Document):
        for page in document.pages:
            self.group_caption_blocks(page)
            # self.group_lists(page)   # keep each ListItem standalone (no merging into ListGroup)
            # self.unmark_lists(page)  # don't demote lone ListItems to Text

    def group_caption_blocks(self, page: PageGroup):
        gap_threshold_px = self.gap_threshold * page.polygon.height
        static_page_structure = page.structure.copy()
        remove_ids = list()

        for i, block_id in enumerate(static_page_structure):
            block = page.get_block(block_id)
            if block.block_type not in [BlockTypes.Table, BlockTypes.Figure, BlockTypes.Picture]:
                continue

            if block.id in remove_ids:
                continue

            block_structure = [block_id]
            selected_polygons = [block.polygon]
            caption_types = [BlockTypes.Caption, BlockTypes.Footnote]

            prev_block = page.get_prev_block(block)
            next_block = page.get_next_block(block)

            if prev_block and \
                prev_block.block_type in caption_types and \
                prev_block.polygon.minimum_gap(block.polygon) < gap_threshold_px and \
                    prev_block.id not in remove_ids:
                block_structure.insert(0, prev_block.id)
                selected_polygons.append(prev_block.polygon)

            if next_block and \
                    next_block.block_type in caption_types and \
                    next_block.polygon.minimum_gap(block.polygon) < gap_threshold_px:
                block_structure.append(next_block.id)
                selected_polygons.append(next_block.polygon)

            if len(block_structure) > 1:
                # Create a merged block
                new_block_cls = get_block_class(BlockTypes[block.block_type.name + "Group"])
                new_polygon = block.polygon.merge(selected_polygons)
                group_block = page.add_block(new_block_cls, new_polygon)
                group_block.structure = block_structure

                # Update the structure of the page to reflect the new block
                page.update_structure_item(block_id, group_block.id)
                remove_ids.extend(block_structure)
        page.remove_structure_items(remove_ids)

    def group_lists(self, page: PageGroup):
        gap_threshold_px = self.list_gap_threshold * page.polygon.height
        static_page_structure = page.structure.copy()
        remove_ids = list()
        for i, block_id in enumerate(static_page_structure):
            block = page.get_block(block_id)
            if block.block_type not in [BlockTypes.ListItem]:
                continue

            if block.id in remove_ids:
                continue

            block_structure = [block_id]
            selected_polygons = [block.polygon]

            for j, next_block_id in enumerate(page.structure[i + 1:]):
                next_block = page.get_block(next_block_id)
                if all([
                    next_block.block_type == BlockTypes.ListItem,
                    next_block.polygon.minimum_gap(selected_polygons[-1]) < gap_threshold_px
                ]):
                    block_structure.append(next_block_id)
                    selected_polygons.append(next_block.polygon)
                else:
                    break

            if len(block_structure) > 1:
                new_polygon = block.polygon.merge(selected_polygons)
                group_block = page.add_block(ListGroup, new_polygon)
                group_block.structure = block_structure

                # Update the structure of the page to reflect the new block
                page.update_structure_item(block_id, group_block.id)
                remove_ids.extend(block_structure)

        page.remove_structure_items(remove_ids)

    def unmark_lists(self, page: PageGroup):
        # If lists aren't grouped, unmark them as list items
        for block_id in page.structure:
            block = page.get_block(block_id)
            if block.block_type == BlockTypes.ListItem:
                generated_block = Text(
                    polygon=block.polygon,
                    page_id=block.page_id,
                    structure=block.structure,
                )
                page.replace_block(block, generated_block)
