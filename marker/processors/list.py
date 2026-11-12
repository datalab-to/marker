from typing import Annotated, List, Tuple

from marker.processors import BaseProcessor
from marker.schema import BlockTypes
from marker.schema.blocks import ListItem
from marker.schema.document import Document


class ListProcessor(BaseProcessor):
    """
    A processor for merging lists across pages and columns
    """
    block_types = (BlockTypes.ListGroup,)
    ignored_block_types: Annotated[
        Tuple[BlockTypes],
        "The list of block types to ignore when merging lists.",
    ] = (BlockTypes.PageHeader, BlockTypes.PageFooter)
    min_x_indent: Annotated[
        float, "The minimum horizontal indentation required to consider a block as a nested list item.",
        "This is expressed as a percentage of the page width and is used to determine hierarchical relationships within a list.",
    ] = 0.01

    def __init__(self, config):
        super().__init__(config)

    def __call__(self, document: Document):
        self.list_group_continuation(document)
        self.list_group_indentation(document)

    def list_group_continuation(self, document: Document):
        for page in document.pages:
            for block in page.contained_blocks(document, self.block_types):
                next_block = document.get_next_block(block, self.ignored_block_types)
                if next_block is None:
                    continue
                if next_block.block_type not in self.block_types:
                    continue
                if next_block.structure is None:
                    continue
                if next_block.ignore_for_output:
                    continue

                column_break, page_break = False, False
                next_block_in_first_quadrant = False

                if next_block.page_id == block.page_id:  # block on the same page
                    # we check for a column break
                    column_break = next_block.polygon.y_start <= block.polygon.y_end
                else:
                    page_break = True
                    next_page = document.get_page(next_block.page_id)
                    next_block_in_first_quadrant = (next_block.polygon.x_start < next_page.polygon.width // 2) and \
                        (next_block.polygon.y_start < next_page.polygon.height // 2)

                block.has_continuation = column_break or (page_break and next_block_in_first_quadrant)

    def list_group_indentation(self, document: Document):
        for page in document.pages:
            for block in page.contained_blocks(document, self.block_types):
                if not block.structure:
                    continue
                if block.ignore_for_output:
                    continue

                stack: List[ListItem] = []
                for list_item_id in block.structure:
                    list_item_block: ListItem = page.get_block(list_item_id)

                    # This can be a line sometimes
                    if list_item_block.block_type != BlockTypes.ListItem:
                        continue

                    while stack and list_item_block.polygon.x_start <= stack[-1].polygon.x_start + (self.min_x_indent * page.polygon.width):
                        stack.pop()

                    if stack and list_item_block.polygon.y_start > stack[-1].polygon.y_start:
                        list_item_block.list_indent_level = stack[-1].list_indent_level
                        if list_item_block.polygon.x_start > stack[-1].polygon.x_start + (self.min_x_indent * page.polygon.width):
                            list_item_block.list_indent_level += 1

                    next_list_item_block = block.get_next_block(page, list_item_block)
                    if (
                        next_list_item_block is not None
                        and next_list_item_block.block_type == BlockTypes.ListItem
                        and next_list_item_block.polygon.x_start > list_item_block.polygon.x_end
                    ):
                        stack = [next_list_item_block]  # reset stack on column breaks
                    else:
                        stack.append(list_item_block)

                stack: List[ListItem] = []
                for list_item_id in block.structure.copy():
                    list_item_block: ListItem = page.get_block(list_item_id)
                    if list_item_block.block_type != BlockTypes.ListItem:
                        continue

                    while stack and list_item_block.list_indent_level <= stack[-1].list_indent_level:
                        stack.pop()

                    if stack:
                        current_parent = stack[-1]
                        current_parent.add_structure(list_item_block)
                        current_parent.polygon = current_parent.polygon.merge([list_item_block.polygon])

                        block.remove_structure_items([list_item_id])
                    stack.append(list_item_block)
