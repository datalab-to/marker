from copy import deepcopy
from typing import Annotated

from marker.processors import BaseProcessor
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.registry import get_block_class


class ListItemLineExplodeProcessor(BaseProcessor):
    """
    Explode every multi-line ListItem into one ListItem per Line child.
    Each new ListItem inherits the line's polygon and points at that single
    Line as its structure. The parent block is dropped from the page structure.

    Prefix-free: relies only on line-level layout from LineBuilder.
    """
    block_types = (BlockTypes.ListItem,)
    min_lines: Annotated[
        int,
        "Minimum number of Line children before a ListItem is exploded.",
    ] = 2

    def __call__(self, document: Document):
        ListItemCls = get_block_class(BlockTypes.ListItem)
        for page in document.pages:
            if not page.structure:
                continue
            new_structure = []
            for bid in list(page.structure):
                block = page.get_block(bid)
                if block is None or block.block_type != BlockTypes.ListItem:
                    new_structure.append(bid)
                    continue

                line_ids = list(block.structure or [])
                lines = []
                for lid in line_ids:
                    ln = page.get_block(lid)
                    if ln is not None and ln.block_type == BlockTypes.Line:
                        lines.append(ln)

                if len(lines) < self.min_lines:
                    new_structure.append(bid)
                    continue

                for line in lines:
                    new_li = page.add_block(ListItemCls, deepcopy(line.polygon))
                    new_li.structure = [line.id]
                    new_structure.append(new_li.id)
                block.removed = True

            page.structure = new_structure
