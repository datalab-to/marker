import math
from typing import Annotated, List

import regex

from marker.processors import BaseProcessor
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.text.line import Line

HYPHEN_END_REGEX = regex.compile(r".*[\p{Ll}|\d][-—¬]\s?$", regex.DOTALL)


class TextProcessor(BaseProcessor):
    """
    A processor for merging text across pages and columns.
    """

    block_types = (BlockTypes.Text, BlockTypes.TextInlineMath)
    ignored_block_types = (BlockTypes.PageHeader, BlockTypes.PageFooter)
    column_gap_ratio: Annotated[
        float,
        "The minimum ratio of the page width to the column gap to consider a column break.",
    ] = 0.02

    def __init__(self, config):
        super().__init__(config)

    def __call__(self, document: Document):
        for page in document.pages:
            for block in page.contained_blocks(document, self.block_types):
                if block.structure is None:
                    continue

                if not len(block.structure) >= 2:  # Skip single lines
                    continue

                next_block = document.get_next_block(block, self.ignored_block_types)
                if next_block is None:  # we've reached the end of the document
                    continue
                if next_block.block_type not in self.block_types:
                    continue  # we found a non-text block
                if next_block.structure is None:
                    continue  # This is odd though, why do we have text blocks with no structure?
                if next_block.ignore_for_output:
                    continue  # skip ignored blocks

                column_gap = block.polygon.width * self.column_gap_ratio

                column_break, page_break = False, False
                next_block_starts_indented = True
                next_block_in_first_quadrant = False
                last_line_is_full_width = False
                last_line_is_hyphentated = False

                if next_block.page_id == block.page_id:  # block on the same page
                    # we check for a column break
                    column_break = math.floor(next_block.polygon.y_start) <= math.ceil(
                        block.polygon.y_start
                    ) and next_block.polygon.x_start > (
                        block.polygon.x_end + column_gap
                    )
                else:
                    page_break = True
                    next_page = document.get_page(next_block.page_id)
                    next_block_in_first_quadrant = (
                        next_block.polygon.x_start < next_page.polygon.width // 2
                    ) and (next_block.polygon.y_start < next_page.polygon.height // 2)

                if not (column_break or page_break):
                    continue

                new_block_lines = next_block.structure_blocks(document)

                # we check for next_block indentation
                if len(new_block_lines):
                    min_x = math.ceil(
                        min([line.polygon.x_start for line in new_block_lines])
                    )
                    next_block_starts_indented = (
                        new_block_lines[0].polygon.x_start > min_x
                    )

                lines: List[Line] = [
                    line
                    for line in block.structure_blocks(document)
                    if line.polygon.width > 1
                ]
                if len(lines):
                    max_x = math.floor(max([line.polygon.x_end for line in lines]))
                    last_line_is_full_width = lines[-1].polygon.x_end >= max_x

                    last_line_is_hyphentated = HYPHEN_END_REGEX.match(
                        lines[-1].raw_text(document).strip()
                    )

                if (
                    (last_line_is_full_width or last_line_is_hyphentated)
                    and not next_block_starts_indented
                    and ((next_block_in_first_quadrant and page_break) or column_break)
                ):
                    block.has_continuation = True
