from typing import List

from marker.schema import BlockTypes
from marker.schema.blocks import Block


class TableCell(Block):
    block_type: BlockTypes = BlockTypes.TableCell
    rowspan: int
    colspan: int
    row_id: int
    col_id: int
    is_header: bool
    text_lines: List[str] | None = None
    block_description: str = "A cell in a table."

    @property
    def text(self):
        return "\n".join(self.text_lines)

    def assemble_html(
        self, document, child_blocks, parent_structure=None, block_config=None
    ):
        tag_cls = "th" if self.is_header else "td"
        tag = f"<{tag_cls}"
        if self.rowspan > 1:
            tag += f" rowspan={self.rowspan}"
        if self.colspan > 1:
            tag += f" colspan={self.colspan}"
        if self.text_lines is None:
            self.text_lines = []
        text = "<br>".join(self.text_lines)
        return f"{tag}>{text}</{tag_cls}>"
