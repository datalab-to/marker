from typing import List

from marker.schema import BlockTypes
from marker.schema.blocks import Block, BlockOutput
from marker.schema.blocks.tablecell import TableCell


class BaseTable(Block):
    block_type: BlockTypes | None = None
    html: str | None = None

    @staticmethod
    def format_cells(
        document, child_blocks, child_cells: List[TableCell] | None = None
    ):
        if child_cells is None:
            child_cells: List[TableCell] = [
                document.get_block(c.id)
                for c in child_blocks
                if c.id.block_type == BlockTypes.TableCell
            ]

        unique_rows = sorted(list(set([c.row_id for c in child_cells])))
        html_repr = "<table><tbody>"
        for row_id in unique_rows:
            row_cells = sorted(
                [c for c in child_cells if c.row_id == row_id], key=lambda x: x.col_id
            )
            html_repr += "<tr>"
            for cell in row_cells:
                html_repr += cell.assemble_html(document, child_blocks, None, None)
            html_repr += "</tr>"
        html_repr += "</tbody></table>"
        return html_repr

    def assemble_html(
        self,
        document,
        child_blocks: List[BlockOutput],
        parent_structure=None,
        block_config: dict | None = None,
    ):
        # Filter out the table cells, so they don't render twice
        child_ref_blocks = [
            block
            for block in child_blocks
            if block.id.block_type == BlockTypes.Reference
        ]
        template = super().assemble_html(
            document, child_ref_blocks, parent_structure, block_config
        )

        child_block_types = set([c.id.block_type for c in child_blocks])
        if self.html:
            # LLM processor
            return template + self.html
        elif len(child_blocks) > 0 and BlockTypes.TableCell in child_block_types:
            # Table processor
            return template + self.format_cells(document, child_blocks)
        else:
            # Default text lines and spans
            return f"<p>{template}</p>"
