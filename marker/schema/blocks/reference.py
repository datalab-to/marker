from marker.schema import BlockTypes
from marker.schema.blocks import Block


class Reference(Block):
    block_type: BlockTypes = BlockTypes.Reference
    ref: str
    block_description: str = "A reference to this block from another block."

    def assemble_html(
        self, document, child_blocks, parent_structure=None, block_config=None
    ):
        template = super().assemble_html(
            document, child_blocks, parent_structure, block_config
        )
        return f"<span id='{self.ref}'>{template}</span>"
