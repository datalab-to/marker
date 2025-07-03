from marker.schema import BlockTypes
from marker.schema.blocks import Block


class Picture(Block):
    block_type: BlockTypes = BlockTypes.Picture
    description: str | None = None
    block_description: str = "An image block that represents a picture."
    html: str | None = None

    def assemble_html(
        self, document, child_blocks, parent_structure, block_config=None
    ):
        if self.html:
            return super().handle_html_output(
                document, child_blocks, parent_structure, block_config
            )

        child_ref_blocks = [
            block
            for block in child_blocks
            if block.id.block_type == BlockTypes.Reference
        ]
        html = super().assemble_html(
            document, child_ref_blocks, parent_structure, block_config
        )

        if self.description:
            return (
                html
                + f"<p role='img' data-original-image-id='{self.id}'>Image {self.id} description: {self.description}</p>"
            )
        return html
