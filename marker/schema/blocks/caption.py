from marker.schema import BlockTypes
from marker.schema.blocks import Block


class Caption(Block):
    block_type: BlockTypes = BlockTypes.Caption
    block_description: str = "A text caption that is directly above or below an image or table. Only used for text describing the image or table.  "
    replace_output_newlines: bool = True
    html: str | None = None

    def assemble_html(self, document, child_blocks, parent_structure, block_config):
        if self.html:
            return super().handle_html_output(
                document, child_blocks, parent_structure, block_config
            )

        return super().assemble_html(
            document, child_blocks, parent_structure, block_config
        )
