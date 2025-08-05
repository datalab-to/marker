from marker.schema import BlockTypes
from marker.schema.blocks import Block


class Picture(Block):
    block_type: BlockTypes = BlockTypes.Picture
    description: str | None = None
    block_description: str = "An image block that represents a picture."

    def assemble_html(self, document, child_blocks, parent_structure):
        child_ref_blocks = [block for block in child_blocks if block.id.block_type == BlockTypes.Reference]
        html = super().assemble_html(document, child_ref_blocks, parent_structure)

        # Use consistent placeholder ID format matching HTMLRenderer
        imgid = str(self.id)
        print("@@@@####picture imgid: ", imgid)

        if self.description:
            # Include both placeholder and description
            placeholder = f"<p>_placeholder_imgid_{imgid}</p>"
            description = f"<p role='img' data-original-image-id='{self.id}'>Image {self.id} description: {self.description}</p>"
            return html + placeholder + description
        # else:
            # Just placeholder
        return f"<p>_placeholder_imgid_{imgid}</p>"
