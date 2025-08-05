from marker.schema import BlockTypes
from marker.schema.blocks import Block


class Figure(Block):
    block_type: BlockTypes = BlockTypes.Figure
    description: str | None = None
    block_description: str = "A chart or other image that contains data."

    def assemble_html(self, document, child_blocks, parent_structure):
        print(f"@@@ Figure.assemble_html called - id: {self.id}, description: {self.description}")
        child_ref_blocks = [block for block in child_blocks if block.id.block_type == BlockTypes.Reference]
        print(f"@@@ Figure child_ref_blocks count: {len(child_ref_blocks)}")
        html = super().assemble_html(document, child_ref_blocks, parent_structure)
        print(f"@@@ Figure super().assemble_html returned: '{html}'")
        
        # Use consistent placeholder ID format matching HTMLRenderer
        imgid = str(self.id)
        print("@@@@####figure imgid: ", imgid)
        
        if self.description:
            # Include both placeholder and description
            placeholder = f"<p>_placeholder_figid_{imgid}</p>"
            description = f"<p role='img' data-original-image-id='{self.id}'>Image {self.id} description: {self.description}</p>"
            print(f"@@@ Figure has description: '{self.description}'")
            print(f"@@@ Generated placeholder: '{placeholder}'")
            print(f"@@@ Generated description: '{description}'")
            # return html + placeholder + description
        # else:

        final_result = f"<p>_placeholder_imgid_{imgid}</p>"
        print(f"@@@ Figure.assemble_html returning: '{final_result}'")
        return final_result
