from marker.schema import BlockTypes
from marker.schema.groups.base import Group


class PictureGroup(Group):
    block_type: BlockTypes = BlockTypes.PictureGroup
    block_description: str = "A picture along with associated captions."
    html: str | None = None

    def assemble_html(
        self, document, child_blocks, parent_structure, block_config=None
    ):
        if self.html:
            return self.html

        child_html = super().assemble_html(
            document, child_blocks, parent_structure, block_config
        )
        return child_html
