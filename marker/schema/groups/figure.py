from marker.schema import BlockTypes
from marker.schema.groups.base import Group


class FigureGroup(Group):
    block_type: BlockTypes = BlockTypes.FigureGroup
    block_description: str = "A group that contains a figure and associated captions."
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
