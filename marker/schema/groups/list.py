from marker.schema import BlockTypes
from marker.schema.groups.base import Group


class ListGroup(Group):
    block_type: BlockTypes = BlockTypes.ListGroup
    has_continuation: bool = False
    block_description: str = "A group of list items that should be rendered together."
    html: str | None = None

    def assemble_html(
        self, document, child_blocks, parent_structure, block_config=None
    ):
        if self.html:
            return self.handle_html_output(
                document, child_blocks, parent_structure, block_config
            )

        template = super().assemble_html(
            document, child_blocks, parent_structure, block_config
        )

        el_attr = f" block-type='{self.block_type}'"
        if self.has_continuation:
            el_attr += " class='has-continuation'"
        return f"<p{el_attr}><ul>{template}</ul></p>"
