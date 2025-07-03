import re

from marker.schema import BlockTypes
from marker.schema.blocks import Block


def replace_bullets(child_blocks):
    # Replace bullet characters with a -
    first_block = None
    while len(child_blocks) > 0:
        first_block = child_blocks[0]
        child_blocks = first_block.children

    if first_block is not None and first_block.id.block_type == BlockTypes.Line:
        bullet_pattern = r"(^|[\n ]|<[^>]*>)[•●○ഠ ം◦■▪▫–—-]( )"
        first_block.html = re.sub(bullet_pattern, r"\1\2", first_block.html)


class ListItem(Block):
    block_type: BlockTypes = BlockTypes.ListItem
    list_indent_level: int = 0
    block_description: str = "A list item that is part of a list.  This block is used to represent a single item in a list."
    html: str | None = None

    def assemble_html(
        self, document, child_blocks, parent_structure, block_config=None
    ):
        template = super().assemble_html(
            document, child_blocks, parent_structure, block_config
        )
        template = template.replace("\n", " ")
        # Remove the first bullet character
        replace_bullets(child_blocks)

        if self.html:
            template = (
                super()
                .handle_html_output(
                    document, child_blocks, parent_structure, block_config
                )
                .strip()
            )
            template = template.replace("<li>", "").replace("</li>", "")

        el_attr = f" block-type='{self.block_type}'"
        has_line = "Line" in template
        has_indent = self.list_indent_level > 0

        # If there is no line in the template, only return the nested list.
        if has_indent:
            if has_line:
                return f"<ul><li{el_attr} class='list-indent-{self.list_indent_level}'>{template}</li></ul>"
            return f"<ul>{template}</ul>"

        if has_line:
            return f"<li{el_attr}>{template}</li>"
        return template