from typing import Optional

from marker.schema import BlockTypes
from marker.schema.blocks import Block


class SectionHeader(Block):
    block_type: BlockTypes = BlockTypes.SectionHeader
    heading_level: Optional[int] = None
    block_description: str = "The header of a section of text or other blocks."
    html: str | None = None

    def assemble_html(
        self, document, child_blocks, parent_structure, block_config=None
    ):
        if self.ignore_for_output:
            return ""

        if self.html:
            return super().handle_html_output(
                document, child_blocks, parent_structure, block_config
            )

        template = super().assemble_html(
            document, child_blocks, parent_structure, block_config
        )
        template = template.replace("\n", " ")
        tag = f"h{self.heading_level}" if self.heading_level else "h2"
        return f"<{tag}>{template}</{tag}>"
