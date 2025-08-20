from marker.schema import BlockTypes
from marker.schema.blocks import Block


class PageFooter(Block):
    block_type: str = BlockTypes.PageFooter
    block_description: str = (
        "Text that appears at the bottom of a page, like a page number."
    )
    replace_output_newlines: bool = True
    ignore_for_output: bool = True
    html: str | None = None

    def assemble_html(self, document, child_blocks, parent_structure, block_config):
        if block_config and block_config.get("keep_pagefooter_in_output"):
            self.ignore_for_output = False

        if self.html and not self.ignore_for_output:
            return self.html

        return super().assemble_html(
            document, child_blocks, parent_structure, block_config
        )
