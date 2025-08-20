from marker.schema import BlockTypes
from marker.schema.blocks import Block


class PageHeader(Block):
    block_type: BlockTypes = BlockTypes.PageHeader
    block_description: str = (
        "Text that appears at the top of a page, like a page title."
    )
    replace_output_newlines: bool = True
    ignore_for_output: bool = True
    html: str | None = None

    def assemble_html(self, document, child_blocks, parent_structure, block_config):
        if block_config and block_config.get("keep_pageheader_in_output"):
            self.ignore_for_output = False

        if self.html and not self.ignore_for_output:
            return self.html

        return super().assemble_html(
            document, child_blocks, parent_structure, block_config
        )
