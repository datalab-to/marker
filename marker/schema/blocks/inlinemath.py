from marker.schema import BlockTypes
from marker.schema.blocks import Block


class InlineMath(Block):
    block_type: BlockTypes = BlockTypes.TextInlineMath
    has_continuation: bool = False
    blockquote: bool = False
    blockquote_level: int = 0
    block_description: str = "A text block that contains inline math.  This is not used for italic text or references - only for text that contains math."
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

        el_attr = f" block-type='{self.block_type}'"
        if self.has_continuation:
            el_attr += " class='has-continuation'"

        if self.blockquote:
            # Add indentation for blockquote levels
            blockquote_prefix = "<blockquote>" * self.blockquote_level
            blockquote_suffix = "</blockquote>" * self.blockquote_level
            return f"{blockquote_prefix}<p{el_attr}>{template}</p>{blockquote_suffix}"
        else:
            return f"<p{el_attr}>{template}</p>"
