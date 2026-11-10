import re
from typing import Optional

from marker.schema import BlockTypes
from marker.schema.blocks import Block

_HEADING_TAG_RE = re.compile(r"<h[1-6]\b", re.IGNORECASE)
_OUTER_P_RE = re.compile(r"^<p[^>]*>(.*)</p>$", re.DOTALL | re.IGNORECASE)


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
            body = self.html.strip()
            if body and not _HEADING_TAG_RE.search(body):
                m = _OUTER_P_RE.match(body)
                if m and "<p" not in m.group(1).lower():
                    body = m.group(1).strip()
                tag = f"h{self.heading_level}" if self.heading_level else "h2"
                child_ref_blocks = [
                    block
                    for block in child_blocks
                    if block.id.block_type == BlockTypes.Reference
                ]
                prefix = super().assemble_html(
                    document, child_ref_blocks, parent_structure, block_config
                )
                return f"{prefix}<{tag}>{body}</{tag}>"
            return super().handle_html_output(
                document, child_blocks, parent_structure, block_config
            )

        template = super().assemble_html(
            document, child_blocks, parent_structure, block_config
        )
        template = template.replace("\n", " ")
        tag = f"h{self.heading_level}" if self.heading_level else "h2"
        return f"<{tag}>{template}</{tag}>"
