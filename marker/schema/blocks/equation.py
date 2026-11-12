import re

from marker.schema import BlockTypes
from marker.schema.blocks import Block

# A single top-level <p>...</p> wrapper around the block's html (the VLM wraps
# its output in <p>, and assemble_html adds its own paragraph wrapper).
_P_WRAPPED = re.compile(r"^<p[^>]*>(.*)</p>$", re.DOTALL)


class Equation(Block):
    block_type: BlockTypes = BlockTypes.Equation
    html: str | None = None
    block_description: str = "A block math equation."

    def assemble_html(
        self, document, child_blocks, parent_structure=None, block_config=None
    ):
        if self.html:
            child_ref_blocks = [
                block
                for block in child_blocks
                if block.id.block_type == BlockTypes.Reference
            ]
            html_out = super().assemble_html(
                document, child_ref_blocks, parent_structure, block_config
            )
            inner = self.html.strip()
            # Unwrap a single outer <p> so we don't emit nested <p><p>.
            m = _P_WRAPPED.match(inner)
            if m and "<p" not in m.group(1):
                inner = m.group(1)
            if (
                "<math" in inner
                and 'display="block"' not in inner
                and "display='block'" not in inner
            ):
                inner = re.sub(
                    r"<math\b",
                    '<math display="block"',
                    inner,
                    count=1,
                    flags=re.IGNORECASE,
                )
            html_out += f"""<p block-type='{self.block_type}'>{inner}</p>"""
            return html_out
        else:
            template = super().assemble_html(
                document, child_blocks, parent_structure, block_config
            )
            return f"<p block-type='{self.block_type}'>{template}</p>"
