"""
Marginal block (e.g. for legal documents).

Rendered as <aside>#</aside> when encountered as a child block
inside a paragraph's structure.  Never appears in page.structure directly.
"""

from marker.schema import BlockTypes
from marker.schema.blocks import Block


class Marginal(Block):
    """
    Marginal numbers are identified by MarginalProcessor and registered 
    in page.children only (not page.structure).  Its ID is inserted at 
    position 0 of the associated paragraph's structure. The renderer 
    emits <aside>#</aside> before the paragraph's own content.
    """

    block_type: BlockTypes = BlockTypes.Marginal
    block_description: str = (
        "A margin number (Randnummer) — rendered at the start of its paragraph."
    )
    marginal_number: str = ""
    associated_block_id: str | None = None

    # Required by ListProcessor.list_group_indentation() which iterates
    # page.children and reads/writes this field on every block it encounters.
    # Marginal blocks are never part of a list, so 0 is always correct.
    list_indent_level: int = 0

    def assemble_html(
        self, document, child_blocks, parent_structure, block_config=None
    ):
        if self.ignore_for_output:
            return ""
        text = (self.marginal_number or self.raw_text(document)).strip()
        return f"<aside>{text}</aside>"
