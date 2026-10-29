from typing import Annotated

from marker.processors import BaseProcessor
from marker.schema import BlockTypes
from marker.schema.document import Document

# Text-carrying leaf types eligible for relabeling. Footnotes are deliberately
# excluded (marker keeps footnotes); pictures/figures/tables are never
# marginalia.
_ELIGIBLE = (BlockTypes.Text, BlockTypes.SectionHeader, BlockTypes.Caption)


class MarginaliaProcessor(BaseProcessor):
    """Relabel running headers/footers the layout model missed.

    A block is marginalia only when ALL of these hold - each guard exists to
    protect a real-content case:
      - it sits entirely inside the top/bottom margin zone (body text is out)
      - it is the extremal text block on the page (nothing above a header /
        below a footer - protects edge-adjacent body paragraphs)
      - it is running-head sized (small type; protects title pages, where
        titles are near the top but large)
      - it is short (a line or two; protects tall edge columns)
    """

    block_types = _ELIGIBLE
    header_zone: Annotated[
        float, "Top fraction of the page that can hold a running header."
    ] = 0.08
    footer_zone: Annotated[
        float, "Bottom fraction of the page that can hold a running footer."
    ] = 0.13
    max_height_frac: Annotated[
        float,
        "Maximum block height (fraction of page height) - running heads are",
        "one or two small lines.",
    ] = 0.035
    max_chars: Annotated[int, "Maximum text length for a marginalia block."] = 150
    # Mirror the renderer flags: a user who asked to keep headers/footers in
    # the output shouldn't have positionally-detected ones suppressed here.
    keep_pageheader_in_output: Annotated[
        bool, "Keep page headers in the final output."
    ] = False
    keep_pagefooter_in_output: Annotated[
        bool, "Keep page footers in the final output."
    ] = False

    def __call__(self, document: Document):
        for page in document.pages:
            text_blocks = [
                b
                for b in page.structure_blocks(document)
                if b.block_type
                in (*_ELIGIBLE, BlockTypes.ListGroup, BlockTypes.TextInlineMath)
                and not getattr(b, "ignore_for_output", False)
            ]
            if len(text_blocks) < 2:
                continue  # single-block pages: nothing is "marginal"

            # Block polygons are 0-origin within the page, so only the page's
            # height is taken from its polygon - never its origin. Subtracting
            # page.polygon.bbox[1] would shift every fraction on pages whose
            # polygon carries a non-zero CropBox origin (the force_ocr path in
            # PdfProvider), which is why the sibling processors compare raw
            # block coordinates against page.polygon.height/width directly.
            page_height = page.polygon.height or 1

            def yfrac(b):
                return (
                    b.polygon.bbox[1] / page_height,
                    b.polygon.bbox[3] / page_height,
                )

            # Body = text blocks not fully inside either margin zone. A
            # candidate must clear ALL body text (side-by-side header parts
            # don't disqualify each other).
            body = [
                b
                for b in text_blocks
                if not (
                    yfrac(b)[1] <= self.header_zone
                    or yfrac(b)[0] >= 1 - self.footer_zone
                )
            ]
            if not body:
                continue  # nothing but marginalia-sized edge text: leave it
            body_top = min(yfrac(b)[0] for b in body)
            body_bottom = max(yfrac(b)[1] for b in body)

            for block in text_blocks:
                if block.block_type not in _ELIGIBLE:
                    continue
                y0, y1 = yfrac(block)
                if (y1 - y0) > self.max_height_frac:
                    continue
                text = block.raw_text(document).strip()
                if not text or len(text) > self.max_chars:
                    continue

                is_header = (
                    y1 <= self.header_zone
                    and y1 <= body_top
                    and not self.keep_pageheader_in_output
                )
                is_footer = (
                    y0 >= 1 - self.footer_zone
                    and y0 >= body_bottom
                    and not self.keep_pagefooter_in_output
                )
                if is_header or is_footer:
                    block.ignore_for_output = True
