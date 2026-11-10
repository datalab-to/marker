import re
from collections import Counter
from itertools import groupby
from typing import Annotated, List

from rapidfuzz import fuzz

from marker.processors import BaseProcessor
from marker.schema import BlockTypes
from marker.schema.blocks import Block
from marker.schema.document import Document


class IgnoreTextProcessor(BaseProcessor):
    """
    A processor for identifying and ignoring common text blocks in a document. 
    These blocks often represent repetitive or non-essential elements, such as headers, footers, or page numbers.
    """
    block_types = (
        BlockTypes.Text, BlockTypes.SectionHeader,
        BlockTypes.TextInlineMath
    )
    common_element_threshold: Annotated[
        float,
        "The minimum ratio of pages a text block must appear on to be considered a common element.",
        "Blocks that meet or exceed this threshold are marked as common elements.",
    ] = 0.2
    common_element_min_blocks: Annotated[
        int,
        "The minimum number of occurrences of a text block within a document to consider it a common element.",
        "This ensures that rare blocks are not mistakenly flagged.",
    ] = 3
    max_streak: Annotated[
        int,
        "The maximum number of consecutive occurrences of a text block allowed before it is classified as a common element.",
        "Helps to identify patterns like repeated headers or footers.",
    ] = 3
    text_match_threshold: Annotated[
        int,
        "The minimum fuzzy match score (0-100) required to classify a text block as similar to a common element.",
        "Higher values enforce stricter matching.",
    ] = 90

    def __call__(self, document: Document):
        first_blocks = []
        last_blocks = []
        for page in document.pages:
            initial_block = None
            last_block = None
            for block in page.contained_blocks(document, self.block_types):
                if block.structure is not None:
                    if initial_block is None:
                        initial_block = block

                    last_block = block

            if initial_block is not None:
                first_blocks.append(initial_block)
            if last_block is not None:
                last_blocks.append(last_block)

        self.filter_common_elements(document, first_blocks)
        self.filter_common_elements(document, last_blocks)

    @staticmethod
    def clean_text(text):
        text = text.replace("\n", "").strip()
        text = re.sub(r"^\d+\s*", "", text)  # remove numbers at the start of the line
        text = re.sub(r"\s*\d+$", "", text)  # remove numbers at the end of the line
        return text

    def filter_common_elements(self, document, blocks: List[Block]):
        # We can't filter if we don't have enough pages to find common elements
        if len(blocks) < self.common_element_min_blocks:
            return

        text = [self.clean_text(b.raw_text(document)) for b in blocks]

        streaks = {}
        for key, group in groupby(text):
            streaks[key] = max(streaks.get(key, 0), len(list(group)))

        counter = Counter(text)
        common = [
            k for k, v in counter.items()
            if (v >= len(blocks) * self.common_element_threshold or streaks[k] >= self.max_streak)
            and v >= self.common_element_min_blocks
        ]
        if len(common) == 0:
            return

        for t, b in zip(text, blocks):
            # Check against all common elements
            if any(fuzz.ratio(t, common_element) > self.text_match_threshold for common_element in common):
                b.ignore_for_output = True
