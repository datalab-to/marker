from statistics import mean
from collections import defaultdict
from typing import List, Dict

from marker.processors import BaseProcessor
from marker.rules import RuleEngine
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.blocks import Block
from marker.schema.groups import PageGroup


class OrderProcessor(BaseProcessor):
    """
    A processor for sorting the blocks in order if needed. This can be configured with rules.
    """
    block_types = tuple()

    def __init__(self, rule_engine: RuleEngine, config=None):
        self.rule_engine = rule_engine
        super().__init__(config)

    def __call__(self, document: Document):
        block_ordering_rules = self.rule_engine.get_rules("block_ordering", [])
        regions_rule = next((rule for rule in block_ordering_rules if rule.get("type") == "define_regions"), None)

        for page in document.pages:
            if regions_rule and page.page_id in regions_rule.get("pages", []):
                self.apply_define_regions_rule(page, document, regions_rule)
            else:
                self.apply_default_ordering(page, document)

    def _sort_blocks_in_reading_order(self, blocks: List[Block], layout_type: str, page_width: float) -> List[Block]:
        if layout_type == "two_column":
            left_column = [b for b in blocks if b.polygon.x_start < page_width / 2]
            right_column = [b for b in blocks if b.polygon.x_start >= page_width / 2]

            left_column.sort(key=lambda b: b.polygon.y_start)
            right_column.sort(key=lambda b: b.polygon.y_start)
            
            return left_column + right_column
        else: # single_column and default
            return sorted(blocks, key=lambda b: b.polygon.y_start)

    def apply_define_regions_rule(self, page: PageGroup, document: Document, rule: Dict):
        all_blocks = [document.get_block(bid) for bid in page.structure]
        page_height = page.polygon.height
        new_structure = []
        
        region_map = {region['name']: region for region in rule.get("regions", [])}
        region_order = rule.get("region_order", [])

        for region_name in region_order:
            region_def = region_map.get(region_name)
            if not region_def:
                continue

            y_start = page_height * (region_def.get("y_start_percent", 0) / 100)
            y_end = page_height * (region_def.get("y_end_percent", 100) / 100)
            
            region_blocks = [
                block for block in all_blocks
                if block.polygon.y_start >= y_start and block.polygon.y_end <= y_end
            ]
            
            layout_type = region_def.get("layout_type", "single_column")
            sorted_region_blocks = self._sort_blocks_in_reading_order(region_blocks, layout_type, page.polygon.width)
            
            new_structure.extend([block.id for block in sorted_region_blocks])

        non_region_block_ids = set(page.structure) - set(new_structure)
        if non_region_block_ids:
            non_region_blocks = [document.get_block(bid) for bid in non_region_block_ids]
            non_region_blocks.sort(key=lambda b: b.polygon.y_start)
            new_structure.extend([b.id for b in non_region_blocks])

        page.structure = new_structure

    def apply_default_ordering(self, page: PageGroup, document: Document):
        for page in document.pages:
            # Skip OCRed pages
            if page.text_extraction_method != "pdftext":
                continue

            # Skip pages without layout slicing
            if not page.layout_sliced:
                continue

            block_idxs = defaultdict(int)
            for block_id in page.structure:
                block = document.get_block(block_id)
                spans = block.contained_blocks(document, (BlockTypes.Span, ))
                if len(spans) == 0:
                    continue

                # Avg span position in original PDF
                block_idxs[block_id] = (spans[0].minimum_position + spans[-1].maximum_position) / 2

            for block_id in page.structure:
                # Already assigned block id via span position
                if block_idxs[block_id] > 0:
                    continue

                block = document.get_block(block_id)
                prev_block = document.get_prev_block(block)
                next_block = document.get_next_block(block)

                block_idx_add = 0
                if prev_block:
                    block_idx_add = 1

                while prev_block and prev_block.id not in block_idxs:
                    prev_block = document.get_prev_block(prev_block)
                    block_idx_add += 1

                if not prev_block:
                    block_idx_add = -1
                    while next_block and next_block.id not in block_idxs:
                        next_block = document.get_next_block(next_block)
                        block_idx_add -= 1

                if not next_block and not prev_block:
                    pass
                elif prev_block:
                    block_idxs[block_id] = block_idxs[prev_block.id] + block_idx_add
                else:
                    block_idxs[block_id] = block_idxs[next_block.id] + block_idx_add

            page.structure = sorted(page.structure, key=lambda x: block_idxs[x])
