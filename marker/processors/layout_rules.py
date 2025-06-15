from typing import List, Dict

from marker.processors import BaseProcessor
from marker.rules import RuleEngine
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.blocks import Block
from marker.schema.groups import PageGroup
from marker.logger import get_logger

logger = get_logger()


class LayoutRuleProcessor(BaseProcessor):
    """
    A processor for applying layout rules from the rule engine.
    This handles things like filtering out regions or defining column layouts.
    """
    block_types = tuple()

    def __init__(self, rule_engine: RuleEngine, config=None):
        self.rule_engine = rule_engine
        super().__init__(config)

    def apply_rules(self, page: PageGroup, document: Document):
        """
        Applies layout rules to a page.
        """
        layout_rules = self.rule_engine.get_rules("layout")
        if not layout_rules:
            return

        for rule in layout_rules:
            if rule.get("type") == "exclude_gutter":
                self.apply_exclude_gutter_rule(page, document, rule)
            elif rule.get("type") == "define_regions":
                self.apply_define_regions_rule(page, document, rule)
            else:
                logger.warning(f"Warning: Unknown layout rule type: {rule.get('type')}")

    def _sort_blocks_in_reading_order(self, blocks: List[Block], layout_type: str, page_width: float) -> List[Block]:
        if layout_type == "two_column":
            # Separate blocks into left and right columns
            left_column = []
            right_column = []
            page_center_x = page_width / 2

            for block in blocks:
                # Use x_start for column assignment for better accuracy with blocks near the center
                if block.polygon.x_start < page_center_x:
                    left_column.append(block)
                else:
                    right_column.append(block)

            # Sort each column top-to-bottom
            left_column.sort(key=lambda b: b.polygon.y_start)
            right_column.sort(key=lambda b: b.polygon.y_start)

            # The correct reading order is the entire left column, followed by the entire right column
            return left_column + right_column
        else: # single_column and default
            return sorted(blocks, key=lambda b: b.polygon.y_start)

    def apply_define_regions_rule(self, page: PageGroup, document: Document, rule: Dict):
        """
        Applies a rule to define and order regions on a page.
        """
        rule_pages = rule.get("pages", [])
        if page.page_id not in rule_pages:
            return

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

        # Find any blocks that weren't part of a defined region and append them
        non_region_block_ids = set(page.structure) - set(new_structure)
        if non_region_block_ids:
            # We will sort them by y-pos and append at the end
            non_region_blocks = [document.get_block(bid) for bid in non_region_block_ids]
            non_region_blocks.sort(key=lambda b: b.polygon.y_start)
            new_structure.extend([b.id for b in non_region_blocks])

        # Replace the page's structure with the new region-based order
        page.structure = new_structure
        # Mark the page as handled by a rule to prevent OrderProcessor from running
        if not hasattr(page, 'layout_rules_applied'):
            page.layout_rules_applied = []
        page.layout_rules_applied.append("define_regions")

    def apply_exclude_gutter_rule(self, page: PageGroup, document: Document, rule: Dict):
        position = rule.get("position")
        width_ratio = rule.get("width_ratio")

        if not position or not width_ratio:
            return

        page_width = page.polygon.width
        blocks_to_remove = []

        for block_id in page.structure:
            block = document.get_block(block_id)
            if position == "left" and block.polygon.x_end < page_width * width_ratio:
                blocks_to_remove.append(block_id)
            elif position == "right" and block.polygon.x_start > page_width * (1 - width_ratio):
                blocks_to_remove.append(block_id)

        page.remove_structure_items(blocks_to_remove)

    def __call__(self, document: Document):
        for page in document.pages:
            self.apply_rules(page, document) 