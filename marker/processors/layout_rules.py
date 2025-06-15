from typing import List, Dict, Optional, Tuple

from marker.processors import BaseProcessor
from marker.rules import RuleEngine
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.blocks import Block


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