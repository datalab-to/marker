from typing import List, Dict

from marker.processors import BaseProcessor
from marker.rules import RuleEngine
from marker.schema.document import Document
from marker.schema.blocks import Block
from marker.schema.groups import PageGroup


class LayoutRuleProcessor(BaseProcessor):
    """
    A processor for applying layout filtering rules from the rule engine.
    e.g. to filter out gutters from a PDF by width ratio.
    """
    block_types = tuple()

    def __init__(self, rule_engine: RuleEngine, config=None):
        self.rule_engine = rule_engine
        super().__init__(config)

    def __call__(self, document: Document):
        # This is inefficient, but the alternative is passing the document to every method
        # and having every method re-fetch the rules
        for page in document.pages:
            self.apply_rules(page, document)

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
        gutter_width = page_width * width_ratio

        if position == "left":
            # Filter out blocks that are mostly in the left gutter
            filtered_structure = [
                block_id for block_id in page.structure
                if document.get_block(block_id).polygon.x_start >= gutter_width
            ]
        elif position == "right":
            # Filter out blocks that are mostly in the right gutter
            filtered_structure = [
                block_id for block_id in page.structure
                if document.get_block(block_id).polygon.x_end <= (page_width - gutter_width)
            ]
        else:
            return

        page.structure = filtered_structure 