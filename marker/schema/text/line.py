import html
import re
from typing import Literal, List

import regex

from marker.schema import BlockTypes
from marker.schema.blocks import Block, BlockOutput
from marker.schema.text.span import sanitize_url

HYPHENS = r"\-\u2010\u2011\u00ad\u2013—¬"
HYPHEN_REGEX = regex.compile(rf".*[{HYPHENS}]\s*$", regex.DOTALL)
NEXT_LOWER_REGEX = regex.compile(r"^\s*[\p{Ll}]")
REMOVE_TAGS_REGEX = re.compile(r"<[^>]+>")


def remove_tags(text):
    return REMOVE_TAGS_REGEX.sub("", text)


def replace_last(string, old, new):
    matches = list(re.finditer(old, string))
    if not matches:
        return string
    last_match = matches[-1]
    return string[: last_match.start()] + new + string[last_match.end() :]


def strip_trailing_hyphens(line_text, next_line_text, line_html) -> str:
    if HYPHEN_REGEX.match(line_text) and NEXT_LOWER_REGEX.match(next_line_text):
        line_html = replace_last(line_html, rf"[{HYPHENS}]\s*", "")

    return line_html


class Line(Block):
    block_type: BlockTypes = BlockTypes.Line
    block_description: str = "A line of text."
    formats: List[Literal["math"]] | None = (
        None  # Sometimes we want to set math format at the line level, not span
    )

    def formatted_text(self, document, skip_urls=False):
        text = ""
        for block in self.contained_blocks(document, (BlockTypes.Span,)):
            block_text = html.escape(block.text)

            if block.has_superscript:
                block_text = re.sub(r"^([0-9\W]+)(.*)", r"<sup>\1</sup>\2", block_text)
                if "<sup>" not in block_text:
                    block_text = f"<sup>{block_text}</sup>"

            if block.url and not skip_urls:
                safe_url = sanitize_url(block.url)
                if safe_url:
                    block_text = f"<a href='{safe_url}'>{block_text}</a>"

            if block.math:
                text += f"<math display='inline'>{block_text}</math>"
            else:
                if block.italic:
                    block_text = f"<i>{block_text}</i>"
                if block.bold:
                    block_text = f"<b>{block_text}</b>"
                text += block_text

        return text

    def assemble_html(self, document, child_blocks, parent_structure, block_config):
        template = ""
        for c in child_blocks:
            template += c.html

        raw_text = remove_tags(template).strip()
        structure_idx = parent_structure.index(self.id)
        if structure_idx < len(parent_structure) - 1:
            next_block_id = parent_structure[structure_idx + 1]
            next_line = document.get_block(next_block_id)
            next_line_raw_text = next_line.raw_text(document)
            template = strip_trailing_hyphens(raw_text, next_line_raw_text, template)
        else:
            template = template.strip(
                " "
            )  # strip any trailing whitespace from the last line
        return template

    def render(
        self, document, parent_structure, section_hierarchy=None, block_config=None
    ):
        child_content = []
        if self.structure is not None and len(self.structure) > 0:
            for block_id in self.structure:
                block = document.get_block(block_id)
                child_content.append(
                    block.render(
                        document, parent_structure, section_hierarchy, block_config
                    )
                )

        return BlockOutput(
            html=self.assemble_html(
                document, child_content, parent_structure, block_config
            ),
            polygon=self.polygon,
            id=self.id,
            children=[],
            section_hierarchy=section_hierarchy,
        )

    def merge(self, other: "Line"):
        self.polygon = self.polygon.merge([other.polygon])

        # Handle merging structure with Nones
        if self.structure is None:
            self.structure = other.structure
        elif other.structure is not None:
            self.structure = self.structure + other.structure

        # Merge formats with Nones
        if self.formats is None:
            self.formats = other.formats
        elif other.formats is not None:
            self.formats = list(set(self.formats + other.formats))
