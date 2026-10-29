import re
from typing import Annotated

import regex

from marker.builders import BaseBuilder
from marker.schema import BlockTypes
from marker.schema.blocks import ListItem, Text
from marker.schema.document import Document
from marker.schema.groups import ListGroup
from marker.schema.groups.page import PageGroup
from marker.schema.registry import get_block_class

HYPHENS = r"-—¬"

# Bullets, numbers ("1.", "1)"), letters ("a.", "A)"), roman numerals
LIST_ITEM_START_PATTERN = re.compile(
    r"^\s*(?:[•●○ഠ ം◦■▪▫–—-]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)]|\(?[ivxlcIVXLC]{1,5}[.)])\s"
)


class StructureBuilder(BaseBuilder):
    """
    A builder for grouping blocks together based on their structure.
    """

    gap_threshold: Annotated[
        float,
        "The minimum gap between blocks to consider them part of the same group.",
    ] = 0.05
    list_gap_threshold: Annotated[
        float,
        "The minimum gap between list items to consider them part of the same group.",
    ] = 0.1
    defer_images: Annotated[
        bool,
        "Defer image blocks that interrupt an active paragraph to the end of the paragraph.",
    ] = True

    def __init__(self, config=None):
        super().__init__(config)

    def __call__(self, document: Document):
        for page in document.pages:
            self.group_caption_blocks(page)
            self.defer_interrupted_images(document, page)
            self.split_list_groups(document, page)
            self.group_lists(page)
            self.unmark_lists(page)

    def defer_interrupted_images(self, document: Document, page: PageGroup):
        if not self.defer_images or not page.structure:
            return

        image_block_types = {
            BlockTypes.Figure,
            BlockTypes.Picture,
            BlockTypes.Diagram,
            BlockTypes.FigureGroup,
            BlockTypes.PictureGroup,
        }
        text_block_types = {
            BlockTypes.Text,
            BlockTypes.TextInlineMath,
        }
        sentence_end_re = re.compile(r"[.!?:][\"')\]]*$")
        hyphen_re = re.compile(rf"[{HYPHENS}]$")

        changed = True
        max_passes = 10
        pass_num = 0
        while changed and pass_num < max_passes:
            changed = False
            pass_num += 1

            for idx, block_id in enumerate(page.structure):
                block = page.get_block(block_id)
                if block.block_type not in image_block_types:
                    continue

                # Check if accompanied by an ungrouped caption directly after
                img_block_ids = [block_id]
                if idx + 1 < len(page.structure):
                    next_b = page.get_block(page.structure[idx + 1])
                    if next_b.block_type == BlockTypes.Caption:
                        img_block_ids.append(next_b.id)

                prev_idx = idx - 1
                if prev_idx < 0:
                    continue

                prev_block = page.get_block(page.structure[prev_idx])
                if prev_block.block_type not in text_block_types:
                    continue

                next_idx = idx + len(img_block_ids)
                if next_idx >= len(page.structure):
                    continue

                next_block = page.get_block(page.structure[next_idx])
                if next_block.block_type not in text_block_types:
                    continue

                prev_text = prev_block.raw_text(document).strip()
                next_text = next_block.raw_text(document).strip()
                if not prev_text or not next_text:
                    continue

                ends_with_sentence_punct = bool(sentence_end_re.search(prev_text))
                starts_with_lowercase = bool(regex.match(r"^\p{Ll}", next_text))
                ends_with_hyphen = bool(hyphen_re.search(prev_text))

                is_interrupted = (
                    ends_with_hyphen
                    or (not ends_with_sentence_punct)
                    or starts_with_lowercase
                )
                if not is_interrupted:
                    continue

                # Find the end of the active paragraph
                curr_idx = next_idx
                while curr_idx < len(page.structure):
                    curr_b = page.get_block(page.structure[curr_idx])
                    if curr_b.block_type not in text_block_types:
                        curr_idx -= 1
                        break

                    curr_text = curr_b.raw_text(document).strip()
                    if sentence_end_re.search(curr_text) and not hyphen_re.search(curr_text):
                        if curr_idx + 1 < len(page.structure):
                            after_b = page.get_block(page.structure[curr_idx + 1])
                            if after_b.block_type not in text_block_types:
                                break
                            after_text = after_b.raw_text(document).strip()
                            if not regex.match(r"^\p{Ll}", after_text):
                                break
                            curr_idx += 1
                        else:
                            break
                    else:
                        curr_idx += 1

                curr_idx = min(curr_idx, len(page.structure) - 1)
                continuation_ids = page.structure[next_idx : curr_idx + 1]
                if not continuation_ids:
                    continue

                # Merge continuation text blocks into prev_block
                merged_polygons = [prev_block.polygon]
                for cont_id in continuation_ids:
                    cont_block = page.get_block(cont_id)
                    if prev_block.structure is not None and cont_block.structure is not None:
                        prev_block.structure.extend(cont_block.structure)
                    elif cont_block.structure is not None:
                        prev_block.structure = cont_block.structure

                    if prev_block.html is not None and cont_block.html is not None:
                        h1 = re.sub(r"</p>\s*$", "", prev_block.html.strip())
                        h2 = re.sub(r"^\s*<p[^>]*>", "", cont_block.html.strip())
                        prev_block.html = f"{h1} {h2}</p>"
                    elif cont_block.html is not None:
                        prev_block.html = cont_block.html

                    merged_polygons.append(cont_block.polygon)
                    cont_block.removed = True

                prev_block.polygon = prev_block.polygon.merge(merged_polygons)

                # Reconstruct page.structure:
                # Remove img_block_ids and continuation_ids, then reinsert img_block_ids right after prev_block
                to_remove = set(img_block_ids + continuation_ids)
                new_structure = [bid for bid in page.structure if bid not in to_remove]
                insert_pos = new_structure.index(prev_block.id) + 1
                for offset, img_id in enumerate(img_block_ids):
                    new_structure.insert(insert_pos + offset, img_id)

                page.structure = new_structure
                changed = True
                break


    def split_list_groups(self, document: Document, page: PageGroup):
        """Layout emits whole ListGroup regions. On pdftext pages those contain
        raw Lines - group consecutive lines into ListItems on bullet boundaries
        so the renderer can emit <li> items. OCR'd groups already carry html."""
        for block_id in page.structure.copy():
            block = page.get_block(block_id)
            if block.block_type != BlockTypes.ListGroup or block.html:
                continue
            if not block.structure:
                # Layout emitted a list region that never received children -
                # drop it rather than rendering an empty <ul>.
                block.removed = True
                page.structure.remove(block_id)
                continue

            lines = block.structure_blocks(page)
            if any(b.block_type != BlockTypes.Line for b in lines):
                continue

            item_line_groups = []
            current_item = []
            found_bullets = 0
            for line in lines:
                if LIST_ITEM_START_PATTERN.match(line.raw_text(document)):
                    found_bullets += 1
                    if current_item:
                        item_line_groups.append(current_item)
                    current_item = [line]
                else:
                    current_item.append(line)
            if current_item:
                item_line_groups.append(current_item)

            if found_bullets == 0:
                # Not actually a list - demote to text
                generated_block = Text(
                    polygon=block.polygon,
                    page_id=block.page_id,
                    structure=block.structure,
                )
                page.replace_block(block, generated_block)
                continue

            item_ids = []
            for item_lines in item_line_groups:
                polygon = item_lines[0].polygon.merge(
                    [line.polygon for line in item_lines]
                )
                list_item = page.add_block(ListItem, polygon)
                list_item.structure = [line.id for line in item_lines]
                item_ids.append(list_item.id)
            block.structure = item_ids

    def group_caption_blocks(self, page: PageGroup):
        gap_threshold_px = self.gap_threshold * page.polygon.height
        static_page_structure = page.structure.copy()
        remove_ids = list()

        for i, block_id in enumerate(static_page_structure):
            block = page.get_block(block_id)
            if block.block_type not in [
                BlockTypes.Table,
                BlockTypes.Figure,
                BlockTypes.Picture,
            ]:
                continue

            if block.id in remove_ids:
                continue

            block_structure = [block_id]
            selected_polygons = [block.polygon]
            caption_types = [BlockTypes.Caption, BlockTypes.Footnote]

            prev_block = page.get_prev_block(block)
            next_block = page.get_next_block(block)

            if (
                prev_block
                and prev_block.block_type in caption_types
                and prev_block.polygon.minimum_gap(block.polygon) < gap_threshold_px
                and prev_block.id not in remove_ids
            ):
                block_structure.insert(0, prev_block.id)
                selected_polygons.append(prev_block.polygon)

            if (
                next_block
                and next_block.block_type in caption_types
                and next_block.polygon.minimum_gap(block.polygon) < gap_threshold_px
            ):
                block_structure.append(next_block.id)
                selected_polygons.append(next_block.polygon)

            if len(block_structure) > 1:
                # Create a merged block
                new_block_cls = get_block_class(
                    BlockTypes[block.block_type.name + "Group"]
                )
                new_polygon = block.polygon.merge(selected_polygons)
                group_block = page.add_block(new_block_cls, new_polygon)
                group_block.structure = block_structure

                # Update the structure of the page to reflect the new block
                page.update_structure_item(block_id, group_block.id)
                remove_ids.extend(block_structure)
        page.remove_structure_items(remove_ids)

    def group_lists(self, page: PageGroup):
        gap_threshold_px = self.list_gap_threshold * page.polygon.height
        static_page_structure = page.structure.copy()
        remove_ids = list()
        for i, block_id in enumerate(static_page_structure):
            block = page.get_block(block_id)
            if block.block_type not in [BlockTypes.ListItem]:
                continue

            if block.id in remove_ids:
                continue

            block_structure = [block_id]
            selected_polygons = [block.polygon]

            for j, next_block_id in enumerate(page.structure[i + 1 :]):
                next_block = page.get_block(next_block_id)
                if all(
                    [
                        next_block.block_type == BlockTypes.ListItem,
                        next_block.polygon.minimum_gap(selected_polygons[-1])
                        < gap_threshold_px,
                    ]
                ):
                    block_structure.append(next_block_id)
                    selected_polygons.append(next_block.polygon)
                else:
                    break

            if len(block_structure) > 1:
                new_polygon = block.polygon.merge(selected_polygons)
                group_block = page.add_block(ListGroup, new_polygon)
                group_block.structure = block_structure

                # Update the structure of the page to reflect the new block
                page.update_structure_item(block_id, group_block.id)
                remove_ids.extend(block_structure)

        page.remove_structure_items(remove_ids)

    def unmark_lists(self, page: PageGroup):
        # If lists aren't grouped, unmark them as list items
        for block_id in page.structure:
            block = page.get_block(block_id)
            if block.block_type == BlockTypes.ListItem:
                generated_block = Text(
                    polygon=block.polygon,
                    page_id=block.page_id,
                    structure=block.structure,
                )
                page.replace_block(block, generated_block)
