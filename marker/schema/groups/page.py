from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image, ImageDraw

from pdftext.schema import Reference
from pydantic import computed_field

from marker.providers import ProviderOutput
from marker.schema import BlockTypes
from marker.schema.blocks import Block, BlockId, Text
from marker.schema.blocks.base import BlockMetadata
from marker.schema.groups.base import Group
from marker.schema.polygon import PolygonBox
from marker.util import matrix_intersection_area, sort_text_lines

LINE_MAPPING_TYPE = List[Tuple[int, ProviderOutput]]


class PageGroup(Group):
    block_type: BlockTypes = BlockTypes.Page
    # This is bytes if it is serialized
    lowres_image: Image.Image | None | bytes = None
    highres_image: Image.Image | None | bytes = None
    children: List[Union[Any, Block]] | None = None
    layout_sliced: bool = (
        False  # Whether the layout model had to slice the image (order may be wrong)
    )
    excluded_block_types: Sequence[BlockTypes] = (
        BlockTypes.Line,
        BlockTypes.Span,
    )
    maximum_assignment_distance: float = 20  # pixels
    block_description: str = "A single page in the document."
    refs: List[Reference] | None = None
    ocr_errors_detected: bool = False

    def incr_block_id(self):
        if self.block_id is None:
            self.block_id = 0
        else:
            self.block_id += 1

    def add_child(self, block: Block):
        if self.children is None:
            self.children = [block]
        else:
            self.children.append(block)

    def get_image(
        self,
        *args,
        highres: bool = False,
        remove_blocks: Sequence[BlockTypes] | None = None,
        **kwargs,
    ):
        image = self.highres_image if highres else self.lowres_image

        # Check if RGB, convert if needed
        if isinstance(image, Image.Image) and image.mode != "RGB":
            image = image.convert("RGB")

        # Avoid double OCR for certain elements
        if remove_blocks:
            image = image.copy()
            draw = ImageDraw.Draw(image)
            bad_blocks = [
                block
                for block in self.current_children
                if block.block_type in remove_blocks
            ]
            for bad_block in bad_blocks:
                poly = bad_block.polygon.rescale(self.polygon.size, image.size).polygon
                poly = [(int(p[0]), int(p[1])) for p in poly]
                draw.polygon(poly, fill="white")

        return image

    @computed_field
    @property
    def current_children(self) -> List[Block]:
        return [child for child in self.children if not child.removed]

    def get_next_block(
        self,
        block: Optional[Block] = None,
        ignored_block_types: Optional[List[BlockTypes]] = None,
    ):
        if ignored_block_types is None:
            ignored_block_types = []

        structure_idx = 0
        if block is not None:
            structure_idx = self.structure.index(block.id) + 1

        # Iterate over blocks following the given block
        for next_block_id in self.structure[structure_idx:]:
            if next_block_id.block_type not in ignored_block_types:
                return self.get_block(next_block_id)

        return None  # No valid next block found

    def get_prev_block(self, block: Block):
        block_idx = self.structure.index(block.id)
        if block_idx > 0:
            return self.get_block(self.structure[block_idx - 1])
        return None

    def add_block(self, block_cls: type[Block], polygon: PolygonBox) -> Block:
        self.incr_block_id()
        block = block_cls(
            polygon=polygon,
            block_id=self.block_id,
            page_id=self.page_id,
        )
        self.add_child(block)
        return block

    def add_full_block(self, block: Block) -> Block:
        self.incr_block_id()
        block.block_id = self.block_id
        self.add_child(block)
        return block

    def get_block(self, block_id: BlockId) -> Block | None:
        block: Block = self.children[block_id.block_id]
        assert block.block_id == block_id.block_id
        return block

    def assemble_html(
        self, document, child_blocks, parent_structure=None, block_config=None
    ):
        template = ""
        for c in child_blocks:
            template += f"<content-ref src='{c.id}'></content-ref>"
        return template

    def compute_line_block_intersections(
        self, blocks: List[Block], provider_outputs: List[ProviderOutput]
    ):
        max_intersections = {}

        block_bboxes = [block.polygon.bbox for block in blocks]
        line_bboxes = [
            provider_output.line.polygon.bbox for provider_output in provider_outputs
        ]

        intersection_matrix = matrix_intersection_area(line_bboxes, block_bboxes)

        for line_idx, line in enumerate(provider_outputs):
            intersection_line = intersection_matrix[line_idx]
            if intersection_line.sum() == 0:
                continue

            max_intersection = intersection_line.argmax()
            max_intersections[line_idx] = (
                intersection_matrix[line_idx, max_intersection],
                blocks[max_intersection].id,
            )
        return max_intersections

    def replace_block(self, block: Block, new_block: Block):
        # Handles incrementing the id
        self.add_full_block(new_block)

        # Replace block id in structure
        super().replace_block(block, new_block)

        # Replace block in structure of children
        for child in self.children:
            child.replace_block(block, new_block)

        # Mark block as removed
        block.removed = True

    def identify_missing_blocks(
        self,
        provider_line_idxs: List[int],
        provider_outputs: List[ProviderOutput],
        assigned_line_idxs: set[int],
    ):
        new_blocks = []
        new_block = None
        for line_idx in provider_line_idxs:
            if line_idx in assigned_line_idxs:
                continue

            # if the unassociated line is a new line with minimal area, we can skip it
            if (
                provider_outputs[line_idx].line.polygon.area <= 1
                and provider_outputs[line_idx].raw_text == "\n"
            ):
                continue

            if new_block is None:
                new_block = [(line_idx, provider_outputs[line_idx])]
            elif all(
                [
                    new_block[-1][0] + 1 == line_idx,
                    provider_outputs[line_idx].line.polygon.center_distance(
                        new_block[-1][1].line.polygon
                    )
                    < self.maximum_assignment_distance,
                ]
            ):
                new_block.append((line_idx, provider_outputs[line_idx]))
            else:
                new_blocks.append(new_block)
                new_block = [(line_idx, provider_outputs[line_idx])]
            assigned_line_idxs.add(line_idx)
        if new_block:
            new_blocks.append(new_block)

        return new_blocks

    def create_missing_blocks(
        self,
        new_blocks: List[LINE_MAPPING_TYPE],
        block_lines: Dict[BlockId, LINE_MAPPING_TYPE],
    ):
        for new_block in new_blocks:
            block = self.add_block(Text, new_block[0][1].line.polygon)
            block.source = "heuristics"
            block_lines[block.id] = new_block

            min_dist_idx = None
            min_dist = None
            for existing_block_id in self.structure:
                existing_block = self.get_block(existing_block_id)
                if existing_block.block_type in self.excluded_block_types:
                    continue
                # We want to assign to blocks closer in y than x
                dist = block.polygon.center_distance(
                    existing_block.polygon, x_weight=5, absolute=True
                )
                if dist > 0 and min_dist_idx is None or dist < min_dist:
                    min_dist = dist
                    min_dist_idx = existing_block.id

            if min_dist_idx is not None:
                existing_idx = self.structure.index(min_dist_idx)
                self.structure.insert(existing_idx + 1, block.id)
            else:
                self.structure.append(block.id)

    def add_initial_blocks(
        self,
        block_lines: Dict[BlockId, LINE_MAPPING_TYPE],
        text_extraction_method: str,
        keep_chars: bool = False,
    ):
        # Add lines to the proper blocks, sorted in order
        for block_id, lines in block_lines.items():
            line_extraction_methods = set(
                [line[1].line.text_extraction_method for line in lines]
            )
            if len(line_extraction_methods) == 1:
                lines = sorted(lines, key=lambda x: x[0])
                lines = [line for _, line in lines]
            else:
                lines = [line for _, line in lines]
                line_polygons = [line.line.polygon for line in lines]
                sorted_line_polygons = sort_text_lines(line_polygons)
                argsort = [line_polygons.index(p) for p in sorted_line_polygons]
                lines = [lines[i] for i in argsort]

            block = self.get_block(block_id)
            for provider_output in lines:
                line = provider_output.line
                spans = provider_output.spans
                self.add_full_block(line)
                block.add_structure(line)
                block.polygon = block.polygon.merge([line.polygon])
                block.text_extraction_method = text_extraction_method
                for span_idx, span in enumerate(spans):
                    self.add_full_block(span)
                    line.add_structure(span)

                    if not keep_chars:
                        continue

                    # Provider doesn't have chars
                    if len(provider_output.chars) == 0:
                        continue

                    # Loop through characters associated with the span
                    for char in provider_output.chars[span_idx]:
                        char.page_id = self.page_id
                        self.add_full_block(char)
                        span.add_structure(char)

    def merge_blocks(
        self,
        provider_outputs: List[ProviderOutput],
        text_extraction_method: str,
        keep_chars: bool = False,
    ):
        provider_line_idxs = list(range(len(provider_outputs)))
        valid_blocks = [
            block
            for block in self.current_children  # ensure we only look at children that haven't been replaced
            if block.block_type not in self.excluded_block_types
        ]

        max_intersections = self.compute_line_block_intersections(
            valid_blocks, provider_outputs
        )

        # Try to assign lines by intersection
        assigned_line_idxs = set()
        block_lines = defaultdict(list)
        for line_idx, provider_output in enumerate(provider_outputs):
            if line_idx in max_intersections:
                block_id = max_intersections[line_idx][1]
                block_lines[block_id].append((line_idx, provider_output))
                assigned_line_idxs.add(line_idx)

        # If no intersection, assign by distance
        for line_idx in set(provider_line_idxs).difference(assigned_line_idxs):
            min_dist = None
            min_dist_idx = None
            provider_output: ProviderOutput = provider_outputs[line_idx]
            line = provider_output.line
            for block in valid_blocks:
                # We want to assign to blocks closer in y than x
                dist = line.polygon.center_distance(block.polygon, x_weight=5)
                if min_dist_idx is None or dist < min_dist:
                    min_dist = dist
                    min_dist_idx = block.id

            if min_dist_idx is not None and min_dist < self.maximum_assignment_distance:
                block_lines[min_dist_idx].append((line_idx, provider_output))
                assigned_line_idxs.add(line_idx)

        # This creates new blocks to hold anything too far away
        new_blocks = self.identify_missing_blocks(
            provider_line_idxs, provider_outputs, assigned_line_idxs
        )
        self.create_missing_blocks(new_blocks, block_lines)

        # Add blocks to the page
        self.add_initial_blocks(block_lines, text_extraction_method, keep_chars)

    def aggregate_block_metadata(self) -> BlockMetadata:
        if self.metadata is None:
            self.metadata = BlockMetadata()

        for block in self.current_children:
            if block.metadata is not None:
                self.metadata = self.metadata.merge(block.metadata)
        return self.metadata
