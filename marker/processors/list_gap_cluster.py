from copy import deepcopy
from statistics import median
from typing import Annotated, List

from marker.processors import BaseProcessor
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.registry import get_block_class


class ListItemGapClusterProcessor(BaseProcessor):
    """
    Split multi-line ListItem blocks into per-item ListItems by clustering
    Line children on vertical gap. Lines whose top-to-prev-bottom gap is at
    most `gap_factor * median_gap` (with a minimum of `min_gap_pt`) are
    treated as continuations of the previous item; larger gaps start a new
    item. Each emitted ListItem's polygon is the merge of its lines'
    polygons.

    Prefix-free: uses only line bboxes.
    """
    block_types = (BlockTypes.ListItem,)
    min_lines: Annotated[
        int, "Minimum number of Line children before clustering applies.",
    ] = 2
    gap_factor: Annotated[
        float,
        "Multiplier on the median inter-line gap. Anything <= this is treated as 'tight'.",
    ] = 1.5
    min_gap_pt: Annotated[
        float,
        "Floor on the 'tight' gap (PDF points). Below this, lines always belong to the same item.",
    ] = 2.0

    def __call__(self, document: Document):
        ListItemCls = get_block_class(BlockTypes.ListItem)
        for page in document.pages:
            if not page.structure:
                continue
            new_structure = []
            for bid in list(page.structure):
                block = page.get_block(bid)
                if block is None or block.block_type != BlockTypes.ListItem:
                    new_structure.append(bid)
                    continue

                line_ids = list(block.structure or [])
                lines = []
                for lid in line_ids:
                    ln = page.get_block(lid)
                    if ln is not None and ln.block_type == BlockTypes.Line:
                        lines.append(ln)

                if len(lines) < self.min_lines:
                    new_structure.append(bid)
                    continue

                # Sort top-to-bottom (defensive — should already be in order)
                lines.sort(key=lambda l: l.polygon.y_start)

                gaps = [
                    max(0.0, lines[i].polygon.y_start - lines[i - 1].polygon.y_end)
                    for i in range(1, len(lines))
                ]
                if not gaps:
                    new_structure.append(bid)
                    continue

                med = median(gaps)
                tight_threshold = max(self.min_gap_pt, med * self.gap_factor)

                clusters: List[List] = [[lines[0]]]
                for i in range(1, len(lines)):
                    g = lines[i].polygon.y_start - lines[i - 1].polygon.y_end
                    if g <= tight_threshold:
                        clusters[-1].append(lines[i])
                    else:
                        clusters.append([lines[i]])

                if len(clusters) == 1:
                    # Nothing to split — keep the original block intact
                    new_structure.append(bid)
                    continue

                for cluster in clusters:
                    merged_poly = deepcopy(cluster[0].polygon)
                    if len(cluster) > 1:
                        merged_poly = merged_poly.merge([l.polygon for l in cluster[1:]])
                    new_li = page.add_block(ListItemCls, merged_poly)
                    new_li.structure = [l.id for l in cluster]
                    new_structure.append(new_li.id)
                block.removed = True

            page.structure = new_structure
