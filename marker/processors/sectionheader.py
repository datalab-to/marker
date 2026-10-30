import re
import warnings
from typing import Annotated, Dict, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

from marker.processors import BaseProcessor
from marker.schema import BlockTypes
from marker.schema.document import Document

HEADING_TAG_PATTERN = re.compile(r"<h(\d)", re.IGNORECASE)

# Ignore sklearn warning about not converging
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class SectionHeaderProcessor(BaseProcessor):
    """
    A processor for recognizing section headers in the document.
    """

    block_types = (BlockTypes.SectionHeader,)
    level_count: Annotated[
        int,
        "The number of levels to use for headings.",
    ] = 4
    merge_threshold: Annotated[
        float,
        "The minimum gap between headings to consider them part of the same group.",
    ] = 0.25
    default_level: Annotated[
        int,
        "The default heading level to use if no heading level is detected.",
    ] = 2
    height_tolerance: Annotated[
        float,
        "The minimum height of a heading to consider it a heading.",
    ] = 0.99

    def __call__(self, document: Document):
        line_heights: Dict[int, float] = {}
        for page in document.pages:
            # Iterate children to grab all section headers
            for block in page.children:
                if block.block_type not in self.block_types:
                    continue
                if block.html:
                    # OCR'd header - keep the model's heading level for the
                    # section hierarchy
                    match = HEADING_TAG_PATTERN.search(block.html)
                    if match:
                        block.heading_level = min(int(match.group(1)), 6)
                    continue
                if block.structure:
                    line_heights[block.id] = block.line_height(document)
                else:
                    line_heights[block.id] = 0
                    block.ignore_for_output = (
                        True  # Don't output an empty section header
                    )

        flat_line_heights = list(line_heights.values())
        heading_ranges = self.bucket_headings(flat_line_heights)

        for page in document.pages:
            # Iterate children to grab all section headers
            for block in page.children:
                if block.block_type not in self.block_types:
                    continue
                if block.html:
                    continue
                block_height = line_heights.get(block.id, 0)
                if block_height > 0:
                    for idx, (min_height, max_height) in enumerate(heading_ranges):
                        if block_height >= min_height * self.height_tolerance:
                            block.heading_level = idx + 1
                            break

                if block.heading_level is None:
                    block.heading_level = self.default_level

    def bucket_headings(self, line_heights: List[float]):
        if len(line_heights) <= self.level_count:
            return []

        data = np.asarray(line_heights).reshape(-1, 1)
        # Clamp the cluster count to the number of distinct (rounded) sizes so
        # that rendering jitter doesn't invent heading levels.
        distinct_sizes = len(np.unique(np.round(data)))
        clusters = min(self.level_count, distinct_sizes)
        if clusters < 2:
            return []
        labels = KMeans(n_clusters=clusters, random_state=0, n_init="auto").fit_predict(
            data
        )
        data_labels = np.concatenate([data, labels.reshape(-1, 1)], axis=1)
        # Sort rows by height, keeping each height paired with its label.
        data_labels = data_labels[data_labels[:, 0].argsort()]

        cluster_means = {
            int(label): float(np.mean(data_labels[data_labels[:, 1] == label, 0]))
            for label in np.unique(labels)
        }
        label_max = None
        label_min = None
        heading_ranges = []
        prev_cluster = None
        for row in data_labels:
            value, label = row
            value = float(value)
            label = int(label)
            if prev_cluster is not None and label != prev_cluster:
                prev_cluster_mean = cluster_means[prev_cluster]
                cluster_mean = cluster_means[label]
                if cluster_mean * self.merge_threshold < prev_cluster_mean:
                    heading_ranges.append((label_min, label_max))
                    label_min = None
                    label_max = None

            label_min = value if label_min is None else min(label_min, value)
            label_max = value if label_max is None else max(label_max, value)
            prev_cluster = label

        if label_min is not None:
            heading_ranges.append((label_min, label_max))

        heading_ranges = sorted(heading_ranges, reverse=True)

        return heading_ranges
