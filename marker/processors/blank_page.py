from typing import Annotated

from PIL import Image
import numpy as np
import cv2

from marker.processors import BaseProcessor
from marker.schema import BlockTypes
from marker.schema.blocks import Block
from marker.schema.document import Document

from marker.logger import get_logger

logger = get_logger()


class BlankPageProcessor(BaseProcessor):
    """
    A processor to filter out blank pages detected as a single layout block
    """

    full_page_block_intersection_threshold: Annotated[
        float, "Threshold to detect blank pages at"
    ] = 0.8
    filter_blank_pages: Annotated[bool, "Remove blank pages detected as images."] = (
        False
    )

    def is_blank(self, image: Image.Image):
        image = np.asarray(image)
        if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
            # Handle empty image case
            return True

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # Adaptive threshold (inverse for text as white)
        binarized = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binarized, connectivity=8
        )
        cleaned = np.zeros_like(binarized)
        for i in range(1, num_labels):  # skip background
            cleaned[labels == i] = 255

        kernel = np.ones((1, 5), np.uint8)
        dilated = cv2.dilate(cleaned, kernel, iterations=3)
        b = dilated / 255
        return b.sum() == 0

    def __call__(self, document: Document):
        if not self.filter_blank_pages:
            return

        for page in document.pages:
            structure_blocks = page.structure_blocks(document)
            if not structure_blocks or len(structure_blocks) > 1:
                continue

            full_page_block: Block = structure_blocks[0]

            conditions = [
                full_page_block.block_type in [BlockTypes.Picture, BlockTypes.Figure],
                self.is_blank(full_page_block.get_image(document)),
                page.polygon.intersection_area(full_page_block.polygon)
                > self.full_page_block_intersection_threshold,
            ]

            if all(conditions):
                logger.debug(f"Removing blank block {full_page_block.id}")
                page.remove_structure_items([full_page_block.id])
                full_page_block.removed = True
