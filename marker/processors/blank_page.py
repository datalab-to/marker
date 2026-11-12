from typing import Annotated

from PIL import Image

from marker.processors import BaseProcessor
from marker.schema import BlockTypes
from marker.schema.blocks import Block
from marker.schema.document import Document
from marker.utils.image import is_blank_image

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

    def is_blank(self, image: Image.Image) -> bool:
        return is_blank_image(image)

    def __call__(self, document: Document):
        if not self.filter_blank_pages:
            return

        for page in document.pages:
            structure_blocks = page.structure_blocks(document)
            if not structure_blocks or len(structure_blocks) > 1:
                continue

            full_page_block: Block = structure_blocks[0]

            if (
                full_page_block.block_type
                in [BlockTypes.Picture, BlockTypes.Figure, BlockTypes.Table]
                and page.polygon.intersection_pct(full_page_block.polygon)
                > self.full_page_block_intersection_threshold
                and self.is_blank(full_page_block.get_image(document))
            ):
                logger.debug(f"Removing blank block {full_page_block.id}")
                page.remove_structure_items([full_page_block.id])
                full_page_block.removed = True
