from typing import Annotated, List, Tuple

from bs4 import BeautifulSoup
from PIL import Image
from surya.layout.schema import LayoutBox, LayoutResult
from surya.recognition import RecognitionPredictor, _detect_repeat_loop

from marker.builders import BaseBuilder
from marker.logger import get_logger
from marker.providers.pdf import PdfProvider
from marker.schema import BlockTypes
from marker.schema.blocks import BlockId
from marker.schema.document import Document
from marker.schema.groups import PageGroup
from marker.schema.labels import (
    block_type_to_surya_label,
    surya_label_to_block_type,
)
from marker.schema.polygon import PolygonBox
from marker.schema.registry import get_block_class

logger = get_logger()

# Attributes the model can emit for debugging that shouldn't reach output
STRIP_ATTRIBUTES = ("data-bbox", "data-label")


class OcrBuilder(BaseBuilder):
    """
    OCRs pages whose embedded text was unusable. Each page is recognized in a
    single full-page VLM call (the most accurate path - tight block crops
    lose context the model needs), and the page structure is rebuilt from the
    returned blocks with their HTML set directly. Pages where full-page OCR
    fails are automatically retried block-by-block against the existing
    layout blocks.
    """

    disable_tqdm: Annotated[
        bool,
        "Disable tqdm progress bars.",
    ] = False
    ocr_full_page: Annotated[
        bool,
        "Use a single full-page OCR request per page (more accurate, and",
        "faster on CPU) and rebuild the page structure from the result. When",
        "False, each existing layout block is OCR'd individually instead.",
        "Applies to both modes; fast mode still skips OCR on clean pages.",
    ] = True
    # In block mode, tables/forms/TOCs are handled by the TableProcessor;
    # pictures, figures, and diagrams contain no text.
    skip_ocr_blocks: Annotated[
        List[BlockTypes],
        "Blocktypes which should not be OCRed by this builder.",
    ] = [
        BlockTypes.Figure,
        BlockTypes.Picture,
        BlockTypes.Diagram,
        BlockTypes.Table,
        BlockTypes.Form,
        BlockTypes.TableOfContents,
    ]
    default_token_budget: Annotated[
        int,
        "Token budget for OCRing a block when the layout model did not provide an estimate.",
    ] = 1200

    def __init__(self, recognition_model: RecognitionPredictor, config=None):
        super().__init__(config)

        self.recognition_model = recognition_model

    def __call__(self, document: Document, provider: PdfProvider):
        self.recognition_model.disable_tqdm = self.disable_tqdm

        # Pages whose embedded text was unusable get OCR'd wholesale.
        full_page_pages = [
            page for page in document.pages if page.text_extraction_method == "surya"
        ]
        # When the whole page needs OCR, a single full-page request is both
        # faster (one decode vs N block decodes — ~7x on llama.cpp, which pays
        # heavy per-request overhead + KV-cache contention for many small
        # requests) and more accurate (tight block crops lose context). This
        # holds in fast mode too: clean pages already skip OCR entirely via
        # pdftext, so a "surya" page is one whose entire content needs OCR.
        # Fast mode's block-by-block decoding is reserved for individual
        # garbled/missing blocks on otherwise-clean pages (ocr_flagged_blocks).
        use_full_page = self.ocr_full_page
        if full_page_pages:
            images, layout_results, block_ids = self.build_block_requests(
                document, full_page_pages
            )
            if use_full_page:
                # The synthetic layouts serve as the per-page fallback if the
                # full-page output fails or loops. When layout was skipped
                # (force_ocr), pass None so surya lazily runs its own layout
                # for just the pages that need the fallback.
                if all(len(layout.bboxes) == 0 for layout in layout_results):
                    layout_results = None
                recognition_results = self.recognition_model(
                    images=images, layout_results=layout_results, full_page=True
                )
                self.replace_page_structure(
                    document, full_page_pages, images, recognition_results
                )
            else:
                recognition_results = self.recognition_model(
                    images=images, layout_results=layout_results, full_page=False
                )
                self.apply_block_html(document, recognition_results, block_ids)

        # Individual blocks on otherwise-good pages whose embedded text was
        # missing or garbled get OCR'd in place, keeping the rest of the
        # page's pdftext content. (In balanced mode LineBuilder promotes any
        # page with flagged blocks to full-page OCR, so this only fires in
        # fast mode - the cheap, surgical repair.)
        self.ocr_flagged_blocks(document)

    def ocr_flagged_blocks(self, document: Document):
        images = []
        layout_results = []
        block_ids = []
        for page in document.pages:
            if page.text_extraction_method != "pdftext":
                continue
            flagged = [
                b
                for b in page.structure_blocks(document)
                if b.text_extraction_method == "surya"
                and b.block_type not in self.skip_ocr_blocks
            ]
            if not flagged:
                continue

            image = page.get_image(highres=True)
            image_size = image.size
            page_size = page.polygon.size
            boxes = []
            page_block_ids = []
            for block in flagged:
                polygon = block.polygon.rescale(page_size, image_size).fit_to_bounds(
                    (0, 0, *image_size)
                )
                boxes.append(
                    LayoutBox(
                        polygon=polygon.polygon,
                        label=block_type_to_surya_label(block.block_type),
                        raw_label=str(block.block_type),
                        position=len(boxes),
                        count=block.layout_token_count or self.default_token_budget,
                    )
                )
                page_block_ids.append(block.id)
            images.append(image)
            layout_results.append(
                LayoutResult(bboxes=boxes, image_bbox=[0, 0, *image_size])
            )
            block_ids.append(page_block_ids)

        if not images:
            return

        recognition_results = self.recognition_model(
            images=images, layout_results=layout_results, full_page=False
        )
        self.apply_block_html(
            document, recognition_results, block_ids, clear_lines=True
        )

    def build_block_requests(
        self, document: Document, pages: List[PageGroup]
    ) -> Tuple[List[Image.Image], List[LayoutResult], List[List[BlockId]]]:
        images = []
        layout_results = []
        block_ids = []
        for page in pages:
            image = page.get_image(highres=True)
            image_size = image.size
            page_size = page.polygon.size

            boxes = []
            page_block_ids = []
            for block in page.structure_blocks(document):
                if block.block_type in self.skip_ocr_blocks:
                    continue
                polygon = block.polygon.rescale(page_size, image_size).fit_to_bounds(
                    (0, 0, *image_size)
                )
                boxes.append(
                    LayoutBox(
                        polygon=polygon.polygon,
                        label=block_type_to_surya_label(block.block_type),
                        raw_label=str(block.block_type),
                        position=len(boxes),
                        count=block.layout_token_count or self.default_token_budget,
                    )
                )
                page_block_ids.append(block.id)

            images.append(image)
            layout_results.append(
                LayoutResult(bboxes=boxes, image_bbox=[0, 0, *image_size])
            )
            block_ids.append(page_block_ids)

        return images, layout_results, block_ids

    def replace_page_structure(
        self,
        document: Document,
        pages: List[PageGroup],
        images: List[Image.Image],
        recognition_results,
    ):
        """Rebuild each OCR'd page's structure from the recognition result.

        Works for both full-page results and the per-page block-mode fallback,
        since both return blocks with polygon, label, and html.
        """
        for page, image, page_result in zip(pages, images, recognition_results):
            image_size = image.size
            page_size = page.polygon.size

            new_structure = []
            for block_result in page_result.blocks:
                block_type = surya_label_to_block_type(block_result.label)
                if block_type is None:
                    continue
                if block_result.error:
                    logger.warning(
                        f"OCR failed for a {block_result.label} block on page {page.page_id}"
                    )
                    continue

                polygon = (
                    PolygonBox(polygon=block_result.polygon)
                    .rescale(image_size, page_size)
                    .fit_to_bounds((0, 0, *page_size))
                )

                block_cls = get_block_class(block_type)
                block = page.add_block(block_cls, polygon)
                block.structure = []
                block.text_extraction_method = "surya"
                if not block_result.skipped:
                    html = self.clean_html(block_result.html)
                    if html:
                        block.html = html
                new_structure.append(block.id)

            # Retire the layout blocks the page structure previously pointed to
            for old_block in page.structure_blocks(document):
                old_block.removed = True
            page.structure = new_structure

    def apply_block_html(
        self,
        document: Document,
        recognition_results,
        block_ids: List[List[BlockId]],
        clear_lines: bool = False,
    ):
        for page_block_ids, page_result in zip(block_ids, recognition_results):
            # Block mode returns one result per requested box, in order
            assert len(page_block_ids) == len(page_result.blocks)
            for block_id, block_result in zip(page_block_ids, page_result.blocks):
                block = document.get_block(block_id)
                if block_result.error:
                    logger.warning(f"OCR failed for block {block_id}")
                    continue
                if block_result.skipped:
                    continue

                html = self.clean_html(block_result.html)
                if not html:
                    continue

                if clear_lines:
                    # Retire the stale pdftext lines this block had before
                    for line in block.contained_blocks(document, (BlockTypes.Line,)):
                        line.removed = True

                block.html = html
                block.structure = []
                block.text_extraction_method = "surya"

    def clean_html(self, html: str) -> str:
        """Clean VLM block HTML: strip debug attributes, balance truncated
        tags, and drop output that devolved into a repetition loop."""
        if not html:
            return ""

        if _detect_repeat_loop(html):
            logger.warning("Dropping OCR block output due to repetition loop")
            return ""

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(True):
            for attribute in STRIP_ATTRIBUTES:
                if attribute in tag.attrs:
                    del tag.attrs[attribute]

        # The VLM marks non-text regions (e.g. signatures) with a bare <img/>;
        # it has no source, so it would render as an empty image (![]())
        for img in soup.find_all("img"):
            if not img.get("src"):
                img.decompose()

        # Re-serializing through BeautifulSoup balances any tags left open by
        # token-budget truncation
        return str(soup).strip()
