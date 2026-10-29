from typing import Annotated, List

import numpy as np
from surya.fast_layout import FastLayoutPredictor
from surya.layout import LayoutPredictor
from surya.layout.schema import LayoutResult, LayoutBox

from marker.builders import BaseBuilder
from marker.logger import get_logger
from marker.providers.pdf import PdfProvider
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.labels import BLANK_PAGE_LABEL, surya_label_to_block_type
from marker.schema.polygon import PolygonBox
from marker.schema.registry import get_block_class

logger = get_logger()

# Block types that are all rendered as HTML tables by the TableProcessor. When
# the layout model emits two heavily-overlapping boxes from this set for one
# region (e.g. a questionnaire tagged as both Table and Form), both render and
# the same table is duplicated in the output.
TABLE_FAMILY_TYPES = (
    BlockTypes.Table,
    BlockTypes.Form,
    BlockTypes.TableOfContents,
)


class LayoutBuilder(BaseBuilder):
    """
    A builder for performing layout detection on PDF pages and merging the results into the document.
    """

    mode: Annotated[
        str,
        "Conversion mode: 'balanced' (GPU) uses the VLM layout model;",
        "'fast' (CPU) uses the lightweight rf-detr/onnx layout detector.",
        "The converter defaults this by device (balanced on GPU, fast on CPU/MPS).",
    ] = "balanced"
    disable_ocr: Annotated[
        bool,
        "Pure text-layer path (no VLM). Forces the lightweight rf-detr layout",
        "detector so the whole pipeline runs on CPU without an inference server.",
    ] = False
    force_layout_block: Annotated[
        str,
        "Skip layout and force every page to be treated as a specific block type.",
    ] = None
    force_ocr: Annotated[
        bool,
        "Set when OCR is forced for the whole document. Layout is skipped -",
        "full-page OCR rebuilds the page structure itself, and surya runs its",
        "own layout lazily for any page whose full-page output fails.",
    ] = False
    ocr_full_page: Annotated[
        bool,
        "Mirrors the OcrBuilder setting - layout can only be skipped when",
        "full-page OCR is in use.",
    ] = True
    use_pdftext_reading_order: Annotated[
        bool,
        "Mirrors the LineBuilder setting. When True (default), fast mode only",
        "runs the learned reading-order head on textless (scanned) pages -",
        "pdftext pages are reordered from the PDF character stream instead.",
        "When False, the head runs on every page in fast mode.",
    ] = True
    disable_tqdm: Annotated[
        bool,
        "Disable tqdm progress bars.",
    ] = False
    expand_block_types: Annotated[
        List[BlockTypes],
        "Block types whose bounds should be expanded to accomodate missing regions",
    ] = [
        BlockTypes.Picture,
        BlockTypes.Figure,
        BlockTypes.ComplexRegion,
        BlockTypes.Diagram,
    ]  # Does not include groups since they are only injected later
    max_expand_frac: Annotated[
        float, "The maximum fraction to expand the layout box bounds by"
    ] = 0.05
    table_merge_overlap_pct: Annotated[
        float,
        "Two table-family layout boxes (Table/Form/TableOfContents) whose mutual "
        "overlap fraction both exceed this value are treated as duplicate "
        "detections of the same region; the lower-confidence one is dropped so "
        "the region is not rendered as a table twice.",
    ] = 0.75

    def __init__(
        self,
        layout_model: LayoutPredictor,
        fast_layout_model: FastLayoutPredictor,
        config=None,
    ):
        self.layout_model = layout_model
        self.fast_layout_model = fast_layout_model

        super().__init__(config)

    def use_fast_layout(self):
        # Fast mode and disable_ocr (pure text-layer, no VLM) both use the
        # lightweight rf-detr layout detector so no VLM server is required.
        return self.mode == "fast" or self.disable_ocr

    def get_layout_model(self):
        return self.fast_layout_model if self.use_fast_layout() else self.layout_model

    def __call__(self, document: Document, provider: PdfProvider):
        if self.force_layout_block is not None:
            # Assign the full content of every page to a single layout type
            layout_results = self.forced_layout(document.pages)
        elif self.force_ocr and self.ocr_full_page and self.mode == "balanced":
            # Balanced + force_ocr: every page is rebuilt by full-page OCR, so
            # don't pay for a layout pass that gets thrown away. Fast mode still
            # needs layout boxes to drive block-mode OCR.
            layout_results = [
                LayoutResult(bboxes=[], image_bbox=page.polygon.bbox)
                for page in document.pages
            ]
        else:
            layout_results = self.surya_layout(document.pages, provider)
        self.add_blocks_to_pages(document.pages, layout_results)
        self.merge_overlapping_table_blocks(document)
        self.expand_layout_blocks(document)

    def merge_overlapping_table_blocks(self, document: Document):
        """Drop duplicate table-family detections. The layout model can tag one
        region as both a Table and a Form (near-identical boxes); both are
        rendered as HTML tables by the TableProcessor, so the same table appears
        twice in the output. When two table-family boxes mutually overlap above
        table_merge_overlap_pct, keep the higher-confidence one and remove the
        other."""
        for page in document.pages:
            if not page.structure:
                continue

            table_blocks = [
                block
                for block in (document.get_block(bid) for bid in page.structure)
                if block is not None and block.block_type in TABLE_FAMILY_TYPES
            ]

            remove_ids = []
            for i, block in enumerate(table_blocks):
                if block.id in remove_ids:
                    continue
                for other in table_blocks[i + 1 :]:
                    if other.id in remove_ids:
                        continue
                    if (
                        block.polygon.intersection_pct(other.polygon)
                        >= self.table_merge_overlap_pct
                        and other.polygon.intersection_pct(block.polygon)
                        >= self.table_merge_overlap_pct
                    ):
                        loser = (
                            other
                            if self._layout_confidence(block)
                            >= self._layout_confidence(other)
                            else block
                        )
                        loser.removed = True
                        remove_ids.append(loser.id)
                        if loser.id == block.id:
                            break

            if remove_ids:
                page.remove_structure_items(remove_ids)

    @staticmethod
    def _layout_confidence(block) -> float:
        if block.top_k:
            return block.top_k.get(block.block_type, 1.0)
        return 1.0

    def forced_layout(self, pages: List[PageGroup]) -> List[LayoutResult]:
        layout_results = []
        for page in pages:
            layout_results.append(
                LayoutResult(
                    image_bbox=page.polygon.bbox,
                    bboxes=[
                        LayoutBox(
                            label=self.force_layout_block,
                            raw_label=self.force_layout_block,
                            position=0,
                            count=0,
                            polygon=page.polygon.polygon,
                        ),
                    ],
                )
            )
        return layout_results

    def surya_layout(
        self, pages: List[PageGroup], provider: PdfProvider
    ) -> List[LayoutResult]:
        model = self.get_layout_model()
        model.disable_tqdm = self.disable_tqdm
        images = [p.get_image(highres=False) for p in pages]
        if not self.use_fast_layout():
            return model(images)

        # Fast mode skips the layout reading-order head wherever pdftext can
        # order the page instead (LineBuilder reorders from the PDF character
        # stream). The head still runs where no pdftext order exists: textless
        # (scanned) pages, whose layout order seeds block-mode OCR and the
        # per-page fallback when full-page OCR fails - and every page when
        # pdftext ordering is turned off.
        if not self.use_pdftext_reading_order:
            return model(images, use_order=True)

        need_order, raster = [], []
        for i, page in enumerate(pages):
            if provider.page_lines.get(page.page_id):
                raster.append(i)
            else:
                need_order.append(i)

        results: List[LayoutResult] = [None] * len(pages)
        for idx, result in zip(
            raster, model([images[i] for i in raster], use_order=False)
        ):
            results[idx] = result
        for idx, result in zip(
            need_order, model([images[i] for i in need_order], use_order=True)
        ):
            results[idx] = result
        return results

    def expand_layout_blocks(self, document: Document):
        for page in document.pages:
            page_blocks = [document.get_block(bid) for bid in page.structure]
            page_size = page.polygon.size

            expand_idxs = [
                i
                for i, block in enumerate(page_blocks)
                if block.block_type in self.expand_block_types
            ]
            if not expand_idxs:
                continue

            if len(page_blocks) == 1:
                block = page_blocks[0]
                block.polygon = block.polygon.expand(
                    self.max_expand_frac, self.max_expand_frac
                ).fit_to_bounds((0, 0, *page_size))
                continue

            # Vectorized pairwise minimum_gap: for non-overlapping boxes the gap
            # is the corner distance when diagonal, else the edge distance.
            bboxes = np.array([b.polygon.bbox for b in page_blocks])  # (N, 4)
            dx = np.maximum(
                bboxes[None, :, 0] - bboxes[:, None, 2],
                bboxes[:, None, 0] - bboxes[None, :, 2],
            )
            dy = np.maximum(
                bboxes[None, :, 1] - bboxes[:, None, 3],
                bboxes[:, None, 1] - bboxes[None, :, 3],
            )
            gaps = np.where(
                (dx > 0) & (dy > 0),
                np.hypot(dx, dy),
                np.maximum(dx, 0) + np.maximum(dy, 0),
            )
            np.fill_diagonal(gaps, np.inf)

            for i in expand_idxs:
                block = page_blocks[i]
                min_gap = gaps[i].min()
                if min_gap <= 0 or not np.isfinite(min_gap):
                    continue

                x_expand_frac = (
                    min_gap / block.polygon.width if block.polygon.width > 0 else 0
                )
                y_expand_frac = (
                    min_gap / block.polygon.height if block.polygon.height > 0 else 0
                )

                block.polygon = block.polygon.expand(
                    min(self.max_expand_frac, x_expand_frac),
                    min(self.max_expand_frac, y_expand_frac),
                ).fit_to_bounds((0, 0, *page_size))

    def add_blocks_to_pages(
        self, pages: List[PageGroup], layout_results: List[LayoutResult]
    ):
        for page, layout_result in zip(pages, layout_results):
            layout_page_size = PolygonBox.from_bbox(layout_result.image_bbox).size
            provider_page_size = page.polygon.size

            if layout_result.error:
                logger.warning(
                    f"Layout inference failed for page {page.page_id}; leaving page empty."
                )

            for bbox in sorted(layout_result.bboxes, key=lambda x: x.position):
                if bbox.label == BLANK_PAGE_LABEL:
                    continue

                block_type = surya_label_to_block_type(bbox.label)
                if block_type is None and bbox.label in BlockTypes.__members__:
                    # force_layout_block can name any marker block type
                    block_type = BlockTypes[bbox.label]
                if block_type is None:
                    logger.warning(
                        f"Unknown layout label {bbox.label} on page {page.page_id}; skipping."
                    )
                    continue

                block_cls = get_block_class(block_type)
                layout_block = page.add_block(
                    block_cls, PolygonBox(polygon=bbox.polygon)
                )
                layout_block.polygon = layout_block.polygon.rescale(
                    layout_page_size, provider_page_size
                ).fit_to_bounds((0, 0, *provider_page_size))
                layout_block.top_k = {block_type: bbox.confidence or 1.0}
                layout_block.layout_token_count = bbox.count
                page.add_structure(layout_block)

            # Ensure page has non-empty structure
            if page.structure is None:
                page.structure = []

            # Ensure page has non-empty children
            if page.children is None:
                page.children = []
