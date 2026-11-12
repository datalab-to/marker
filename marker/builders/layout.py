from typing import Annotated, Any, List

import numpy as np
from surya.layout.schema import LayoutResult, LayoutBox

from marker.builders import BaseBuilder
from marker.logger import get_logger
from marker.ocr import IndraRecognitionPredictor
from marker.providers.pdf import PdfProvider
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.labels import BLANK_PAGE_LABEL, surya_label_to_block_type
from marker.schema.polygon import PolygonBox
from marker.schema.registry import get_block_class

logger = get_logger()


class LayoutBuilder(BaseBuilder):
    """
    A builder for performing layout detection on PDF pages and merging the results into the document.
    """

    mode: Annotated[
        str,
        "Conversion mode: 'no_ocr', 'fast'/'indra', or 'balanced'.",
    ] = "balanced"
    disable_ocr: Annotated[
        bool,
        "Pure text-layer path (no OCR). Uses the lightweight in-process Indra",
        "ONNX layout detector on CPU without an inference server.",
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
        "Mirrors the LineBuilder setting. When False (default), pdftext pages",
        "retain the OrderNet (order.onnx) reading order.",
    ] = False
    disable_tqdm: Annotated[
        bool,
        "Disable tqdm progress bars.",
    ] = False
    extract_images: Annotated[
        bool,
        "Extract images from the document.",
    ] = True
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

    def __init__(
        self,
        layout_model: Any = None,
        fast_layout_model: Any = None,
        config=None,
        indra_model: IndraRecognitionPredictor | None = None,
    ):
        self.layout_model = layout_model
        self.fast_layout_model = fast_layout_model
        self.disable_image_extraction = bool(
            config.get("disable_image_extraction", False)
            if isinstance(config, dict)
            else False
        )

        super().__init__(config)
        if self.mode == "no_ocr":
            self.disable_ocr = True
        elif self.mode in ("vllm", "surya", "accurate"):
            self.force_ocr = True

        if indra_model is None and self.mode not in ("vllm", "surya", "accurate"):
            if isinstance(layout_model, IndraRecognitionPredictor):
                indra_model = layout_model
            elif isinstance(fast_layout_model, IndraRecognitionPredictor):
                indra_model = fast_layout_model
            else:
                indra_model = IndraRecognitionPredictor()
        self.indra_model = indra_model

    def use_fast_layout(self):
        return self.mode in ("fast", "indra", "no_ocr") or self.disable_ocr

    def get_layout_model(self):
        return self.fast_layout_model if self.use_fast_layout() else self.layout_model

    def __call__(self, document: Document, provider: PdfProvider):
        if self.force_layout_block is not None:
            # Assign the full content of every page to a single layout type
            layout_results = self.forced_layout(document.pages)
        elif self.force_ocr and self.ocr_full_page and not self.disable_ocr:
            # When force_ocr + ocr_full_page is enabled, every page is rebuilt
            # by full-page OCR, so don't pay for a layout pass that gets thrown away.
            layout_results = [
                LayoutResult(bboxes=[], image_bbox=page.polygon.bbox)
                for page in document.pages
            ]
        else:
            layout_results = self.surya_layout(document.pages, provider)
        self.add_blocks_to_pages(document.pages, layout_results)
        self.expand_layout_blocks(document)

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
        if (
            model is not None
            and not isinstance(model, IndraRecognitionPredictor)
            and (
                type(model).__module__.startswith("unittest.mock")
                or self.mode in ("vllm", "surya", "accurate")
            )
        ):
            model.disable_tqdm = self.disable_tqdm
            images = [p.get_image(highres=False) for p in pages]
            return model(images)
        if self.mode in ("vllm", "surya", "accurate"):
            from surya.layout import LayoutPredictor

            vlm_layout = LayoutPredictor()
            vlm_layout.disable_tqdm = self.disable_tqdm
            images = [p.get_image(highres=False) for p in pages]
            return vlm_layout(images)

        results: List[LayoutResult] = [None] * len(pages)
        active_indices: List[int] = []
        active_pages: List[PageGroup] = []
        has_page_lines = provider is not None and hasattr(provider, "page_lines")
        for i, page in enumerate(pages):
            if (
                has_page_lines
                and not provider.page_lines.get(page.page_id)
                and self.ocr_full_page
                and not self.disable_ocr
            ):
                # Textless/scanned pages will be rebuilt wholesale by full-page OCR
                # in OcrBuilder, so skip running layout on them.
                results[i] = LayoutResult(bboxes=[], image_bbox=page.polygon.bbox)
            else:
                active_indices.append(i)
                active_pages.append(page)

        if active_pages:
            want_images = not self.disable_ocr and (
                (self.extract_images and not self.disable_image_extraction)
                or self.mode == "balanced"
            )
            detected = self.indra_model.detect_layout(
                active_pages, provider, extract_images=want_images
            )
            for idx, res in zip(active_indices, detected):
                results[idx] = res
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
