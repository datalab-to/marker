from copy import deepcopy
from typing import Annotated, List, Tuple

import numpy as np
from PIL import Image
import cv2

from surya.detection import DetectionPredictor
from surya.ocr_error import OCRErrorPredictor

from marker.builders import BaseBuilder
from marker.providers import ProviderOutput, ProviderPageLines
from marker.providers.pdf import PdfProvider
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.polygon import PolygonBox
from marker.schema.registry import get_block_class
from marker.schema.text.line import Line
from marker.settings import settings
from marker.util import matrix_intersection_area, sort_text_lines


class LineBuilder(BaseBuilder):
    """
    A builder for detecting text lines. Merges the detected lines with the lines from the provider
    """

    detection_batch_size: Annotated[
        int,
        "The batch size to use for the detection model.",
        "Default is None, which will use the default batch size for the model.",
    ] = None
    ocr_error_batch_size: Annotated[
        int,
        "The batch size to use for the ocr error detection model.",
        "Default is None, which will use the default batch size for the model.",
    ] = None
    layout_coverage_min_lines: Annotated[
        int,
        "The minimum number of PdfProvider lines that must be covered by the layout model",
        "to consider the lines from the PdfProvider valid.",
    ] = 1
    layout_coverage_threshold: Annotated[
        float,
        "The minimum coverage ratio required for the layout model to consider",
        "the lines from the PdfProvider valid.",
    ] = 0.25
    min_document_ocr_threshold: Annotated[
        float,
        "If less pages than this threshold are good, OCR will happen in the document.  Otherwise it will not.",
    ] = 0.85
    provider_line_provider_line_min_overlap_pct: Annotated[
        float,
        "The percentage of a provider line that has to be covered by a detected line",
    ] = 0.1
    excluded_for_coverage: Annotated[
        Tuple[BlockTypes],
        "A list of block types to exclude from the layout coverage check.",
    ] = (
        BlockTypes.Figure,
        BlockTypes.Picture,
        BlockTypes.Table,
        BlockTypes.FigureGroup,
        BlockTypes.TableGroup,
        BlockTypes.PictureGroup,
    )
    ocr_remove_blocks: Tuple[BlockTypes, ...] = (
        BlockTypes.Table,
        BlockTypes.Form,
        BlockTypes.TableOfContents,
        BlockTypes.Equation,
    )
    disable_tqdm: Annotated[
        bool,
        "Disable tqdm progress bars.",
    ] = False
    disable_ocr: Annotated[
        bool,
        "Disable OCR for the document. This will only use the lines from the provider.",
    ] = False
    keep_chars: Annotated[bool, "Keep individual characters."] = False

    def __init__(
        self,
        detection_model: DetectionPredictor,
        ocr_error_model: OCRErrorPredictor,
        config=None,
    ):
        super().__init__(config)

        self.detection_model = detection_model
        self.ocr_error_model = ocr_error_model

    def __call__(self, document: Document, provider: PdfProvider):
        # Disable inline detection for documents where layout model doesn't detect any equations
        # Also disable if we won't use the inline detections (if we aren't using the LLM)
        provider_lines, ocr_lines = self.get_all_lines(document, provider)
        self.merge_blocks(document, provider_lines, ocr_lines)

    def get_detection_batch_size(self):
        if self.detection_batch_size is not None:
            return self.detection_batch_size
        elif settings.TORCH_DEVICE_MODEL == "cuda":
            return 10
        return 4

    def get_ocr_error_batch_size(self):
        if self.ocr_error_batch_size is not None:
            return self.ocr_error_batch_size
        elif settings.TORCH_DEVICE_MODEL == "cuda":
            return 14
        return 4

    def get_detection_results(
        self, page_images: List[Image.Image], run_detection: List[bool]
    ):
        self.detection_model.disable_tqdm = self.disable_tqdm
        page_detection_results = self.detection_model(
            images=page_images, batch_size=self.get_detection_batch_size()
        )

        assert len(page_detection_results) == sum(run_detection)
        detection_results = []
        idx = 0
        for good in run_detection:
            if good:
                detection_results.append(page_detection_results[idx])
                idx += 1
            else:
                detection_results.append(None)
        assert idx == len(page_images)

        assert len(run_detection) == len(detection_results)
        return detection_results

    def get_all_lines(self, document: Document, provider: PdfProvider):
        ocr_error_detection_results = self.ocr_error_detection(
            document.pages, provider.page_lines
        )

        boxes_to_ocr = {page.page_id: [] for page in document.pages}
        page_lines = {page.page_id: [] for page in document.pages}

        LineClass: Line = get_block_class(BlockTypes.Line)

        layout_good = []
        for document_page, ocr_error_detection_label in zip(
            document.pages, ocr_error_detection_results.labels
        ):
            document_page.ocr_errors_detected = ocr_error_detection_label == "bad"
            provider_lines: List[ProviderOutput] = provider.page_lines.get(
                document_page.page_id, []
            )
            provider_lines_good = all(
                [
                    bool(provider_lines),
                    not document_page.ocr_errors_detected,
                    self.check_layout_coverage(document_page, provider_lines),
                    self.check_line_overlaps(
                        document_page, provider_lines
                    ),  # Ensure provider lines don't overflow the page or intersect
                ]
            )
            if self.disable_ocr:
                provider_lines_good = True

            layout_good.append(provider_lines_good)

        run_detection = [not good for good in layout_good]
        page_images = [
            page.get_image(highres=False, remove_blocks=self.ocr_remove_blocks)
            for page, bad in zip(document.pages, run_detection)
            if bad
        ]

        # Note: run_detection is longer than page_images, since it has a value for each page, not just good ones
        # Detection results and inline detection results are for every page (we use run_detection to make the list full length)
        detection_results = self.get_detection_results(page_images, run_detection)

        assert len(detection_results) == len(layout_good) == len(document.pages)
        for document_page, detection_result, provider_lines_good in zip(
            document.pages, detection_results, layout_good
        ):
            provider_lines: List[ProviderOutput] = provider.page_lines.get(
                document_page.page_id, []
            )

            # Setup detection results
            detection_boxes = []
            if detection_result:
                detection_boxes = [
                    PolygonBox(polygon=box.polygon) for box in detection_result.bboxes
                ]

            detection_boxes = sort_text_lines(detection_boxes)

            if provider_lines_good:
                document_page.text_extraction_method = "pdftext"

                # Mark extraction method as pdftext, since all lines are good
                for provider_line in provider_lines:
                    provider_line.line.text_extraction_method = "pdftext"

                page_lines[document_page.page_id] = provider_lines
            else:
                document_page.text_extraction_method = "surya"
                boxes_to_ocr[document_page.page_id].extend(detection_boxes)

        # Dummy lines to merge into the document - Contains no spans, will be filled in later by OCRBuilder
        ocr_lines = {document_page.page_id: [] for document_page in document.pages}
        for page_id, page_ocr_boxes in boxes_to_ocr.items():
            page_size = provider.get_page_bbox(page_id).size
            image_size = document.get_page(page_id).get_image(highres=False).size
            for box_to_ocr in page_ocr_boxes:
                line_polygon = PolygonBox(polygon=box_to_ocr.polygon).rescale(
                    image_size, page_size
                )
                ocr_lines[page_id].append(
                    ProviderOutput(
                        line=LineClass(
                            polygon=line_polygon,
                            page_id=page_id,
                            text_extraction_method="surya",
                        ),
                        spans=[],
                        chars=[],
                    )
                )

        return page_lines, ocr_lines

    def ocr_error_detection(
        self, pages: List[PageGroup], provider_page_lines: ProviderPageLines
    ):
        page_texts = []
        for document_page in pages:
            provider_lines = provider_page_lines.get(document_page.page_id, [])
            page_text = "\n".join(
                " ".join(s.text for s in line.spans) for line in provider_lines
            )
            page_texts.append(page_text)

        self.ocr_error_model.disable_tqdm = self.disable_tqdm
        ocr_error_detection_results = self.ocr_error_model(
            page_texts, batch_size=int(self.get_ocr_error_batch_size())
        )
        return ocr_error_detection_results

    def check_line_overlaps(
        self, document_page: PageGroup, provider_lines: List[ProviderOutput]
    ) -> bool:
        provider_bboxes = [line.line.polygon.bbox for line in provider_lines]
        # Add a small margin to account for minor overflows
        page_bbox = document_page.polygon.expand(5, 5).bbox

        for bbox in provider_bboxes:
            if bbox[0] < page_bbox[0]:
                return False
            if bbox[1] < page_bbox[1]:
                return False
            if bbox[2] > page_bbox[2]:
                return False
            if bbox[3] > page_bbox[3]:
                return False

        intersection_matrix = matrix_intersection_area(provider_bboxes, provider_bboxes)
        for i, line in enumerate(provider_lines):
            intersect_counts = np.sum(
                intersection_matrix[i]
                > self.provider_line_provider_line_min_overlap_pct
            )

            # There should be one intersection with itself
            if intersect_counts > 2:
                return False

        return True

    def check_layout_coverage(
        self,
        document_page: PageGroup,
        provider_lines: List[ProviderOutput],
    ):
        covered_blocks = 0
        total_blocks = 0
        large_text_blocks = 0

        layout_blocks = [
            document_page.get_block(block) for block in document_page.structure
        ]
        layout_blocks = [
            b for b in layout_blocks if b.block_type not in self.excluded_for_coverage
        ]

        layout_bboxes = [block.polygon.bbox for block in layout_blocks]
        provider_bboxes = [line.line.polygon.bbox for line in provider_lines]

        if len(layout_bboxes) == 0:
            return True

        if len(provider_bboxes) == 0:
            return False

        intersection_matrix = matrix_intersection_area(layout_bboxes, provider_bboxes)

        for idx, layout_block in enumerate(layout_blocks):
            total_blocks += 1
            intersecting_lines = np.count_nonzero(intersection_matrix[idx] > 0)

            if intersecting_lines >= self.layout_coverage_min_lines:
                covered_blocks += 1

            if (
                layout_block.polygon.intersection_pct(document_page.polygon) > 0.8
                and layout_block.block_type == BlockTypes.Text
            ):
                large_text_blocks += 1

        coverage_ratio = covered_blocks / total_blocks if total_blocks > 0 else 1
        text_okay = coverage_ratio >= self.layout_coverage_threshold

        # Model will sometimes say there is a single block of text on the page when it is blank
        if not text_okay and (total_blocks == 1 and large_text_blocks == 1):
            text_okay = True
        return text_okay

    def is_blank_slice(self, slice_image: Image.Image):
        image = np.asarray(slice_image)
        if (
            image is None
            or image.size == 0
            or image.shape[0] == 0
            or image.shape[1] == 0
        ):
            # Handle empty image case
            return True

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

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

    def filter_blank_lines(self, page: PageGroup, lines: List[ProviderOutput]):
        page_size = (page.polygon.width, page.polygon.height)
        page_image = page.get_image()
        image_size = page_image.size

        good_lines = []
        for line in lines:
            line_polygon_rescaled = deepcopy(line.line.polygon).rescale(
                page_size, image_size
            )
            line_bbox = line_polygon_rescaled.fit_to_bounds((0, 0, *image_size)).bbox

            if not self.is_blank_slice(page_image.crop(line_bbox)):
                good_lines.append(line)

        return good_lines

    def merge_blocks(
        self,
        document: Document,
        page_provider_lines: ProviderPageLines,
        page_ocr_lines: ProviderPageLines,
    ):
        for document_page in document.pages:
            provider_lines: List[ProviderOutput] = page_provider_lines[
                document_page.page_id
            ]
            ocr_lines: List[ProviderOutput] = page_ocr_lines[document_page.page_id]

            # Only one or the other will have lines
            # Filter out blank lines which come from bad provider boxes, or invisible text
            merged_lines = self.filter_blank_lines(
                document_page, provider_lines + ocr_lines
            )

            # Text extraction method is overridden later for OCRed documents
            document_page.merge_blocks(
                merged_lines,
                text_extraction_method="pdftext" if provider_lines else "surya",
                keep_chars=self.keep_chars,
            )
