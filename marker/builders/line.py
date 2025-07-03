from collections import defaultdict
from copy import deepcopy
from itertools import chain
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
    enable_table_ocr: Annotated[
        bool,
        "Whether to skip OCR on tables.  The TableProcessor will re-OCR them.  Only enable if the TableProcessor is not running.",
    ] = False
    format_lines: Annotated[
        bool, "Enable good provider lines to be checked and fixed by the OCR model"
    ] = False
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
    provider_line_detected_line_min_overlap_pct: Annotated[
        float,
        "The percentage of a provider line that has to be covered by a detected line",
    ] = 0.1
    line_vertical_merge_threshold: Annotated[
        int, "The maximum pixel distance between y1s for two lines to be merged"
    ] = 8
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
                ]
            )
            layout_good.append(provider_lines_good)

        # Don't OCR if only a few pages are bad
        if sum(layout_good) > len(document.pages) * self.min_document_ocr_threshold:
            layout_good = [True] * len(document.pages)

        run_detection = [(not good or self.format_lines) for good in layout_good]
        page_images = [
            page.get_image(highres=False, remove_blocks=self.ocr_remove_blocks)
            for page, good in zip(document.pages, run_detection)
            if good
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
            page_size = provider.get_page_bbox(document_page.page_id).size
            image_size = (
                PolygonBox.from_bbox(detection_result.image_bbox).size
                if detection_result
                else page_size
            )

            # Setup detection results
            if detection_result:
                detection_boxes = [
                    PolygonBox(polygon=box.polygon) for box in detection_result.bboxes
                ]
            else:
                detection_boxes = []
            detection_boxes = sort_text_lines(detection_boxes)

            if provider_lines_good:
                document_page.text_extraction_method = "pdftext"

                # Add in the provider lines - merge ones that get broken by pdftext
                # Also track the lines that were detected but have no provider overlaps
                merged_provider_lines, detected_only_lines = (
                    self.merge_provider_lines_detected_lines(
                        provider_lines,
                        detection_boxes,
                        image_size,
                        page_size,
                        document_page.page_id,
                    )
                )

                # If fixing lines, mark every line to be passed to the OCR model
                for provider_line in merged_provider_lines:
                    provider_line.line.text_extraction_method = (
                        "hybrid" if self.format_lines else "pdftext"
                    )
                page_lines[document_page.page_id] = (
                    merged_provider_lines + detected_only_lines
                )
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

    def merge_provider_lines_detected_lines(
        self,
        provider_lines: List[ProviderOutput],
        detected_lines: List[PolygonBox],
        image_size,
        page_size,
        page_id,
    ):
        # If no lines detected, skip the merging
        if not detected_lines:
            return provider_lines, []

        # If no provider lines, return all detected text lines
        if not provider_lines:
            detected_only_lines = []
            LineClass: Line = get_block_class(BlockTypes.Line)
            for detected_line in detected_lines:
                detected_line_polygon = PolygonBox(
                    polygon=detected_line.polygon
                ).rescale(image_size, page_size)
                detected_only_lines.append(
                    ProviderOutput(
                        line=LineClass(
                            polygon=detected_line_polygon,
                            page_id=page_id,
                            text_extraction_method="surya",
                        ),
                        spans=[],
                        chars=[],
                    )
                )

            return [], detected_only_lines

        out_provider_lines = []
        horizontal_provider_lines = []
        for j, provider_line in enumerate(provider_lines):
            # Multiply to account for small blocks inside equations, but filter out big vertical lines
            if provider_line.line.polygon.height < provider_line.line.polygon.width * 5:
                horizontal_provider_lines.append((j, provider_line))
            else:
                out_provider_lines.append((j, provider_line))

        provider_line_boxes = [
            p.line.polygon.bbox for _, p in horizontal_provider_lines
        ]
        detected_line_boxes = [
            PolygonBox(polygon=line.polygon).rescale(image_size, page_size).bbox
            for line in detected_lines
        ]

        provider_detected_overlaps = matrix_intersection_area(
            provider_line_boxes, detected_line_boxes
        )

        # Find provider lines to merge together
        # Merge lines has keys of detected line index, and values are lists of provider line indices
        # The provider lines overlap with the detected line index
        provider_lines_within_detected_lines = defaultdict(list)
        for i in range(len(provider_line_boxes)):
            max_overlap_pct = np.max(provider_detected_overlaps[i]) / max(
                1, horizontal_provider_lines[i][1].line.polygon.area
            )
            if max_overlap_pct <= self.provider_line_detected_line_min_overlap_pct:
                continue

            best_overlap_detected_line = np.argmax(provider_detected_overlaps[i])
            provider_lines_within_detected_lines[best_overlap_detected_line].append(i)

        # If a detected line contains multiple provider lines, group them by vertical proximity
        # Only merge provider lines that are vertically close together (within threshold)
        # This prevents merging separate text lines that happen to fall under one detected region
        # We only want to merge a single line of text horizontally, not multiple lines vertically
        # For example, small math symbols, etc, can get broken in separate lines in PDFs
        filtered_provider_lines_within_detected_lines = defaultdict(list)
        for line_idx in provider_lines_within_detected_lines:
            merge_segment = []
            prev_line = None
            for ml in provider_lines_within_detected_lines[line_idx]:
                line = horizontal_provider_lines[ml][1].line.polygon
                if prev_line:
                    close = (
                        abs(line.y_start - prev_line.y_start)
                        < self.line_vertical_merge_threshold
                        or abs(line.y_end - prev_line.y_end)
                        < self.line_vertical_merge_threshold
                    )
                else:
                    # First line
                    close = True

                prev_line = line
                if close:
                    merge_segment.append(ml)
                else:
                    if merge_segment:
                        filtered_provider_lines_within_detected_lines[line_idx].append(
                            merge_segment
                        )
                    merge_segment = [ml]
            if merge_segment:
                filtered_provider_lines_within_detected_lines[line_idx].append(
                    merge_segment
                )

        # Find all the potential merges of provider lines
        already_merged = set()
        potential_merges = []
        for line_idx in filtered_provider_lines_within_detected_lines:
            potential_merges.extend(
                chain.from_iterable(
                    filtered_provider_lines_within_detected_lines[line_idx]
                )
            )
        potential_merges = set(potential_merges)

        # Provider lines that are not in any merge group should be outputted as-is
        out_provider_lines.extend(
            [
                hp
                for i, hp in enumerate(horizontal_provider_lines)
                if i not in potential_merges
            ]
        )

        detected_lines_remerged = set()

        def bbox_for_merge_section(
            merge_section: List[int],
            all_merge_sections: List[List[int]],
            text_line: PolygonBox,
        ):
            # Don't just take the whole detected line if we have multiple sections inside
            if len(all_merge_sections) == 1 and len(all_merge_sections[0]) == 1:
                # This is to cover for the special case where multiple detected lines fall under the same provider line
                # This happens sometimes since providers lines are long, and will merge across whitespace
                # We merge in all detected lines into a single polygon before assigning to the provider line
                idx = merge_section[0]
                overlap_idxs = np.nonzero(provider_detected_overlaps[idx])[0]
                # Account for lines that overlap, but have been assigned to a different provider line already
                lines = [text_line]

                # Find other detected lines within this provider line, and merge them
                # Track used lines to ensure no duplicates
                other_merge_detected_idxs = [
                    i
                    for i in overlap_idxs
                    if i not in provider_lines_within_detected_lines
                    and i not in detected_lines_remerged
                ]
                other_merge_detected_lines = [
                    detected_lines[i] for i in other_merge_detected_idxs
                ]
                lines += other_merge_detected_lines
                detected_lines_remerged.update(set(other_merge_detected_idxs))

                merged = lines[0] if len(lines) == 1 else lines[0].merge(lines[1:])
                return merged.rescale(image_size, page_size)
            else:
                poly = None
                for section_idx in merge_section:
                    section_polygon = deepcopy(
                        horizontal_provider_lines[section_idx][1].line.polygon
                    )
                    if poly is None:
                        poly: PolygonBox = section_polygon
                    else:
                        poly = poly.merge([section_polygon])
                return poly

        for line_idx in filtered_provider_lines_within_detected_lines:
            detected_line = detected_lines[line_idx]
            for merge_section in filtered_provider_lines_within_detected_lines[
                line_idx
            ]:
                merge_section = [m for m in merge_section if m not in already_merged]
                if len(merge_section) == 0:
                    continue
                elif len(merge_section) == 1:
                    horizontal_provider_idx = merge_section[0]
                    out_idx, merged_line = horizontal_provider_lines[
                        horizontal_provider_idx
                    ]
                    # Set the polygon to the detected line - This is because provider polygons are sometimes incorrect
                    # TODO Add metadata for this
                    merged_line.line.polygon = bbox_for_merge_section(
                        merge_section,
                        filtered_provider_lines_within_detected_lines[line_idx],
                        detected_line,
                    )
                    out_provider_lines.append((out_idx, merged_line))
                    already_merged.add(merge_section[0])
                else:
                    merge_section = sorted(merge_section)
                    merged_line = None
                    min_idx = min(merge_section)
                    out_idx = horizontal_provider_lines[min_idx][0]
                    for idx in merge_section:
                        provider_line = deepcopy(horizontal_provider_lines[idx][1])
                        if merged_line is None:
                            merged_line = provider_line
                        else:
                            # Combine the spans of the provider line with the merged line
                            merged_line = merged_line.merge(provider_line)
                        already_merged.add(idx)  # Prevent double merging
                    # Set the polygon to the detected line - This is because provider polygons are sometimes incorrect
                    # TODO Add metadata for this
                    merged_line.line.polygon = bbox_for_merge_section(
                        merge_section,
                        filtered_provider_lines_within_detected_lines[line_idx],
                        detected_line,
                    )
                    out_provider_lines.append((out_idx, merged_line))

        # Sort to preserve original order
        out_provider_lines = sorted(out_provider_lines, key=lambda x: x[0])
        out_provider_lines = [p for _, p in out_provider_lines]

        # Detected lines that do not overlap with any provider lines shoudl be outputted as-is
        detected_only_lines = []
        LineClass: Line = get_block_class(BlockTypes.Line)
        for j in range(len(detected_line_boxes)):
            # Ensure we don't do max on an empty array
            if provider_detected_overlaps[:, j].size == 0:
                continue

            if np.max(provider_detected_overlaps[:, j]) == 0:
                detected_line_polygon = PolygonBox.from_bbox(detected_line_boxes[j])
                detected_only_lines.append(
                    ProviderOutput(
                        line=LineClass(
                            polygon=detected_line_polygon,
                            page_id=page_id,
                            text_extraction_method="surya",
                        ),
                        spans=[],
                        chars=[],
                    )
                )

        return out_provider_lines, detected_only_lines
