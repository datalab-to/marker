import copy
from typing import Annotated, List

from ftfy import fix_text
from PIL import Image
from surya.common.surya.schema import TaskNames
from surya.recognition import RecognitionPredictor, OCRResult, TextChar

from marker.builders import BaseBuilder
from marker.providers.pdf import PdfProvider
from marker.schema import BlockTypes
from marker.schema.blocks import BlockId
from marker.schema.blocks.base import Block
from marker.schema.document import Document
from marker.schema.groups import PageGroup
from marker.schema.registry import get_block_class
from marker.schema.text.char import Char
from marker.schema.text.line import Line
from marker.schema.text.span import Span
from marker.settings import settings
from marker.schema.polygon import PolygonBox
from marker.util import get_opening_tag_type, get_closing_tag_type
from marker.utils.model_call import call_model_sync


class OcrBuilder(BaseBuilder):
    """
    A builder for performing OCR on PDF pages and merging the results into the document.
    """

    recognition_batch_size: Annotated[
        int,
        "The batch size to use for the recognition model.",
        "Default is None, which will use the default batch size for the model.",
    ] = None
    disable_tqdm: Annotated[
        bool,
        "Disable tqdm progress bars.",
    ] = False
    # We can skip tables here, since the TableProcessor will re-OCR
    skip_ocr_blocks: Annotated[
        List[BlockTypes],
        "Blocktypes to skip OCRing by the model in this stage."
        "By default, this avoids recognizing lines inside equations/tables (handled later), figures, and pictures",
        "Note that we **do not** have to skip group types, since they are not built by this point"
    ] = [
        BlockTypes.Equation,
        BlockTypes.Figure,
        BlockTypes.Picture,
        BlockTypes.Table,
    ]
    full_ocr_block_types: Annotated[
        List[BlockTypes],
        "Blocktypes for which OCR is done at the **block level** instead of line-level."
        "This feature is still in beta, and should be used sparingly."
    ] = [
        BlockTypes.SectionHeader,
        BlockTypes.ListItem,
        BlockTypes.Footnote,
        BlockTypes.Text,
        BlockTypes.TextInlineMath,
        BlockTypes.Code,
        BlockTypes.Caption,
    ]
    ocr_task_name: Annotated[
        str,
        "The OCR mode to use, see surya for details.  Set to 'ocr_without_boxes' for potentially better performance, at the expense of formatting.",
    ] = TaskNames.ocr_with_boxes
    keep_chars: Annotated[bool, "Keep individual characters."] = False
    disable_ocr_math: Annotated[bool, "Disable inline math recognition in OCR"] = False
    drop_repeated_text: Annotated[bool, "Drop repeated text in OCR results."] = False
    block_mode_intersection_thresh: Annotated[float, "Max intersection before falling back to line mode"] = 0.5
    block_mode_max_lines: Annotated[int, "Max lines within a block before falling back to line mode"] = 15
    block_mode_max_height_frac: Annotated[float, "Max height of a block as a percentage of the page before falling back to line mode"] = 0.5

    def __init__(self, recognition_model: RecognitionPredictor = None, artifact_dict: dict = None, config=None):
        super().__init__(config)
        
        self.recognition_model = recognition_model
        self.artifact_dict = artifact_dict or {}

    def __call__(self, document: Document, provider: PdfProvider):
        pages_to_ocr = [page for page in document.pages if page.text_extraction_method == 'surya']
        ocr_page_images, block_polygons, block_ids, block_original_texts = (
            self.get_ocr_images_polygons_ids(document, pages_to_ocr, provider)
        )
        self.ocr_extraction(
            document,
            pages_to_ocr,
            ocr_page_images,
            block_polygons,
            block_ids,
            block_original_texts,
        )

    def get_recognition_batch_size(self):
        if self.recognition_batch_size is not None:
            return self.recognition_batch_size
        elif settings.TORCH_DEVICE_MODEL == "cuda":
            return 48
        elif settings.TORCH_DEVICE_MODEL == "mps":
            return 16
        return 32

    def select_ocr_blocks_by_mode(
        self, page: PageGroup, block: Block, block_lines: List[Block], page_max_intersection_pct: float
    ):
        if any([
            page_max_intersection_pct > self.block_mode_intersection_thresh,
            block.block_type not in self.full_ocr_block_types,
            len(block_lines) > self.block_mode_max_lines,
            block.polygon.height >= self.block_mode_max_height_frac * page.polygon.height
        ]):
            # Line mode
            return block_lines

        # Block mode
        return [block]

    def get_ocr_images_polygons_ids(
        self, document: Document, pages: List[PageGroup], provider: PdfProvider
    ):
        highres_images, highres_polys, block_ids, block_original_texts = [], [], [], []
        for document_page in pages:
            page_highres_image = document_page.get_image(highres=True)
            page_highres_polys = []
            page_block_ids = []
            page_block_original_texts = []

            page_size = provider.get_page_bbox(document_page.page_id).size
            image_size = page_highres_image.size
            max_intersection_pct = document_page.compute_max_structure_block_intersection_pct()
            for block in document_page.structure_blocks(document):
                if block.block_type in self.skip_ocr_blocks:
                    # Skip OCR
                    continue

                block_lines = block.contained_blocks(document, [BlockTypes.Line])
                blocks_to_ocr = self.select_ocr_blocks_by_mode(document_page, block, block_lines, max_intersection_pct)

                block.text_extraction_method = "surya"
                for block in blocks_to_ocr:
                    # Fit the polygon to image bounds since PIL image crop expands by default which might create bad images for the OCR model.
                    block_polygon_rescaled = (
                        copy.deepcopy(block.polygon)
                        .rescale(page_size, image_size)
                        .fit_to_bounds((0, 0, *image_size))
                    )
                    block_bbox_rescaled = block_polygon_rescaled.polygon
                    block_bbox_rescaled = [
                        [int(x) for x in point] for point in block_bbox_rescaled
                    ]

                    page_highres_polys.append(block_bbox_rescaled)
                    page_block_ids.append(block.id)
                    page_block_original_texts.append("")

            highres_images.append(page_highres_image)
            highres_polys.append(page_highres_polys)
            block_ids.append(page_block_ids)
            block_original_texts.append(page_block_original_texts)

        return highres_images, highres_polys, block_ids, block_original_texts

    def ocr_extraction(
        self,
        document: Document,
        pages: List[PageGroup],
        images: List[any],
        block_polygons: List[List[List[List[int]]]],  # polygons
        block_ids: List[List[BlockId]],
        block_original_texts: List[List[str]],
    ):
        if sum(len(b) for b in block_polygons) == 0:
            return

        self.recognition_model.disable_tqdm = self.disable_tqdm
        # Use the recognition_model directly if it's available, otherwise use the model from artifact_dict
        if self.recognition_model is not None:
            recognition_results: List[OCRResult] = call_model_sync(
                "recognition_model",
                {"recognition_model": self.recognition_model},
                images=images,
                task_names=[self.ocr_task_name] * len(images),
                polygons=block_polygons,
                input_text=block_original_texts,
                recognition_batch_size=int(self.get_recognition_batch_size()),
                sort_lines=False,
                math_mode=not self.disable_ocr_math,
                drop_repeated_text=self.drop_repeated_text,
                max_sliding_window=2148,
                max_tokens=2048
            )
        elif "recognition_model" in self.artifact_dict:
            recognition_results: List[OCRResult] = call_model_sync(
                "recognition_model",
                self.artifact_dict,
                images=images,
                task_names=[self.ocr_task_name] * len(images),
                polygons=block_polygons,
                input_text=block_original_texts,
                recognition_batch_size=int(self.get_recognition_batch_size()),
                sort_lines=False,
                math_mode=not self.disable_ocr_math,
                drop_repeated_text=self.drop_repeated_text,
                max_sliding_window=2148,
                max_tokens=2048
            )
        else:
            raise ValueError("No recognition_model available for processing")

        assert len(recognition_results) == len(images) == len(pages) == len(block_ids), (
            f"Mismatch in OCR lengths: {len(recognition_results)}, {len(images)}, {len(pages)}, {len(block_ids)}"
        )
        for document_page, page_recognition_result, page_block_ids, image in zip(
            pages, recognition_results, block_ids, images
        ):
            for block_id, block_ocr_result in zip(
                page_block_ids, page_recognition_result.text_lines
            ):
                if block_ocr_result.original_text_good:
                    continue
                if not fix_text(block_ocr_result.text):
                    continue
                
                block = document_page.get_block(block_id)
                # This is a nested list of spans, so multiple lines are supported
                all_line_spans = self.spans_from_html_chars(
                    block_ocr_result.chars, document_page, image
                )
                if block.block_type == BlockTypes.Line:
                    # flatten all spans across lines
                    flat_spans = [s for line_spans in all_line_spans for s in line_spans]
                    self.replace_line_spans(document, document_page, block, flat_spans)
                else:
                    # Clear out any old lines. Mark as removed for the json ocr renderer
                    for line in block.contained_blocks(document_page, block_types=[BlockTypes.Line]):
                        line.removed = True
                    block.structure = []

                    for line_spans in all_line_spans:
                        # TODO Replace this polygon with the polygon for each line, constructed from the spans
                        # This needs the OCR model bbox predictions to improve first
                        new_line = Line(
                            polygon=block.polygon,
                            page_id=block.page_id,
                            text_extraction_method="surya"
                        )
                        document_page.add_full_block(new_line)
                        block.add_structure(new_line)
                        self.replace_line_spans(document, document_page, new_line, line_spans)

    # TODO Fix polygons when we cut the span into multiple spans
    def link_and_break_span(self, span: Span, text: str, match_text, url: str):
        before_text, _, after_text = text.partition(match_text)
        before_span, after_span = None, None
        if before_text:
            before_span = copy.deepcopy(span)
            before_span.structure = []  # Avoid duplicate characters
            before_span.text = before_text
        if after_text:
            after_span = copy.deepcopy(span)
            after_span.text = after_text
            after_span.structure = []  # Avoid duplicate characters

        match_span = copy.deepcopy(span)
        match_span.text = match_text
        match_span.url = url

        return before_span, match_span, after_span

    # Pull all refs from old spans and attempt to insert back into appropriate place in new spans
    def replace_line_spans(
        self, document: Document, page: PageGroup, line: Line, new_spans: List[Span]
    ):
        old_spans = line.contained_blocks(document, [BlockTypes.Span])
        text_ref_matching = {span.text: span.url for span in old_spans if span.url}

        # Insert refs into new spans, since the OCR model does not (cannot) generate these
        final_new_spans = []
        for span in new_spans:
            # Use for copying attributes into new spans
            original_span = copy.deepcopy(span)
            remaining_text = span.text
            while remaining_text:
                matched = False
                for match_text, url in text_ref_matching.items():
                    if match_text in remaining_text:
                        matched = True
                        before, current, after = self.link_and_break_span(
                            original_span, remaining_text, match_text, url
                        )
                        if before:
                            final_new_spans.append(before)
                        final_new_spans.append(current)
                        if after:
                            remaining_text = after.text
                        else:
                            remaining_text = ""  # No more text left
                        # Prevent repeat matches
                        del text_ref_matching[match_text]
                        break
                if not matched:
                    remaining_span = copy.deepcopy(original_span)
                    remaining_span.text = remaining_text
                    final_new_spans.append(remaining_span)
                    break

        # Clear the old spans from the line
        line.structure = []
        for span in final_new_spans:
            page.add_full_block(span)
            line.structure.append(span.id)

    def assign_chars(self, span: Span, current_chars: List[Char]):
        if self.keep_chars:
            span.structure = [c.id for c in current_chars]

        return []

    def store_char(self, char: Char, current_chars: List[Char], page: PageGroup):
        if self.keep_chars:
            current_chars.append(char)
            page.add_full_block(char)

    def spans_from_html_chars(
        self, chars: List[TextChar], page: PageGroup, image: Image.Image
    ) -> List[List[Span]]:
        # Turn input characters from surya into spans - also store the raw characters
        SpanClass: Span = get_block_class(BlockTypes.Span)
        CharClass: Char = get_block_class(BlockTypes.Char)

        all_line_spans = []
        current_line_spans = []
        formats = {"plain"}
        current_span = None
        current_chars = []
        image_size = image.size

        for idx, char in enumerate(chars):
            char_box = PolygonBox(polygon=char.polygon).rescale(
                image_size, page.polygon.size
            )
            marker_char = CharClass(
                text=char.text,
                idx=idx,
                page_id=page.page_id,
                polygon=char_box,
            )

            if char.text == "<br>":
                if current_span:
                    current_chars = self.assign_chars(current_span, current_chars)
                    current_line_spans.append(current_span)
                    current_span = None
                if current_line_spans:
                    current_line_spans[-1].text += "\n"
                    all_line_spans.append(current_line_spans)
                    current_line_spans = []
                continue

            is_opening_tag, format = get_opening_tag_type(char.text)
            if is_opening_tag and format not in formats:
                formats.add(format)
                if current_span:
                    current_chars = self.assign_chars(current_span, current_chars)
                    current_line_spans.append(current_span)
                    current_span = None

                if format == "math":
                    current_span = SpanClass(
                        text="",
                        formats=list(formats),
                        page_id=page.page_id,
                        polygon=char_box,
                        minimum_position=0,
                        maximum_position=0,
                        font="Unknown",
                        font_weight=0,
                        font_size=0,
                    )
                    self.store_char(marker_char, current_chars, page)
                continue

            is_closing_tag, format = get_closing_tag_type(char.text)
            if is_closing_tag:
                # Useful since the OCR model sometimes returns closing tags without an opening tag
                try:
                    formats.remove(format)
                except Exception:
                    continue
                if current_span:
                    current_chars = self.assign_chars(current_span, current_chars)
                    current_line_spans.append(current_span)
                    current_span = None
                continue

            if not current_span:
                current_span = SpanClass(
                    text=fix_text(char.text),
                    formats=list(formats),
                    page_id=page.page_id,
                    polygon=char_box,
                    minimum_position=0,
                    maximum_position=0,
                    font="Unknown",
                    font_weight=0,
                    font_size=0,
                )
                self.store_char(marker_char, current_chars, page)
                continue

            current_span.text = fix_text(current_span.text + char.text)
            self.store_char(marker_char, current_chars, page)

            # Tokens inside a math span don't have valid boxes, so we skip the merging
            if "math" not in formats:
                current_span.polygon = current_span.polygon.merge([char_box])

        # Add the last span to the list
        if current_span:
            self.assign_chars(current_span, current_chars)
            current_line_spans.append(current_span)

        # flush last line
        if current_line_spans:
            current_line_spans[-1].text += "\n"
            all_line_spans.append(current_line_spans)

        return all_line_spans
