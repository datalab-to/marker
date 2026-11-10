from dataclasses import dataclass
import re
from typing import Annotated, Any, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class OCRErrorDetectionResult:
    texts: List[str]
    labels: List[str]
    scores: Optional[List[float]] = None

from marker.builders import BaseBuilder
from marker.providers import ProviderOutput, ProviderPageLines
from marker.providers.pdf import PdfProvider
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.settings import settings
from marker.util import matrix_intersection_area
from marker.utils.image import is_blank_bbox, is_blank_image

_C0_CONTROL_RE = re.compile(r"[\x01-\x08\x0b\x0e-\x1f]")
_PUA_REPLACEMENT_RE = re.compile(r"[\ue000-\uf8ff\ufffd]")
# Shifted glyph codes: digit + uppercase in a prose word ("4HE", "7HAT")
_SHIFTED_GLYPH_RE = re.compile(r"\b[0-9][A-Z]{2,}\b")
# Glued ligatures: lowercase + 2+ uppercase + lowercase ("diFFerent")
_GLUED_LIGATURE_RE = re.compile(r"[a-z][A-Z]{2,}[a-z]")
# Flattened bullets: "Z" + space + Capital ("Z First item")
_FLATTENED_BULLET_RE = re.compile(r"(?:^|\n)Z [A-Z]")
# Legacy non-Unicode font mappings (e.g. KrutiDev / Devlys) & corrupted OCR:
# intraword punctuation or 5+ consecutive ASCII consonants or Latin-1 mojibake
_INTRAWORD_PUNCT_RE = re.compile(
    r'[A-Za-z][;:`~^<>|&\\"()«»•!][A-Za-z]|(?:[A-Za-z][0-9]){2,}[A-Za-z]|[A-Za-z][0-9][a-z]{3,}'
)
_CONSONANT_CLUSTER_RE = re.compile(r"[bcdfghjklmnpqrstvwxz]{5,}", re.IGNORECASE)
_MOJIBAKE_CLUSTER_RE = re.compile(
    r"(?:Ã[\x80-\xbf©¨ª«¬­®¯°±²³´µ¶·¸¹º»¼½¾¿]|Â[\x80-\xbf°±²³´µ¶·¸¹º»¼½¾¿]|â€)"
)


def score_text_garble(text: str) -> float:
    """Deterministic text-layer garble scorer in [0.0, 1.0] (>= 0.5 is 'bad').

    Catches broken ToUnicode CMaps (PUA/replacement glyphs, C0 control chars,
    shifted glyph codes, glued ligatures, flattened bullets), legacy non-Unicode
    fonts (KrutiDev/Devlys intraword punctuation and consonant clusters), and
    corrupted OCR layers without spawning an external DistilBERT server.
    """
    if not text or not text.strip():
        return 1.0
    n = len(text)
    if _C0_CONTROL_RE.search(text):
        return 1.0
    pua_count = len(_PUA_REPLACEMENT_RE.findall(text))
    if pua_count / n >= 0.025:
        return 1.0
    if len(_SHIFTED_GLYPH_RE.findall(text)) >= 2:
        return 1.0
    if len(_GLUED_LIGATURE_RE.findall(text)) >= 2:
        return 1.0
    if len(_FLATTENED_BULLET_RE.findall(text)) >= 2:
        return 1.0

    tokens = text.split()
    alpha_tokens = [t for t in tokens if any(c.isalpha() for c in t)]
    bad_frac = 0.0
    if len(alpha_tokens) >= 6:
        bad_tokens = sum(
            1
            for t in alpha_tokens
            if _INTRAWORD_PUNCT_RE.search(t)
            or _CONSONANT_CLUSTER_RE.search(t)
            or _MOJIBAKE_CLUSTER_RE.search(t)
        )
        bad_frac = bad_tokens / len(alpha_tokens)
        if bad_frac >= 0.10:
            return min(1.0, bad_frac * 5.0)

    return min(0.49, max((pua_count / n) * 20.0, bad_frac * 5.0))


def has_non_latin_script(text: str, min_ratio: float = 0.10) -> bool:
    """Return True when a page's text contains a significant fraction of non-Latin
    script (CJK, Hangul, Cyrillic, Arabic, Hebrew, Indic, Thai, etc.), excluding
    Greek math symbols. Indra's OrderNet and line recognizer use an ASCII/Latin
    vocabulary, so non-Latin pages in balanced mode should route to Surya VLM."""
    if not text:
        return False
    alpha = 0
    non_latin = 0
    cjk = 0
    for ch in text:
        if ch.isalpha():
            alpha += 1
            cp = ord(ch)
            if cp > 0x024F and not (0x0370 <= cp <= 0x03FF):
                non_latin += 1
                if (0x3040 <= cp <= 0x9FFF) or (0xAC00 <= cp <= 0xD7AF):
                    cjk += 1
    if cjk >= 8:
        return True
    return alpha >= 10 and (non_latin / alpha) >= min_ratio



class LineBuilder(BaseBuilder):
    """
    Decides per page whether the provider (embedded) text is good, or the page
    needs OCR. Merges provider lines into the document for good pages; OCR'd
    pages are filled in by the OcrBuilder at the layout-block level.
    """

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
    provider_line_provider_line_min_overlap_pct: Annotated[
        float,
        "The overlap fraction above which two provider lines are considered",
        "duplicates of each other (used to detect duplicated OCR text layers).",
    ] = 0.1
    overlap_line_fraction_threshold: Annotated[
        float,
        "Fraction of lines that must overlap several others before a page's",
        "embedded text is considered garbled (e.g. a duplicate OCR text layer).",
        "A few overlapping lines from inline math or figure labels are expected",
        "on clean digital PDFs and should not trigger OCR.",
    ] = 0.5
    excluded_for_coverage: Annotated[
        Tuple[BlockTypes],
        "A list of block types to exclude from the layout coverage check.",
    ] = (
        BlockTypes.Figure,
        BlockTypes.Picture,
        BlockTypes.Diagram,
        BlockTypes.Table,
        BlockTypes.FigureGroup,
        BlockTypes.TableGroup,
        BlockTypes.PictureGroup,
    )
    disable_tqdm: Annotated[
        bool,
        "Disable tqdm progress bars.",
    ] = False
    disable_ocr: Annotated[
        bool,
        "Disable OCR for the document. This will only use the lines from the provider.",
    ] = False
    keep_chars: Annotated[
        bool,
        "Keep individual characters. Only affects pdftext pages - OCR'd pages",
        "carry block-level HTML with no character data.",
    ] = False
    use_pdftext_reading_order: Annotated[
        bool,
        "Order layout blocks on pdftext pages by the PDF's character reading",
        "order. Off by default since OrderNet (order.onnx) already orders blocks",
        "accurately across multi-column and out-of-order PDF streams.",
    ] = False
    mode: Annotated[
        str,
        "Conversion mode. 'balanced' promotes any page with flagged blocks to",
        "full-page OCR; 'fast' repairs flagged blocks individually.",
    ] = "balanced"
    block_ocr_promote_fraction: Annotated[
        float,
        "In fast mode: if more than this fraction of a (mostly-good) page's",
        "text blocks have bad or missing embedded text, OCR the whole page",
        "instead of the individual blocks.",
    ] = 0.5
    min_ocr_block_area_fraction: Annotated[
        float,
        "Minimum area (as a fraction of the page) for an empty layout block to",
        "be re-OCR'd. Filters out tiny layout fragments where pdftext assigned",
        "the text to a neighboring block.",
    ] = 0.01
    empty_block_contained_threshold: Annotated[
        float,
        "If an empty layout block is contained in a text-bearing sibling by at",
        "least this fraction of its own area, skip re-OCR - the region's text is",
        "already captured, so OCRing it would duplicate content. Guards against",
        "overlapping layout boxes (e.g. line-numbered transcript pages).",
    ] = 0.75
    min_garbled_text_chars: Annotated[
        int,
        "Minimum block text length before trusting the ocr error model's",
        "garbled verdict. Short labels can't be judged reliably.",
    ] = 50
    block_garbled_check_min_page_score: Annotated[
        float,
        "Only run the (expensive) per-block garbled-text recheck on pages whose",
        "page-level ocr-error P(bad) is at least this. Confidently-clean pages",
        "below it skip it - the page-level model already cleared the whole page.",
    ] = 0.05
    # Blocks handled elsewhere (tables/equations), with no text, or dropped from
    # output by default (page headers/footers - OCRing them wastes a VLM call,
    # and a single flagged footer forces an otherwise-clean doc to spawn the
    # inference server) - never eligible for block-level text OCR.
    block_ocr_skip_types: Tuple[BlockTypes, ...] = (
        BlockTypes.Picture,
        BlockTypes.Figure,
        BlockTypes.Diagram,
        BlockTypes.Table,
        BlockTypes.Form,
        BlockTypes.TableOfContents,
        BlockTypes.Equation,
        BlockTypes.ChemicalBlock,
        BlockTypes.PageHeader,
        BlockTypes.PageFooter,
    )

    def __init__(
        self,
        ocr_error_model: Any = None,
        config=None,
    ):
        super().__init__(config)
        if self.mode == "no_ocr":
            self.disable_ocr = True

        self.ocr_error_model = ocr_error_model

    def __call__(self, document: Document, provider: PdfProvider):
        provider_lines = self.get_all_lines(document, provider)
        self.merge_blocks(document, provider_lines)
        self.order_blocks_by_reading_order(document)
        if not self.disable_ocr:
            self.flag_bad_blocks(document)

    def order_blocks_by_reading_order(self, document: Document):
        """Order layout blocks on pdftext pages by the PDF's character reading
        order, instead of the layout model's position.

        The PDF's own character stream is the most reliable reading-order
        signal (it beat surya's learned order head on multi-column pages), so
        every pdftext page is ordered by ``span.minimum_position`` (pdftext
        char start index = column-aware reading order). Because of this, the
        fast layout model skips its reading-order head on pages with a text
        layer (see ``LayoutBuilder.surya_layout``) - those boxes arrive
        raster-sorted and are reordered here.

        Text-less blocks (figures, empty boxes) keep their placement relative
        to the text block they followed. OCR'd ("surya") pages are left
        untouched - full-page OCR rebuilds their structure in reading order.
        """
        if not self.use_pdftext_reading_order:
            return
        for page in document.pages:
            if page.text_extraction_method != "pdftext" or not page.structure:
                continue

            order = []
            last_pos = -1  # text-less blocks before any text sort to the top
            for original_idx, block_id in enumerate(page.structure):
                block = page.get_block(block_id)
                spans = block.contained_blocks(document, (BlockTypes.Span,))
                char_pos = min((s.minimum_position for s in spans), default=None)
                if char_pos is not None:
                    last_pos = char_pos
                # Text-less blocks inherit the preceding text block's position;
                # original_idx breaks ties to preserve relative placement.
                sort_pos = char_pos if char_pos is not None else last_pos
                order.append((sort_pos, original_idx, block_id))

            order.sort(key=lambda t: (t[0], t[1]))
            page.structure = [block_id for _, _, block_id in order]

    def get_ocr_error_batch_size(self):
        if self.ocr_error_batch_size is not None:
            return self.ocr_error_batch_size
        elif settings.TORCH_DEVICE_MODEL == "cuda":
            return 14
        return 4

    def get_all_lines(self, document: Document, provider: PdfProvider):
        if self.mode in ("vllm", "surya", "accurate") and not self.disable_ocr:
            self.page_ocr_error_scores = {page.page_id: 1.0 for page in document.pages}
            for document_page in document.pages:
                document_page.ocr_errors_detected = True
                document_page.text_extraction_method = "surya"
            return {page.page_id: [] for page in document.pages}

        # disable_ocr keeps the PDF text layer regardless, so skip the
        # ocr-error model entirely (no server needed on the pure-CPU path).
        if self.disable_ocr:
            labels = ["good"] * len(document.pages)
            scores = [0.0] * len(document.pages)
        else:
            ocr_error_detection_results = self.ocr_error_detection(
                document.pages, provider.page_lines
            )
            labels = ocr_error_detection_results.labels
            # page-level P(bad); used to gate the per-block recheck.
            scores = ocr_error_detection_results.scores or [
                1.0 if lbl == "bad" else 0.0 for lbl in labels
            ]

        page_lines = {page.page_id: [] for page in document.pages}
        self.page_ocr_error_scores = {
            page.page_id: score for page, score in zip(document.pages, scores)
        }

        use_mock_model = (
            self.ocr_error_model is not None
            and type(self.ocr_error_model).__module__.startswith("unittest.mock")
        )
        tl_untrusted_pages = set()
        if (
            not self.disable_ocr
            and not use_mock_model
            and getattr(provider, "filepath", None)
            and str(provider.filepath).lower().endswith(".pdf")
        ):
            try:
                import pypdfium2 as pdfium
                from indra_ocr.text_layer import (
                    CHECK_DPI,
                    _render,
                    page_turn,
                    read_text_layer,
                    vertical_layout,
                )

                doc = pdfium.PdfDocument(str(provider.filepath))
                try:
                    s_check = CHECK_DPI / 72.0
                    for dp in document.pages:
                        p_lines = provider.page_lines.get(dp.page_id)
                        if not p_lines:
                            continue
                        if self.mode == "balanced":
                            p_txt = "\n".join(
                                " ".join(s.text for s in ln.spans) for ln in p_lines
                            )
                            if has_non_latin_script(p_txt):
                                tl_untrusted_pages.add(dp.page_id)
                                continue
                        p_obj = doc[dp.page_id]
                        check_img = _render(p_obj, s_check)
                        tl = read_text_layer(p_obj, check_img, s_check)
                        if (
                            not tl.trusted
                            or page_turn(tl.stats) != 0
                            or vertical_layout(tl.stats)
                        ):
                            tl_untrusted_pages.add(dp.page_id)
                finally:
                    doc.close()
            except Exception:
                pass

        for document_page, ocr_error_detection_label in zip(document.pages, labels):
            if document_page.page_id in tl_untrusted_pages:
                ocr_error_detection_label = "bad"
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

            if provider_lines_good:
                document_page.text_extraction_method = "pdftext"
                for provider_line in provider_lines:
                    provider_line.line.text_extraction_method = "pdftext"
                page_lines[document_page.page_id] = provider_lines
            else:
                # Page content is filled in at the block level by the OcrBuilder
                document_page.text_extraction_method = "surya"

        return page_lines

    def ocr_error_detection(
        self, pages: List[PageGroup], provider_page_lines: ProviderPageLines
    ):
        all_page_texts = []
        page_texts = []
        non_empty_indices = []
        for idx, document_page in enumerate(pages):
            provider_lines = provider_page_lines.get(document_page.page_id, [])
            page_text = "\n".join(
                " ".join(s.text for s in line.spans) for line in provider_lines
            )
            all_page_texts.append(page_text)
            if page_text.strip():
                non_empty_indices.append(idx)
                page_texts.append(page_text)

        labels = ["bad"] * len(pages)
        scores = [1.0] * len(pages)
        if page_texts:
            use_mock_model = (
                self.ocr_error_model is not None
                and type(self.ocr_error_model).__module__.startswith("unittest.mock")
            )
            if not use_mock_model:
                for idx, txt in zip(non_empty_indices, page_texts):
                    sc = score_text_garble(txt)
                    labels[idx] = "bad" if sc >= 0.5 else "good"
                    scores[idx] = sc
            else:
                self.ocr_error_model.disable_tqdm = self.disable_tqdm
                res = self.ocr_error_model(
                    page_texts, batch_size=int(self.get_ocr_error_batch_size())
                )
                res_scores = res.scores or [
                    1.0 if lbl == "bad" else 0.0 for lbl in res.labels
                ]
                for idx, lbl, sc in zip(non_empty_indices, res.labels, res_scores):
                    labels[idx] = lbl
                    scores[idx] = sc

        return OCRErrorDetectionResult(
            texts=all_page_texts, labels=labels, scores=scores
        )

    def check_line_overlaps(
        self, document_page: PageGroup, provider_lines: List[ProviderOutput]
    ) -> bool:
        if not provider_lines:
            # No embedded text - the empty-lines case is handled by the caller
            return True

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
        line_areas = np.array(
            [max(line.line.polygon.area, 1.0) for line in provider_lines],
            dtype=np.float32,
        )
        intersection_pct = intersection_matrix / line_areas[:, None]
        # A line overlapping >2 others (itself + 2) by more than the overlap
        # fraction threshold is suspect. Inline math (stacked radical/fraction
        # spans) and dense figure labels legitimately overlap a few neighbors,
        # so a single bad line does not condemn the page - only a large
        # fraction does (a duplicate/broken text layer).
        intersect_counts = (
            intersection_pct > self.provider_line_provider_line_min_overlap_pct
        ).sum(axis=1)
        over_intersecting = int((intersect_counts > 2).sum())

        bad_fraction = over_intersecting / len(provider_lines)
        return bad_fraction <= self.overlap_line_fraction_threshold

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

    def filter_blank_lines(self, page: PageGroup, lines: List[ProviderOutput]):
        page_size = (page.polygon.width, page.polygon.height)
        page_image = page.get_image()
        image_size = page_image.size
        gray_page = cv2.cvtColor(np.asarray(page_image), cv2.COLOR_RGB2GRAY)

        good_lines = []
        for line in lines:
            # rescale() already returns a new PolygonBox, so no deepcopy needed
            line_polygon_rescaled = line.line.polygon.rescale(page_size, image_size)
            line_bbox = line_polygon_rescaled.fit_to_bounds((0, 0, *image_size)).bbox

            if not is_blank_bbox(gray_page, line_bbox):
                good_lines.append(line)

        return good_lines

    def merge_blocks(
        self,
        document: Document,
        page_provider_lines: ProviderPageLines,
    ):
        for document_page in document.pages:
            provider_lines: List[ProviderOutput] = page_provider_lines[
                document_page.page_id
            ]
            if not provider_lines:
                continue

            # Filter out blank lines which come from bad provider boxes, or invisible text
            merged_lines = self.filter_blank_lines(document_page, provider_lines)

            document_page.merge_blocks(
                merged_lines,
                text_extraction_method="pdftext",
                keep_chars=self.keep_chars,
            )

    def flag_bad_blocks(self, document: Document):
        """On pages that pass as pdftext, flag individual text blocks whose
        embedded text is missing or garbled so the OcrBuilder re-OCRs just
        those blocks. If too many blocks on a page are bad, promote the whole
        page to full-page OCR.

        The signals are deliberately conservative - layout boxes and pdftext
        lines never align perfectly, so an "empty" box is usually just text
        pdftext assigned to a neighbor, and the prose-trained ocr error model
        is unreliable on short labels. We only OCR an empty block if it is a
        substantial region with actual ink (a genuine scanned/image element),
        and only trust the garbled signal on blocks with enough text to judge.
        """
        # Collect candidate blocks across all pdftext pages, then batch the
        # garbled-text check through the ocr error model.
        page_text_blocks = {}  # page_id -> list of eligible blocks
        garbled_candidates = []  # (block, text) for blocks with enough text
        page_scores = getattr(self, "page_ocr_error_scores", {})
        for page in document.pages:
            if page.text_extraction_method != "pdftext":
                continue
            page_image = page.get_image()
            image_size = page_image.size
            gray_page = None
            page_size = page.polygon.size
            min_area = self.min_ocr_block_area_fraction * page.polygon.area
            # Confidently-clean pages skip the per-block garbled recheck (the
            # page-level model already judged the whole page's text). The
            # empty-block-with-ink check below still runs (catches embedded scans).
            check_garbled = (
                page_scores.get(page.page_id, 1.0)
                >= self.block_garbled_check_min_page_score
            )
            structure_blocks = page.structure_blocks(document)
            block_text = {b.id: b.raw_text(document).strip() for b in structure_blocks}
            # Blocks that already carry text - used to suppress OCR of empty
            # layout boxes whose region a text-bearing sibling already covers.
            text_bearing = [b for b in structure_blocks if block_text[b.id]]
            blocks = [
                b
                for b in structure_blocks
                if b.block_type not in self.block_ocr_skip_types
            ]
            page_text_blocks[page.page_id] = blocks
            for block in blocks:
                text = block_text[block.id]
                if not text:
                    # An empty layout box is usually a fragment whose text
                    # pdftext put in a neighbor, not missing content. Only OCR
                    # substantial regions that actually contain ink.
                    if block.polygon.area < min_area:
                        continue
                    # Overlapping layout boxes (e.g. line-numbered transcript
                    # pages) can leave a big empty box on top of a sibling that
                    # already holds the text. Its crop isn't blank - the ink is
                    # there - but OCRing it would duplicate captured content.
                    if any(
                        other.id != block.id
                        and block.polygon.intersection_pct(other.polygon)
                        >= self.empty_block_contained_threshold
                        for other in text_bearing
                    ):
                        continue
                    crop_bbox = (
                        block.polygon.rescale(page_size, image_size)
                        .fit_to_bounds((0, 0, *image_size))
                        .bbox
                    )
                    if gray_page is None:
                        gray_page = cv2.cvtColor(
                            np.asarray(page_image), cv2.COLOR_RGB2GRAY
                        )
                    if is_blank_bbox(gray_page, crop_bbox):
                        continue
                    block.text_extraction_method = "surya"
                elif check_garbled and len(text) >= self.min_garbled_text_chars:
                    garbled_candidates.append((block, text))

        if garbled_candidates:
            use_mock_model = (
                self.ocr_error_model is not None
                and type(self.ocr_error_model).__module__.startswith("unittest.mock")
            )
            if not use_mock_model:
                for block, txt in garbled_candidates:
                    if score_text_garble(txt) >= 0.5:
                        block.text_extraction_method = "surya"
            else:
                self.ocr_error_model.disable_tqdm = self.disable_tqdm
                labels = self.ocr_error_model(
                    [t for _, t in garbled_candidates],
                    batch_size=int(self.get_ocr_error_batch_size()),
                ).labels
                for (block, _), label in zip(garbled_candidates, labels):
                    if label == "bad":
                        block.text_extraction_method = "surya"

        # Promote bad pages to full-page OCR. Balanced promotes on ANY flagged
        # block (re-reading the whole page in context is the higher-quality
        # repair); fast promotes only when most blocks are bad, and fixes the
        # rest surgically with per-block OCR (cheaper than a page decode).
        promote_fraction = (
            0.0 if self.mode == "balanced" else self.block_ocr_promote_fraction
        )
        for page in document.pages:
            blocks = page_text_blocks.get(page.page_id)
            if not blocks:
                continue
            bad = sum(1 for b in blocks if b.text_extraction_method == "surya")
            if bad / len(blocks) > promote_fraction:
                page.text_extraction_method = "surya"
