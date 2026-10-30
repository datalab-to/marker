import re
from typing import Annotated, List, Tuple

import numpy as np

from surya.ocr_error import OCRErrorPredictor

from marker.builders import BaseBuilder
from marker.providers import ProviderOutput, ProviderPageLines
from marker.providers.pdf import PdfProvider
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.settings import settings
from marker.util import matrix_intersection_area
from marker.utils.image import is_blank_image


# Strong deterministic signatures of "lying font" extraction damage: CID
# subset fonts without a ToUnicode CMap whose embedded glyph names disagree
# with the drawn outlines (a glyph named "exclam" that draws small-cap "A").
# Extractors that trust glyph names emit text shifted one ASCII band down
# (7HAT=WHAT, 4HE=THE, 9OU=YOU), fi/fl ligature glyphs glued into cap runs
# (CLARIlED, lELD, INmUENCES), quote/ellipsis glyph codes (hWRONGv,
# GOVERNMENTv, CONSCIENCEx-UST), and punctuation glyph codes surfacing as raw
# C0 control bytes (a comma as U+000C). The prose-trained ocr error model
# rates this text "good" - it is mostly valid English words in order - so a
# deterministic check is needed to route these blocks to OCR.
_DETERMINISTIC_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_DETERMINISTIC_TOKEN_EXCEPTION = re.compile(r"[0-9]{1,2}(?:ST|ND|RD|TH|AM|PM)")
_DETERMINISTIC_TOKEN_PATTERNS = (
    # Glyph codes one ASCII band down: 4HE=THE, 7HAT=WHAT, 5NDER=UNDER.
    # Ordinal and clock-time suffixes (1ST, 9AM) and URL percent escapes
    # (%2FTXT) are excluded in has_deterministic_garble; a single short
    # token (4GB) is not enough - real damage shifts several tokens in the
    # same run.
    re.compile(r"(?<!%)\b[0-9][A-Z]{2,}\b"),
    # fi/fl ligature glyphs glued into cap runs: CLARIlED, lELD, INmUENCES.
    re.compile(r"\b[A-Z]{2,}[lm][A-Z]+\b|\bl[A-Z]{2,}\b"),
    # Quote/ellipsis glyph codes: hANALYSISv, GOVERNMENTv, CONSCIENCEx-UST.
    re.compile(r"\bh[A-Z]{4,}|[A-Z]{2,}v\b|\b[A-Z]{3,}x(?:-[A-Z]{2,})?"),
)
# Bullet glyphs flattened into run-on "s " text: "s item s item s item".
# Three or more, so a lone " s word" (e.g. German seconds, "0,3 s wird")
# never triggers a re-OCR.
_DETERMINISTIC_FLATTENED_BULLET = re.compile(r"(?:^|[\s.;:()])s [A-Za-z0-9(]")
_DETERMINISTIC_MIN_FLATTENED_BULLETS = 3


def has_deterministic_garble(text: str) -> bool:
    """Detect lying-font extraction damage that needs re-OCR, no model needed.

    Tuned against a 353-document extraction corpus (English and German
    prose, standards, tables): apart from two product-name lines
    ("3DEXPERIENCE"), nothing clean triggers it. On the damaged document
    that motivated it, every one of the 18 bad text blocks (out of 621)
    is caught.
    """
    if _DETERMINISTIC_CONTROL_CHARS.search(text):
        return True
    tokens = [
        token
        for token in _DETERMINISTIC_TOKEN_PATTERNS[0].findall(text)
        if not _DETERMINISTIC_TOKEN_EXCEPTION.fullmatch(token)
    ]
    if len(tokens) >= 2 or any(len(token) >= 5 for token in tokens):
        return True
    if _DETERMINISTIC_TOKEN_PATTERNS[1].search(text):
        return True
    if _DETERMINISTIC_TOKEN_PATTERNS[2].search(text):
        return True
    return (
        len(_DETERMINISTIC_FLATTENED_BULLET.findall(text))
        >= _DETERMINISTIC_MIN_FLATTENED_BULLETS
    )



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
        "order. On by default: it beats learned reading-order models on",
        "multi-column pages (olmocr-bench order tests: 75% vs 56%). OCR'd",
        "pages get their order from full-page OCR instead. Turning this off",
        "orders every page by the layout model (in fast mode this enables its",
        "learned reading-order head for all pages).",
    ] = True
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
    deterministic_garble_filter: Annotated[
        bool,
        "Deterministically re-OCR blocks whose embedded text carries strong",
        '"lying font" damage signatures: C0 control bytes, or glyph-code-shifted',
        "tokens like 7HAT/lELD/hWRONGv from no-ToUnicode CID fonts. These never",
        "reach the ocr error model, which rates such text as good because it",
        "is mostly valid English words in the right order.",
    ] = True
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
        ocr_error_model: OCRErrorPredictor,
        config=None,
    ):
        super().__init__(config)

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

        for document_page, ocr_error_detection_label in zip(document.pages, labels):
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
        # A line overlapping >2 others (itself + 2) is suspect. Inline math
        # (stacked radical/fraction spans) and dense figure labels legitimately
        # overlap a few neighbors, so a single bad line does not condemn the
        # page - only a large fraction does (a duplicate/broken text layer).
        intersect_counts = (
            intersection_matrix > self.provider_line_provider_line_min_overlap_pct
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

        good_lines = []
        for line in lines:
            # rescale() already returns a new PolygonBox, so no deepcopy needed
            line_polygon_rescaled = line.line.polygon.rescale(page_size, image_size)
            line_bbox = line_polygon_rescaled.fit_to_bounds((0, 0, *image_size)).bbox

            if not is_blank_image(page_image.crop(line_bbox)):
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
        Strong lying-font signatures (control bytes, glyph-code-shifted
        tokens) are checked deterministically and always trigger re-OCR.
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
                    if is_blank_image(page_image.crop(crop_bbox)):
                        continue
                    block.text_extraction_method = "surya"
                elif self.deterministic_garble_filter and has_deterministic_garble(text):
                    # Lying-font damage is deterministic - flag it even when
                    # the page-level model cleared the page or the block is
                    # shorter than the ocr error model trusts.
                    block.text_extraction_method = "surya"
                elif check_garbled and len(text) >= self.min_garbled_text_chars:
                    garbled_candidates.append((block, text))

        if garbled_candidates:
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
