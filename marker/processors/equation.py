import re
from typing import Annotated, Tuple

from bs4 import BeautifulSoup
from ftfy import fix_text, TextFixerConfig
from surya.layout.schema import LayoutBox, LayoutResult
from surya.recognition import RecognitionPredictor

from marker.logger import get_logger
from marker.ocr import IndraRecognitionPredictor, select_recognition_model
from marker.processors import BaseProcessor
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.labels import block_type_to_surya_label

logger = get_logger()


class EquationProcessor(BaseProcessor):
    """
    A processor for recognizing equations (and other always-OCR blocks) on
    pages that kept their embedded pdftext text. OCR'd pages already have
    equation html from the OcrBuilder.
    """

    block_types: Annotated[
        Tuple[BlockTypes],
        "The block types to process.",
    ] = (BlockTypes.Equation, BlockTypes.ChemicalBlock)
    equation_token_budget: Annotated[
        int,
        "Token budget for recognizing a single equation block.",
    ] = 8192
    disable_tqdm: Annotated[
        bool,
        "Whether to disable the tqdm progress bar.",
    ] = False
    disable_ocr: Annotated[
        bool,
        "Disable OCR entirely - no VLM calls at all, including equations.",
    ] = False
    ocr_equations: Annotated[
        bool,
        "OCR equation blocks (block-mode). On in both modes: pdftext cannot",
        "represent math, so equations have no text-layer substitute. Set False",
        "to skip; disable_ocr also disables this.",
    ] = True
    mode: Annotated[str, "Conversion mode ('balanced' | 'fast' | 'indra')."] = "balanced"
    ocr_backend: Annotated[
        str,
        "OCR backend: 'auto' (Indra ONNX in fast mode, Surya VLM in balanced mode), 'indra', or 'surya'.",
    ] = "auto"
    ocr_inline_math: Annotated[
        bool,
        "OCR text blocks that appear to contain inline math (pdftext extracts",
        "inline math as garbled glyphs). None = auto (on for balanced only).",
    ] = None
    inline_math_block_types: Annotated[
        Tuple[BlockTypes],
        "Text-like block types eligible for inline-math OCR.",
    ] = (
        BlockTypes.Text,
        BlockTypes.TextInlineMath,
        BlockTypes.Caption,
        BlockTypes.Footnote,
        BlockTypes.ListItem,
    )
    inline_math_min_unicode_frac: Annotated[
        float,
        "Fraction of a block's chars in math unicode ranges to flag inline math.",
    ] = 0.02
    inline_math_doc_ratio: Annotated[
        float,
        "If more than this fraction of text blocks are flagged, OCR ALL text",
        "blocks (math-heavy docs scatter inline math everywhere).",
    ] = 0.3

    def __init__(
        self,
        recognition_model: RecognitionPredictor,
        config=None,
        indra_model: IndraRecognitionPredictor = None,
    ):
        super().__init__(config)
        if self.mode == "no_ocr":
            self.disable_ocr = True

        self.recognition_model = recognition_model
        self.indra_model = indra_model

    def get_recognition_model(self, full_page: bool = False):
        return select_recognition_model(
            self.recognition_model,
            self.indra_model,
            self.mode,
            self.ocr_backend,
            full_page=full_page,
        )

    def _inline_math_enabled(self) -> bool:
        if self.ocr_inline_math is not None:
            return self.ocr_inline_math
        return (
            not self.disable_ocr
            and self.mode == "balanced"
            and not isinstance(self.get_recognition_model(), IndraRecognitionPredictor)
        )

    def __call__(self, document: Document):
        if self.disable_ocr or not self.ocr_equations:
            # No VLM: leave equation blocks as their pdftext content.
            return

        inline_math_blocks = (
            self._collect_inline_math_blocks(document)
            if self._inline_math_enabled()
            else set()
        )

        images = []
        layout_results = []
        block_ids = []

        for page in document.pages:
            target_blocks = [
                block
                for block in page.contained_blocks(
                    document, self.block_types + self.inline_math_block_types
                )
                if not block.html
                and (
                    block.block_type in self.block_types
                    or block.id in inline_math_blocks
                )
            ]
            if not target_blocks:
                continue

            page_image = page.get_image(highres=True)
            page_size = page.polygon.size
            image_size = page_image.size

            boxes = []
            page_block_ids = []
            for i, block in enumerate(target_blocks):
                polygon = block.polygon.rescale(page_size, image_size).fit_to_bounds(
                    (0, 0, *image_size)
                )
                boxes.append(
                    LayoutBox(
                        polygon=polygon.polygon,
                        label=block_type_to_surya_label(block.block_type),
                        raw_label=str(block.block_type),
                        position=i,
                        count=block.layout_token_count or self.equation_token_budget,
                    )
                )
                page_block_ids.append(block.id)

            images.append(page_image)
            layout_results.append(
                LayoutResult(bboxes=boxes, image_bbox=[0, 0, *image_size])
            )
            block_ids.append(page_block_ids)

        if not images:
            return

        rec_model = self.get_recognition_model()
        rec_model.disable_tqdm = self.disable_tqdm
        try:
            recognition_results = rec_model(
                images=images, layout_results=layout_results, full_page=False
            )
        except Exception as exc:
            if (
                self.mode not in ("vllm", "surya", "accurate")
                and self.indra_model is not None
                and rec_model is not self.indra_model
                and type(exc).__name__ == "SpawnError"
            ):
                recognition_results = self.indra_model(
                    images=images, layout_results=layout_results, full_page=False
                )
            else:
                raise

        for page_block_ids, page_result in zip(block_ids, recognition_results):
            assert len(page_block_ids) == len(page_result.blocks), (
                "Every equation block should have a corresponding prediction"
            )
            for block_id, block_result in zip(page_block_ids, page_result.blocks):
                if block_result.error or not block_result.html:
                    logger.warning(f"Equation recognition failed for {block_id}")
                    continue
                block = document.get_block(block_id)
                for child in block.contained_blocks(
                    document, (BlockTypes.Line, BlockTypes.Span)
                ):
                    child.removed = True
                block.structure = []
                if block.block_type == BlockTypes.Equation:
                    block.html = self.fix_latex(block_result.html)
                else:
                    block.html = block_result.html.strip()

    # Math-specific font-name hints (TeX math italics/symbols, AMS, STIX, etc.).
    _MATH_FONT_HINTS = (
        "cmmi",
        "cmsy",
        "cmex",
        "msam",
        "msbm",
        "stix",
        "mathjax",
        "math",
        "symbol",
    )

    @staticmethod
    def _is_math_char(c: str) -> bool:
        o = ord(c)
        return (
            0x0370 <= o <= 0x03FF  # Greek
            or 0x2070 <= o <= 0x209F  # super/subscripts
            or 0x2100 <= o <= 0x214F  # letterlike (ℝ, ℓ, …)
            or 0x2190 <= o <= 0x21FF  # arrows
            or 0x2200 <= o <= 0x22FF  # math operators
            or 0x2A00 <= o <= 0x2AFF  # supplemental math operators
        )

    def _block_has_inline_math(self, block, document) -> bool:
        spans = block.contained_blocks(document, (BlockTypes.Span,))
        if not spans:
            return False
        text = ""
        for s in spans:
            fname = (s.font or "").lower()
            if any(h in fname for h in self._MATH_FONT_HINTS):
                return True  # math font is a high-precision signal
            if getattr(s, "math", False):
                return True
            text += s.text or ""
        if not text.strip():
            return False
        math_chars = sum(1 for c in text if self._is_math_char(c))
        return math_chars / len(text) > self.inline_math_min_unicode_frac

    def _collect_inline_math_blocks(self, document: Document) -> set:
        """Ids of text blocks to OCR for inline math. Flags blocks with a math
        signal; if a large fraction of a page's text blocks are flagged, OCR all
        of them (math-heavy pages scatter inline math into ordinary prose)."""
        flagged = set()
        for page in document.pages:
            if page.text_extraction_method != "pdftext":
                continue  # OCR'd pages already have html
            text_blocks = [
                b
                for b in page.contained_blocks(document, self.inline_math_block_types)
                if not b.html
            ]
            if not text_blocks:
                continue
            page_flagged = [
                b for b in text_blocks if self._block_has_inline_math(b, document)
            ]
            if len(page_flagged) / len(text_blocks) > self.inline_math_doc_ratio:
                page_flagged = text_blocks  # escalate: whole page is math-heavy
            flagged.update(b.id for b in page_flagged)
        return flagged

    def fix_latex(self, math_html: str):
        math_html = math_html.strip()
        soup = BeautifulSoup(math_html, "html.parser")
        opening_math_tag = soup.find("math")

        # No math block found
        if not opening_math_tag:
            return ""

        # The model wraps its output in <p>; Equation.assemble_html adds its own
        # paragraph wrapper, so unwrap here to avoid nested <p><p>.
        for p in soup.find_all("p"):
            p.unwrap()

        # Force block format
        opening_math_tag.attrs["display"] = "block"
        fixed_math_html = str(soup)

        # Sometimes model outputs newlines at the beginning/end of tags
        fixed_math_html = re.sub(
            r"^<math display=\"block\">\\n(?![a-zA-Z])",
            '<math display="block">',
            fixed_math_html,
        )
        fixed_math_html = re.sub(r"\\n</math>$", "</math>", fixed_math_html)
        fixed_math_html = re.sub(r"<br>", "", fixed_math_html)
        fixed_math_html = fix_text(
            fixed_math_html, config=TextFixerConfig(unescape_html=True)
        )
        return fixed_math_html
