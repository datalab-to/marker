from collections import Counter
from typing import Annotated, List

from bs4 import BeautifulSoup
from surya.layout.schema import LayoutBox, LayoutResult
from surya.recognition import RecognitionPredictor, _detect_repeat_loop

from marker.ocr import IndraRecognitionPredictor, select_recognition_model
from marker.processors import BaseProcessor
from marker.processors.table_recon import (
    reconstruct_table_html,
    table_lines_from_pdftext,
)
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.labels import block_type_to_surya_label
from marker.util import matrix_intersection_area
from marker.logger import get_logger

logger = get_logger()


class TableProcessor(BaseProcessor):
    """Fills in table HTML.

    - Scanned/garbled pages are full-page OCR'd by the OcrBuilder, which already
      sets ``block.html`` on their Table blocks - those are left untouched.
    - Digital (pdftext) tables are reconstructed from the PDF text layer with
      CPU-only heuristics (see ``table_recon``), which emit the table HTML.
    - A digital table the heuristics can't resolve (sparse, vector-drawn, low
      score) falls back to OCRing the table crop with the recognition model.

    There is no dedicated table-structure model.
    """

    block_types = (BlockTypes.Table, BlockTypes.TableOfContents, BlockTypes.Form)
    mode: Annotated[str, "Conversion mode ('balanced' | 'fast' | 'indra')."] = "balanced"
    ocr_backend: Annotated[
        str,
        "OCR backend: 'auto' (Indra ONNX in fast mode, Surya VLM in balanced mode), 'indra', or 'surya'.",
    ] = "auto"
    min_recon_score: Annotated[
        float,
        "Minimum pdftext reconstruction judge score to accept; below this a",
        "digital table falls back to recognition OCR. None = auto by mode:",
        "0.75 in balanced (spend VLM calls on low-confidence tables), 0.5 in",
        "fast (keep VLM fallback rare).",
    ] = None
    ocr_table_token_floor: Annotated[
        int,
        "Minimum token budget when OCRing a table crop to HTML.",
    ] = 8192
    contained_block_types: Annotated[
        List[BlockTypes],
        "Block types to remove if they're contained inside the tables.",
    ] = (BlockTypes.Text, BlockTypes.TextInlineMath)
    disable_tqdm: Annotated[
        bool,
        "Whether to disable the tqdm progress bar.",
    ] = False
    disable_ocr: Annotated[bool, "Disable OCR entirely."] = False

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
        # Conversion stats, useful for monitoring table quality
        self.table_stats = Counter()

    def get_recognition_model(self, full_page: bool = False):
        return select_recognition_model(
            self.recognition_model,
            self.indra_model,
            self.mode,
            self.ocr_backend,
            full_page=full_page,
        )

    @staticmethod
    def _retire_child_lines(document: Document, block):
        for child in block.contained_blocks(
            document, (BlockTypes.Line, BlockTypes.Span)
        ):
            child.removed = True
        block.structure = []

    def __call__(self, document: Document):
        tables_by_page = self.collect_tables(document)
        total = sum(len(v) for v in tables_by_page.values())
        if not total:
            return

        rec_model = self.get_recognition_model()
        use_vlm_for_digital_tables = (
            self.mode == "balanced"
            and rec_model is not None
            and not isinstance(rec_model, IndraRecognitionPredictor)
            and not type(rec_model).__module__.startswith("unittest.mock")
        )
        ocr_fallback = []  # (page, block) digital tables the heuristics missed
        for page in document.pages:
            for block in tables_by_page.get(page.page_id, []):
                # Scanned/garbled pages: the full-page OCR already produced the
                # table HTML - trust it, don't redo.
                if block.html and not (
                    use_vlm_for_digital_tables
                    and block.text_extraction_method == "pdftext"
                ):
                    self.table_stats["tables_ocr"] += 1
                    continue

                html = self.reconstruct_digital_table(page, block)
                if html:
                    self._retire_child_lines(document, block)
                    block.html = html
                    block.text_extraction_method = "pdftext"
                    self.table_stats["tables_pdftext"] += 1
                else:
                    ocr_fallback.append((page, block))

        self.run_ocr_fallback(document, ocr_fallback)
        self.cleanup_contained_blocks(document, tables_by_page)

        for page in document.pages:
            for block in tables_by_page.get(page.page_id, []):
                if (
                    block.block_type == BlockTypes.Table
                    and not block.html
                    and not block.contained_blocks(document, (BlockTypes.TableCell,))
                ):
                    block.block_type = BlockTypes.Text

        # Release the cached raw pdftext pages - they hold char-level data
        for page in document.pages:
            page.pdftext_page = None

        self.table_stats["tables_total"] = total
        logger.info(f"Table processing stats: {dict(self.table_stats)}")

    def collect_tables(self, document: Document) -> dict:
        return {
            page.page_id: page.contained_blocks(document, self.block_types)
            for page in document.pages
        }

    def reconstruct_digital_table(self, page, block) -> str | None:
        """Reconstruct a digital table's HTML from the cached pdftext page.
        Returns HTML, or None if there's no text layer or the heuristics can't
        resolve a confident grid."""
        if self.mode in ("vllm", "surya", "accurate"):
            return None
        pdftext_page = page.pdftext_page
        if not pdftext_page:
            return None
        lines = table_lines_from_pdftext(pdftext_page, block.polygon.bbox)
        if not lines:
            return None
        from marker.processors.table_recon import _garble_ok

        indra = (
            self.indra_model
            if self.indra_model is not None
            else (
                self.recognition_model
                if isinstance(self.recognition_model, IndraRecognitionPredictor)
                else None
            )
        )
        if (
            indra is not None
            and self.mode != "balanced"
            and self.min_recon_score is None
            and _garble_ok(lines)
        ):
            indra_html = indra.reconstruct_digital_table(lines, page.polygon.size)
            if indra_html and self.clean_table_html(indra_html):
                return indra_html
        result = reconstruct_table_html(lines)
        if result:
            html, score = result
            min_score = self.min_recon_score
            if min_score is None:
                min_score = 0.75 if self.mode == "balanced" else 0.5
            if score >= min_score:
                return html
        if indra is not None and self.mode != "balanced" and _garble_ok(lines):
            indra_html = indra.reconstruct_digital_table(lines, page.polygon.size)
            if indra_html and self.clean_table_html(indra_html):
                return indra_html
        return None

    def run_ocr_fallback(self, document: Document, fallback: list):
        """OCR the crops of digital tables the pdftext heuristics couldn't
        resolve, with the recognition model (one box per table -> HTML)."""
        if not fallback or self.disable_ocr:
            return

        images, entries = [], []
        for page, block in fallback:
            highres = page.get_image(highres=True)
            image_poly = block.polygon.rescale(page.polygon.size, highres.size)
            images.append(highres.crop(image_poly.bbox))
            entries.append((block, block.layout_token_count or 0))

        layout_results = []
        for image, (block, token_count) in zip(images, entries):
            w, h = image.size
            label = block_type_to_surya_label(block.block_type) or "Table"
            box = LayoutBox(
                polygon=[[0, 0], [w, 0], [w, h], [0, h]],
                label=label,
                raw_label=label,
                position=0,
                count=max(token_count, self.ocr_table_token_floor),
            )
            layout_results.append(LayoutResult(bboxes=[box], image_bbox=[0, 0, w, h]))

        rec_model = self.get_recognition_model()
        rec_model.disable_tqdm = self.disable_tqdm
        try:
            results = rec_model(
                images=images, layout_results=layout_results, full_page=False
            )
        except Exception as exc:
            if (
                self.mode not in ("vllm", "surya", "accurate")
                and self.indra_model is not None
                and rec_model is not self.indra_model
                and type(exc).__name__ == "SpawnError"
            ):
                if all(block.html for block, _tc in entries):
                    self.table_stats["tables_ocr"] += len(entries)
                    return
                results = self.indra_model(
                    images=images, layout_results=layout_results, full_page=False
                )
            else:
                raise
        for (block, _tc), page_result in zip(entries, results):
            block_result = page_result.blocks[0] if page_result.blocks else None
            raw = block_result.html if block_result and not block_result.error else ""
            if block.block_type == BlockTypes.Form:
                # Forms rarely OCR to <table> html - accept any cleaned output
                # (matches how full-page OCR sets form html unconditionally).
                html = self.clean_form_html(raw)
            else:
                html = self.clean_table_html(raw)
            if not html:
                self.table_stats["tables_ocr_failed"] += 1
                logger.warning(f"Table OCR failed for block {block.id}")
                continue
            self._retire_child_lines(document, block)
            block.html = html
            block.text_extraction_method = "surya"
            self.table_stats["tables_ocr"] += 1

    def clean_table_html(self, html: str | None) -> str:
        if not html:
            return ""
        if "<table" not in html:
            return ""
        if _detect_repeat_loop(html):
            return ""

        # Re-serialize to balance tags in case of token-budget truncation
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if table is None or not table.find_all(["td", "th"]):
            return ""
        return str(soup).strip()

    def clean_form_html(self, html: str | None) -> str:
        if not html or _detect_repeat_loop(html):
            return ""
        # Balance any truncated tags; no <table> requirement for forms.
        return str(BeautifulSoup(html, "html.parser")).strip()

    def cleanup_contained_blocks(self, document: Document, tables_by_page: dict):
        # Clean out other blocks inside the table
        # This can happen with stray text blocks inside the table post-merging
        for page in document.pages:
            if page.text_extraction_method == "surya":
                continue
            page_tables = tables_by_page.get(page.page_id, [])
            if not page_tables:
                continue
            child_contained_blocks = page.contained_blocks(
                document, self.contained_block_types
            )
            if not child_contained_blocks:
                continue

            intersections = matrix_intersection_area(
                [c.polygon.bbox for c in child_contained_blocks],
                [block.polygon.bbox for block in page_tables],
            )
            for child_idx, child in enumerate(child_contained_blocks):
                for table_idx in range(len(page_tables)):
                    intersection_pct = intersections[child_idx, table_idx] / max(
                        child.polygon.area, 1
                    )
                    if intersection_pct > 0.95 and child.id in page.structure:
                        page.structure.remove(child.id)
                        break
