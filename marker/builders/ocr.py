import re
from typing import Annotated, List, Tuple

from bs4 import BeautifulSoup
from PIL import Image
from surya.layout.schema import LayoutBox, LayoutResult
from surya.recognition import (
    BlockOCRResult,
    PageOCRResult,
    RecognitionPredictor,
    _detect_repeat_loop,
)

from marker.builders import BaseBuilder
from marker.logger import get_logger
from marker.ocr import IndraRecognitionPredictor, select_recognition_model
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
    ] = 8192
    mode: Annotated[
        str,
        "Conversion mode ('no_ocr' | 'fast' | 'indra' | 'balanced' | 'vllm').",
    ] = "balanced"
    ocr_backend: Annotated[
        str,
        "OCR backend: 'auto' (Indra ONNX in fast mode, Surya VLM in balanced mode), 'indra', or 'surya'.",
    ] = "auto"

    def __init__(
        self,
        recognition_model: RecognitionPredictor,
        config=None,
        indra_model: IndraRecognitionPredictor = None,
    ):
        super().__init__(config)

        self.recognition_model = recognition_model
        self.indra_model = indra_model

    def get_recognition_model(self, full_page: bool = True):
        return select_recognition_model(
            self.recognition_model,
            self.indra_model,
            self.mode,
            self.ocr_backend,
            full_page=full_page,
        )

    def can_convert_pdf_with_indra(
        self, provider: PdfProvider, layout_builder=None
    ) -> bool:
        rec_model = self.get_recognition_model()
        return bool(
            self.mode in ("indra", "fast")
            and self.ocr_full_page
            and isinstance(rec_model, IndraRecognitionPredictor)
            and getattr(layout_builder, "force_layout_block", None) is None
            and isinstance(provider, PdfProvider)
            and getattr(provider, "filepath", None)
            and str(provider.filepath).lower().endswith(".pdf")
        )

    def can_convert_balanced_pdftext_with_indra(
        self, provider: PdfProvider
    ) -> bool:
        rec_model = self.get_recognition_model()
        is_mock = (
            rec_model is not None
            and type(rec_model).__module__.startswith("unittest.mock")
        )
        return bool(
            self.mode == "balanced"
            and self.ocr_full_page
            and self.indra_model is not None
            and not is_mock
            and isinstance(provider, PdfProvider)
            and getattr(provider, "filepath", None)
            and str(provider.filepath).lower().endswith(".pdf")
            and not bool(getattr(provider, "force_ocr", False))
        )

    def convert_pdf_with_indra(self, document: Document, provider: PdfProvider):
        rec_model = self.get_recognition_model()
        indra_model: IndraRecognitionPredictor = self.indra_model or rec_model
        use_vlm_for_ocr = (
            self.mode == "balanced"
            and not isinstance(rec_model, IndraRecognitionPredictor)
        )
        if use_vlm_for_ocr:
            target_pages = [
                p for p in document.pages if p.text_extraction_method == "pdftext"
            ]
        else:
            target_pages = list(document.pages)
        if not target_pages:
            return

        page_ids = [page.page_id for page in target_pages]
        force_ocr = bool(getattr(provider, "force_ocr", False))
        recognition_results, sources = indra_model.convert_pdf_pages(
            provider.filepath, page_ids, force_ocr=force_ocr
        )
        for page, page_result, src in zip(target_pages, recognition_results, sources):
            if use_vlm_for_ocr:
                from marker.builders.line import has_non_latin_script

                page_html_text = " ".join(
                    b.html or "" for b in (page_result.blocks or [])
                )
                if src != "text" or has_non_latin_script(page_html_text):
                    page.text_extraction_method = "surya"
                    continue
            method = "pdftext" if src in ("text", "hybrid") else "surya"
            page.text_extraction_method = method
            page_size = page.polygon.size
            if page_result.image_bbox and len(page_result.image_bbox) == 4:
                image_size = (
                    max(
                        1.0,
                        float(page_result.image_bbox[2] - page_result.image_bbox[0]),
                    ),
                    max(
                        1.0,
                        float(page_result.image_bbox[3] - page_result.image_bbox[1]),
                    ),
                )
            else:
                image_size = page_size

            new_structure = []
            for block_result in page_result.blocks:
                block_type = surya_label_to_block_type(block_result.label)
                if block_type is None or block_result.error:
                    continue
                polygon = (
                    PolygonBox(polygon=block_result.polygon)
                    .rescale(image_size, page_size)
                    .fit_to_bounds((0, 0, *page_size))
                )
                block_cls = get_block_class(block_type)
                block = page.add_block(block_cls, polygon)
                block.structure = []
                block.text_extraction_method = method
                if not block_result.skipped:
                    html = self.clean_html(block_result.html)
                    if html:
                        block.html = html
                        if use_vlm_for_ocr and (
                            block_type in (BlockTypes.Equation, BlockTypes.TextInlineMath)
                            or "<math" in html
                            or any(
                                c in html
                                for c in (
                                    "\u222b",
                                    "\u2211",
                                    "\u220f",
                                    "\u221a",
                                    "\u2264",
                                    "\u2265",
                                    "\u2260",
                                    "\u00b1",
                                    "\u221e",
                                    "\u2202",
                                    "\u2207",
                                    "\u2208",
                                    "\u2209",
                                    "\u2282",
                                    "\u2286",
                                    "\u222a",
                                    "\u2229",
                                    "\u2248",
                                    "\u2261",
                                )
                            )
                        ):
                            block.text_extraction_method = "surya"
                new_structure.append(block.id)

            for old_block in page.structure_blocks(document):
                old_block.removed = True
            page.structure = new_structure

    def __call__(self, document: Document, provider: PdfProvider):
        rec_model = self.get_recognition_model()
        if self.can_convert_pdf_with_indra(provider):
            self.convert_pdf_with_indra(document, provider)
            return
        if self.can_convert_balanced_pdftext_with_indra(provider):
            self.convert_pdf_with_indra(document, provider)

        rec_model.disable_tqdm = self.disable_tqdm

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
                try:
                    recognition_results = rec_model(
                        images=images, layout_results=layout_results, full_page=True
                    )
                except Exception as exc:
                    if (
                        self.mode not in ("vllm", "surya", "accurate")
                        and self.indra_model is not None
                        and rec_model is not self.indra_model
                        and type(exc).__name__ == "SpawnError"
                    ):
                        logger.warning(
                            f"Surya inference server unavailable ({exc}); falling back to Indra full-page OCR."
                        )
                        recognition_results = self.indra_model(
                            images=images,
                            layout_results=layout_results,
                            full_page=True,
                        )
                    else:
                        raise
                self.replace_page_structure(
                    document, full_page_pages, images, recognition_results
                )
            else:
                recognition_results = rec_model(
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

        rec_model = self.get_recognition_model(full_page=False)
        rec_model.disable_tqdm = self.disable_tqdm
        try:
            recognition_results = rec_model(
                images=images, layout_results=layout_results, full_page=False
            )
        except Exception as exc:
            if (
                self.indra_model is not None
                and rec_model is not self.indra_model
                and type(exc).__name__ == "SpawnError"
            ):
                if all(
                    document.get_block(bid).html
                    for page_bids in block_ids
                    for bid in page_bids
                ):
                    return
                recognition_results = self.indra_model(
                    images=images, layout_results=layout_results, full_page=False
                )
            else:
                raise
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

    def fuse_vlm_with_indra(
        self,
        images: List[Image.Image],
        vlm_results: List[PageOCRResult],
        provider: PdfProvider | None = None,
        pages: List[PageGroup] | None = None,
    ) -> List[PageOCRResult]:
        """Fuse Surya VLM full-page blocks with Indra's untruncated page structure."""
        if self.indra_model is None:
            return vlm_results

        def _strip_tags(s: str) -> str:
            return " ".join(re.sub(r"<[^>]+>", " ", s or "").split())

        def _real_math_count(html_s: str) -> int:
            cnt = 0
            for m in re.findall(
                r"<math\b[^>]*>(.*?)</math>", html_s or "", flags=re.S | re.I
            ):
                ms = m.strip()
                if len(ms) >= 4 or any(
                    c in ms for c in ("\\", "=", "^", "_", "+", "{", "}")
                ):
                    cnt += 1
            return cnt

        indra_results = self.indra_model(images=images, full_page=True)
        fused_pages: List[PageOCRResult] = []

        for img, vlm_res, indra_res in zip(images, vlm_results, indra_results):
            w, h = img.size
            page_bbox = [0.0, 0.0, float(w), float(h)]
            vlm_blocks = list(vlm_res.blocks) if vlm_res and vlm_res.blocks else []
            if not vlm_blocks:
                fused_pages.append(indra_res)
                continue
            if not indra_res or not indra_res.blocks:
                fused_pages.append(vlm_res)
                continue

            if indra_res.image_bbox and len(indra_res.image_bbox) == 4:
                iw = max(1.0, float(indra_res.image_bbox[2] - indra_res.image_bbox[0]))
                ih = max(1.0, float(indra_res.image_bbox[3] - indra_res.image_bbox[1]))
                if abs(iw - w) > 1.0 or abs(ih - h) > 1.0:
                    sx, sy = w / iw, h / ih
                    for ib in indra_res.blocks:
                        ib.polygon = [[pt[0] * sx, pt[1] * sy] for pt in ib.polygon]
                    indra_res.image_bbox = page_bbox

            vlm_txt = _strip_tags(" ".join(b.html or "" for b in vlm_blocks))
            indra_txt = _strip_tags(" ".join(b.html or "" for b in indra_res.blocks))
            vlm_math_cnt = sum(_real_math_count(b.html) for b in vlm_blocks)
            indra_math_cnt = sum(_real_math_count(b.html) for b in indra_res.blocks)

            vlm_bboxes = [
                (
                    min(pt[0] for pt in vb.polygon),
                    min(pt[1] for pt in vb.polygon),
                    max(pt[0] for pt in vb.polygon),
                    max(pt[1] for pt in vb.polygon),
                )
                for vb in vlm_blocks
            ]
            indra_bboxes = [
                (
                    min(pt[0] for pt in ib.polygon),
                    min(pt[1] for pt in ib.polygon),
                    max(pt[0] for pt in ib.polygon),
                    max(pt[1] for pt in ib.polygon),
                )
                for ib in indra_res.blocks
            ]

            covered_indra_indices = set()
            for j, (ix0, iy0, ix1, iy1) in enumerate(indra_bboxes):
                i_area = max(1.0, (ix1 - ix0) * (iy1 - iy0))
                inter_sum = 0.0
                for vx0, vy0, vx1, vy1 in vlm_bboxes:
                    ox = max(0.0, min(ix1, vx1) - max(ix0, vx0))
                    oy = max(0.0, min(iy1, vy1) - max(iy0, vy0))
                    inter_sum += ox * oy
                if (inter_sum / i_area) >= 0.30:
                    covered_indra_indices.add(j)

            covered_indra_txt = _strip_tags(
                " ".join(
                    indra_res.blocks[j].html or ""
                    for j in sorted(covered_indra_indices)
                )
            )
            has_indra_table = any(
                b.label in ("Table", "Form", "TableOfContents")
                for b in indra_res.blocks
            )
            has_indra_formula = any(
                b.label in ("Formula", "Equation") for b in indra_res.blocks
            )

            if (
                not has_indra_table
                and not has_indra_formula
                and len(vlm_txt) >= len(indra_txt) * 0.90
                and (
                    len(vlm_txt) > max(30, len(covered_indra_txt) * 1.12)
                    or vlm_math_cnt >= indra_math_cnt + 2
                )
            ):
                out_blocks = list(vlm_blocks)
                for j, ib in enumerate(indra_res.blocks):
                    if j not in covered_indra_indices:
                        ib.reading_order = len(out_blocks)
                        out_blocks.append(ib)
                fused_pages.append(
                    PageOCRResult(blocks=out_blocks, image_bbox=page_bbox)
                )
                continue

            replaced_indra = {}
            extra_vlm = []
            used_indra = set()

            for vb, (vx0, vy0, vx1, vy1) in zip(vlm_blocks, vlm_bboxes):
                if vb.skipped or not (vb.html or "").strip():
                    continue
                v_area = max(1.0, (vx1 - vx0) * (vy1 - vy0))
                group_j = []
                has_partial = False
                total_inter_with_indra = 0.0

                for j, (ix0, iy0, ix1, iy1) in enumerate(indra_bboxes):
                    ox = max(0.0, min(ix1, vx1) - max(ix0, vx0))
                    oy = max(0.0, min(iy1, vy1) - max(iy0, vy0))
                    inter = ox * oy
                    if inter <= 0.0:
                        continue
                    total_inter_with_indra += inter
                    i_area = max(1.0, (ix1 - ix0) * (iy1 - iy0))
                    ratio = inter / i_area
                    if ratio >= 0.45:
                        if j in used_indra:
                            has_partial = True
                        else:
                            group_j.append(j)
                    elif ratio >= 0.20:
                        has_partial = True

                vb_txt = _strip_tags(vb.html or "")
                vb_math = _real_math_count(vb.html)

                if not group_j and (total_inter_with_indra / v_area) < 0.15:
                    if len(vb_txt) >= 6 or vb_math > 0:
                        extra_vlm.append((vy0, vx0, vb))
                    continue

                if group_j and not has_partial and len(group_j) <= 3:
                    g_blocks = [indra_res.blocks[j] for j in group_j]
                    g_txt = _strip_tags(" ".join(b.html or "" for b in g_blocks))
                    g_math = sum(_real_math_count(b.html) for b in g_blocks)
                    g_has_table = any(
                        b.label in ("Table", "Form", "TableOfContents")
                        for b in g_blocks
                    )
                    g_has_formula = any(
                        b.label in ("Formula", "Equation") for b in g_blocks
                    )
                    g_has_heading = any(
                        b.label in ("SectionHeader", "Title") for b in g_blocks
                    )

                    if len(group_j) > 1 and (
                        g_has_heading or g_has_table or g_has_formula
                    ):
                        continue

                    upgrade = False
                    if g_has_table:
                        if (
                            "</table>" in (vb.html or "").lower()
                            and len(vb_txt) >= max(8, len(g_txt) * 0.65)
                        ):
                            upgrade = True
                    elif g_has_formula:
                        if (
                            vb_math > 0
                            and "</math>" in (vb.html or "").lower()
                            and len(vb_txt) >= max(3, len(g_txt) * 0.45)
                        ):
                            upgrade = True
                    elif vb_math > g_math and len(vb_txt) >= max(
                        4, len(g_txt) * (0.65 if len(group_j) == 1 else 0.85)
                    ):
                        upgrade = True
                    elif (
                        vb_math >= g_math
                        and len(vb_txt) > max(15, len(g_txt) * 1.18)
                        and len(vb_txt) <= max(80, len(g_txt) * 3.0)
                    ):
                        upgrade = True

                    if upgrade:
                        first_j = group_j[0]
                        if len(group_j) == 1:
                            orig_b = indra_res.blocks[first_j]
                            vb = BlockOCRResult(
                                polygon=orig_b.polygon,
                                label=orig_b.label,
                                raw_label=orig_b.raw_label,
                                reading_order=orig_b.reading_order,
                                html=vb.html,
                                skipped=False,
                                error=False,
                                confidence=1.0,
                            )
                        replaced_indra[first_j] = vb
                        for j in group_j:
                            used_indra.add(j)

            fused_blocks = []
            for j, ib in enumerate(indra_res.blocks):
                if j in replaced_indra:
                    vb = replaced_indra[j]
                    vb.reading_order = len(fused_blocks)
                    fused_blocks.append(vb)
                elif j not in used_indra:
                    ib.reading_order = len(fused_blocks)
                    fused_blocks.append(ib)

            for _, _, vb in sorted(extra_vlm, key=lambda x: (x[0], x[1])):
                vb.reading_order = len(fused_blocks)
                fused_blocks.append(vb)

            fused_pages.append(
                PageOCRResult(blocks=fused_blocks, image_bbox=page_bbox)
            )

        return fused_pages

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

        # Re-serializing through BeautifulSoup balances any tags left open by
        # token-budget truncation
        return str(soup).strip()
