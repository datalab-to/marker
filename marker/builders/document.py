from typing import Annotated, Sequence

from marker.builders import BaseBuilder
from marker.builders.layout import LayoutBuilder
from marker.builders.line import LineBuilder
from marker.builders.ocr import OcrBuilder
from marker.providers.pdf import PdfProvider
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.registry import get_block_class


class DocumentBuilder(BaseBuilder):
    """
    Constructs a Document given a PdfProvider, LayoutBuilder, and OcrBuilder.
    """

    lowres_image_dpi: Annotated[
        int,
        "DPI setting for low-resolution page images used for layout detection.",
    ] = 96
    highres_image_dpi: Annotated[
        int,
        "DPI setting for high-resolution page images used for OCR.",
    ] = 192
    disable_ocr: Annotated[
        bool,
        "Disable OCR processing.",
    ] = False
    mode: Annotated[
        str,
        "Conversion mode ('no_ocr' | 'fast' | 'indra' | 'balanced' | 'vllm').",
    ] = "balanced"

    def __init__(self, config=None):
        super().__init__(config)
        if self.mode == "no_ocr":
            self.disable_ocr = True

    # Block types whose default-path processors crop the high-res image:
    # TableProcessor (Table/TableOfContents/Form) and EquationProcessor
    # (Equation/ChemicalBlock). Figures/Pictures/Diagrams are only consumed by
    # opt-in LLM processors, which are served by the per-page lazy loader, so
    # they are deliberately excluded here (a figure-only page skips the render).
    highres_block_types: Annotated[
        Sequence[BlockTypes],
        "Block types whose presence on a page requires the high-res image.",
    ] = (
        BlockTypes.Table,
        BlockTypes.Form,
        BlockTypes.TableOfContents,
        BlockTypes.Equation,
        BlockTypes.ChemicalBlock,
    )

    def __call__(
        self,
        provider: PdfProvider,
        layout_builder: LayoutBuilder,
        line_builder: LineBuilder,
        ocr_builder: OcrBuilder,
    ):
        document = self.build_document(provider)
        if not self.disable_ocr and ocr_builder.can_convert_pdf_with_indra(
            provider, layout_builder
        ):
            ocr_builder.convert_pdf_with_indra(document, provider)
            return document
        layout_builder(document, provider)
        line_builder(document, provider)
        # Now that layout + the OCR decision are known, render high-res only for
        # the pages that actually need it (one batched pass). Anything missed
        # (e.g. opt-in LLM processors touching a clean page) falls back to the
        # per-page lazy loader set in build_document.
        self.render_highres(document, provider)
        if not self.disable_ocr:
            ocr_builder(document, provider)
        return document

    def page_needs_highres(self, page: PageGroup) -> bool:
        if page.text_extraction_method == "surya" or self.mode in (
            "vllm",
            "surya",
            "accurate",
        ):
            return True
        highres_types = set(self.highres_block_types)
        return any(block.block_type in highres_types for block in (page.children or []))

    def render_highres(self, document: Document, provider: PdfProvider):
        needed = [
            page
            for page in document.pages
            if page.highres_image is None and self.page_needs_highres(page)
        ]
        if not needed:
            return
        images = provider.get_images(
            [page.page_id for page in needed], self.highres_image_dpi
        )
        for page, image in zip(needed, images):
            page.highres_image = image

    def build_document(self, provider: PdfProvider):
        PageGroupClass: PageGroup = get_block_class(BlockTypes.Page)
        lowres_images = provider.get_images(provider.page_range, self.lowres_image_dpi)

        def make_loader(prov: PdfProvider):
            # Non-persistent: get_images opens/renders/closes per call.
            return lambda page_id: prov.get_images([page_id], self.highres_image_dpi)[0]

        loader = make_loader(provider)
        initial_pages = []
        for i, p in enumerate(provider.page_range):
            page = PageGroupClass(
                page_id=p,
                lowres_image=lowres_images[i],
                highres_image=None,
                polygon=provider.get_page_bbox(p),
                refs=provider.get_page_refs(p),
                pdftext_page=getattr(provider, "raw_pdftext_pages", {}).get(p),
            )
            page._highres_loader = loader
            initial_pages.append(page)
        DocumentClass: Document = get_block_class(BlockTypes.Document)
        return DocumentClass(filepath=provider.filepath, pages=initial_pages)
