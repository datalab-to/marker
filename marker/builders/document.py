from typing import Annotated

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
        "DPI setting for low-resolution page images used for Layout and Line Detection.",
    ] = 96
    highres_image_dpi: Annotated[
        int,
        "DPI setting for high-resolution page images used for OCR.",
    ] = 192
    disable_ocr: Annotated[
        bool,
        "Disable OCR processing.",
    ] = False
    page_batch_size: Annotated[
        int,
        "Number of pages to process at once. 0 = all pages (default, fastest but most memory). "
        ">0 = process in batches, compressing images after each batch to dramatically reduce "
        "RAM usage on large documents (97-page doc: ~13 GB -> ~200 MB).",
    ] = 0

    def __call__(self, provider: PdfProvider, layout_builder: LayoutBuilder, line_builder: LineBuilder, ocr_builder: OcrBuilder):
        if self.page_batch_size > 0:
            return self._build_streaming(provider, layout_builder, line_builder, ocr_builder)
        else:
            return self._build_all_at_once(provider, layout_builder, line_builder, ocr_builder)

    def _build_all_at_once(self, provider, layout_builder, line_builder, ocr_builder):
        """Original behavior: load all images, process all pages."""
        document = self._create_document_structure(provider, load_images=True)
        layout_builder(document, provider)
        line_builder(document, provider)
        if not self.disable_ocr:
            ocr_builder(document, provider)
        return document

    def _build_streaming(self, provider, layout_builder, line_builder, ocr_builder):
        """Batch mode: create pages without images, load and process in batches."""
        document = self._create_document_structure(provider, load_images=False)
        batch_size = self.page_batch_size

        for start in range(0, len(document.pages), batch_size):
            end = min(start + batch_size, len(document.pages))
            batch = document.pages[start:end]

            # Load images for this batch only
            ids = [p.page_id for p in batch]
            lowres = provider.get_images(ids, self.lowres_image_dpi)
            highres = provider.get_images(ids, self.highres_image_dpi)
            for i, page in enumerate(batch):
                page.lowres_image = lowres[i]
                page.highres_image = highres[i]

            # Builders iterate document.pages — temporarily scope to the batch
            original_pages = document.pages
            document.pages = batch
            try:
                layout_builder(document, provider)
                line_builder(document, provider)
                if not self.disable_ocr:
                    ocr_builder(document, provider)
            finally:
                document.pages = original_pages

            # Compress images to bytes to free CPU RAM (~100x reduction)
            for page in batch:
                page.compress_images()

        return document

    def _create_document_structure(self, provider: PdfProvider, load_images: bool):
        """Create Document with page structure. If load_images=False, pages start with no images."""
        PageGroupClass: PageGroup = get_block_class(BlockTypes.Page)

        if load_images:
            lowres_images = provider.get_images(provider.page_range, self.lowres_image_dpi)
            highres_images = provider.get_images(provider.page_range, self.highres_image_dpi)
        else:
            lowres_images = [None] * len(provider.page_range)
            highres_images = [None] * len(provider.page_range)

        initial_pages = [
            PageGroupClass(
                page_id=p,
                lowres_image=lowres_images[i],
                highres_image=highres_images[i],
                polygon=provider.get_page_bbox(p),
                refs=provider.get_page_refs(p)
            ) for i, p in enumerate(provider.page_range)
        ]
        DocumentClass: Document = get_block_class(BlockTypes.Document)
        return DocumentClass(filepath=provider.filepath, pages=initial_pages)
