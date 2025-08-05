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
from marker.utils import send_callback, flush_cuda_memory
from datetime import datetime
import pytz
# 获取北京时区
beijing_tz = pytz.timezone('Asia/Shanghai')


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
    original_image_dpi: Annotated[
        int,
        "DPI setting for original-resolution page images for high-quality processing.",
    ] = 300
    disable_ocr: Annotated[
        bool,
        "Disable OCR processing.",
    ] = False

    def __call__(self, provider: PdfProvider, layout_builder: LayoutBuilder, line_builder: LineBuilder, ocr_builder: OcrBuilder, callback_url: str | None = None, docId: str | None = None, second_layout_builder = None):
        document = self.build_document(provider)
        flush_cuda_memory()
        time_str = datetime.now(beijing_tz).strftime("%H:%M:%S")
        send_callback(callback_url, {
            'status': True,
            'messages': 'success',
            'docId': docId,
            'progress': 21,
            'progress_text': '完成文字Detection ' + time_str
        })

        layout_builder(document, provider)
        # 如果有第二个layout_builder（分子识别），则调用第二个layout_builder
        if second_layout_builder:
            second_layout_builder(document, provider)
            flush_cuda_memory()
        
        line_builder(document, provider)

        flush_cuda_memory()
        time_str = datetime.now(beijing_tz).strftime("%H:%M:%S")
        send_callback(callback_url, {
            'status': True,
            'messages': 'success',
            'docId': docId,
            'progress': 42,
            'progress_text': '完成Layout解析 ' + time_str
        })

        if not self.disable_ocr:
            ocr_builder(document, provider)
        
        return document

    def build_document(self, provider: PdfProvider):
        PageGroupClass: PageGroup = get_block_class(BlockTypes.Page)
        lowres_images = provider.get_images(provider.page_range, self.lowres_image_dpi)
        highres_images = provider.get_images(provider.page_range, self.highres_image_dpi)
        original_images = provider.get_images(provider.page_range, self.original_image_dpi)
        initial_pages = [
            PageGroupClass(
                page_id=p,
                lowres_image=lowres_images[i],
                highres_image=highres_images[i],
                original_image=original_images[i],
                polygon=provider.get_page_bbox(p),
                refs=provider.get_page_refs(p)
            ) for i, p in enumerate(provider.page_range)
        ]
        DocumentClass: Document = get_block_class(BlockTypes.Document)
        return DocumentClass(filepath=provider.filepath, pages=initial_pages)
