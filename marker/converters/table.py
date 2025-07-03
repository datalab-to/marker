from typing import Tuple, List

from marker.builders.document import DocumentBuilder
from marker.builders.line import LineBuilder
from marker.builders.ocr import OcrBuilder
from marker.converters.pdf import PdfConverter
from marker.processors import BaseProcessor
from marker.processors.llm.llm_complex import LLMComplexRegionProcessor
from marker.processors.llm.llm_form import LLMFormProcessor
from marker.processors.llm.llm_table import LLMTableProcessor
from marker.processors.llm.llm_table_merge import LLMTableMergeProcessor
from marker.processors.table import TableProcessor
from marker.providers.registry import provider_from_filepath
from marker.schema import BlockTypes


class TableConverter(PdfConverter):
    default_processors: Tuple[BaseProcessor, ...] = (
        TableProcessor,
        LLMTableProcessor,
        LLMTableMergeProcessor,
        LLMFormProcessor,
        LLMComplexRegionProcessor,
    )
    converter_block_types: List[BlockTypes] = (
        BlockTypes.Table,
        BlockTypes.Form,
        BlockTypes.TableOfContents,
    )

    def build_document(self, filepath: str):
        provider_cls = provider_from_filepath(filepath)
        layout_builder = self.resolve_dependencies(self.layout_builder_class)
        line_builder = self.resolve_dependencies(LineBuilder)
        ocr_builder = self.resolve_dependencies(OcrBuilder)
        document_builder = DocumentBuilder(self.config)
        document_builder.disable_ocr = True

        provider = provider_cls(filepath, self.config)
        document = document_builder(provider, layout_builder, line_builder, ocr_builder)

        for page in document.pages:
            page.structure = [
                p for p in page.structure if p.block_type in self.converter_block_types
            ]

        for processor in self.processor_list:
            processor(document)

        return document

    def __call__(self, filepath: str):
        document = self.build_document(filepath)
        self.page_count = len(document.pages)

        renderer = self.resolve_dependencies(self.renderer)
        return renderer(document)
