import re
from typing import Annotated

from marker.builders.document import DocumentBuilder
from marker.builders.line import LineBuilder
from marker.builders.ocr import OcrBuilder
from marker.builders.structure import StructureBuilder
from marker.converters.pdf import PdfConverter
from marker.extractors.document import DocumentExtractor
from marker.extractors.page import PageExtractor
from marker.providers.registry import provider_from_filepath

from marker.renderers.extraction import ExtractionRenderer, ExtractionOutput
from marker.renderers.markdown import MarkdownRenderer

from marker.logger import get_logger

logger = get_logger()


class ExtractionConverter(PdfConverter):
    pattern: str = r"{\d+\}-{48}\n\n"
    existing_markdown: Annotated[
        str, "Markdown that was already converted for extraction."
    ] = None

    def build_document(self, filepath: str):
        provider_cls = provider_from_filepath(filepath)
        layout_builder = self.resolve_dependencies(self.layout_builder_class)
        line_builder = self.resolve_dependencies(LineBuilder)
        ocr_builder = self.resolve_dependencies(OcrBuilder)
        provider = provider_cls(filepath, self.config)
        document = DocumentBuilder(self.config)(
            provider, layout_builder, line_builder, ocr_builder
        )
        structure_builder_cls = self.resolve_dependencies(StructureBuilder)
        structure_builder_cls(document)

        for processor in self.processor_list:
            processor(document)

        return document, provider

    def __call__(self, filepath: str) -> ExtractionOutput:
        self.config["paginate_output"] = True  # Ensure we can split the output properly
        self.config["output_format"] = (
            "markdown"  # Output must be markdown for extraction
        )
        markdown = self.existing_markdown

        if not markdown:
            document, provider = self.build_document(filepath)
            self.page_count = len(document.pages)
            renderer = self.resolve_dependencies(MarkdownRenderer)
            output = renderer(document)
            markdown = output.markdown

        output_pages = re.split(self.pattern, markdown)[1:]  # Split output into pages

        # This needs an LLM service for extraction, this sets it in the extractor
        if self.artifact_dict.get("llm_service") is None:
            self.artifact_dict["llm_service"] = self.resolve_dependencies(
                self.default_llm_service
            )

        page_extractor = self.resolve_dependencies(PageExtractor)
        document_extractor = self.resolve_dependencies(DocumentExtractor)
        renderer = self.resolve_dependencies(ExtractionRenderer)

        # Inference in parallel
        notes = page_extractor(output_pages)
        document_output = document_extractor(notes)

        merged = renderer(document_output, markdown)
        return merged
