from typing import Tuple

from marker.builders.document import DocumentBuilder
from marker.builders.line import LineBuilder
from marker.builders.ocr import OcrBuilder
from marker.converters.pdf import PdfConverter
from marker.processors import BaseProcessor
from marker.processors.equation import EquationProcessor
from marker.providers.registry import provider_from_filepath
from marker.renderers.ocr_json import OCRJSONRenderer


class OCRConverter(PdfConverter):
    default_processors: Tuple[BaseProcessor, ...] = (EquationProcessor,)

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None:
            config = {}
        else:
            config = dict(config)
        config["force_ocr"] = True
        kwargs["config"] = config
        if not kwargs.get("renderer"):
            kwargs["renderer"] = "marker.renderers.ocr_json.OCRJSONRenderer"
        super().__init__(*args, **kwargs)
        self.renderer = OCRJSONRenderer

    def build_document(self, filepath: str):
        provider_cls = provider_from_filepath(filepath)
        layout_builder = self.resolve_dependencies(self.layout_builder_class)
        line_builder = self.resolve_dependencies(LineBuilder)
        ocr_builder = self.resolve_dependencies(OcrBuilder)
        document_builder = DocumentBuilder(self.config)

        with provider_cls(filepath, self.config) as provider:
            document = document_builder(
                provider, layout_builder, line_builder, ocr_builder
            )

        for processor in self.processor_list:
            processor(document)

        return document
