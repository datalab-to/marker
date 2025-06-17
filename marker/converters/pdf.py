import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disables a tokenizers warning

from collections import defaultdict
from typing import Annotated, Any, Dict, List, Optional, Type, Tuple

from marker.processors import BaseProcessor
from marker.processors.llm.llm_table_merge import LLMTableMergeProcessor
from marker.providers.registry import provider_from_filepath
from marker.builders.document import DocumentBuilder
from marker.builders.layout import LayoutBuilder
from marker.builders.llm_layout import LLMLayoutBuilder
from marker.builders.molecule_layout import MoleculeLayoutBuilder
from marker.builders.line import LineBuilder
from marker.builders.ocr import OcrBuilder
from marker.builders.structure import StructureBuilder
from marker.converters import BaseConverter
from marker.processors.blockquote import BlockquoteProcessor
from marker.processors.code import CodeProcessor
from marker.processors.debug import DebugProcessor
from marker.processors.document_toc import DocumentTOCProcessor
from marker.processors.equation import EquationProcessor
from marker.processors.footnote import FootnoteProcessor
from marker.processors.ignoretext import IgnoreTextProcessor
from marker.processors.line_numbers import LineNumbersProcessor
from marker.processors.list import ListProcessor
from marker.processors.llm.llm_complex import LLMComplexRegionProcessor
from marker.processors.llm.llm_form import LLMFormProcessor
from marker.processors.llm.llm_image_description import LLMImageDescriptionProcessor
from marker.processors.llm.llm_table import LLMTableProcessor
from marker.processors.llm.llm_inlinemath import LLMInlineMathLinesProcessor
from marker.processors.page_header import PageHeaderProcessor
from marker.processors.reference import ReferenceProcessor
from marker.processors.sectionheader import SectionHeaderProcessor
from marker.processors.table import TableProcessor
from marker.processors.text import TextProcessor
from marker.processors.llm.llm_equation import LLMEquationProcessor
from marker.renderers.markdown import MarkdownRenderer
from marker.schema import BlockTypes
from marker.schema.blocks import Block
from marker.schema.registry import register_block_class
from marker.util import strings_to_classes
from marker.processors.llm.llm_handwriting import LLMHandwritingProcessor
from marker.processors.order import OrderProcessor
from marker.services.gemini import GoogleGeminiService
from marker.processors.line_merge import LineMergeProcessor
from marker.processors.llm.llm_mathblock import LLMMathBlockProcessor

from datetime import datetime
from marker.utils import send_callback, flush_cuda_memory
from marker import settings
import pytz
import traceback
# 获取北京时区
beijing_tz = pytz.timezone('Asia/Shanghai')


class PdfConverter(BaseConverter):
    """
    A converter for processing and rendering PDF files into Markdown, JSON, HTML and other formats.
    """
    # A mapping to override the default block classes for specific block types
    override_map: Dict[BlockTypes, Type[Block]] = defaultdict()
    
    # Enable higher quality processing with LLMs
    use_llm: bool = False
    
    # Enable chemical molecule detection using specialized layout model
    use_molecule_detection: bool = False
    
    default_processors: Tuple[BaseProcessor, ...] = (
        OrderProcessor,
        LineMergeProcessor,
        BlockquoteProcessor,
        CodeProcessor,
        DocumentTOCProcessor,
        EquationProcessor,
        FootnoteProcessor,
        IgnoreTextProcessor,
        LineNumbersProcessor,
        ListProcessor,
        PageHeaderProcessor,
        SectionHeaderProcessor,
        TableProcessor,
        LLMTableProcessor,
        LLMTableMergeProcessor,
        LLMFormProcessor,
        TextProcessor,
        LLMInlineMathLinesProcessor,
        LLMComplexRegionProcessor,
        LLMImageDescriptionProcessor,
        LLMEquationProcessor,
        LLMHandwritingProcessor,
        LLMMathBlockProcessor,
        ReferenceProcessor,
        DebugProcessor,
    )

    def __init__(
        self,
        artifact_dict: Dict[str, Any],
        processor_list: Optional[List[str]] = None,
        renderer: str | None = None,
        llm_service: str | None = None,
        config=None,
        callback_url: str | None = None,
        docId: str | None = None,
        mol_detect: bool = False
    ):
        super().__init__(config)

        if config is None:
            config = {}

        for block_type, override_block_type in self.override_map.items():
            register_block_class(block_type, override_block_type)

        if processor_list:
            processor_list = strings_to_classes(processor_list)
        else:
            processor_list = self.default_processors

        if renderer:
            renderer = strings_to_classes([renderer])[0]
        else:
            renderer = MarkdownRenderer

        if llm_service:
            llm_service_cls = strings_to_classes([llm_service])[0]
            llm_service = self.resolve_dependencies(llm_service_cls)
        elif config.get("use_llm", False):
            llm_service = self.resolve_dependencies(GoogleGeminiService)

        # Inject llm service into artifact_dict so it can be picked up by processors, etc.
        artifact_dict["llm_service"] = llm_service
        self.llm_service = llm_service

        self.artifact_dict = artifact_dict
        self.renderer = renderer

        processor_list = self.initialize_processors(processor_list)
        self.processor_list = processor_list

        self.layout_builder_class = LayoutBuilder
        if self.use_llm:
            self.layout_builder_class = LLMLayoutBuilder
        
        # 分子layout builder
        self.molecule_layout_builder = None
        self.use_molecule_detection = mol_detect
        if config.get("use_molecule_detection", False) or self.use_molecule_detection:
            # img2mol processor的配置
            processor_config = artifact_dict.get("processor_config", {})
            
            # 设置默认配置
            if not processor_config.get("device"):
                processor_config["device"] = "cuda" if settings.TORCH_DEVICE_MODEL == "cuda" else "cpu"
            
            try:
                self.molecule_layout_builder = self.resolve_dependencies(
                    MoleculeLayoutBuilder
                )
            except Exception as e:
                traceback.print_exc()
                print(f"Warning: Failed to initialize MoleculeLayoutBuilder: {e}")
                self.molecule_layout_builder = None
        
        self.callback_url = callback_url
        self.docId = docId

    def build_document(self, filepath: str, file_type: str='pdf'):
        provider_cls = provider_from_filepath(filepath, file_type=file_type)
        layout_builder = self.resolve_dependencies(self.layout_builder_class)
        line_builder = self.resolve_dependencies(LineBuilder)
        ocr_builder = self.resolve_dependencies(OcrBuilder)
        provider = provider_cls(filepath, self.config)
        document = DocumentBuilder(self.config)(
            provider, 
            layout_builder, 
            line_builder, 
            ocr_builder, 
            callback_url=self.callback_url, 
            docId=self.docId,
            second_layout_builder=self.molecule_layout_builder
        )
        
        flush_cuda_memory()
        time_str = datetime.now(beijing_tz).strftime("%H:%M:%S")
        send_callback(self.callback_url, {
            'status': True,
            'messages': 'success',
            'docId': self.docId,
            'progress': 68,
            'progress_text': '完成OCR ' + time_str
        })

        structure_builder_cls = self.resolve_dependencies(StructureBuilder)
        structure_builder_cls(document)

        flush_cuda_memory()
        time_str = datetime.now(beijing_tz).strftime("%H:%M:%S")
        send_callback(self.callback_url, {
            'status': True,
            'messages': 'success',
            'docId': self.docId,
            'progress': 91,
            'progress_text': '完成文档重建 ' + time_str
        })

        # 移除 DebugProcessor 和 IgnoreTextProcessor
        self.processor_list = [p for p in self.processor_list 
                              if not isinstance(p, (DebugProcessor, IgnoreTextProcessor))]
        
        for processor in self.processor_list:
            processor(document)

        return document

    def __call__(self, filepath: str, file_type: str='pdf'):
        document = self.build_document(filepath, file_type=file_type)
        renderer = self.resolve_dependencies(self.renderer)
        return renderer(document)
