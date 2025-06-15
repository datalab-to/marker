import inspect
from typing import Optional, List, Type

from pydantic import BaseModel

from marker.processors import BaseProcessor
from marker.processors.llm import BaseLLMSimpleBlockProcessor
from marker.processors.llm.llm_meta import LLMSimpleBlockMetaProcessor
from marker.rules import RuleEngine
from marker.util import assign_config, download_font


class BaseConverter:
    def __init__(self, config: Optional[BaseModel | dict] = None):
        assign_config(self, config)
        self.config = config
        self.llm_service = None
        self.rule_engine: RuleEngine | None = None

        # Download render font, needed for some providers
        download_font()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def resolve_dependencies(self, cls):
        init_signature = inspect.signature(cls.__init__)
        parameters = init_signature.parameters

        resolved_kwargs = {}
        for param_name, param in parameters.items():
            if param_name == 'self':
                continue
            elif param_name == 'config':
                resolved_kwargs[param_name] = self.config
            elif param.name == 'rule_engine':
                resolved_kwargs[param_name] = self.rule_engine
            elif param.name in self.artifact_dict:
                resolved_kwargs[param_name] = self.artifact_dict[param_name]
            elif param.default != inspect.Parameter.empty:
                resolved_kwargs[param_name] = param.default
            else:
                raise ValueError(f"Cannot resolve dependency for parameter: {param_name}")

        return cls(**resolved_kwargs)

    def initialize_processors(self, processor_cls_lst: List[Type[BaseProcessor]]) -> List[BaseProcessor]:
        processors = []
        for processor_cls in processor_cls_lst:
            processors.append(self.resolve_dependencies(processor_cls))

        simple_llm_processors = [p for p in processors if issubclass(type(p), BaseLLMSimpleBlockProcessor)]
        other_processors = [p for p in processors if not issubclass(type(p), BaseLLMSimpleBlockProcessor)]

        if not simple_llm_processors:
            return processors

        llm_positions = [i for i, p in enumerate(processors) if issubclass(type(p), BaseLLMSimpleBlockProcessor)]
        insert_position = max(0, llm_positions[-1] - len(simple_llm_processors) + 1)

        meta_processor = LLMSimpleBlockMetaProcessor(
            processor_lst=simple_llm_processors,
            llm_service=self.llm_service,
            config=self.config,
        )
        other_processors.insert(insert_position, meta_processor)
        return other_processors