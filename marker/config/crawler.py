import importlib
import inspect
import pkgutil
from functools import cached_property
from typing import Annotated, Dict, Set, Type, get_args, get_origin

from marker.builders import BaseBuilder
from marker.converters import BaseConverter
from marker.extractors import BaseExtractor
from marker.processors import BaseProcessor
from marker.providers import BaseProvider
from marker.renderers import BaseRenderer
from marker.services import BaseService


class ConfigCrawler:
    def __init__(
        self,
        base_classes=(
            BaseBuilder,
            BaseProcessor,
            BaseConverter,
            BaseProvider,
            BaseRenderer,
            BaseService,
            BaseExtractor,
        ),
    ):
        self.base_classes = base_classes
        self.class_config_map: Dict[str, dict] = {}

        self._crawl_config()

    def _crawl_config(self):
        for base in self.base_classes:
            base_class_type = base.__name__.removeprefix("Base")
            self.class_config_map.setdefault(base_class_type, {})
            for class_name, class_type in self._find_subclasses(base).items():
                if class_name.startswith("Base"):
                    continue

                self.class_config_map[base_class_type].setdefault(
                    class_name, {"class_type": class_type, "config": {}}
                )
                for attr, attr_type in self._gather_super_annotations(
                    class_type
                ).items():
                    default = getattr(class_type, attr)
                    metadata = (f"Default is {default}.",)

                    if get_origin(attr_type) is Annotated:
                        if any("Default" in desc for desc in attr_type.__metadata__):
                            metadata = attr_type.__metadata__
                        else:
                            metadata = attr_type.__metadata__ + metadata
                        attr_type = get_args(attr_type)[0]

                    formatted_type = self._format_type(attr_type)
                    self.class_config_map[base_class_type][class_name]["config"][
                        attr
                    ] = (attr_type, formatted_type, default, metadata)

    @staticmethod
    def _gather_super_annotations(cls: Type) -> Dict[str, Type]:
        """
        Collect all annotated attributes from `cls` and its superclasses, bottom-up.
        Subclass attributes overwrite superclass attributes with the same name.
        """
        # We'll walk the MRO from base -> derived so subclass attributes overwrite
        # the same attribute name from superclasses.
        annotations = {}
        for base in reversed(cls.__mro__):
            if base is object:
                continue
            if hasattr(base, "__annotations__"):
                for name, annotation in base.__annotations__.items():
                    annotations[name] = annotation
        return annotations

    @cached_property
    def attr_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for base_type_dict in self.class_config_map.values():
            for class_map in base_type_dict.values():
                for attr in class_map["config"].keys():
                    counts[attr] = counts.get(attr, 0) + 1
        return counts

    @cached_property
    def attr_set(self) -> Set[str]:
        attr_set: Set[str] = set()
        for base_type_dict in self.class_config_map.values():
            for class_name, class_map in base_type_dict.items():
                for attr in class_map["config"].keys():
                    attr_set.add(attr)
                    attr_set.add(f"{class_name}_{attr}")
        return attr_set

    def _find_subclasses(self, base_class):
        subclasses = {}
        module_name = base_class.__module__
        package = importlib.import_module(module_name)
        if hasattr(package, "__path__"):
            for _, module_name, _ in pkgutil.walk_packages(
                package.__path__, module_name + "."
            ):
                try:
                    module = importlib.import_module(module_name)
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, base_class) and obj is not base_class:
                            subclasses[name] = obj
                except ImportError:
                    pass
        return subclasses

    def _format_type(self, t: Type) -> str:
        """Format a typing type like Optional[int] into a readable string."""

        if get_origin(t):  # Handle Optional and types with origins separately
            return f"{t}".removeprefix("typing.")
        else:  # Regular types like int, str
            return t.__name__


crawler = ConfigCrawler()
