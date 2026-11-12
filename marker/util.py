import inspect
import os
import tempfile
from importlib import import_module
from typing import List, Annotated
import re

import numpy as np
import requests
from pydantic import BaseModel

from marker.schema.polygon import PolygonBox
from marker.settings import settings


def strings_to_classes(items: List[str]) -> List[type]:
    from marker.builders import BaseBuilder
    from marker.converters import BaseConverter
    from marker.processors import BaseProcessor
    from marker.providers import BaseProvider
    from marker.renderers import BaseRenderer
    from marker.services import BaseService

    allowed_bases = (
        BaseBuilder,
        BaseConverter,
        BaseProcessor,
        BaseProvider,
        BaseRenderer,
        BaseService,
    )
    classes = []
    for item in items:
        if "." not in item:
            raise ValueError(f"Invalid class path '{item}'; expected 'module.ClassName'")
        module_name, class_name = item.rsplit(".", 1)
        module = import_module(module_name)
        cls = getattr(module, class_name)
        if not inspect.isclass(cls):
            raise ValueError(f"Resolved object '{item}' is not a class")
        if not (
            module_name.startswith("marker.")
            or issubclass(cls, allowed_bases)
        ):
            raise ValueError(
                f"Class '{item}' must be a Marker component subclass or reside in 'marker.*'"
            )
        classes.append(cls)
    return classes


def classes_to_strings(items: List[type]) -> List[str]:
    for item in items:
        if not inspect.isclass(item):
            raise ValueError(f"Item {item} is not a class")

    return [f"{item.__module__}.{item.__name__}" for item in items]


def verify_config_keys(obj):
    annotations = inspect.get_annotations(obj.__class__)

    none_vals = ""
    for attr_name, annotation in annotations.items():
        if isinstance(annotation, type(Annotated[str, ""])):
            value = getattr(obj, attr_name)
            if value is None:
                none_vals += f"{attr_name}, "

    assert len(none_vals) == 0, (
        f"In order to use {obj.__class__.__name__}, you must set the configuration values `{none_vals}`."
    )


def assign_config(cls, config: BaseModel | dict | None):
    cls_name = cls.__class__.__name__
    if config is None:
        return
    elif isinstance(config, BaseModel):
        dict_config = config.dict()
    elif isinstance(config, dict):
        dict_config = config
    else:
        raise ValueError("config must be a dict or a pydantic BaseModel")

    for k in dict_config:
        if hasattr(cls, k):
            setattr(cls, k, dict_config[k])
    for k in dict_config:
        if cls_name not in k:
            continue
        # Enables using class-specific keys, like "MarkdownRenderer_remove_blocks"
        split_k = k.removeprefix(cls_name + "_")

        if hasattr(cls, split_k):
            setattr(cls, split_k, dict_config[k])


def parse_range_str(range_str: str) -> List[int]:
    range_lst = range_str.split(",")
    page_lst = []
    for i in range_lst:
        if "-" in i:
            start, end = i.split("-")
            page_lst += list(range(int(start), int(end) + 1))
        else:
            page_lst.append(int(i))
    page_lst = sorted(list(set(page_lst)))  # Deduplicate page numbers and sort in order
    return page_lst


def matrix_intersection_area(
    boxes1: List[List[float]], boxes2: List[List[float]]
) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1 = boxes1[:, np.newaxis, :]  # Shape: (N, 1, 4)
    boxes2 = boxes2[np.newaxis, :, :]  # Shape: (1, M, 4)

    min_x = np.maximum(boxes1[..., 0], boxes2[..., 0])  # Shape: (N, M)
    min_y = np.maximum(boxes1[..., 1], boxes2[..., 1])
    max_x = np.minimum(boxes1[..., 2], boxes2[..., 2])
    max_y = np.minimum(boxes1[..., 3], boxes2[..., 3])

    width = np.maximum(0, max_x - min_x)
    height = np.maximum(0, max_y - min_y)

    return width * height  # Shape: (N, M)


def sort_text_lines(lines: List[PolygonBox], tolerance=1.25):
    # Sorts in reading order.  Not 100% accurate, this should only
    # be used as a starting point for more advanced sorting.
    vertical_groups = {}
    for line in lines:
        group_key = round(line.bbox[1] / tolerance) * tolerance
        if group_key not in vertical_groups:
            vertical_groups[group_key] = []
        vertical_groups[group_key].append(line)

    # Sort each group horizontally and flatten the groups into a single list
    sorted_lines = []
    for _, group in sorted(vertical_groups.items()):
        sorted_group = sorted(group, key=lambda x: x.bbox[0])
        sorted_lines.extend(sorted_group)

    return sorted_lines


def download_font():
    if not os.path.exists(settings.FONT_PATH):
        font_dir = os.path.dirname(settings.FONT_PATH)
        os.makedirs(font_dir, exist_ok=True)
        font_dl_path = f"{settings.ARTIFACT_URL}/{settings.FONT_NAME}"
        fd, tmp_path = tempfile.mkstemp(dir=font_dir, suffix=".tmp")
        try:
            with (
                requests.get(font_dl_path, stream=True, timeout=15) as r,
                os.fdopen(fd, "wb") as f,
            ):
                fd = None
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            os.replace(tmp_path, settings.FONT_PATH)
        except Exception:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise


# Modification of unwrap_math from surya.recognition
MATH_SYMBOLS = ["^", "_", "\\", "{", "}"]
MATH_TAG_PATTERN = re.compile(r"<math\b[^>]*>.*?</math>", re.DOTALL)
LATEX_ESCAPES = {
    r"\%": "%",
    r"\$": "$",
    r"\_": "_",
    r"\&": "&",
    r"\#": "#",
    r"\‰": "‰",
}


def normalize_latex_escapes(s: str) -> str:
    for k, v in LATEX_ESCAPES.items():
        s = s.replace(k, v)
    return s


def unwrap_math(text: str, math_symbols: List[str] = MATH_SYMBOLS) -> str:
    """Unwrap a single <math>...</math> block if it's not really math."""
    if MATH_TAG_PATTERN.match(text):
        # Remove tags
        inner = re.sub(r"^\s*<math\b[^>]*>|</math>\s*$", "", text, flags=re.DOTALL)

        # Strip a single leading/trailing \\ plus surrounding whitespace
        inner_stripped = re.sub(r"^\s*\\\\\s*|\s*\\\\\s*$", "", inner)

        # Unwrap \text{...}
        unwrapped = re.sub(r"\\text[a-zA-Z]*\s*\{(.*?)\}", r"\1", inner_stripped)

        # Normalize escapes
        normalized = normalize_latex_escapes(unwrapped)

        # If no math symbols remain → unwrap fully
        if not any(symb in normalized for symb in math_symbols):
            return normalized.strip()

    # Otherwise, return as-is
    return text
