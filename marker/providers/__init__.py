from copy import deepcopy
from typing import List, Optional, Dict

from PIL import Image
from pydantic import BaseModel

from pdftext.schema import Reference

from marker.logger import configure_logging
from marker.schema.polygon import PolygonBox
from marker.schema.text import Span
from marker.schema.text.char import Char
from marker.schema.text.line import Line
from marker.settings import settings
from marker.util import assign_config

configure_logging()


class ProviderOutput(BaseModel):
    line: Line
    spans: List[Span]
    chars: Optional[List[List[Char]]] = None

    @property
    def raw_text(self):
        return "".join(span.text for span in self.spans)

    def __hash__(self):
        return hash(tuple(self.line.polygon.bbox))

    def merge(self, other: "ProviderOutput"):
        new_output = deepcopy(self)
        other_copy = deepcopy(other)

        new_output.spans.extend(other_copy.spans)
        if new_output.chars is not None and other_copy.chars is not None:
            new_output.chars.extend(other_copy.chars)
        elif other_copy.chars is not None:
            new_output.chars = other_copy.chars

        new_output.line.polygon = new_output.line.polygon.merge(
            [other_copy.line.polygon]
        )
        return new_output


ProviderPageLines = Dict[int, List[ProviderOutput]]


class BaseProvider:
    def __init__(self, filepath: str | bytes, config: Optional[BaseModel | dict] = None):
        assign_config(self, config)
        self.filepath = filepath

    def __len__(self):
        pass

    def get_images(self, idxs: List[int], dpi: int) -> List[Image.Image]:
        pass

    def get_page_bbox(self, idx: int) -> PolygonBox | None:
        pass

    def get_page_lines(self, idx: int) -> List[Line]:
        pass

    def get_page_refs(self, idx: int) -> List[Reference]:
        pass

    def __enter__(self):
        return self

    @staticmethod
    def get_font_css():
        from weasyprint import CSS
        from weasyprint.text.fonts import FontConfiguration

        font_config = FontConfiguration()
        css = CSS(
            string=f"""
            @font-face {{
                font-family: GoNotoCurrent-Regular;
                src: url({settings.FONT_PATH});
                font-display: swap;
            }}
            body {{
                font-family: {settings.FONT_NAME.split(".")[0]}, sans-serif;
                font-variant-ligatures: none;
                font-feature-settings: "liga" 0;
                text-rendering: optimizeLegibility;
            }}
            """,
            font_config=font_config,
        )
        return css
