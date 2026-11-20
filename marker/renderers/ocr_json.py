from typing import Annotated, List, Tuple

from pydantic import BaseModel

from marker.renderers import BaseRenderer
from marker.schema import BlockTypes
from marker.schema.document import Document


class OCRJSONCharOutput(BaseModel):
    id: str
    block_type: str
    text: str
    polygon: List[List[float]]
    bbox: List[float]


class OCRJSONLineOutput(BaseModel):
    id: str
    block_type: str
    html: str
    polygon: List[List[float]]
    bbox: List[float]
    children: List["OCRJSONCharOutput"] | None = None


class OCRJSONPageOutput(BaseModel):
    id: str
    block_type: str
    polygon: List[List[float]]
    bbox: List[float]
    children: List[OCRJSONLineOutput] | None = None


class OCRJSONOutput(BaseModel):
    children: List[OCRJSONPageOutput]
    block_type: str = str(BlockTypes.Document)
    metadata: dict | None = None


class OCRJSONRenderer(BaseRenderer):
    """
    A renderer for OCR JSON output.
    """

    image_blocks: Annotated[
        Tuple[BlockTypes],
        "The list of block types to consider as images.",
    ] = (BlockTypes.Picture, BlockTypes.Figure, BlockTypes.Diagram)
    page_blocks: Annotated[
        Tuple[BlockTypes],
        "The list of block types to consider as pages.",
    ] = (BlockTypes.Page,)

    def extract_json(self, document: Document) -> List[OCRJSONPageOutput]:
        pages = []
        for page in document.pages:
            page_equations = [
                b
                for b in (page.children or [])
                if b.block_type == BlockTypes.Equation and not b.removed
            ]
            equation_lines = []
            for equation in page_equations:
                if not equation.structure:
                    continue

                equation_lines += [
                    line
                    for line in equation.structure
                    if line.block_type == BlockTypes.Line
                ]

            page_lines = [
                block
                for block in (page.children or [])
                if block.block_type == BlockTypes.Line
                and block.id not in equation_lines
                and not block.removed
            ]

            # OCR'd blocks carry html directly and have no line children
            html_blocks = [
                block
                for block in (page.children or [])
                if block.block_type
                not in (
                    BlockTypes.Line,
                    BlockTypes.Span,
                    BlockTypes.Char,
                    BlockTypes.Equation,
                )
                and not block.removed
                and not block.structure
                and getattr(block, "html", None)
            ]

            lines = []
            for block in html_blocks:
                lines.append(
                    OCRJSONLineOutput(
                        id=str(block.id),
                        block_type=str(block.block_type),
                        html=block.html,
                        polygon=block.polygon.polygon,
                        bbox=block.polygon.bbox,
                    )
                )
            for line in page_lines + page_equations:
                line_obj = OCRJSONLineOutput(
                    id=str(line.id),
                    block_type=str(line.block_type),
                    html="",
                    polygon=line.polygon.polygon,
                    bbox=line.polygon.bbox,
                )
                if line in page_equations:
                    line_obj.html = line.html
                else:
                    line_obj.html = line.formatted_text(document)
                    spans = (
                        [document.get_block(span_id) for span_id in line.structure]
                        if line.structure
                        else []
                    )
                    children = []
                    for span in spans:
                        if not span.structure:
                            continue

                        span_chars = [
                            document.get_block(char_id) for char_id in span.structure
                        ]
                        children.extend(
                            [
                                OCRJSONCharOutput(
                                    id=str(char.id),
                                    block_type=str(char.block_type),
                                    text=char.text,
                                    polygon=char.polygon.polygon,
                                    bbox=char.polygon.bbox,
                                )
                                for char in span_chars
                            ]
                        )
                    line_obj.children = children
                lines.append(line_obj)

            page = OCRJSONPageOutput(
                id=str(page.id),
                block_type=str(page.block_type),
                polygon=page.polygon.polygon,
                bbox=page.polygon.bbox,
                children=lines,
            )
            pages.append(page)

        return pages

    def __call__(self, document: Document) -> OCRJSONOutput:
        return OCRJSONOutput(children=self.extract_json(document), metadata=None)
