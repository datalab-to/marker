from typing import Annotated, Dict, List, Optional, Tuple

from pydantic import BaseModel

from marker.renderers import BaseRenderer
from marker.schema import BlockTypes
from marker.schema.blocks import Block, BlockOutput
from marker.schema.document import Document
from marker.schema.registry import get_block_class


class JSONBlockOutput(BaseModel):
    id: str
    block_type: str
    html: str
    polygon: List[List[float]]
    bbox: List[float]
    children: List["JSONBlockOutput"] | None = None
    section_hierarchy: Dict[int, str] | None = None
    images: dict | None = None


class JSONOutput(BaseModel):
    children: List[JSONBlockOutput]
    block_type: str = str(BlockTypes.Document)
    metadata: dict
    llm_hierarchy: Optional[List] = None


def reformat_section_hierarchy(section_hierarchy):
    new_section_hierarchy = {}
    for key, value in section_hierarchy.items():
        new_section_hierarchy[key] = str(value)
    return new_section_hierarchy


class JSONRenderer(BaseRenderer):
    """
    A renderer for JSON output.
    """

    image_blocks: Annotated[
        Tuple[BlockTypes],
        "The list of block types to consider as images.",
    ] = (BlockTypes.Picture, BlockTypes.Figure)
    page_blocks: Annotated[
        Tuple[BlockTypes],
        "The list of block types to consider as pages.",
    ] = (BlockTypes.Page,)

    def extract_json(self, document: Document, block_output: BlockOutput):
        cls = get_block_class(block_output.id.block_type)
        if cls.__base__ == Block:
            html, images = self.extract_block_html(document, block_output)
            return JSONBlockOutput(
                html=html,
                polygon=block_output.polygon.polygon,
                bbox=block_output.polygon.bbox,
                id=str(block_output.id),
                block_type=str(block_output.id.block_type),
                images=images,
                section_hierarchy=reformat_section_hierarchy(
                    block_output.section_hierarchy
                ),
            )
        else:
            children = []
            for child in block_output.children:
                child_output = self.extract_json(document, child)
                children.append(child_output)

            return JSONBlockOutput(
                html=block_output.html,
                polygon=block_output.polygon.polygon,
                bbox=block_output.polygon.bbox,
                id=str(block_output.id),
                block_type=str(block_output.id.block_type),
                children=children,
                section_hierarchy=reformat_section_hierarchy(
                    block_output.section_hierarchy
                ),
            )

    def __call__(self, document: Document) -> JSONOutput:
        document_output = document.render(self.block_config)
        json_output = []
        for page_output in document_output.children:
            json_output.append(self.extract_json(document, page_output))
        return JSONOutput(
            children=json_output,
            metadata=self.generate_document_metadata(document, document_output),
            llm_hierarchy=document.llm_hierarchy
        )
