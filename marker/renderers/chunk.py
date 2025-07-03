from typing import List, Dict

from pydantic import BaseModel

from marker.output import json_to_html
from marker.renderers.json import JSONRenderer, JSONBlockOutput
from marker.schema.document import Document


class FlatBlockOutput(BaseModel):
    id: str
    block_type: str
    html: str
    page: int
    polygon: List[List[float]]
    bbox: List[float]
    section_hierarchy: Dict[int, str] | None = None
    images: dict | None = None


class ChunkOutput(BaseModel):
    blocks: List[FlatBlockOutput]
    metadata: dict


def json_to_chunks(
    block: JSONBlockOutput, page_id: int = 0
) -> FlatBlockOutput | List[FlatBlockOutput]:
    if block.block_type == "Page":
        children = block.children
        page_id = int(block.id.split("/")[-1])
        return [json_to_chunks(child, page_id=page_id) for child in children]
    else:
        return FlatBlockOutput(
            id=block.id,
            block_type=block.block_type,
            html=json_to_html(block),
            page=page_id,
            polygon=block.polygon,
            bbox=block.bbox,
            section_hierarchy=block.section_hierarchy,
            images=block.images,
        )


class ChunkRenderer(JSONRenderer):
    def __call__(self, document: Document) -> ChunkOutput:
        document_output = document.render(self.block_config)
        json_output = []
        for page_output in document_output.children:
            json_output.append(self.extract_json(document, page_output))

        # This will get the top-level blocks from every page
        chunk_output = []
        for item in json_output:
            chunk_output.extend(json_to_chunks(item))

        return ChunkOutput(
            blocks=chunk_output,
            metadata=self.generate_document_metadata(document, document_output),
        )
