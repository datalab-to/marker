import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, TypedDict, List, Sequence

from pydantic import BaseModel
from tqdm import tqdm
from PIL import Image

from marker.output import json_to_html
from marker.processors import BaseProcessor
from marker.schema import BlockTypes
from marker.schema.blocks import Block, BlockId
from marker.schema.document import Document
from marker.schema.groups import PageGroup
from marker.services import BaseService
from marker.util import assign_config
from marker.logger import get_logger

logger = get_logger()


class PromptData(TypedDict):
    prompt: str
    image: Image.Image
    block: Block
    schema: BaseModel
    page: PageGroup
    additional_data: dict | None


class BlockData(TypedDict):
    page: PageGroup
    block: Block


class BaseLLMProcessor(BaseProcessor):
    """
    A processor for using LLMs to convert blocks.
    """

    max_concurrency: Annotated[
        int,
        "The maximum number of concurrent requests to make to the Gemini model.",
    ] = 3
    image_expansion_ratio: Annotated[
        float,
        "The ratio to expand the image by when cropping.",
    ] = 0.01
    use_llm: Annotated[
        bool,
        "Whether to use the LLM model.",
    ] = False
    disable_tqdm: Annotated[
        bool,
        "Whether to disable the tqdm progress bar.",
    ] = False
    block_types = None

    def __init__(self, llm_service: BaseService, config=None):
        super().__init__(config)

        self.llm_service = None
        if not self.use_llm:
            return

        self.llm_service = llm_service

    def extract_image(
        self,
        document: Document,
        image_block: Block,
        remove_blocks: Sequence[BlockTypes] | None = None,
    ) -> Image.Image:
        return image_block.get_image(
            document,
            highres=True,
            expansion=(self.image_expansion_ratio, self.image_expansion_ratio),
            remove_blocks=remove_blocks,
        )

    def normalize_block_json(self, block: Block, document: Document, page: PageGroup):
        """
        Get the normalized JSON representation of a block for the LLM.
        """
        page_width = page.polygon.width
        page_height = page.polygon.height
        block_bbox = block.polygon.bbox

        # Normalize bbox to 0-1000 range
        normalized_bbox = [
            (block_bbox[0] / page_width) * 1000,
            (block_bbox[1] / page_height) * 1000,
            (block_bbox[2] / page_width) * 1000,
            (block_bbox[3] / page_height) * 1000,
        ]

        block_json = {
            "id": str(block.id),
            "block_type": str(block.id.block_type),
            "bbox": normalized_bbox,
            "html": json_to_html(block.render(document)),
        }

        return block_json

    def load_blocks(self, response: dict):
        return [json.loads(block) for block in response["blocks"]]

    def handle_rewrites(self, blocks: list, document: Document):
        for block_data in blocks:
            try:
                block_id = block_data["id"].strip().lstrip("/")
                _, page_id, block_type, block_id = block_id.split("/")
                block_id = BlockId(
                    page_id=page_id,
                    block_id=block_id,
                    block_type=getattr(BlockTypes, block_type),
                )
                block = document.get_block(block_id)
                if not block:
                    logger.debug(f"Block {block_id} not found in document")
                    continue

                if hasattr(block, "html"):
                    block.html = block_data["html"]
            except Exception as e:
                logger.debug(f"Error parsing block ID {block_data['id']}: {e}")
                continue


class BaseLLMComplexBlockProcessor(BaseLLMProcessor):
    """
    A processor for using LLMs to convert blocks with more complex logic.
    """

    def __call__(self, document: Document):
        if not self.use_llm or self.llm_service is None:
            return

        try:
            self.rewrite_blocks(document)
        except Exception as e:
            logger.warning(f"Error rewriting blocks in {self.__class__.__name__}: {e}")

    def process_rewriting(self, document: Document, page: PageGroup, block: Block):
        raise NotImplementedError()

    def rewrite_blocks(self, document: Document):
        # Don't show progress if there are no blocks to process
        total_blocks = sum(
            len(page.contained_blocks(document, self.block_types))
            for page in document.pages
        )
        if total_blocks == 0:
            return

        pbar = tqdm(
            total=total_blocks,
            desc=f"{self.__class__.__name__} running",
            disable=self.disable_tqdm
        )
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            for future in as_completed(
                [
                    executor.submit(self.process_rewriting, document, page, block)
                    for page in document.pages
                    for block in page.contained_blocks(document, self.block_types)
                ]
            ):
                future.result()  # Raise exceptions if any occurred
                pbar.update(1)

        pbar.close()


class BaseLLMSimpleBlockProcessor(BaseLLMProcessor):
    """
    A processor for using LLMs to convert single blocks.
    """

    # Override init since we don't need an llmservice here
    def __init__(self, config=None):
        assign_config(self, config)

    def __call__(self, result: dict, prompt_data: PromptData, document: Document):
        try:
            self.rewrite_block(result, prompt_data, document)
        except Exception as e:
            logger.warning(f"Error rewriting block in {self.__class__.__name__}: {e}")
            traceback.print_exc()

    def inference_blocks(self, document: Document) -> List[BlockData]:
        blocks = []
        for page in document.pages:
            for block in page.contained_blocks(document, self.block_types):
                blocks.append({"page": page, "block": block})
        return blocks

    def block_prompts(self, document: Document) -> List[PromptData]:
        raise NotImplementedError()

    def rewrite_block(
        self, response: dict, prompt_data: PromptData, document: Document
    ):
        raise NotImplementedError()
