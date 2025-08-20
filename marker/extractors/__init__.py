from typing import Annotated, Sequence

from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.groups import PageGroup
from PIL import Image

from marker.services import BaseService
from marker.util import assign_config


class BaseExtractor:
    """
    An extractor that uses a provided service to extract structured data from documents.
    """

    max_concurrency: Annotated[
        int,
        "The maximum number of concurrent requests to make to the Gemini model.",
    ] = 3
    disable_tqdm: Annotated[
        bool,
        "Whether to disable the tqdm progress bar.",
    ] = False

    def __init__(self, llm_service: BaseService, config=None):
        assign_config(self, config)
        self.llm_service = llm_service

    def extract_image(
        self,
        document: Document,
        page: PageGroup,
        remove_blocks: Sequence[BlockTypes] | None = None,
        highres: bool = False,  # Default False to save tokens
    ) -> Image.Image:
        return page.get_image(
            document,
            highres=highres,
            remove_blocks=remove_blocks,
        )

    def __call__(self, document: Document, *args, **kwargs):
        raise NotImplementedError
