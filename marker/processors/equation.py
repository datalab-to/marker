from typing import Annotated, List, Tuple
from PIL import Image
import re
from bs4 import BeautifulSoup

from ftfy import fix_text, TextFixerConfig
from surya.recognition import RecognitionPredictor, OCRResult

from marker.processors import BaseProcessor
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.settings import settings

MATH_TAG_PATTERN = re.compile(r"<math[^>]*>(.*?)</math>")


class EquationProcessor(BaseProcessor):
    """
    A processor for recognizing equations in the document.
    """

    block_types: Annotated[
        Tuple[BlockTypes],
        "The block types to process.",
    ] = (BlockTypes.Equation,)
    model_max_length: Annotated[
        int,
        "The maximum number of tokens to allow for the Recognition model.",
    ] = 1024
    equation_batch_size: Annotated[
        int,
        "The batch size to use for the recognition model while processing equations.",
        "Default is None, which will use the default batch size for the model.",
    ] = None
    disable_tqdm: Annotated[
        bool,
        "Whether to disable the tqdm progress bar.",
    ] = False
    drop_repeated_text: Annotated[bool, "Drop repeated text in OCR results."] = False

    def __init__(self, recognition_model: RecognitionPredictor, config=None):
        super().__init__(config)

        self.recognition_model = recognition_model

    def get_batch_size(self):
        # Set to 1/4th of OCR batch size due to sequence length with tiling
        if self.equation_batch_size is not None:
            return self.equation_batch_size
        elif settings.TORCH_DEVICE_MODEL == "cuda":
            return 16
        elif settings.TORCH_DEVICE_MODEL == "mps":
            return 6
        return 6

    def __call__(self, document: Document):
        images = []
        equation_boxes = []
        equation_block_ids = []
        total_equation_blocks = 0

        for page in document.pages:
            page_image = page.get_image(highres=True)
            page_size = page.polygon.width, page.polygon.height
            image_size = page_image.size

            page_equation_boxes = []
            page_equation_block_ids = []
            equation_blocks = page.contained_blocks(document, self.block_types)
            for block in equation_blocks:
                page_equation_boxes.append(
                    block.polygon.rescale(page_size, image_size).bbox
                )
                page_equation_block_ids.append(block.id)
                total_equation_blocks += 1

            images.append(page_image)
            equation_boxes.append(page_equation_boxes)
            equation_block_ids.append(page_equation_block_ids)

        if total_equation_blocks == 0:
            return

        predictions = self.get_latex_batched(images, equation_boxes)
        for page_predictions, page_equation_block_ids in zip(
            predictions, equation_block_ids
        ):
            assert len(page_predictions) == len(page_equation_block_ids), (
                "Every equation block should have a corresponding prediction"
            )
            for block_prediction, block_id in zip(
                page_predictions, page_equation_block_ids
            ):
                block = document.get_block(block_id)
                block.html = self.fix_latex(block_prediction)

    def fix_latex(self, math_html: str):
        math_html = math_html.strip()
        soup = BeautifulSoup(math_html, "html.parser")
        opening_math_tag = soup.find("math")

        # No math block found
        if not opening_math_tag:
            return ""

        # Force block format
        opening_math_tag.attrs["display"] = "block"
        fixed_math_html = str(soup)

        # Sometimes model outputs newlines at the beginning/end of tags
        fixed_math_html = re.sub(
            r"^<math display=\"block\">\\n(?![a-zA-Z])",
            '<math display="block">',
            fixed_math_html,
        )
        fixed_math_html = re.sub(r"\\n</math>$", "</math>", fixed_math_html)
        fixed_math_html = fix_text(
            fixed_math_html, config=TextFixerConfig(unescape_html=True)
        )
        return fixed_math_html

    def get_latex_batched(
        self,
        page_images: List[Image.Image],
        bboxes: List[List[List[float]]],
    ):
        self.recognition_model.disable_tqdm = self.disable_tqdm
        predictions: List[OCRResult] = self.recognition_model(
            images=page_images,
            bboxes=bboxes,
            task_names=["block_without_boxes"] * len(page_images),
            recognition_batch_size=self.get_batch_size(),
            sort_lines=False,
            drop_repeated_text=self.drop_repeated_text,
        )

        equation_predictions = [
            [line.text.strip() for line in page_prediction.text_lines]
            for page_prediction in predictions
        ]

        return equation_predictions
