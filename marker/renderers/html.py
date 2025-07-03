import textwrap

from PIL import Image
from typing import Annotated, Tuple

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from pydantic import BaseModel

from marker.renderers import BaseRenderer
from marker.schema import BlockTypes
from marker.schema.blocks import BlockId
from marker.settings import settings

# Ignore beautifulsoup warnings
import warnings

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# Suppress DecompressionBombError
Image.MAX_IMAGE_PIXELS = None


class HTMLOutput(BaseModel):
    html: str
    images: dict
    metadata: dict


class HTMLRenderer(BaseRenderer):
    """
    A renderer for HTML output.
    """

    page_blocks: Annotated[
        Tuple[BlockTypes],
        "The block types to consider as pages.",
    ] = (BlockTypes.Page,)
    paginate_output: Annotated[
        bool,
        "Whether to paginate the output.",
    ] = False

    def extract_image(self, document, image_id):
        image_block = document.get_block(image_id)
        cropped = image_block.get_image(
            document, highres=self.image_extraction_mode == "highres"
        )
        return cropped

    def extract_html(self, document, document_output, level=0):
        soup = BeautifulSoup(document_output.html, "html.parser")

        content_refs = soup.find_all("content-ref")
        ref_block_id = None
        images = {}
        for ref in content_refs:
            src = ref.get("src")
            sub_images = {}
            content = ""
            for item in document_output.children:
                if item.id == src:
                    content, sub_images_ = self.extract_html(document, item, level + 1)
                    sub_images.update(sub_images_)
                    ref_block_id: BlockId = item.id
                    break

            if ref_block_id.block_type in self.image_blocks:
                if self.extract_images:
                    image = self.extract_image(document, ref_block_id)
                    image_name = f"{ref_block_id.to_path()}.{settings.OUTPUT_IMAGE_FORMAT.lower()}"
                    images[image_name] = image
                    ref.replace_with(
                        BeautifulSoup(
                            f"<p>{content}<img src='{image_name}'></p>", "html.parser"
                        )
                    )
                else:
                    # This will be the image description if using llm mode, or empty if not
                    ref.replace_with(BeautifulSoup(f"{content}", "html.parser"))
            elif ref_block_id.block_type in self.page_blocks:
                images.update(sub_images)
                if self.paginate_output:
                    content = f"<div class='page' data-page-id='{ref_block_id.page_id}'>{content}</div>"
                ref.replace_with(BeautifulSoup(f"{content}", "html.parser"))
            else:
                images.update(sub_images)
                ref.replace_with(BeautifulSoup(f"{content}", "html.parser"))

        output = str(soup)
        if level == 0:
            output = self.merge_consecutive_tags(output, "b")
            output = self.merge_consecutive_tags(output, "i")
            output = self.merge_consecutive_math(
                output
            )  # Merge consecutive inline math tags
            output = textwrap.dedent(f"""
            <!DOCTYPE html>
            <html>
                <head>
                    <meta charset="utf-8" />
                </head>
                <body>
                    {output}
                </body>
            </html>
""")

        return output, images

    def merge_continuations(self, soup):
        for p in soup.find_all("p", class_="has-continuation"):
            if p.get("block-type") not in ["Text", "TextInlineMath"]:
                # No special handling for list groups
                continue
            next_p = p.find_next_sibling("p")
            next_tag = p.find_next_sibling()
            if next_tag and next_tag.name == "p" and next_p.get("block-type") in ["Text", "TextInlineMath"]:
                for child in next_p.contents:
                    p.append(child)

                next_p.decompose()

                # Only remove the continuation class if we merge
                p.attrs.pop('class', None)

    def __call__(self, document) -> HTMLOutput:
        document_output = document.render(self.block_config)
        full_html, images = self.extract_html(document, document_output)
        soup = BeautifulSoup(full_html, 'html.parser')
        self.merge_continuations(soup)
        full_html = soup.prettify() # Add indentation to the HTML
        return HTMLOutput(
            html=full_html,
            images=images,
            metadata=self.generate_document_metadata(document, document_output),
        )
