import html as html_lib
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

    def insert_block_id(self, soup, block_id: BlockId):
        """
        Insert a block ID into the soup as a data attribute.
        """
        if block_id.block_type in [BlockTypes.Line, BlockTypes.Span]:
            return soup

        if self.add_block_ids:
            # Find the outermost tag (first tag that isn't a NavigableString)
            outermost_tag = None
            for element in soup.contents:
                if hasattr(element, "name") and element.name:
                    outermost_tag = element
                    break

            # If we found an outermost tag, add the data-block-id attribute
            if outermost_tag:
                outermost_tag["data-block-id"] = str(block_id)

            # If soup only contains text or no tags, wrap in a span
            elif soup.contents:
                wrapper = soup.new_tag("span")
                wrapper["data-block-id"] = str(block_id)

                contents = list(soup.contents)
                for content in contents:
                    content.extract()
                    wrapper.append(content)
                soup.append(wrapper)
        return soup

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
                    # Extract description from content and use as alt text to avoid duplication
                    alt_text = ""
                    content_soup = BeautifulSoup(content, "html.parser")
                    desc_tag = content_soup.find("p", attrs={"role": "img"})
                    if desc_tag:
                        alt_text = desc_tag.get_text().strip()
                        # Remove the "Image ... description: " prefix
                        prefix_end = alt_text.find("description: ")
                        if prefix_end != -1:
                            alt_text = alt_text[prefix_end + len("description: "):]
                        desc_tag.decompose()
                        content = str(content_soup)
                    element = BeautifulSoup(
                        f"<p>{content}<img src='{image_name}' alt='{html_lib.escape(alt_text, quote=True)}'></p>", "html.parser"
                    )
                    ref.replace_with(self.insert_block_id(element, ref_block_id))
                else:
                    # This will be the image description if using llm mode, or empty if not
                    element = BeautifulSoup(f"{content}", "html.parser")
                    ref.replace_with(self.insert_block_id(element, ref_block_id))
            elif ref_block_id.block_type in self.page_blocks:
                images.update(sub_images)
                if self.paginate_output:
                    content = f"<div class='page' data-page-id='{ref_block_id.page_id}'>{content}</div>"
                element = BeautifulSoup(f"{content}", "html.parser")
                ref.replace_with(self.insert_block_id(element, ref_block_id))
            else:
                images.update(sub_images)
                element = BeautifulSoup(f"{content}", "html.parser")
                ref.replace_with(self.insert_block_id(element, ref_block_id))

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

    def __call__(self, document) -> HTMLOutput:
        document_output = document.render(self.block_config)
        full_html, images = self.extract_html(document, document_output)
        soup = BeautifulSoup(full_html, "html.parser")
        full_html = soup.prettify()  # Add indentation to the HTML
        return HTMLOutput(
            html=full_html,
            images=images,
            metadata=self.generate_document_metadata(document, document_output),
        )
