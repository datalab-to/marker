import re
import textwrap

from PIL import Image
from typing import Annotated, Tuple

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from pydantic import BaseModel

from marker.renderers import BaseRenderer, CONTENT_REF_RE
from marker.schema import BlockTypes
from marker.schema.blocks import BlockId
from marker.settings import settings

# Ignore beautifulsoup warnings
import warnings

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# Raise PIL limit to support high-DPI PDF page renders while preventing decompression bombs
Image.MAX_IMAGE_PIXELS = 200_000_000


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

    def insert_block_id_str(self, content: str, block_id: BlockId) -> str:
        """String wrapper around insert_block_id for the splice path."""
        if block_id.block_type in (BlockTypes.Line, BlockTypes.Span):
            return content
        return str(
            self.insert_block_id(BeautifulSoup(content, "html.parser"), block_id)
        )

    def splice_content_refs(self, document, document_output, images: dict) -> str:
        """Resolve <content-ref> placeholders to their child HTML by string
        substitution. This replaces a per-node BeautifulSoup parse that was the
        dominant render cost on the deep line/span tree. BeautifulSoup is only
        used for the optional add_block_ids annotation path."""
        children = {str(c.id): c for c in (document_output.children or [])}

        def repl(match: "re.Match") -> str:
            child = children.get(match.group(1))
            if child is None:
                return ""
            block_id: BlockId = child.id
            if block_id.block_type in self.image_blocks:
                # Image blocks keep only the extracted crop; their sub-tree's
                # images are discarded (matches prior behavior), so recurse with
                # a throwaway image dict.
                content = self.splice_content_refs(document, child, {})
                if self.extract_images:
                    image = self.extract_image(document, block_id)
                    image_name = (
                        f"{block_id.to_path()}.{settings.OUTPUT_IMAGE_FORMAT.lower()}"
                    )
                    images[image_name] = image
                    content = f"<p>{content}<img src='{image_name}'></p>"
            elif block_id.block_type in self.page_blocks:
                content = self.splice_content_refs(document, child, images)
                if self.paginate_output:
                    content = (
                        f"<div class='page' data-page-id='{block_id.page_id}'>"
                        f"{content}</div>"
                    )
            else:
                content = self.splice_content_refs(document, child, images)
            if self.add_block_ids:
                content = self.insert_block_id_str(content, block_id)
            return content

        return CONTENT_REF_RE.sub(repl, document_output.html)

    def extract_html(self, document, document_output):
        images = {}
        output = self.splice_content_refs(document, document_output, images)
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
        return HTMLOutput(
            html=full_html,
            images=images,
            metadata=self.generate_document_metadata(document, document_output),
        )
