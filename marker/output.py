import json
import os

from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel
from PIL import Image

from marker.renderers.extraction import ExtractionOutput
from marker.renderers.html import HTMLOutput
from marker.renderers.json import JSONOutput, JSONBlockOutput
from marker.renderers.markdown import MarkdownOutput
from marker.renderers.ocr_json import OCRJSONOutput
from marker.schema.blocks import BlockOutput
from marker.settings import settings


def unwrap_outer_tag(html: str):
    soup = BeautifulSoup(html, "html.parser")
    contents = list(soup.contents)
    if len(contents) == 1 and isinstance(contents[0], Tag) and contents[0].name == "p":
        # Unwrap the p tag
        soup.p.unwrap()

    return str(soup)


def json_to_html(block: JSONBlockOutput | BlockOutput):
    # Utility function to take in json block output and give html for the block.
    if not getattr(block, "children", None):
        return block.html
    else:
        child_html = [json_to_html(child) for child in block.children]
        child_ids = [child.id for child in block.children]

        soup = BeautifulSoup(block.html, "html.parser")
        content_refs = soup.find_all("content-ref")
        for ref in content_refs:
            src_id = ref.attrs["src"]
            if src_id in child_ids:
                child_soup = BeautifulSoup(
                    child_html[child_ids.index(src_id)], "html.parser"
                )
                ref.replace_with(child_soup)
        return str(soup)


def output_exists(output_dir: str, fname_base: str):
    exts = ["md", "html", "json"]
    for ext in exts:
        if os.path.exists(os.path.join(output_dir, f"{fname_base}.{ext}")):
            return True
    return False


def text_from_rendered(rendered: BaseModel):
    from marker.renderers.chunk import ChunkOutput  # Has an import from this file

    if isinstance(rendered, MarkdownOutput):
        return rendered.markdown, "md", rendered.images
    elif isinstance(rendered, HTMLOutput):
        return rendered.html, "html", rendered.images
    elif isinstance(rendered, JSONOutput):
        return rendered.model_dump_json(exclude=["metadata"], indent=2), "json", {}
    elif isinstance(rendered, ChunkOutput):
        return rendered.model_dump_json(exclude=["metadata"], indent=2), "json", {}
    elif isinstance(rendered, OCRJSONOutput):
        return rendered.model_dump_json(exclude=["metadata"], indent=2), "json", {}
    elif isinstance(rendered, ExtractionOutput):
        return rendered.document_json, "json", {}
    else:
        raise ValueError("Invalid output type")


def convert_if_not_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def save_output(rendered: BaseModel, output_dir: str, fname_base: str):
    text, ext, images = text_from_rendered(rendered)
    text = text.encode(settings.OUTPUT_ENCODING, errors="replace").decode(
        settings.OUTPUT_ENCODING
    )

    with open(
        os.path.join(output_dir, f"{fname_base}.{ext}"),
        "w+",
        encoding=settings.OUTPUT_ENCODING,
    ) as f:
        f.write(text)
    with open(
        os.path.join(output_dir, f"{fname_base}_meta.json"),
        "w+",
        encoding=settings.OUTPUT_ENCODING,
    ) as f:
        f.write(json.dumps(rendered.metadata, indent=2))

    for img_name, img in images.items():
        img = convert_if_not_rgb(img)  # RGBA images can't save as JPG
        img.save(os.path.join(output_dir, img_name), settings.OUTPUT_IMAGE_FORMAT)
