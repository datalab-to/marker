import io
import json
import os
import uuid
from typing import Dict, List, Optional

from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel
from PIL import Image

from marker.renderers import CONTENT_REF_RE
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


def _splice_json_html(block: JSONBlockOutput | BlockOutput) -> str:
    children = getattr(block, "children", None)
    if not children:
        return block.html
    child_html = {str(child.id): _splice_json_html(child) for child in children}

    def repl(match) -> str:
        return child_html.get(match.group(1), match.group(0))

    return CONTENT_REF_RE.sub(repl, block.html)


def json_to_html(block: JSONBlockOutput | BlockOutput):
    # Utility function to take in json block output and give html for the block.
    # Resolves <content-ref> placeholders by string substitution (fast, no
    # per-node BeautifulSoup re-parse; this runs per block inside the LLM
    # processor loops), then normalizes once. Output matches the prior version.
    children = getattr(block, "children", None)
    if not children:
        return block.html
    return str(BeautifulSoup(_splice_json_html(block), "html.parser"))


# The file extension `save_output` writes for each `--output_format`.
OUTPUT_FORMAT_EXTENSIONS = {
    "markdown": "md",
    "html": "html",
    "json": "json",
    "chunks": "json",
}


def output_exists(
    output_dir: str, fname_base: str, output_format: Optional[str] = None
) -> bool:
    """Whether a complete conversion is already on disk.

    `output_format` matters because a markdown conversion says nothing about
    whether the json one was ever run; without it, every known extension counts.
    The metadata file is written last, so its absence means an earlier run was
    interrupted partway through and left outputs that should not be reused.
    """
    if not os.path.exists(os.path.join(output_dir, f"{fname_base}_meta.json")):
        return False

    format_ext = OUTPUT_FORMAT_EXTENSIONS.get(output_format)
    exts = (
        [format_ext]
        if format_ext
        else list(dict.fromkeys(OUTPUT_FORMAT_EXTENSIONS.values()))
    )
    return any(
        os.path.exists(os.path.join(output_dir, f"{fname_base}.{ext}")) for ext in exts
    )


def atomic_write_bytes(path: str, content: bytes) -> None:
    """Write a file that is never observable in a half-written state.

    An output truncated by an interrupted run is indistinguishable from a
    finished one, which is exactly what `output_exists` would then skip over.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    # Named rather than `mkstemp`ed, which would hand the finished file the
    # 0600 permissions of a temporary one instead of the usual umask default.
    tmp_path = os.path.join(
        directory, f".{os.path.basename(path)}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with open(tmp_path, "wb") as f:
            f.write(content)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def atomic_write_text(path: str, content: str) -> None:
    atomic_write_bytes(path, content.encode(settings.OUTPUT_ENCODING, errors="replace"))


def write_failure_report(
    output_dir: str,
    failures: List[Dict[str, str]],
    fname: str = "conversion_failures.json",
) -> str:
    """Record which inputs failed, so a batch can be re-run against the list."""
    path = os.path.join(output_dir, fname)
    atomic_write_text(path, json.dumps(failures, indent=2))
    return path


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
    else:
        raise ValueError("Invalid output type")


def convert_if_not_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def save_output(rendered: BaseModel, output_dir: str, fname_base: str) -> None:
    text, ext, images = text_from_rendered(rendered)

    atomic_write_text(os.path.join(output_dir, f"{fname_base}.{ext}"), text)

    for img_name, img in images.items():
        img = convert_if_not_rgb(img)  # RGBA images can't save as JPG
        buffer = io.BytesIO()
        img.save(buffer, settings.OUTPUT_IMAGE_FORMAT)
        atomic_write_bytes(os.path.join(output_dir, img_name), buffer.getvalue())

    # Written last, so its presence marks a conversion that reached disk whole.
    atomic_write_text(
        os.path.join(output_dir, f"{fname_base}_meta.json"),
        json.dumps(rendered.metadata, indent=2),
    )
