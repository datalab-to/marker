import base64
import html
import io
import os
import re
import sys
import tempfile

from typing import Any
from PIL import Image
import click
import pypdfium2

try:
    import streamlit as st
    from streamlit.runtime.uploaded_file_manager import UploadedFile
except ImportError:
    class _DummyStreamlit:
        @staticmethod
        def cache_data():
            return lambda fn: fn

        @staticmethod
        def cache_resource():
            return lambda fn: fn

    st = _DummyStreamlit()
    UploadedFile = Any

from marker.config.parser import ConfigParser
from marker.config.printer import CustomClickPrinter
from marker.models import create_model_dict
from marker.providers.registry import provider_from_filepath
from marker.settings import settings

_DANGEROUS_TAGS_RE = re.compile(
    r"<\s*/?\s*(?:script|iframe|object|embed|applet|meta|link|base|form)\b[^>]*>",
    re.IGNORECASE,
)
_EVENT_ATTR_RE = re.compile(
    r"""\s+on[a-z]+\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+)""",
    re.IGNORECASE,
)
_JS_URL_RE = re.compile(
    r"""(href|src)\s*=\s*(['"]?)\s*(?:javascript|vbscript)\s*:[^'">\s]*\2""",
    re.IGNORECASE,
)


def sanitize_html_for_display(text: str) -> str:
    if not text:
        return ""
    cleaned = _DANGEROUS_TAGS_RE.sub("", text)
    cleaned = _EVENT_ATTR_RE.sub("", cleaned)
    cleaned = _JS_URL_RE.sub(r'\1="#"', cleaned)
    return cleaned


@st.cache_data()
def parse_args():
    # Use to grab common cli options
    @ConfigParser.common_options
    def options_func():
        pass

    def extract_click_params(decorated_function):
        if hasattr(decorated_function, "__click_params__"):
            return decorated_function.__click_params__
        return []

    cmd = CustomClickPrinter("Marker app.")
    extracted_params = extract_click_params(options_func)
    cmd.params.extend(extracted_params)
    ctx = click.Context(cmd)
    try:
        cmd_args = sys.argv[1:]
        cmd.parse_args(ctx, cmd_args)
        return ctx.params
    except click.exceptions.ClickException as e:
        return {"error": str(e)}


@st.cache_resource()
def load_models():
    return create_model_dict()


def open_pdf(pdf_file):
    stream = io.BytesIO(pdf_file.getvalue())
    return pypdfium2.PdfDocument(stream)


def img_to_html(img, img_alt):
    img_bytes = io.BytesIO()
    img.save(img_bytes, format=settings.OUTPUT_IMAGE_FORMAT)
    img_bytes = img_bytes.getvalue()
    encoded = base64.b64encode(img_bytes).decode()
    safe_alt = html.escape(str(img_alt or ""), quote=True)
    img_html = f'<img src="data:image/{settings.OUTPUT_IMAGE_FORMAT.lower()};base64,{encoded}" alt="{safe_alt}" style="max-width: 100%;">'
    return img_html


@st.cache_data()
def get_page_image(pdf_file, page_num, dpi=96):
    file_type = (pdf_file.type or "").lower()
    if "pdf" in file_type:
        doc = open_pdf(pdf_file)
        page = doc[page_num]
        png_image = (
            page.render(
                scale=dpi / 72,
            )
            .to_pil()
            .convert("RGB")
        )
    elif file_type.startswith("image/"):
        png_image = Image.open(pdf_file).convert("RGB")
    else:
        ext = os.path.splitext(os.path.basename(pdf_file.name or "temp.pdf"))[1] or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=ext) as tmp:
            tmp.write(pdf_file.getvalue())
            tmp.flush()
            provider_cls = provider_from_filepath(tmp.name)
            with provider_cls(tmp.name) as provider:
                idx = max(0, min(page_num, len(provider) - 1))
                png_image = provider.get_images([idx], dpi)[0].convert("RGB")
    return png_image


@st.cache_data()
def page_count(pdf_file: UploadedFile):
    file_type = (pdf_file.type or "").lower()
    if "pdf" in file_type:
        doc = open_pdf(pdf_file)
        return max(0, len(doc) - 1)
    elif file_type.startswith("image/"):
        return 0
    else:
        ext = os.path.splitext(os.path.basename(pdf_file.name or "temp.pdf"))[1] or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=ext) as tmp:
            tmp.write(pdf_file.getvalue())
            tmp.flush()
            provider_cls = provider_from_filepath(tmp.name)
            with provider_cls(tmp.name) as provider:
                return max(0, len(provider) - 1)
