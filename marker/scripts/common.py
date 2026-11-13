import base64
import html
import io
import re

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


def img_to_html(img, img_alt):
    img_bytes = io.BytesIO()
    img.save(img_bytes, format=settings.OUTPUT_IMAGE_FORMAT)
    img_bytes = img_bytes.getvalue()
    encoded = base64.b64encode(img_bytes).decode()
    safe_alt = html.escape(str(img_alt or ""), quote=True)
    img_html = f'<img src="data:image/{settings.OUTPUT_IMAGE_FORMAT.lower()};base64,{encoded}" alt="{safe_alt}" style="max-width: 100%;">'
    return img_html
