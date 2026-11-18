import os
from urllib.parse import unquote, urlparse
import zipfile

from marker.settings import settings


def alphanum_ratio(text: str) -> float:
    text = text.replace(" ", "").replace("\n", "")
    if not text:
        return 1.0
    alphanumeric_count = sum(1 for c in text if c.isalnum())
    return alphanumeric_count / len(text)


def _is_allowed_weasyprint_url(url: str) -> bool:
    """Only data: URIs and the bundled font file may be fetched during rendering."""
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme == "data":
        return True
    if scheme == "file":
        target = os.path.realpath(unquote(parsed.path))
        font_real = os.path.realpath(settings.FONT_PATH)
        return target == font_real and os.path.isfile(font_real)
    return False


def safe_weasyprint_url_fetcher():
    """Build a WeasyPrint URL fetcher that blocks external network requests and local file inclusion.

    WeasyPrint >= 70 requires fetchers to be ``URLFetcher`` instances, so this returns a
    new instance per call. WeasyPrint is imported lazily since it is an optional dependency.
    """
    from weasyprint.urls import URLFetcher

    class SafeURLFetcher(URLFetcher):
        def fetch(self, url, headers=None):
            if not _is_allowed_weasyprint_url(url):
                raise ValueError(
                    f"Blocked external or local URL in document conversion: {url}"
                )
            return super().fetch(url, headers)

    return SafeURLFetcher()


def validate_zip_archive(
    filepath: str,
    max_uncompressed_bytes: int = 500 * 1024 * 1024,
    max_entries: int = 10000,
) -> None:
    """Validate a ZIP-based container (DOCX, XLSX, PPTX, EPUB) against zip bombs."""
    if not zipfile.is_zipfile(filepath):
        return
    with zipfile.ZipFile(filepath, "r") as zf:
        infos = zf.infolist()
        if len(infos) > max_entries:
            raise ValueError(
                f"Archive contains too many entries ({len(infos)} > {max_entries})"
            )
        total_size = sum(info.file_size for info in infos)
        if total_size > max_uncompressed_bytes:
            raise ValueError(
                f"Archive uncompressed size ({total_size} bytes) exceeds limit ({max_uncompressed_bytes} bytes)"
            )
