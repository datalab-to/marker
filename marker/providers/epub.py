import base64
import os
import tempfile

from bs4 import BeautifulSoup

from marker.providers.pdf import PdfProvider
from marker.providers.utils import safe_weasyprint_url_fetcher, validate_zip_archive

css = """
@page {
    size: A4;
    margin: 2cm;
}

img {
    max-width: 100%;
    max-height: 25cm;
    object-fit: contain;
    margin: 12pt auto;
}

div, p {
    max-width: 100%;
    word-break: break-word;
    font-size: 10pt;
}

table {
    width: 100%;
    border-collapse: collapse;
    break-inside: auto;
    font-size: 10pt;
}

tr {
    break-inside: avoid;
    page-break-inside: avoid;
}

td {
    border: 0.75pt solid #000;
    padding: 6pt;
}
"""


class EpubProvider(PdfProvider):
    def __init__(self, filepath: str, config=None):
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        self.temp_pdf_path = temp_pdf.name
        temp_pdf.close()

        # Convert Epub to PDF
        try:
            validate_zip_archive(filepath)
            self.convert_epub_to_pdf(filepath)
            super().__init__(self.temp_pdf_path, config)
        except Exception as e:
            self._cleanup_temp_pdf()
            raise RuntimeError(f"Failed to convert {filepath} to PDF: {e}") from e

    def _cleanup_temp_pdf(self):
        if getattr(self, "temp_pdf_path", None) and os.path.exists(self.temp_pdf_path):
            try:
                os.remove(self.temp_pdf_path)
            except OSError:
                pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_temp_pdf()

    def __del__(self):
        self._cleanup_temp_pdf()

    def convert_epub_to_pdf(self, filepath):
        from weasyprint import CSS, HTML
        from ebooklib import epub
        import ebooklib

        ebook = epub.read_epub(filepath)

        styles = []
        html_content = ""
        img_tags = {}

        for item in ebook.get_items():
            if item.get_type() == ebooklib.ITEM_IMAGE:
                img_data = base64.b64encode(item.get_content()).decode("utf-8")
                img_tags[item.file_name] = f"data:{item.media_type};base64,{img_data}"
            elif item.get_type() == ebooklib.ITEM_STYLE:
                styles.append(item.get_content().decode("utf-8"))

        for item in ebook.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                html_content += item.get_content().decode("utf-8")

        soup = BeautifulSoup(html_content, "html.parser")
        for img in soup.find_all("img"):
            src = img.get("src")
            if src:
                normalized_src = src.replace("../", "")
                if normalized_src in img_tags:
                    img["src"] = img_tags[normalized_src]

        for image in soup.find_all("image"):
            src = image.get("xlink:href")
            if src:
                normalized_src = src.replace("../", "")
                if normalized_src in img_tags:
                    image["xlink:href"] = img_tags[normalized_src]

        html_content = str(soup)
        full_style = "".join([css])  # + styles)

        # we convert the epub to HTML
        HTML(
            string=html_content,
            base_url=filepath,
            url_fetcher=safe_weasyprint_url_fetcher,
        ).write_pdf(
            self.temp_pdf_path,
            stylesheets=[CSS(string=full_style), self.get_font_css()],
        )
