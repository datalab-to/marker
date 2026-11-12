import base64
import os
import re
import tempfile
from io import BytesIO

from PIL import Image
from marker.logger import get_logger

from marker.providers.pdf import PdfProvider
from marker.providers.utils import safe_weasyprint_url_fetcher, validate_zip_archive

logger = get_logger()

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


class DocumentProvider(PdfProvider):
    def __init__(self, filepath: str, config=None):
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        self.temp_pdf_path = temp_pdf.name
        temp_pdf.close()

        # Convert DOCX to PDF
        try:
            validate_zip_archive(filepath)
            self.convert_docx_to_pdf(filepath)
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

    def convert_docx_to_pdf(self, filepath: str):
        from weasyprint import CSS, HTML
        import mammoth

        with open(filepath, "rb") as docx_file:
            # we convert the docx to HTML
            result = mammoth.convert_to_html(docx_file)
            html = result.value

            # We convert the HTML into a PDF
            HTML(
                string=self._preprocess_base64_images(html),
                url_fetcher=safe_weasyprint_url_fetcher,
            ).write_pdf(
                self.temp_pdf_path, stylesheets=[CSS(string=css), self.get_font_css()]
            )

    @staticmethod
    def _preprocess_base64_images(html_content):
        pattern = r'data:([^;]+);base64,([^"\'>\s]+)'

        def convert_image(match):
            try:
                img_data = base64.b64decode(match.group(2))

                with BytesIO(img_data) as bio:
                    with Image.open(bio) as img:
                        output = BytesIO()
                        img.save(output, format=img.format)
                        new_base64 = base64.b64encode(output.getvalue()).decode()
                        return f"data:{match.group(1)};base64,{new_base64}"

            except Exception as e:
                logger.error(f"Failed to process image: {e}")
                return ""  # we ditch broken images as that breaks the PDF creation down the line

        return re.sub(pattern, convert_image, html_content)
