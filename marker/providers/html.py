import os
import tempfile

from marker.providers.pdf import PdfProvider
from marker.providers.utils import safe_weasyprint_url_fetcher


class HTMLProvider(PdfProvider):
    def __init__(self, filepath: str, config=None):
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        self.temp_pdf_path = temp_pdf.name
        temp_pdf.close()

        # Convert HTML to PDF
        try:
            self.convert_html_to_pdf(filepath)
            super().__init__(self.temp_pdf_path, config)
        except Exception as e:
            self._cleanup_temp_pdf()
            if isinstance(e, RuntimeError):
                raise
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

    def convert_html_to_pdf(self, filepath: str):
        from weasyprint import HTML

        font_css = self.get_font_css()
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            html_str = f.read()
        HTML(
            string=html_str,
            url_fetcher=safe_weasyprint_url_fetcher,
        ).write_pdf(self.temp_pdf_path, stylesheets=[font_css])
