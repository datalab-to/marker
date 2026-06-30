import os
import tempfile

from marker.providers.pdf import PdfProvider


class HTMLProvider(PdfProvider):
    def __init__(self, filepath: str, config=None):
        self.source_filepath = filepath
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        self.temp_pdf_path = temp_pdf.name
        temp_pdf.close()

        # Convert HTML to PDF
        try:
            self.convert_html_to_pdf(filepath)
        except Exception as e:
            raise RuntimeError(f"Failed to convert {filepath} to PDF: {e}")

        # Initialize the PDF provider with the temp pdf path
        super().__init__(self.temp_pdf_path, config)

    def __del__(self):
        if os.path.exists(self.temp_pdf_path):
            os.remove(self.temp_pdf_path)

    def convert_html_to_pdf(self, filepath: str):
        from weasyprint import HTML

        font_css = self.get_font_css()
        HTML(filename=filepath, encoding="utf-8").write_pdf(
            self.temp_pdf_path, stylesheets=[font_css]
        )
