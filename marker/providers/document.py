import base64
import os
import re
import tempfile
from io import BytesIO

from PIL import Image

from marker.providers.pdf import PdfProvider

css = '''
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
    overflow-wrap: break-word;
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
'''


class DocumentProvider(PdfProvider):
    def __init__(self, filepath: str, config=None):
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=f".pdf")
        self.temp_pdf_path = temp_pdf.name
        temp_pdf.close()

        # Convert DOCX to PDF
        try:
            self.convert_docx_to_pdf(filepath)
        except Exception as e:
            raise RuntimeError(f"Failed to convert {filepath} to PDF: {e}")

        # Initialize the PDF provider with the temp pdf path
        super().__init__(self.temp_pdf_path, config)

    def __del__(self):
        # if os.path.exists(self.temp_pdf_path):
        #     os.remove(self.temp_pdf_path)
        pass

    def _convert_image_to_base64(self, image):
        """Convert mammoth image to base64 data URI"""
        try:
            with image.open() as image_bytes:
                import base64
                return "data:" + image.content_type + ";base64," + base64.b64encode(image_bytes.read()).decode()
        except Exception as e:
            print(f"Failed to convert image: {e}")
            return ""

    def convert_docx_to_pdf(self, filepath: str):
        from weasyprint import CSS, HTML
        import mammoth
        import re

        with open(filepath, "rb") as docx_file:
            # Configure style mapping to preserve heading levels
            style_map = """
            p[style-name='Heading 1'] => h1:fresh
            p[style-name='Heading 2'] => h2:fresh  
            p[style-name='Heading 3'] => h3:fresh
            p[style-name='Heading 4'] => h4:fresh
            p[style-name='Heading 5'] => h5:fresh
            p[style-name='Heading 6'] => h6:fresh
            p[style-name='标题 1'] => h1:fresh
            p[style-name='标题 2'] => h2:fresh
            p[style-name='标题 3'] => h3:fresh
            p[style-name='标题 4'] => h4:fresh
            p[style-name='标题 5'] => h5:fresh
            p[style-name='标题 6'] => h6:fresh
            p:empty => p:fresh
            """
            
            # Configure mammoth options for better conversion
            convert_options = {
                "convert_image": mammoth.images.img_element(lambda image: {
                    "src": self._convert_image_to_base64(image)
                }),
                "ignore_empty_paragraphs": False,
                "style_map": style_map
            }
            
            # we convert the docx to HTML with better options
            result = mammoth.convert_to_html(docx_file, **convert_options)
            html = result.value
            
            # Print conversion messages if any
            if result.messages:
                print(f"Mammoth conversion messages: {result.messages}", flush=True)
            
            # Post-process HTML to normalize headings and preserve empty lines
            html = self._normalize_html(html)
            
            # Debug: Print HTML content length and preview
            print(f"Generated HTML length: {len(html)} characters", flush=True)
            if html:
                preview = html[:500].replace('\n', ' ')
                print(f"HTML preview: {preview}...", flush=True)
            else:
                print("WARNING: Generated HTML is empty!", flush=True)

            # We convert the HTML into a PDF
            processed_html = self._preprocess_base64_images(html)
            if not processed_html.strip():
                print("ERROR: Processed HTML is empty, adding fallback content", flush=True)
                processed_html = "<html><body><p>Document conversion failed - no content extracted</p></body></html>"
            
            # Wrap in proper HTML structure for better PDF conversion
            if not processed_html.startswith('<html'):
                processed_html = f"<html><head><meta charset='utf-8'></head><body>{processed_html}</body></html>"
            
            print(f"Final HTML for PDF conversion (length: {len(processed_html)}):", flush=True)
            print(f"HTML content: {processed_html[:1000]}...", flush=True)
            
            # Use improved CSS with better heading styling and spacing
            simple_css = '''
            @page { 
                size: A4; 
                margin: 2cm; 
            }
            body { 
                font-family: serif; 
                font-size: 12pt; 
                line-height: 1.6;
            }
            h1 { 
                font-size: 20pt;
                font-weight: bold;
                margin: 1.5em 0 1em 0; 
                line-height: 1.3;
            }
            h2 { 
                font-size: 16pt;
                font-weight: bold;
                margin: 1.2em 0 0.8em 0; 
                line-height: 1.4;
                padding: 0.2em 0;
            }
            h3 { 
                font-size: 14pt;
                font-weight: bold;
                margin: 1em 0 0.6em 0; 
                line-height: 1.3;
            }
            h4, h5, h6 { 
                font-size: 13pt;
                font-weight: bold;
                margin: 0.8em 0 0.4em 0; 
                line-height: 1.3;
            }
            p { 
                margin: 0.5em 0; 
                font-family: serif;
                line-height: 1.6;
            }
            p:empty {
                margin: 0.5em 0;
                min-height: 1em;
            }
            strong {
                font-weight: bold;
            }
            '''
            
            try:
                print("Starting HTML to PDF conversion...", flush=True)
                html_doc = HTML(string=processed_html)
                print("HTML document created successfully", flush=True)
                
                html_doc.write_pdf(
                    self.temp_pdf_path,
                    stylesheets=[CSS(string=simple_css)]
                )
                print(f"PDF conversion completed: {self.temp_pdf_path}", flush=True)
                
                # Check if PDF file was created and has content
                import os
                if os.path.exists(self.temp_pdf_path):
                    pdf_size = os.path.getsize(self.temp_pdf_path)
                    print(f"Generated PDF size: {pdf_size} bytes", flush=True)
                else:
                    print("ERROR: PDF file was not created!", flush=True)
                    
            except Exception as e:
                print(f"ERROR during HTML to PDF conversion: {e}", flush=True)
                import traceback
                traceback.print_exc()
                
                # Create a minimal fallback PDF
                fallback_html = f"<html><body><h1>Document Conversion</h1><p>Original content length: {len(html)} characters</p><p>Error: {str(e)}</p></body></html>"
                try:
                    HTML(string=fallback_html).write_pdf(self.temp_pdf_path, stylesheets=[CSS(string=simple_css)])
                    print("Created fallback PDF", flush=True)
                except Exception as fallback_error:
                    print(f"Even fallback PDF creation failed: {fallback_error}", flush=True)

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
                        return f'data:{match.group(1)};base64,{new_base64}'

            except Exception as e:
                print(e)
                return ""  # we ditch broken images as that breaks the PDF creation down the line

        return re.sub(pattern, convert_image, html_content)

    def _normalize_html(self, html):
        """Normalize HTML to ensure consistent heading levels and preserve empty lines"""
        import re
        
        # Convert all h2 tags to have consistent styling (force them to be treated equally)
        # This helps prevent marker from incorrectly assigning different levels
        html = re.sub(r'<h2([^>]*)>', r'<h2 class="doc-heading">', html)
        
        # Preserve empty paragraphs by adding non-breaking space
        html = re.sub(r'<p></p>', '<p>&nbsp;</p>', html)
        html = re.sub(r'<p>\s*</p>', '<p>&nbsp;</p>', html)
        
        # Handle cases where mammoth might create empty paragraphs with just whitespace
        html = re.sub(r'<p>(\s*)</p>', r'<p>&nbsp;</p>', html)
        
        # Add proper spacing after headings
        html = re.sub(r'</h([1-6])>', r'</h\1>', html)
        
        # Ensure paragraphs have at least some content for proper rendering
        html = re.sub(r'<p>\s*<\/p>', '<p>&nbsp;</p>', html)
        
        # Add extra spacing for better readability
        html = re.sub(r'</h2>', r'</h2>\n<p>&nbsp;</p>', html)
        
        print(f"Normalized HTML preview: {html[:800]}...", flush=True)
        
        return html
