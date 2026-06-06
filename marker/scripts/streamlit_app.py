import os

from marker.scripts.common import (
    load_models,
    parse_args,
    img_to_html,
    get_page_image,
    page_count,
)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["IN_STREAMLIT"] = "true"

from marker.settings import settings
from streamlit.runtime.uploaded_file_manager import UploadedFile

import json
import re
import tempfile
from typing import Any, Dict

import streamlit as st
from PIL import Image

from marker.converters.pdf import PdfConverter
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered


def convert_pdf(fname: str, config_parser: ConfigParser) -> (str, Dict[str, Any], dict):
    config_dict = config_parser.generate_config_dict()
    config_dict["pdftext_workers"] = 1
    converter_cls = PdfConverter
    converter = converter_cls(
        config=config_dict,
        artifact_dict=model_dict,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )
    return converter(fname)


def markdown_insert_images(markdown, images):
    image_tags = re.findall(
        r'(!\[(?P<image_title>[^\]]*)\]\((?P<image_path>[^\)"\s]+)\s*([^\)]*)\))',
        markdown,
    )

    for image in image_tags:
        image_markdown = image[0]
        image_alt = image[1]
        image_path = image[2]
        if image_path in images:
            markdown = markdown.replace(
                image_markdown, img_to_html(images[image_path], image_alt)
            )
    return markdown


st.set_page_config(layout="wide", page_title="Marker — Document Converter", page_icon="📄")

# --- Full-page Loading Screen ---
# Hide ALL Streamlit UI elements while loading, and show an animated overlay
_LOADING_SCREEN = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Hide Streamlit chrome while loading */
.marker-is-loading header[data-testid="stHeader"],
.marker-is-loading section[data-testid="stSidebar"],
.marker-is-loading .stMainBlockContainer {
    visibility: hidden !important;
}

/* Full-page overlay */
.marker-loading-overlay {
    position: fixed;
    inset: 0;
    z-index: 1000000;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    font-family: 'Inter', sans-serif;
    transition: opacity 0.6s ease-out;
}
.marker-loading-overlay.fade-out {
    opacity: 0;
    pointer-events: none;
}

/* Animated logo */
.marker-load-icon {
    font-size: 4.5rem;
    margin-bottom: 1.5rem;
    animation: marker-pulse 2s ease-in-out infinite;
}
@keyframes marker-pulse {
    0%, 100% { transform: scale(1); filter: drop-shadow(0 0 10px rgba(139,92,246,0.4)); }
    50%      { transform: scale(1.12); filter: drop-shadow(0 0 28px rgba(139,92,246,0.85)); }
}

/* Title with gradient text */
.marker-load-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #818cf8, #c084fc);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}
.marker-load-subtitle {
    font-size: 1rem;
    color: #94a3b8;
    font-weight: 300;
    letter-spacing: 0.06em;
    margin-bottom: 2.5rem;
}

/* Indeterminate progress bar */
.marker-load-track {
    width: 300px;
    height: 4px;
    background: rgba(255,255,255,0.07);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 1.4rem;
}
.marker-load-bar {
    width: 38%;
    height: 100%;
    background: linear-gradient(90deg, #a78bfa, #818cf8);
    border-radius: 4px;
    animation: marker-slide 1.6s ease-in-out infinite;
}
@keyframes marker-slide {
    0%   { transform: translateX(-110%); }
    100% { transform: translateX(330%); }
}

/* Status text with animated dots */
.marker-load-status {
    font-size: 0.88rem;
    color: #64748b;
    font-weight: 400;
    letter-spacing: 0.04em;
}
.marker-load-dots::after {
    content: '';
    animation: marker-dots 1.5s steps(4, end) infinite;
}
@keyframes marker-dots {
    0%  { content: ''; }
    25% { content: '.'; }
    50% { content: '..'; }
    75% { content: '...'; }
}
</style>

<script>
// Tag <html> so CSS above can hide Streamlit chrome
document.documentElement.classList.add('marker-is-loading');
</script>

<div class="marker-loading-overlay" id="markerLoadingScreen">
    <div class="marker-load-icon">📄</div>
    <div class="marker-load-title">Marker</div>
    <div class="marker-load-subtitle">Document → Markdown Converter</div>
    <div class="marker-load-track">
        <div class="marker-load-bar"></div>
    </div>
    <div class="marker-load-status">Loading AI models<span class="marker-load-dots"></span></div>
</div>
"""

_LOADING_DISMISS = """
<script>
(function() {
    // Fade out the overlay
    var el = document.getElementById('markerLoadingScreen');
    if (el) {
        el.classList.add('fade-out');
        setTimeout(function(){ el.remove(); }, 650);
    }
    // Reveal Streamlit UI
    document.documentElement.classList.remove('marker-is-loading');
})();
</script>
"""

loading_placeholder = st.empty()
loading_placeholder.html(_LOADING_SCREEN)

# Load models (this is the slow part — cached after first run)
model_dict = load_models()
cli_options = parse_args()

# Dismiss loading screen with fade-out, then clear placeholder
loading_placeholder.html(_LOADING_DISMISS)
import time as _time; _time.sleep(0.7)
loading_placeholder.empty()

# --- Main App UI ---
col1, col2 = st.columns([0.5, 0.5])

st.markdown("""
# 📄 Marker Demo

This app will let you try marker, a PDF or image → Markdown, HTML, JSON converter. It works with any language, and extracts images, tables, equations, etc.

Find the project [here](https://github.com/VikParuchuri/marker).
""")

in_file: UploadedFile = st.sidebar.file_uploader(
    "PDF, document, or image file:",
    type=["pdf", "png", "jpg", "jpeg", "gif", "pptx", "docx", "xlsx", "html", "epub"],
)

if in_file is None:
    st.stop()

filetype = in_file.type

with col1:
    page_count = page_count(in_file)
    page_number = st.number_input(
        f"Page number out of {page_count}:", min_value=0, value=0, max_value=page_count
    )
    pil_image = get_page_image(in_file, page_number)
    st.image(pil_image, use_container_width=True)

page_range = st.sidebar.text_input(
    "Page range to parse, comma separated like 0,5-10,20",
    value=f"{page_number}-{page_number}",
)
output_format = st.sidebar.selectbox(
    "Output format", ["markdown", "json", "html", "chunks"], index=0
)
run_marker = st.sidebar.button("Run Marker")

use_llm = st.sidebar.checkbox(
    "Use LLM", help="Use LLM for higher quality processing", value=False
)
force_ocr = st.sidebar.checkbox("Force OCR", help="Force OCR on all pages", value=False)
strip_existing_ocr = st.sidebar.checkbox(
    "Strip existing OCR",
    help="Strip existing OCR text from the PDF and re-OCR.",
    value=False,
)
debug = st.sidebar.checkbox("Debug", help="Show debug information", value=False)
disable_ocr_math = st.sidebar.checkbox(
    "Disable math",
    help="Disable math in OCR output - no inline math",
    value=False,
)

if not run_marker:
    st.stop()

# Run Marker
with st.status("Running Marker...", expanded=True) as status:
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_pdf = os.path.join(tmp_dir, "temp.pdf")
        st.write("⏳ Preparing document...")
        with open(temp_pdf, "wb") as f:
            f.write(in_file.getvalue())

        cli_options.update(
            {
                "output_format": output_format,
                "page_range": page_range,
                "force_ocr": force_ocr,
                "debug": debug,
                "output_dir": settings.DEBUG_DATA_FOLDER if debug else None,
                "use_llm": use_llm,
                "strip_existing_ocr": strip_existing_ocr,
                "disable_ocr_math": disable_ocr_math,
            }
        )
        config_parser = ConfigParser(cli_options)

        st.write("🔄 Running Marker conversion... This may take a moment.")
        rendered = convert_pdf(temp_pdf, config_parser)
        page_range = config_parser.generate_config_dict()["page_range"]
        first_page = page_range[0] if page_range else 0

        st.write("✅ Processing output...")
    status.update(label="Conversion complete!", state="complete", expanded=False)

text, ext, images = text_from_rendered(rendered)
with col2:
    if output_format == "markdown":
        text = markdown_insert_images(text, images)
        st.markdown(text, unsafe_allow_html=True)
    elif output_format == "json":
        st.json(text)
    elif output_format == "html":
        st.html(text)
    elif output_format == "chunks":
        st.json(text)

# Download button
file_base = os.path.splitext(in_file.name)[0]
ext_map = {"markdown": ".md", "json": ".json", "html": ".html", "chunks": ".json"}
mime_map = {
    "markdown": "text/markdown",
    "json": "application/json",
    "html": "text/html",
    "chunks": "application/json",
}
download_ext = ext_map[output_format]
download_mime = mime_map[output_format]
download_data = json.dumps(text, indent=2) if output_format in ("json", "chunks") else text

with col2:
    st.download_button(
        label=f"⬇️ Download {output_format.title()} file",
        data=download_data,
        file_name=f"{file_base}{download_ext}",
        mime=download_mime,
    )

if debug:
    with col1:
        debug_data_path = rendered.metadata.get("debug_data_path")
        if debug_data_path:
            pdf_image_path = os.path.join(debug_data_path, f"pdf_page_{first_page}.png")
            img = Image.open(pdf_image_path)
            st.image(img, caption="PDF debug image", use_container_width=True)
            layout_image_path = os.path.join(
                debug_data_path, f"layout_page_{first_page}.png"
            )
            img = Image.open(layout_image_path)
            st.image(img, caption="Layout debug image", use_container_width=True)
        st.write("Raw output:")
        st.code(text, language=output_format)
