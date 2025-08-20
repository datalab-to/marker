import json
import os

from streamlit_ace import st_ace
from pydantic import BaseModel

from marker.converters.extraction import ExtractionConverter
from marker.scripts.common import (
    parse_args,
    load_models,
    get_page_image,
    page_count,
    get_root_class,
)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["IN_STREAMLIT"] = "true"

from streamlit.runtime.uploaded_file_manager import UploadedFile

import tempfile
from typing import Any, Dict

import streamlit as st

from marker.config.parser import ConfigParser


def extract_data(
    fname: str, config: dict, schema: str, markdown: str | None = None
) -> (str, Dict[str, Any], dict):
    config["pdftext_workers"] = 1
    config["page_schema"] = schema
    config["existing_markdown"] = markdown
    config_parser = ConfigParser(config)
    config_dict = config_parser.generate_config_dict()

    converter_cls = ExtractionConverter
    converter = converter_cls(
        config=config_dict,
        artifact_dict=model_dict,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )
    return converter(fname)


st.set_page_config(layout="wide")
col1, col2 = st.columns([0.5, 0.5])

model_dict = load_models()
cli_options = parse_args()

st.markdown("""
# Marker Extraction Demo

This app will let you use marker to do structured extraction.

Warning: This can execute untrusted code entered into the schema panel.
""")

in_file: UploadedFile = st.sidebar.file_uploader(
    "PDF, document, or image file:",
    type=["pdf", "png", "jpg", "jpeg", "gif", "pptx", "docx", "xlsx", "html", "epub"],
)

# Initialize session state variables
if "rendered_pydantic_schema" not in st.session_state:
    st.session_state.rendered_pydantic_schema = ""

if "markdown" not in st.session_state:
    st.session_state.markdown = ""

if "current_file_id" not in st.session_state:
    st.session_state.current_file_id = None

# Detect file changes and clear markdown when new file is uploaded
if in_file is not None:
    # Create a unique identifier for the current file
    current_file_id = f"{in_file.name}_{in_file.size}_{hash(in_file.getvalue())}"

    # Check if this is a new file
    if st.session_state.current_file_id != current_file_id:
        st.session_state.current_file_id = current_file_id
        st.session_state.markdown = ""  # Clear markdown for new file
else:
    # No file uploaded, clear the current file ID
    if st.session_state.current_file_id is not None:
        st.session_state.current_file_id = None
        st.session_state.markdown = ""  # Clear markdown when no file
        st.session_state.rendered_pydantic_schema = ""

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
with col2:
    tab1, tab2 = st.tabs(["JSON Schema", "Pydantic Schema"])

    # Initialize schema variable
    schema = None

    with tab1:
        st.write("Enter an existing JSON schema here:")
        default_json_value = (
            st.session_state.rendered_pydantic_schema
            if st.session_state.rendered_pydantic_schema
            else ""
        )
        json_schema_input = st.text_area(
            "JSON Schema",
            value=default_json_value,
            height=300,
            placeholder='{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}',
            key="json_schema_input",
            label_visibility="collapsed",
        )

        # Set schema if JSON input is provided
        if json_schema_input and json_schema_input.strip():
            try:
                # Validate JSON
                json.loads(json_schema_input)
                schema = json_schema_input.strip()
                st.success("✅ Valid JSON schema detected")
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON: {e}")
                schema = None

    with tab2:
        st.write("Enter pydantic schema here:")
        pydantic_schema_input = st_ace(
            value="""from pydantic import BaseModel

class Schema(BaseModel):
    # Add your fields here
    # Example:
    name: str
    age: int
    # email: str
    pass""",
            language="python",
            height=300,
            key="pydantic_editor",
        )

        render_schema = st.button("🔄 Render Pydantic schema to JSON")

        if render_schema and pydantic_schema_input:
            try:
                pydantic_root: BaseModel = get_root_class(pydantic_schema_input)
                json_schema = pydantic_root.model_json_schema()
                schema = json.dumps(json_schema, indent=2)
                st.success("✅ Schema rendered successfully!")
                st.json(json_schema)
                st.session_state.rendered_pydantic_schema = schema
            except Exception as e:
                st.error(f"❌ Could not parse your schema: {e}")
                schema = None
        elif (
            pydantic_schema_input
            and pydantic_schema_input.strip()
            and not render_schema
        ):
            # If there's Pydantic code but not rendered yet, show a message
            if (
                "class Schema(BaseModel):" in pydantic_schema_input
                and "pass" not in pydantic_schema_input
            ):
                st.info(
                    "💡 Click 'Render Pydantic schema to JSON' to convert your Pydantic model to JSON schema"
                )

# Move the run logic outside of col2
run_marker = st.sidebar.button("Run Extraction")

use_llm = st.sidebar.checkbox(
    "Use LLM", help="Use LLM for higher quality text", value=False
)
force_ocr = st.sidebar.checkbox("Force OCR", help="Force OCR on all pages", value=False)
strip_existing_ocr = st.sidebar.checkbox(
    "Strip existing OCR",
    help="Strip existing OCR text from the PDF and re-OCR.",
    value=False,
)

# Check if schema is provided before running
if run_marker:
    if not schema:
        st.error(
            "❌ Please provide a schema in either the JSON Schema or Pydantic Schema tab before running extraction."
        )
        st.stop()

    # Run Marker
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_pdf = os.path.join(tmp_dir, "temp.pdf")
        with open(temp_pdf, "wb") as f:
            f.write(in_file.getvalue())

        cli_options.update(
            {
                "force_ocr": force_ocr,
                "use_llm": use_llm,
                "strip_existing_ocr": strip_existing_ocr,
            }
        )

        try:
            rendered = extract_data(
                temp_pdf, cli_options, schema, st.session_state.markdown
            )

            with col2:
                st.write("## Output JSON")
                st.json(rendered.model_dump(exclude=["original_markdown"]))
                st.session_state.markdown = rendered.original_markdown

        except Exception as e:
            st.error(f"❌ Extraction failed: {e}")

else:
    # Show instruction when not running
    if not schema:
        st.info("📝 Please provide a schema and click 'Run Extraction' to begin.")
