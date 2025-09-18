import ast
import base64
import io
import re
import sys
from typing import Optional

from PIL import Image
import click
import pypdfium2
import streamlit as st
from pydantic import BaseModel
from streamlit.runtime.uploaded_file_manager import UploadedFile

from marker.config.parser import ConfigParser
from marker.config.printer import CustomClickPrinter
from marker.models import create_model_dict
from marker.settings import settings


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
    return create_model_dict(config={})


def open_pdf(pdf_file):
    stream = io.BytesIO(pdf_file.getvalue())
    return pypdfium2.PdfDocument(stream)


def img_to_html(img, img_alt):
    img_bytes = io.BytesIO()
    img.save(img_bytes, format=settings.OUTPUT_IMAGE_FORMAT)
    img_bytes = img_bytes.getvalue()
    encoded = base64.b64encode(img_bytes).decode()
    img_html = f'<img src="data:image/{settings.OUTPUT_IMAGE_FORMAT.lower()};base64,{encoded}" alt="{img_alt}" style="max-width: 100%;">'
    return img_html


@st.cache_data()
def get_page_image(pdf_file, page_num, dpi=96):
    if "pdf" in pdf_file.type:
        doc = open_pdf(pdf_file)
        page = doc[page_num]
        png_image = (
            page.render(
                scale=dpi / 72,
            )
            .to_pil()
            .convert("RGB")
        )
    else:
        png_image = Image.open(pdf_file).convert("RGB")
    return png_image


@st.cache_data()
def page_count(pdf_file: UploadedFile):
    if "pdf" in pdf_file.type:
        doc = open_pdf(pdf_file)
        return len(doc) - 1
    else:
        return 1


def pillow_image_to_base64_string(img: Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def extract_root_pydantic_class(schema_code: str) -> Optional[str]:
    try:
        # Parse the code into an AST
        tree = ast.parse(schema_code)

        # Find all class definitions that inherit from BaseModel
        class_names = set()
        class_info = {}  # Store information about each class

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if this class inherits from BaseModel
                is_pydantic = False
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == "BaseModel":
                        is_pydantic = True
                        break

                if is_pydantic:
                    class_names.add(node.name)
                    class_info[node.name] = {
                        "references": set(),  # Classes this class references
                        "fields": [],  # Field names in this class
                    }

                    # Extract field information
                    for item in node.body:
                        if isinstance(item, ast.AnnAssign) and isinstance(
                            item.target, ast.Name
                        ):
                            field_name = item.target.id
                            class_info[node.name]["fields"].append(field_name)

                            # Check if this field references another class
                            annotation_str = ast.unparse(item.annotation)

                            # Look for List[ClassName], Optional[ClassName], Dict[Any, ClassName], etc.
                            for other_class in class_names:
                                pattern = rf"(?:List|Dict|Set|Tuple|Optional|Union)?\[.*{other_class}.*\]|{other_class}"
                                if re.search(pattern, annotation_str):
                                    class_info[node.name]["references"].add(other_class)

        if len(class_names) == 1:
            return list(class_names)[0]

        referenced_classes = set()
        for class_name, info in class_info.items():
            referenced_classes.update(info["references"])

        # Find classes that reference others but aren't referenced themselves (potential roots)
        root_candidates = set()
        for class_name, info in class_info.items():
            if info["references"] and class_name not in referenced_classes:
                root_candidates.add(class_name)

        # If we found exactly one root candidate, return it
        if len(root_candidates) == 1:
            return list(root_candidates)[0]

        return None
    except Exception as e:
        print(f"Error parsing schema: {e}")
        return None


def get_root_class(schema_code: str) -> Optional[BaseModel]:
    root_class_name = extract_root_pydantic_class(schema_code)

    if not root_class_name:
        return None

    if "from pydantic" not in schema_code:
        schema_code = "from pydantic import BaseModel\n" + schema_code
    if "from typing" not in schema_code:
        schema_code = (
            "from typing import List, Dict, Optional, Set, Tuple, Union, Any\n\n"
            + schema_code
        )

    # Execute the code in a new namespace
    namespace = {}
    exec(schema_code, namespace)

    # Return the root class object
    return namespace.get(root_class_name)
