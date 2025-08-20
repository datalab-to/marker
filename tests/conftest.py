import tempfile
from typing import Dict, Type

from PIL import Image, ImageDraw

import datasets
import pytest

from marker.builders.document import DocumentBuilder
from marker.builders.layout import LayoutBuilder
from marker.builders.line import LineBuilder
from marker.builders.ocr import OcrBuilder
from marker.builders.structure import StructureBuilder
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.providers.registry import provider_from_filepath
from marker.renderers.chunk import ChunkRenderer
from marker.renderers.html import HTMLRenderer
from marker.schema import BlockTypes
from marker.schema.blocks import Block
from marker.renderers.markdown import MarkdownRenderer
from marker.renderers.json import JSONRenderer
from marker.schema.registry import register_block_class
from marker.util import classes_to_strings, strings_to_classes


@pytest.fixture(scope="session")
def model_dict():
    model_dict = create_model_dict()
    yield model_dict
    del model_dict


@pytest.fixture(scope="session")
def layout_model(model_dict):
    yield model_dict["layout_model"]


@pytest.fixture(scope="session")
def detection_model(model_dict):
    yield model_dict["detection_model"]


@pytest.fixture(scope="session")
def recognition_model(model_dict):
    yield model_dict["recognition_model"]


@pytest.fixture(scope="session")
def table_rec_model(model_dict):
    yield model_dict["table_rec_model"]


@pytest.fixture(scope="session")
def ocr_error_model(model_dict):
    yield model_dict["ocr_error_model"]


@pytest.fixture(scope="function")
def config(request):
    config_mark = request.node.get_closest_marker("config")
    config = config_mark.args[0] if config_mark else {}

    override_map: Dict[BlockTypes, Type[Block]] = config.get("override_map", {})
    for block_type, override_block_type in override_map.items():
        register_block_class(block_type, override_block_type)

    return config


@pytest.fixture(scope="session")
def pdf_dataset():
    return datasets.load_dataset("datalab-to/pdfs", split="train")


@pytest.fixture(scope="function")
def temp_doc(request, pdf_dataset):
    filename_mark = request.node.get_closest_marker("filename")
    filename = filename_mark.args[0] if filename_mark else "adversarial.pdf"

    idx = pdf_dataset["filename"].index(filename)
    suffix = filename.split(".")[-1]

    temp_pdf = tempfile.NamedTemporaryFile(suffix=f".{suffix}")
    temp_pdf.write(pdf_dataset["pdf"][idx])
    temp_pdf.flush()
    yield temp_pdf


@pytest.fixture(scope="function")
def doc_provider(request, config, temp_doc):
    provider_cls = provider_from_filepath(temp_doc.name)
    yield provider_cls(temp_doc.name, config)


@pytest.fixture(scope="function")
def pdf_document(
    request,
    config,
    doc_provider,
    layout_model,
    ocr_error_model,
    recognition_model,
    detection_model,
):
    layout_builder = LayoutBuilder(layout_model, config)
    line_builder = LineBuilder(detection_model, ocr_error_model, config)
    ocr_builder = OcrBuilder(recognition_model, config)
    builder = DocumentBuilder(config)
    structure_builder = StructureBuilder(config)
    document = builder(doc_provider, layout_builder, line_builder, ocr_builder)
    structure_builder(document)
    yield document


@pytest.fixture(scope="function")
def pdf_converter(request, config, model_dict, renderer, llm_service):
    if llm_service:
        llm_service = classes_to_strings([llm_service])[0]
    yield PdfConverter(
        artifact_dict=model_dict,
        processor_list=None,
        renderer=classes_to_strings([renderer])[0],
        config=config,
        llm_service=llm_service,
    )


@pytest.fixture(scope="function")
def renderer(request, config):
    if request.node.get_closest_marker("output_format"):
        output_format = request.node.get_closest_marker("output_format").args[0]
        if output_format == "markdown":
            return MarkdownRenderer
        elif output_format == "json":
            return JSONRenderer
        elif output_format == "html":
            return HTMLRenderer
        elif output_format == "chunks":
            return ChunkRenderer
        else:
            raise ValueError(f"Unknown output format: {output_format}")
    else:
        return MarkdownRenderer


@pytest.fixture(scope="function")
def llm_service(request, config):
    llm_service = config.get("llm_service")
    if not llm_service:
        yield None
    else:
        yield strings_to_classes([llm_service])[0]


@pytest.fixture(scope="function")
def temp_image():
    img = Image.new("RGB", (512, 512), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "Hello, World!", fill="black", font_size=24)
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        img.save(f.name)
        f.flush()
        yield f
