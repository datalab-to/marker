import json
import os
from typing import Dict

import click

from marker.config.crawler import crawler
from marker.converters.pdf import PdfConverter
from marker.logger import get_logger
from marker.renderers.chunk import ChunkRenderer
from marker.renderers.html import HTMLRenderer
from marker.renderers.json import JSONRenderer
from marker.renderers.markdown import MarkdownRenderer
from marker.settings import settings
from marker.util import classes_to_strings, parse_range_str, strings_to_classes

logger = get_logger()


class ConfigParser:
    def __init__(self, cli_options: dict):
        self.cli_options = cli_options

    @staticmethod
    def common_options(fn):
        fn = click.option(
            "--output_dir",
            type=click.Path(exists=False),
            required=False,
            default=settings.OUTPUT_DIR,
            help="Directory to save output.",
        )(fn)
        fn = click.option("--debug", "-d", is_flag=True, help="Enable debug mode.")(fn)
        fn = click.option(
            "--output_format",
            type=click.Choice(["markdown", "json", "html", "chunks"]),
            default="markdown",
            help="Format to output results in.",
        )(fn)
        fn = click.option(
            "--processors",
            type=str,
            default=None,
            help="Comma separated list of processors to use.  Must use full module path.",
        )(fn)
        fn = click.option(
            "--config_json",
            type=str,
            default=None,
            help="Path to JSON file with additional configuration.",
        )(fn)
        fn = click.option(
            "--disable_multiprocessing",
            is_flag=True,
            default=False,
            help="Disable multiprocessing.",
        )(fn)
        fn = click.option(
            "--disable_image_extraction",
            is_flag=True,
            default=False,
            help="Disable image extraction.",
        )(fn)

        # these are options that need a list transformation, i.e splitting/parsing a string
        fn = click.option(
            "--page_range",
            type=str,
            default=None,
            help="Page range to convert, specify comma separated page numbers or ranges.  Example: 0,5-10,20",
        )(fn)

        # we put common options here
        fn = click.option(
            "--converter_cls",
            type=str,
            default=None,
            help="Converter class to use.  Defaults to PDF converter.",
        )(fn)
        fn = click.option(
            "--llm_service",
            type=str,
            default=None,
            help="LLM service to use - should be full import path, like marker.services.gemini.GoogleGeminiService",
        )(fn)
        return fn

    def generate_config_dict(self) -> Dict[str, any]:
        config = {}
        output_dir = self.cli_options.get("output_dir", settings.OUTPUT_DIR)
        for k, v in self.cli_options.items():
            if not v:
                continue

            match k:
                case "debug":
                    config["debug_pdf_images"] = True
                    config["debug_layout_images"] = True
                    config["debug_json"] = True
                    config["debug_data_folder"] = output_dir
                case "page_range":
                    config["page_range"] = parse_range_str(v)
                case "config_json":
                    with open(v, "r", encoding="utf-8") as f:
                        config.update(json.load(f))
                case "disable_multiprocessing":
                    config["pdftext_workers"] = 1
                case "disable_image_extraction":
                    config["extract_images"] = False
                case _:
                    if k in crawler.attr_set:
                        config[k] = v

        # Backward compatibility for google_api_key
        if settings.GOOGLE_API_KEY:
            config["gemini_api_key"] = settings.GOOGLE_API_KEY

        return config

    def get_llm_service(self):
        # Only return an LLM service when use_llm is enabled
        if not self.cli_options.get("use_llm", False):
            return None

        service_cls = self.cli_options.get("llm_service", None)
        if service_cls is None:
            service_cls = "marker.services.gemini.GoogleGeminiService"
        return service_cls

    def get_renderer(self):
        match self.cli_options["output_format"]:
            case "json":
                r = JSONRenderer
            case "markdown":
                r = MarkdownRenderer
            case "html":
                r = HTMLRenderer
            case "chunks":
                r = ChunkRenderer
            case _:
                raise ValueError("Invalid output format")
        return classes_to_strings([r])[0]

    def get_processors(self):
        processors = self.cli_options.get("processors", None)
        if processors is not None:
            processors = processors.split(",")
            for p in processors:
                try:
                    strings_to_classes([p])
                except Exception as e:
                    logger.error(f"Error loading processor: {p} with error: {e}")
                    raise

        return processors

    def get_converter_cls(self):
        converter_cls = self.cli_options.get("converter_cls", None)
        if converter_cls is not None:
            try:
                return strings_to_classes([converter_cls])[0]
            except Exception as e:
                logger.error(
                    f"Error loading converter: {converter_cls} with error: {e}"
                )
                raise

        return PdfConverter

    def get_output_folder(self, filepath: str):
        output_dir = self.cli_options.get("output_dir", settings.OUTPUT_DIR)
        fname_base = os.path.splitext(os.path.basename(filepath))[0]
        output_dir = os.path.join(output_dir, fname_base)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def get_base_filename(self, filepath: str):
        basename = os.path.basename(filepath)
        return os.path.splitext(basename)[0]
