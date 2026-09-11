import sys
from contextlib import suppress
import click
import pytest

from marker.config.printer import CustomClickPrinter
from marker.config.crawler import crawler
from marker.config.parser import ConfigParser

# CPU-only: config/CLI parsing, no models or inference server.
pytestmark = pytest.mark.cpu


def capture_kwargs(argv):
    command = click.command(cls=CustomClickPrinter)
    captured_kwargs = {}

    def parse_args(**kwargs):
        captured_kwargs.update(kwargs)
        return kwargs

    original_argv = sys.argv
    sys.argv = argv
    try:
        with suppress(SystemExit):
            command(ConfigParser.common_options(parse_args))()
    finally:
        sys.argv = original_argv

    return captured_kwargs


def test_config_parser():
    sys.argv = [
        "test",
        "--disable_multiprocessing",
        "--output_dir",
        "output_dir",
        "--height_tolerance",
        "0.5",
    ]
    kwargs = capture_kwargs(sys.argv)
    parser = ConfigParser(kwargs)
    config_dict = parser.generate_config_dict()

    # Validate kwarg capturing
    assert kwargs["disable_multiprocessing"]
    assert kwargs["output_dir"] == "output_dir"

    assert config_dict["pdftext_workers"] == 1  # disabling multiprocessing does this
    assert config_dict["height_tolerance"] == 0.5


def test_config_none():
    kwargs = capture_kwargs(["test"])

    for key in crawler.attr_set:
        # We force some options to become flags for ease of use on the CLI
        value = None
        assert kwargs.get(key) is value


def test_config_llm():
    kwargs = capture_kwargs(["test", "--use_llm"])
    parser = ConfigParser(kwargs)
    config_dict = parser.generate_config_dict()

    # Validate kwarg capturing
    assert config_dict["use_llm"]


def test_config_force_ocr():
    kwargs = capture_kwargs(["test", "--force_ocr"])
    parser = ConfigParser(kwargs)
    config_dict = parser.generate_config_dict()

    # Validate kwarg capturing
    assert config_dict["force_ocr"]


def test_output_paths_trailing_space(tmp_path):
    parser = ConfigParser({"output_dir": str(tmp_path)})
    fpath = "docs/Rental Application .pdf"

    assert parser.get_base_filename(fpath) == "Rental Application"
    assert parser.get_output_folder(fpath) == str(tmp_path / "Rental Application")
