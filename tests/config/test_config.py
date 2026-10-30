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

    assert kwargs["output_dir"] is None
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


def test_output_folder_defaults_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parser = ConfigParser({"output_dir": None})

    assert parser.get_output_folder("/tmp/example.pdf") == str(tmp_path / "example")


def test_output_folder_uses_explicit_output_dir(tmp_path):
    output_dir = tmp_path / "out"
    parser = ConfigParser({"output_dir": str(output_dir)})

    assert parser.get_output_folder("/tmp/example.pdf") == str(output_dir / "example")
