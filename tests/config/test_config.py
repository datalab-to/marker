import sys
from contextlib import suppress
import click

from marker.config.printer import CustomClickPrinter
from marker.config.crawler import crawler
from marker.config.parser import ConfigParser


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
    assert isinstance(config_dict["pdftext_workers"], int)
    assert config_dict["height_tolerance"] == 0.5
    assert isinstance(config_dict["height_tolerance"], float)


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


def test_config_invalid_height_tolerance():
    kwargs = capture_kwargs(["test", "--height_tolerance", "not_a_float"])
    assert "height_tolerance" not in kwargs
