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


def test_config_mode_auto_and_headers_footers(tmp_path):
    kwargs = capture_kwargs(["test", "--mode", "auto", "--keep_headers_footers"])
    parser = ConfigParser(kwargs)
    config_dict = parser.generate_config_dict()

    # 'auto' leaves mode unset so PdfConverter resolves by device
    assert "mode" not in config_dict
    assert config_dict["keep_pageheader_in_output"] is True
    assert config_dict["keep_pagefooter_in_output"] is True

    # Test relative subdirectory output preservation
    parser_out = ConfigParser({"output_dir": str(tmp_path / "out")})
    base_in = tmp_path / "in"
    nested_pdf = base_in / "sub" / "nested" / "report.pdf"
    out_folder = parser_out.get_output_folder(str(nested_pdf), base_in_folder=str(base_in))
    assert out_folder.endswith("sub/nested/report")


def test_discover_input_files_and_skills(tmp_path):
    from click.testing import CliRunner
    from marker.scripts.convert import convert_cli, discover_input_files
    from marker.scripts.convert_single import convert_single_cli

    in_dir = tmp_path / "docs"
    sub_dir = in_dir / "nested"
    sub_dir.mkdir(parents=True)

    (in_dir / "a.pdf").write_bytes(b"%PDF-1.4")
    (in_dir / "b.png").write_bytes(b"png")
    (in_dir / ".DS_Store").write_bytes(b"junk")
    (in_dir / "notes.txt").write_text("ignore me")
    (sub_dir / "c.docx").write_bytes(b"docx")

    non_rec = discover_input_files(str(in_dir), recursive=False)
    assert [f.split("/")[-1] for f in non_rec] == ["a.pdf", "b.png"]

    rec = discover_input_files(str(in_dir), recursive=True)
    assert [f.split("/")[-1] for f in rec] == ["a.pdf", "b.png", "c.docx"]

    single = discover_input_files(str(in_dir / "a.pdf"), recursive=False)
    assert len(single) == 1 and single[0].endswith("a.pdf")

    runner = CliRunner()
    res_skill = runner.invoke(convert_single_cli, ["--skill"])
    assert res_skill.exit_code == 0
    assert "name: marker" in res_skill.output
    assert "marker_single" in res_skill.output

    skill_target = tmp_path / "custom_skills"
    res_install = runner.invoke(
        convert_cli, ["--install_skill", str(skill_target)]
    )
    assert res_install.exit_code == 0
    assert (skill_target / "marker" / "SKILL.md").is_file()

    res_ver = runner.invoke(convert_single_cli, ["--version"])
    assert res_ver.exit_code == 0
    assert "2.2.0" in res_ver.output
