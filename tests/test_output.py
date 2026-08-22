import json
import os

import pytest
from PIL import Image

from marker.output import atomic_write_text, output_exists, save_output
from marker.renderers.markdown import MarkdownOutput


def markdown_output(images=None):
    return MarkdownOutput(
        markdown="# Title\n\nBody text.\n",
        images=images or {},
        metadata={"table_of_contents": []},
    )


def test_save_output_writes_every_part(tmp_path):
    images = {"_page_0_Figure_1.jpeg": Image.new("RGB", (4, 4), "white")}
    save_output(markdown_output(images), str(tmp_path), "doc")

    assert (tmp_path / "doc.md").read_text().startswith("# Title")
    assert json.loads((tmp_path / "doc_meta.json").read_text()) == {
        "table_of_contents": []
    }
    assert (tmp_path / "_page_0_Figure_1.jpeg").exists()


def test_output_exists_is_format_aware(tmp_path):
    save_output(markdown_output(), str(tmp_path), "doc")

    assert output_exists(str(tmp_path), "doc", "markdown")
    # A markdown conversion says nothing about whether the json one was run.
    assert not output_exists(str(tmp_path), "doc", "json")
    assert not output_exists(str(tmp_path), "doc", "chunks")
    assert not output_exists(str(tmp_path), "doc", "html")


def test_output_exists_without_a_format_accepts_any(tmp_path):
    save_output(markdown_output(), str(tmp_path), "doc")

    assert output_exists(str(tmp_path), "doc")
    assert not output_exists(str(tmp_path), "other")


def test_output_exists_rejects_an_interrupted_conversion(tmp_path):
    # A run killed after the markdown was written but before the images were.
    with pytest.raises(AttributeError):
        save_output(
            markdown_output({"broken.jpeg": "not an image"}), str(tmp_path), "doc"
        )

    assert (tmp_path / "doc.md").exists()
    assert not output_exists(str(tmp_path), "doc", "markdown")


def test_atomic_write_text_never_leaves_a_partial_file(tmp_path, monkeypatch):
    target = tmp_path / "doc.md"

    def failing_replace(src, dst):
        raise OSError("interrupted")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError):
        atomic_write_text(str(target), "half a document")

    assert not target.exists()
    assert os.listdir(tmp_path) == []


def test_atomic_write_text_replaces_an_existing_file(tmp_path):
    target = tmp_path / "doc.md"
    target.write_text("stale content that is much longer than the new content")

    atomic_write_text(str(target), "fresh")

    assert target.read_text() == "fresh"
    assert os.listdir(tmp_path) == ["doc.md"]


def test_atomic_write_keeps_the_usual_file_permissions(tmp_path):
    reference = tmp_path / "reference.md"
    with open(reference, "w") as f:
        f.write("written the ordinary way")
    target = tmp_path / "doc.md"

    atomic_write_text(str(target), "written atomically")

    assert oct(target.stat().st_mode) == oct(reference.stat().st_mode)
