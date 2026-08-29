"""Tests for locating sample documents (see `tests/dataset.py`).

These exercise the resolution logic itself, so they must keep passing whether
or not the gated HuggingFace dataset happens to be reachable.
"""

import pytest

from tests import dataset
from tests.dataset import (
    DocumentUnavailable,
    load_document,
    local_dir,
    unavailable_reason,
)

# A name no real corpus will ever contain, so the "missing" assertions below
# hold regardless of whether the Hub is reachable.
MISSING_DOCUMENT = "no-such-sample-document-4f3a1b.pdf"


@pytest.fixture
def sample_dir(tmp_path, monkeypatch):
    """Point the resolver at a directory this test controls."""
    monkeypatch.setenv(dataset.LOCAL_DIR_ENV_VAR, str(tmp_path))
    return tmp_path


@pytest.mark.cpu
def test_local_dir_follows_env_var(sample_dir):
    assert local_dir() == sample_dir


@pytest.mark.cpu
def test_local_dir_defaults_under_tests_data(monkeypatch):
    monkeypatch.delenv(dataset.LOCAL_DIR_ENV_VAR, raising=False)
    assert local_dir() == dataset.DEFAULT_LOCAL_DIR
    assert local_dir().name == "pdfs"


@pytest.mark.cpu
def test_loads_document_from_local_dir(sample_dir):
    (sample_dir / "adversarial.pdf").write_bytes(b"%PDF-1.7 local copy")

    assert load_document("adversarial.pdf") == b"%PDF-1.7 local copy"


@pytest.mark.cpu
def test_local_dir_takes_precedence_over_the_hub(sample_dir, monkeypatch):
    """A local file wins, so the suite runs without any Hub access at all."""
    (sample_dir / "thinkpython.pdf").write_bytes(b"%PDF-1.7 local wins")

    def fail_if_called():  # pragma: no cover - asserted not to run
        raise AssertionError("the Hub must not be consulted for a local file")

    monkeypatch.setattr(dataset, "_load_hf_dataset", fail_if_called)

    assert load_document("thinkpython.pdf") == b"%PDF-1.7 local wins"


@pytest.mark.cpu
def test_non_pdf_samples_load_by_name(sample_dir):
    """The corpus also carries .docx/.epub/.html/.pptx/.xlsx samples."""
    for name in ("gatsby.docx", "manual.epub", "china.html", "single_sheet.xlsx"):
        (sample_dir / name).write_bytes(f"contents of {name}".encode())

    for name in ("gatsby.docx", "manual.epub", "china.html", "single_sheet.xlsx"):
        assert load_document(name) == f"contents of {name}".encode()


@pytest.mark.cpu
def test_missing_document_raises_with_both_sources_named(sample_dir):
    with pytest.raises(DocumentUnavailable) as excinfo:
        load_document(MISSING_DOCUMENT)

    message = str(excinfo.value)
    assert MISSING_DOCUMENT in message
    assert "local:" in message and "hub:" in message
    assert dataset.LOCAL_DIR_ENV_VAR in message


@pytest.mark.cpu
def test_unavailable_reason_is_none_when_present(sample_dir):
    (sample_dir / "pres.pdf").write_bytes(b"%PDF-1.7")

    assert unavailable_reason("pres.pdf") is None
    assert unavailable_reason(MISSING_DOCUMENT) is not None


@pytest.mark.cpu
@pytest.mark.parametrize("filename", ["../secrets.pdf", "nested/doc.pdf", ""])
def test_rejects_names_that_escape_the_local_dir(sample_dir, filename):
    """Marker-supplied names are resolved inside the directory, never above it."""
    with pytest.raises(DocumentUnavailable):
        load_document(filename)


@pytest.fixture
def local_adversarial_pdf(sample_dir):
    """Stage a local document *before* `temp_doc` resolves it."""
    (sample_dir / "adversarial.pdf").write_bytes(b"%PDF-1.7 via fixture")
    return sample_dir


@pytest.mark.cpu
@pytest.mark.filename("adversarial.pdf")
def test_temp_doc_serves_a_local_document(local_adversarial_pdf, temp_doc):
    """The `temp_doc` fixture reads the local directory, not just the Hub."""
    assert temp_doc.name.endswith(".pdf")
    with open(temp_doc.name, "rb") as f:
        assert f.read() == b"%PDF-1.7 via fixture"


@pytest.mark.cpu
def test_hub_failure_is_reported_not_raised(sample_dir, monkeypatch):
    """A gated or unreachable Hub must not raise out of fixture setup."""
    monkeypatch.setattr(
        dataset,
        "_load_hf_dataset",
        lambda: (None, "DatasetNotFoundError: cannot be accessed"),
    )

    hub_dataset, reason = dataset.hf_dataset()

    assert hub_dataset is None
    assert "cannot be accessed" in reason


@pytest.mark.cpu
def test_falls_back_to_the_hub_when_the_local_dir_lacks_it(sample_dir, monkeypatch):
    """The Hub still serves documents, keyed the way the upstream corpus is."""
    stub = {
        dataset.DATASET_FILENAME_COLUMN: ["other.pdf", "form_1040.pdf"],
        dataset.DATASET_CONTENT_COLUMN: [b"other", b"%PDF-1.7 from the hub"],
    }
    monkeypatch.setattr(dataset, "_load_hf_dataset", lambda: (stub, ""))

    assert load_document("form_1040.pdf") == b"%PDF-1.7 from the hub"


@pytest.mark.cpu
def test_document_absent_from_the_hub_names_the_dataset(sample_dir, monkeypatch):
    stub = {
        dataset.DATASET_FILENAME_COLUMN: ["other.pdf"],
        dataset.DATASET_CONTENT_COLUMN: [b"other"],
    }
    monkeypatch.setattr(dataset, "_load_hf_dataset", lambda: (stub, ""))

    with pytest.raises(DocumentUnavailable, match=dataset.DATASET_NAME):
        load_document(MISSING_DOCUMENT)
