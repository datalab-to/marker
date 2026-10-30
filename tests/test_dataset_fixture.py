"""Tests for contributor-safe sample-document dataset loading."""

from __future__ import annotations

import pytest
from datasets.exceptions import DatasetNotFoundError

from tests import conftest as test_config
from tests import utils


def test_load_pdf_fixture_dataset_uses_the_existing_hub_source(monkeypatch) -> None:
    sentinel = object()
    calls: list[tuple[str, str]] = []

    def load_dataset(dataset_id: str, *, split: str) -> object:
        calls.append((dataset_id, split))
        return sentinel

    monkeypatch.setattr(utils.datasets, "load_dataset", load_dataset)

    assert utils.load_pdf_fixture_dataset() is sentinel
    assert calls == [("datalab-to/pdfs", "train")]


def test_load_pdf_fixture_dataset_hides_hub_error_details(monkeypatch) -> None:
    def load_dataset(dataset_id: str, *, split: str) -> object:
        raise DatasetNotFoundError("private-hub-error-detail")

    monkeypatch.setattr(utils.datasets, "load_dataset", load_dataset)

    with pytest.raises(utils.PdfFixtureDatasetUnavailable) as exc_info:
        utils.load_pdf_fixture_dataset()

    assert str(exc_info.value) == (
        "sample document dataset datalab-to/pdfs is unavailable"
    )
    assert "private-hub-error-detail" not in str(exc_info.value)


def test_pdf_dataset_fixture_skips_when_the_hub_source_is_unavailable(
    monkeypatch,
) -> None:
    def unavailable() -> object:
        raise utils.PdfFixtureDatasetUnavailable(
            "sample document dataset datalab-to/pdfs is unavailable"
        )

    monkeypatch.setattr(test_config, "load_pdf_fixture_dataset", unavailable)

    with pytest.raises(pytest.skip.Exception) as exc_info:
        test_config.load_or_skip_pdf_dataset()

    assert str(exc_info.value) == (
        "sample document dataset datalab-to/pdfs is unavailable"
    )
