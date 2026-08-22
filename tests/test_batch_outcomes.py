import json
import os

from marker.scripts.batch_outcomes import (
    ConversionResult,
    failure_report_name,
    report_outcomes,
)


def test_a_clean_batch_writes_no_report(tmp_path):
    results = [
        ConversionResult("/in/a.pdf", page_count=12),
        ConversionResult("/in/b.pdf", skipped=True),
    ]

    assert report_outcomes(results, str(tmp_path)) is None
    assert os.listdir(tmp_path) == []


def test_failures_are_recorded_with_their_file(tmp_path):
    results = [
        ConversionResult("/in/a.pdf", page_count=12),
        ConversionResult("/in/b.pdf", error="PdfiumError: Incorrect password"),
        ConversionResult("/in/c.pdf", error="ValueError: no pages"),
    ]

    path = report_outcomes(results, str(tmp_path))

    assert json.loads(open(path).read()) == [
        {"file": "/in/b.pdf", "error": "PdfiumError: Incorrect password"},
        {"file": "/in/c.pdf", "error": "ValueError: no pages"},
    ]


def test_report_is_written_into_a_missing_output_dir(tmp_path):
    output_dir = tmp_path / "not_yet_created"

    path = report_outcomes(
        [ConversionResult("/in/a.pdf", error="boom")], str(output_dir)
    )

    assert os.path.dirname(path) == str(output_dir)


def test_chunks_do_not_overwrite_each_others_reports():
    assert failure_report_name(0, 1) == "conversion_failures.json"
    assert failure_report_name(0, 4) == "conversion_failures_chunk_0.json"
    assert failure_report_name(3, 4) == "conversion_failures_chunk_3.json"
