"""End-to-end quality integration test on vendored olmOCR-bench pages.

Marker converts three real single-page PDFs (multi-column reading order, tiny
text, headers/footers) and must pass every olmOCR-bench rule attached to them.
The `present`/`absent`/`order`/`baseline` rule types are reimplemented here
faithfully (see tests/data/olmocr_bench/README.md for provenance); the `table`
and `math` types, which need olmOCR-bench's own KaTeX/table checker, are skipped.
"""

import json
import re
import unicodedata
from pathlib import Path

import pytest
from rapidfuzz import fuzz

from marker.converters.pdf import PdfConverter
from marker.output import text_from_rendered

DATA_DIR = Path(__file__).parent.parent / "data" / "olmocr_bench"

# Fancy punctuation -> ASCII, matching olmOCR-bench's normalizer so vendored
# reference strings compare against marker output the same way the real checker
# would (e.g. curly quotes, en/em dashes, micro sign vs greek mu).
_REPLACEMENTS = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "＿": "_",
    "–": "-",
    "—": "-",
    "‑": "-",
    "‒": "-",
    "−": "-",
    "µ": "μ",
}


def _normalize(text: str) -> str:
    text = re.sub(r"<br/?>", " ", text)
    text = re.sub(r"</?[bi]>", "", text)
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)  # bold
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)  # italics
    text = re.sub(r"\s+", " ", text)
    text = unicodedata.normalize("NFC", text)
    for fancy, ascii_char in _REPLACEMENTS.items():
        text = text.replace(fancy, ascii_char)
    return text


def _threshold(query: str, max_diffs: int) -> float:
    return 1.0 - (max_diffs / (len(query) if query else 1))


def _check(test: dict, md: str) -> tuple[bool, str]:
    """Return (passed, failure_message) for one olmOCR-bench rule."""
    typ = test["type"]
    md_norm = _normalize(md)
    max_diffs = test.get("max_diffs", 0)

    if typ in ("present", "absent"):
        query = _normalize(test["text"])
        haystack = md_norm
        first_n, last_n = test.get("first_n"), test.get("last_n")
        if first_n and last_n:
            haystack = haystack[:first_n] + haystack[-last_n:]
        elif first_n:
            haystack = haystack[:first_n]
        elif last_n:
            haystack = haystack[-last_n:]
        ratio = fuzz.partial_ratio(query, haystack) / 100.0
        threshold = _threshold(query, max_diffs)
        if typ == "present":
            return ratio >= threshold, (
                f"expected present (ratio {ratio:.3f} < {threshold:.3f}): {query[:60]!r}"
            )
        return ratio < threshold, (
            f"expected absent (ratio {ratio:.3f} >= {threshold:.3f}): {query[:60]!r}"
        )

    if typ == "order":
        before, after = _normalize(test["before"]), _normalize(test["after"])
        before_match = fuzz.partial_ratio_alignment(before, md_norm)
        after_match = fuzz.partial_ratio_alignment(after, md_norm)
        if before_match is None or before_match.score / 100.0 < _threshold(
            before, max_diffs
        ):
            return False, f"'before' text not found: {before[:60]!r}"
        if after_match is None or after_match.score / 100.0 < _threshold(
            after, max_diffs
        ):
            return False, f"'after' text not found: {after[:60]!r}"
        return before_match.dest_start < after_match.dest_start, (
            f"'before' (@{before_match.dest_start}) should precede "
            f"'after' (@{after_match.dest_start})"
        )

    if typ == "baseline":
        alnum = "".join(c for c in md if c.isalnum()).strip()
        return len(alnum) > 0, "page rendered blank"

    raise ValueError(f"unsupported test type: {typ}")


def _load_tests_by_pdf() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    with open(DATA_DIR / "tests.jsonl") as f:
        for line in f:
            record = json.loads(line)
            grouped.setdefault(record["pdf"], []).append(record)
    return grouped


TESTS_BY_PDF = _load_tests_by_pdf()


@pytest.mark.integration
@pytest.mark.parametrize("pdf_name", sorted(TESTS_BY_PDF))
def test_olmocr_bench_page(model_dict, pdf_name):
    import os
    from marker.settings import settings

    mode = (
        "balanced"
        if (settings.TORCH_DEVICE_MODEL == "cuda" or os.environ.get("SURYA_INFERENCE_URL"))
        else "indra"
    )
    converter = PdfConverter(
        artifact_dict=model_dict,
        config={
            "mode": mode,
            "output_format": "markdown",
            "disable_image_extraction": True,
            "disable_tqdm": True,
            "pdftext_workers": 1,
        },
    )
    markdown, _, _ = text_from_rendered(converter(str(DATA_DIR / "pdfs" / pdf_name)))

    failures = []
    for test in TESTS_BY_PDF[pdf_name]:
        passed, message = _check(test, markdown)
        if not passed:
            failures.append(f"[{test['type']}] {message}")
    assert not failures, (
        f"{pdf_name}: {len(failures)} olmOCR-bench rule(s) failed:\n"
        + "\n".join(failures)
    )
