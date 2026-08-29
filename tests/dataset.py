"""Locating the sample documents that the test suite converts.

Sample documents live in the ``datalab-to/pdfs`` dataset on the HuggingFace
Hub.  That dataset is gated: without an access token the Hub answers HTTP 401,
and every test that needs a sample document becomes unrunnable.  Resolving
documents through this module means a missing corpus turns into a skip with an
actionable reason instead of an error out of fixture setup.

Sources are consulted in order:

1. A local directory of sample documents, named exactly as the tests request
   them (``adversarial.pdf``, ``gatsby.docx``, ...).  It defaults to
   ``tests/data/pdfs`` and is overridable with ``MARKER_TEST_PDF_DIR``.
2. The ``datalab-to/pdfs`` dataset on the HuggingFace Hub.

See ``tests/README.md`` for how to make either source available.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

DATASET_NAME = "datalab-to/pdfs"
DATASET_SPLIT = "train"

#: Column holding the raw document bytes.  Named "pdf" upstream, but it also
#: carries the .docx/.epub/.html/.pptx/.xlsx samples.
DATASET_CONTENT_COLUMN = "pdf"
DATASET_FILENAME_COLUMN = "filename"

#: Document used by tests that do not carry a ``filename`` marker.
DEFAULT_DOCUMENT = "adversarial.pdf"

LOCAL_DIR_ENV_VAR = "MARKER_TEST_PDF_DIR"
DEFAULT_LOCAL_DIR = Path(__file__).parent / "data" / "pdfs"

SETUP_HINT = (
    f"Grant access with `huggingface-cli login` after requesting the dataset at "
    f"https://huggingface.co/datasets/{DATASET_NAME}, or drop the file into the "
    f"local directory (override it with ${LOCAL_DIR_ENV_VAR}). See tests/README.md."
)


class DocumentUnavailable(Exception):
    """A requested sample document was not found in any source."""


def local_dir() -> Path:
    """Directory searched for sample documents before the Hub is consulted."""
    override = os.environ.get(LOCAL_DIR_ENV_VAR)
    return Path(override).expanduser() if override else DEFAULT_LOCAL_DIR


def _local_path(filename: str) -> Optional[Path]:
    """Resolve ``filename`` inside the local directory, or None if it is not a
    plain file name (guards against a marker escaping the directory)."""
    if not filename or Path(filename).name != filename:
        return None
    return local_dir() / filename


def _from_local(filename: str) -> Tuple[Optional[bytes], str]:
    """Return ``(content, reason)``; ``content`` is None when unavailable."""
    path = _local_path(filename)
    if path is None:
        return None, f"{filename!r} is not a plain file name"
    if not path.is_file():
        return None, f"no such file: {path}"
    try:
        return path.read_bytes(), ""
    except OSError as exc:
        return None, f"could not read {path}: {exc}"


@lru_cache(maxsize=1)
def _load_hf_dataset() -> Tuple[Optional[object], str]:
    """Load the Hub dataset once per process.

    Returns ``(dataset, reason)``.  Any failure -- gated dataset, no token, no
    network -- is reported as a reason rather than raised, so that a caller can
    fall back or skip.
    """
    try:
        import datasets
    except ImportError as exc:  # pragma: no cover - datasets is a test dep
        return None, f"the `datasets` package is unavailable: {exc}"

    try:
        return datasets.load_dataset(DATASET_NAME, split=DATASET_SPLIT), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _from_hf_dataset(filename: str) -> Tuple[Optional[bytes], str]:
    """Return ``(content, reason)``; ``content`` is None when unavailable."""
    dataset, reason = _load_hf_dataset()
    if dataset is None:
        return None, reason

    try:
        idx = dataset[DATASET_FILENAME_COLUMN].index(filename)
    except ValueError:
        return None, f"{DATASET_NAME} has no entry named {filename!r}"
    except Exception as exc:
        return None, f"could not read {DATASET_NAME}: {type(exc).__name__}: {exc}"

    return dataset[DATASET_CONTENT_COLUMN][idx], ""


def unavailable_reason(filename: str) -> Optional[str]:
    """Explain why ``filename`` cannot be loaded, or None if it can.

    Consulting both sources is what makes the message actionable, so this does
    the same work as :func:`load_document` and is cheap to call afterwards --
    the Hub lookup is cached per process.
    """
    _, local_reason = _from_local(filename)
    if not local_reason:
        return None

    _, hub_reason = _from_hf_dataset(filename)
    if not hub_reason:
        return None

    return (
        f"sample document {filename!r} is unavailable "
        f"(local: {local_reason}; hub: {hub_reason}). {SETUP_HINT}"
    )


def load_document(filename: str) -> bytes:
    """Return the bytes of sample document ``filename``.

    Raises :class:`DocumentUnavailable`, naming both sources, if neither has it.
    """
    content, local_reason = _from_local(filename)
    if content is not None:
        return content

    content, hub_reason = _from_hf_dataset(filename)
    if content is not None:
        return content

    raise DocumentUnavailable(
        f"sample document {filename!r} is unavailable "
        f"(local: {local_reason}; hub: {hub_reason}). {SETUP_HINT}"
    )


def hf_dataset() -> Tuple[Optional[object], str]:
    """Return ``(dataset, reason)`` for the raw Hub dataset.

    ``dataset`` is None when the Hub could not be reached; ``reason`` then says
    why.  Exposed for tests that need the corpus itself rather than one
    document.
    """
    return _load_hf_dataset()
