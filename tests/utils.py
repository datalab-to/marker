import tempfile
from typing import Any, Dict, Optional

import pytest

from marker.providers.pdf import PdfProvider
from tests.dataset import (
    DEFAULT_DOCUMENT,
    DocumentUnavailable,
    load_document,
    unavailable_reason,
)


def require_documents(*filenames: str) -> None:
    """Skip the current test unless every named sample document can be loaded.

    Call this before handing filenames to a worker process: a skip raised in a
    worker would surface in the parent as a failure, not a skip.
    """
    for filename in filenames:
        reason = unavailable_reason(filename)
        if reason:
            pytest.skip(reason)


def setup_pdf_provider(
    filename: str = DEFAULT_DOCUMENT,
    config: Optional[Dict[str, Any]] = None,
) -> PdfProvider:
    """Build a PdfProvider over a sample document.

    Raises :class:`DocumentUnavailable` when the document cannot be loaded;
    callers running in-process should guard with :func:`require_documents` so
    that turns into a skip.
    """
    content = load_document(filename)

    temp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf")
    temp_pdf.write(content)
    temp_pdf.flush()

    return PdfProvider(temp_pdf.name, config)


__all__ = ["DocumentUnavailable", "require_documents", "setup_pdf_provider"]
