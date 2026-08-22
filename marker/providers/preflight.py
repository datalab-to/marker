"""Checks a PDF must pass before PDFium is asked to parse it.

PDFium answers "Data format error" for an empty file, a download that saved an
error page under a .pdf name, and a genuinely damaged document alike, several
seconds into a conversion and from deep inside a worker.  These checks are the
ones that can be made from the bytes with certainty, so that a file marker
cannot read is named as such at the start.

They deliberately refuse nothing PDFium would accept.  There are no size, page
count or resolution limits here: a document marker converts today must still
convert.  A PDF encrypted with an empty user password is one of those -- it
opens for everyone else, so it opens here, and nothing below looks for it.
"""

import os

PDF_SIGNATURE = b"%PDF-"
# PDFium reads a header that is not at byte zero, but stops looking after the
# first kilobyte.  Verified against pypdfium2: offset 300 opens, offset 2000
# does not.  Matching that window means preflight refuses only what PDFium
# would refuse anyway.
SIGNATURE_SEARCH_BYTES = 1024


# What PDFium reports for a document it cannot decrypt, from pypdfium2's own
# error table: FPDF_ERR_PASSWORD and FPDF_ERR_SECURITY.
ENCRYPTION_ERRORS = ("incorrect password", "unsupported security scheme")


class UnreadablePdfError(ValueError):
    """Raised for a file that cannot be a PDF, before any parser sees it."""


def preflight_pdf(filepath: str) -> None:
    """Raise `UnreadablePdfError` if this file cannot be a PDF.

    Passing says only that a parser should be given the file, not that the
    document is intact -- that is PDFium's judgement to make.
    """
    if not os.path.isfile(filepath):
        raise UnreadablePdfError(f"{filepath} is not a file that can be read.")

    if os.path.getsize(filepath) == 0:
        raise UnreadablePdfError(f"{filepath} is empty.")

    with open(filepath, "rb") as f:
        head = f.read(SIGNATURE_SEARCH_BYTES)

    if PDF_SIGNATURE not in head:
        raise UnreadablePdfError(
            f"{filepath} is not a PDF: no %PDF- header in its first "
            f"{SIGNATURE_SEARCH_BYTES} bytes."
        )


def unreadable_pdf_message(filepath: str, error: Exception) -> str:
    """Say what PDFium's refusal to open a file means, in the caller's terms."""
    reported = str(error).lower()
    if any(signal in reported for signal in ENCRYPTION_ERRORS):
        return (
            f"{filepath} is encrypted and could not be opened.  Marker does not "
            f"ask for a password; supply a decrypted copy instead."
        )
    return f"{filepath} could not be read as a PDF: {error}"
