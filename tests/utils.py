from marker.providers.pdf import PdfProvider
import tempfile

import datasets
from datasets.exceptions import DatasetNotFoundError


class PdfFixtureDatasetUnavailable(RuntimeError):
    """Raised when the gated sample-document dataset cannot be loaded."""


def load_pdf_fixture_dataset():
    """Load the existing sample dataset without reflecting Hub error details."""

    try:
        return datasets.load_dataset("datalab-to/pdfs", split="train")
    except DatasetNotFoundError:
        raise PdfFixtureDatasetUnavailable(
            "sample document dataset datalab-to/pdfs is unavailable"
        ) from None


def setup_pdf_provider(
    filename="adversarial.pdf",
    config=None,
) -> PdfProvider:
    dataset = load_pdf_fixture_dataset()
    idx = dataset["filename"].index(filename)

    temp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf")
    temp_pdf.write(dataset["pdf"][idx])
    temp_pdf.flush()

    provider = PdfProvider(temp_pdf.name, config)
    return provider
