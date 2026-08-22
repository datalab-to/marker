"""What a batch conversion has to say about the files it was given.

Kept apart from `convert.py` because that module configures threading and
device environment variables at import time, and this needs neither.
"""

from typing import Dict, List, NamedTuple, Optional

from marker.logger import get_logger
from marker.output import write_failure_report

logger = get_logger()


class ConversionResult(NamedTuple):
    """The outcome of one worker converting one input file."""

    fpath: str
    page_count: int = 0
    skipped: bool = False
    error: Optional[str] = None


def failure_report_name(chunk_idx: int, num_chunks: int) -> str:
    """Chunks of one run share an output directory, so they need distinct names."""
    if num_chunks == 1:
        return "conversion_failures.json"
    return f"conversion_failures_chunk_{chunk_idx}.json"


def failure_records(results: List[ConversionResult]) -> List[Dict[str, str]]:
    return [
        {"file": result.fpath, "error": result.error}
        for result in results
        if result.error
    ]


def report_outcomes(
    results: List[ConversionResult],
    output_dir: str,
    chunk_idx: int = 0,
    num_chunks: int = 1,
) -> Optional[str]:
    """Summarise a batch and leave its failures behind as a re-runnable list.

    Without this, the only record of a failed file is a traceback somewhere in
    the scrollback of a run that may have covered hundreds of documents.
    Returns the path to the failure report, or None if nothing failed.
    """
    skipped = [result for result in results if result.skipped]
    if skipped:
        logger.info(f"Skipped {len(skipped)} files with existing output.")

    failures = failure_records(results)
    if not failures:
        return None

    path = write_failure_report(
        output_dir, failures, fname=failure_report_name(chunk_idx, num_chunks)
    )
    logger.warning(
        f"{len(failures)} of {len(results)} files failed to convert.  See {path}"
    )
    return path
