import os
from typing import Any, Mapping


def _sanitize_header_value(value: Any, *, max_len: int = 512) -> str:
    text = str(value)
    text = text.replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def sanitize_headers(headers: Mapping[str, Any] | None) -> dict[str, str]:
    if not headers:
        return {}
    cleaned: dict[str, str] = {}
    for key, value in headers.items():
        if value is None:
            continue
        cleaned[str(key)] = _sanitize_header_value(value)
    return cleaned


def build_marker_trace_headers(
    *,
    source_path: str | None = None,
    processor: str | None = None,
    block_id: str | None = None,
    page_id: int | str | None = None,
    extra: Mapping[str, Any] | None = None,
    max_source_len: int = 256,
) -> dict[str, str]:
    headers: dict[str, Any] = {}

    if source_path:
        source_path = str(source_path)
        headers["X-Marker-Source-File"] = os.path.basename(source_path)
        headers["X-Marker-Source"] = _sanitize_header_value(source_path, max_len=max_source_len)

    if processor:
        headers["X-Marker-Processor"] = processor

    if block_id:
        headers["X-Marker-Block"] = block_id

    if page_id is not None:
        headers["X-Marker-Page"] = page_id

    if extra:
        for k, v in extra.items():
            if v is None:
                continue
            headers[f"X-Marker-{k}"] = v

    return sanitize_headers(headers)

