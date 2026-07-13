"""W3C trace context propagation and JSON logging for this service.

When OTEL_ENABLED=true, headers and worker context also bridge OpenTelemetry.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Iterator

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_TRACEPARENT = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)
span_id_var: ContextVar[str | None] = ContextVar("span_id", default=None)
trace_flags_var: ContextVar[str] = ContextVar("trace_flags", default="01")


def _hex(length: int) -> str:
    return secrets.token_hex(length // 2)


def _parse(value: str | None) -> tuple[str, str] | None:
    match = _TRACEPARENT.fullmatch(value.lower()) if value else None
    if not match or match.group(1) == "0" * 32 or match.group(2) == "0" * 16:
        return None
    return match.group(1), match.group(3)


def current_traceparent() -> str | None:
    trace_id, span_id = trace_id_var.get(), span_id_var.get()
    return f"00-{trace_id}-{span_id}-{trace_flags_var.get()}" if trace_id and span_id else None


def get_propagation_headers() -> dict[str, str]:
    """Build outbound headers from OTel context first, then local ContextVars."""
    headers: dict[str, str] = {}
    try:
        from opentelemetry.propagate import inject

        inject(headers)
    except Exception:
        pass
    if "traceparent" not in headers:
        if traceparent := current_traceparent():
            headers["traceparent"] = traceparent
    if request_id := request_id_var.get():
        headers["X-Request-ID"] = request_id
    return headers


def attach_trace_to_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload["_trace_ctx"] = get_propagation_headers()
    return payload


@contextmanager
def activate_trace_from_headers(headers: dict[str, str] | None) -> Iterator[None]:
    """Restore ContextVar + OTel context from propagation headers."""
    headers = headers or {}
    parsed = _parse(headers.get("traceparent"))
    trace_id, flags = parsed if parsed else (_hex(32), "01")
    tokens = (
        request_id_var.set(headers.get("X-Request-ID") or str(uuid.uuid4())),
        trace_id_var.set(trace_id),
        span_id_var.set(_hex(16)),
        trace_flags_var.set(flags),
    )
    otel_token = None
    span_cm = None
    try:
        try:
            from opentelemetry import context, trace
            from opentelemetry.propagate import extract

            otel_token = context.attach(extract(headers))
            tracer = trace.get_tracer("worker")
            span_cm = tracer.start_as_current_span("worker.process")
            span_cm.__enter__()
        except Exception:
            otel_token = None
            span_cm = None
        yield
    finally:
        if span_cm is not None:
            span_cm.__exit__(None, None, None)
        if otel_token is not None:
            try:
                from opentelemetry import context

                context.detach(otel_token)
            except Exception:
                pass
        request_id_var.reset(tokens[0])
        trace_id_var.reset(tokens[1])
        span_id_var.reset(tokens[2])
        trace_flags_var.reset(tokens[3])


@contextmanager
def activate_trace_from_payload(payload: dict[str, Any]) -> Iterator[None]:
    """Restore ContextVar + OTel context for one worker task."""
    with activate_trace_from_headers(payload.get("_trace_ctx") or {}):
        yield


class TraceContextMiddleware(BaseHTTPMiddleware):
    """Continue inbound trace context and return it to callers."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        parsed = _parse(request.headers.get("traceparent"))
        trace_id, flags = parsed if parsed else (_hex(32), "01")
        tokens = (
            request_id_var.set(request.headers.get("X-Request-ID") or str(uuid.uuid4())),
            trace_id_var.set(trace_id),
            span_id_var.set(_hex(16)),
            trace_flags_var.set(flags),
        )
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id_var.get() or ""
            response.headers["traceparent"] = (
                get_propagation_headers().get("traceparent") or current_traceparent() or ""
            )
            return response
        finally:
            request_id_var.reset(tokens[0])
            trace_id_var.reset(tokens[1])
            span_id_var.reset(tokens[2])
            trace_flags_var.reset(tokens[3])


class JsonTraceFormatter(logging.Formatter):
    """Format logs as JSON enriched with the active trace context."""

    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        otel_trace_id = None
        otel_span_id = None
        try:
            from otel_setup import current_otel_ids

            otel_trace_id, otel_span_id = current_otel_ids()
        except Exception:
            pass
        return json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "service": self.service,
                "request_id": request_id_var.get(),
                "trace_id": otel_trace_id or trace_id_var.get(),
                "span_id": otel_span_id or span_id_var.get(),
                "traceparent": get_propagation_headers().get("traceparent") or current_traceparent(),
                "pid": os.getpid(),
                "logger": record.name,
                "message": record.getMessage(),
            },
            ensure_ascii=False,
            default=str,
        )
