"""OpenTelemetry bootstrap for Starlette downstream services.

Enable with OTEL_ENABLED=true. Default OTLP endpoint: http://127.0.0.1:4318.
"""

from __future__ import annotations

import os
from typing import Any

_INITIALIZED = False


def otel_enabled() -> bool:
    """Return whether OpenTelemetry auto-instrumentation is enabled."""
    return os.getenv("OTEL_ENABLED", "false").lower() in {"1", "true", "yes", "on"}


def setup_otel(default_service_name: str) -> bool:
    """Initialize Tracer/Meter providers and instrument HTTP clients.

    Args:
        default_service_name: Fallback service.name when OTEL_SERVICE_NAME is unset.

    Returns:
        True when instrumentation was applied in this process.
    """
    global _INITIALIZED
    if _INITIALIZED or not otel_enabled():
        return False

    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    except ImportError:
        HTTPXClientInstrumentor = None  # type: ignore[misc, assignment]

    service_name = os.getenv("OTEL_SERVICE_NAME", default_service_name)
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318").rstrip("/")
    traces_endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", f"{endpoint}/v1/traces")
    metrics_endpoint = os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", f"{endpoint}/v1/metrics")
    environment = os.getenv("ENVIRONMENT") or os.getenv("PRODUCT_ENV") or "dev"

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.namespace": os.getenv("OTEL_SERVICE_NAMESPACE", "scigpt"),
            "deployment.environment": environment,
        }
    )

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=traces_endpoint)))
    trace.set_tracer_provider(tracer_provider)

    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=metrics_endpoint),
        export_interval_millis=int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL", "60000")),
    )
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[metric_reader]))

    RequestsInstrumentor().instrument()
    AioHttpClientInstrumentor().instrument()
    if HTTPXClientInstrumentor is not None:
        HTTPXClientInstrumentor().instrument()
    LoggingInstrumentor().instrument(set_logging_format=False)

    _INITIALIZED = True
    return True


def instrument_starlette_app(app: Any) -> Any:
    """Wrap a Starlette/ASGI app with OpenTelemetry middleware when enabled."""
    if not otel_enabled():
        return app
    try:
        from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware
    except ImportError:
        return app

    excluded = os.getenv("OTEL_PYTHON_EXCLUDED_URLS", "/api/ping,/favicon.ico")
    return OpenTelemetryMiddleware(app, excluded_urls=excluded)


def current_otel_ids() -> tuple[str | None, str | None]:
    """Return (trace_id, span_id) hex strings from the active OTel span, if any."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx is None or not ctx.is_valid:
            return None, None
        return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")
    except Exception:
        return None, None
