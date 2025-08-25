from __future__ import annotations

"""Utility helpers for Langfuse instrumentation."""
from typing import Any, Optional

try:
    from langfuse import Langfuse  # type: ignore
except Exception:  # pragma: no cover - langfuse optional
    Langfuse = None  # type: ignore


def start_span(langfuse: Optional[Langfuse], name: str, inputs: Optional[dict] = None):
    """Start a Langfuse span compatible with v2 and v3 clients."""
    if not langfuse:
        return None
    if hasattr(langfuse, "start_span"):
        return langfuse.start_span(name=name, input=inputs)
    span = langfuse.span(name)
    if inputs:
        span.log_inputs(inputs)
    return span


def finish_span(span: Any, outputs: Optional[dict] = None, error: Exception | None = None) -> None:
    """Finish a Langfuse span started with ``start_span``."""
    if not span:
        return
    if hasattr(span, "update"):
        if outputs is not None:
            span.update(output=outputs)
        if error is not None:
            span.update(level="ERROR", status_message=str(error))
    else:
        if outputs is not None:
            span.log_outputs(outputs)
        if error is not None:
            span.log_exception(error)
    span.end()

