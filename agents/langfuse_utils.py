from __future__ import annotations

"""Utility helpers for Langfuse instrumentation."""
from typing import Any, Optional

try:  # pragma: no cover - langfuse optional
    from langfuse import Langfuse  # type: ignore
except Exception:  # pragma: no cover - langfuse optional
    Langfuse = None  # type: ignore


def start_trace(
    langfuse: Optional[Langfuse], name: str, inputs: Optional[dict] = None
):
    """Start a Langfuse trace compatible with v2 and v3 clients."""
    if not langfuse:
        return None
    if hasattr(langfuse, "start_trace"):
        return langfuse.start_trace(name=name, input=inputs)
    if hasattr(langfuse, "trace"):
        trace = langfuse.trace(name)
        if inputs:
            trace.log_inputs(inputs)
        return trace
    return None


def start_span(parent: Any, name: str, inputs: Optional[dict] = None):
    """Start a Langfuse span under ``parent`` if supported."""
    if not parent:
        return None
    if hasattr(parent, "start_span"):
        return parent.start_span(name=name, input=inputs)
    if hasattr(parent, "span"):
        span = parent.span(name)
        if inputs:
            span.log_inputs(inputs)
        return span
    return None


def finish_trace(
    trace: Any, outputs: Optional[dict] = None, error: Exception | None = None
) -> None:
    """Finish a Langfuse trace started with ``start_trace``."""
    if not trace:
        return
    if hasattr(trace, "update"):
        if outputs is not None:
            trace.update(output=outputs)
        if error is not None:
            trace.update(level="ERROR", status_message=str(error))
    else:
        if outputs is not None:
            trace.log_outputs(outputs)
        if error is not None:
            trace.log_exception(error)
    if hasattr(trace, "end"):
        trace.end()


def finish_span(
    span: Any, outputs: Optional[dict] = None, error: Exception | None = None
) -> None:
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
    if hasattr(span, "end"):
        span.end()


