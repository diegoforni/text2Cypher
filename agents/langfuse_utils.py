from __future__ import annotations

"""Utility helpers for Langfuse instrumentation."""
from typing import Any, Optional

try:
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
    trace = langfuse.trace(name)
    if inputs:
        trace.log_inputs(inputs)
    return trace


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
    trace.end()


