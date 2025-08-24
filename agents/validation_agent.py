"""Validation agent that executes Cypher fragments."""
from typing import Optional, Tuple

from langfuse import Langfuse

from .langfuse_utils import start_span, finish_span
from neo4j import Driver


class ValidationAgent:
    """Execute Cypher fragments and return results or errors."""

    def __init__(self, driver: Driver, langfuse: Optional[Langfuse] = None):
        self.driver = driver
        self.langfuse = langfuse

    def validate(self, fragment: str) -> Tuple[bool, list | str]:
        span = start_span(self.langfuse, "validate", {"fragment": fragment})
        try:
            if not fragment or not fragment.strip():
                raise ValueError("Empty query")
            with self.driver.session() as session:
                result = session.run(fragment)
                rows = result.data() or []
            finish_span(span, {"rows": rows})
            return True, rows
        except Exception as e:  # pragma: no cover - network errors
            finish_span(span, error=e)
            return False, str(e)
