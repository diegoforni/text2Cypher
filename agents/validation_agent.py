"""Validation agent that executes Cypher fragments."""
from typing import Any, Optional, Tuple

from neo4j import Driver

from config import NEO4J_DB
from .langfuse_utils import start_span, finish_span


class ValidationAgent:
    """Execute Cypher fragments and return results or errors."""

    def __init__(self, driver: Driver, trace: Optional[Any] = None):
        self.driver = driver
        self.trace = trace

    def validate(self, fragment: str) -> Tuple[bool, list | str]:
        span = start_span(self.trace, "validate", {"fragment": fragment})
        try:
            if not fragment or not fragment.strip():
                raise ValueError("Empty query")
            with self.driver.session(database=NEO4J_DB) as session:
                result = session.run(fragment)
                rows = [r.data() for r in result]
            finish_span(span, {"rows": rows})
            return True, rows
        except Exception as e:  # pragma: no cover - network errors
            finish_span(span, error=e)
            return False, str(e)
