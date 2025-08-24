"""Validation agent that executes Cypher fragments."""
from typing import Optional, Tuple

from langfuse import Langfuse

from .langfuse_utils import start_trace, finish_trace
from neo4j import Driver


class ValidationAgent:
    """Execute Cypher fragments and return results or errors."""

    def __init__(self, driver: Driver, langfuse: Optional[Langfuse] = None):
        self.driver = driver
        self.langfuse = langfuse

    def validate(self, fragment: str) -> Tuple[bool, list | str]:
        trace = start_trace(self.langfuse, "validate", {"fragment": fragment})
        try:
            with self.driver.session() as session:
                result = session.run(fragment)
                rows = [r.data() for r in result]
            finish_trace(trace, {"rows": rows})
            return True, rows
        except Exception as e:  # pragma: no cover - network errors
            finish_trace(trace, error=e)
            return False, str(e)
