"""Validation agent that executes Cypher fragments."""
from typing import Optional, Tuple

from langfuse import Langfuse
from neo4j import Driver


class ValidationAgent:
    """Execute Cypher fragments and return results or errors."""

    def __init__(self, driver: Driver, langfuse: Optional[Langfuse] = None):
        self.driver = driver
        self.langfuse = langfuse

    def validate(self, fragment: str) -> Tuple[bool, list | str]:
        if self.langfuse:
            span = self.langfuse.span("validate")
            span.log_inputs({"fragment": fragment})
        try:
            with self.driver.session() as session:
                result = session.run(fragment)
                rows = [r.data() for r in result]
            if self.langfuse:
                span.log_outputs({"rows": rows})
                span.end()
            return True, rows
        except Exception as e:  # pragma: no cover - network errors
            if self.langfuse:
                span.log_exception(e)
                span.end()
            return False, str(e)
