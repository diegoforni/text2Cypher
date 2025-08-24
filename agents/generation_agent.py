"""Generation agent that creates Cypher fragments for each subproblem."""
from typing import List, Optional
import re

from langchain_core.language_models import BaseChatModel
from langfuse import Langfuse
from neo4j import Driver

from config import NEO4J_DB
from .langfuse_utils import start_trace, finish_trace


def _lexical_match(term: str, candidates: List[str]) -> str:
    """Return candidate with longest common substring to term."""
    best = ""
    best_len = 0
    for c in candidates:
        lcs = _longest_common_substring(term.lower(), c.lower())
        if len(lcs) > best_len:
            best, best_len = c, len(lcs)
    return best


def _longest_common_substring(a: str, b: str) -> str:
    m = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    longest = 0
    end = 0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                m[i][j] = m[i - 1][j - 1] + 1
                if m[i][j] > longest:
                    longest = m[i][j]
                    end = i
    return a[end - longest:end]


class GenerationAgent:
    """Generate Cypher query fragments using the schema as context."""

    def __init__(
        self, llm: BaseChatModel, driver: Driver, langfuse: Optional[Langfuse] = None
    ):
        self.llm = llm
        self.driver = driver
        self.langfuse = langfuse

    def _fetch_candidates(self, term: str) -> List[str]:
        query = (
            "MATCH (n) "
            "UNWIND keys(n) AS k "
            "WITH DISTINCT toString(n[k]) AS val "
            "WHERE val IS NOT NULL AND toLower(val) CONTAINS toLower($term) "
            "RETURN val"
        )
        with self.driver.session(database=NEO4J_DB) as session:
            result = session.run(query, term=term)
            return [r["val"] for r in result]

    def _match_db_value(self, value: str) -> str:
        candidates = self._fetch_candidates(value)
        if not candidates:
            return value
        return _lexical_match(value, candidates)

    def _apply_db_matching(self, fragment: str) -> str:
        def repl(match: re.Match[str]) -> str:
            val = match.group(1)
            return f"'{self._match_db_value(val)}'"

        return re.sub(r"'([^']+)'", repl, fragment)

    def generate(self, subproblem: str, schema: str) -> str:
        trace = start_trace(self.langfuse, "generate", {"subproblem": subproblem})
        system_message = (
            "You are a Cypher query expert. Use the provided database schema to solve the given subproblem. "
            "Always produce a complete Cypher fragment ending with a RETURN clause. "
            "Do not include explanations or commentary. "
            "CRITICAL: Do not nest MATCH inside WHERE clauses. "
            "Allowed clauses: MATCH, WHERE, WITH, RETURN, ORDER BY, LIMIT. "
            "Properties like 'technique' or 'protocol' are on relationships [:ATTACKS {technique: \"value\"}]. "
            "Attack relationships follow the pattern (ip:IP)-[:ATTACKS]->(country:Country). "
            "Use WITH or multiple MATCH statements for complex logic and preserve provided values exactly."
        )
        prompt = (
            f"Database schema:\n{schema}\n\n"
            f"Subproblem:\n{subproblem}\n\n"
            "Return ONLY the Cypher fragment."
        )
        response = self.llm.invoke([
            ("system", system_message),
            ("user", prompt),
        ])
        fragment = response.content if hasattr(response, "content") else str(response)
        fragment = self._apply_db_matching(fragment)
        finish_trace(trace, {"fragment": fragment})
        return fragment
