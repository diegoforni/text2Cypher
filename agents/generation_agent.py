"""Generation agent that creates Cypher fragments for each subproblem."""
from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langfuse import Langfuse


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

    def __init__(self, llm: BaseChatModel, langfuse: Optional[Langfuse] = None):
        self.llm = llm
        self.langfuse = langfuse

    def generate(self, subproblem: str, schema: str) -> str:
        if self.langfuse:
            span = self.langfuse.span("generate")
            span.log_inputs({"subproblem": subproblem})
        prompt = (
            "Schema: {schema}\n"
            "Subproblem: {subproblem}\n"
            "Generate a Cypher fragment that addresses the subproblem.".format(
                schema=schema, subproblem=subproblem
            )
        )
        response = self.llm.invoke([
            ("system", "You generate Cypher query fragments."),
            ("user", prompt),
        ])
        fragment = response.content if hasattr(response, "content") else str(response)
        if self.langfuse:
            span.log_outputs({"fragment": fragment})
            span.end()
        return fragment
