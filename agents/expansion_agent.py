"""Expansion agent that clarifies and enriches user requests."""
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langfuse import Langfuse

from .langfuse_utils import start_span, finish_span


class ExpansionAgent:
    """Ask clarifying questions and produce a schema-grounded description."""

    def __init__(self, llm: BaseChatModel, langfuse: Optional[Langfuse] = None):
        self.llm = llm
        self.langfuse = langfuse

    def expand(self, request: str, schema: str) -> str:
        system_message = (
            "You are a data analysis expert specializing in graph databases and cybersecurity data. "
            "Your role is to analyze user questions and provide comprehensive context that will help "
            "other agents generate accurate Cypher queries."
        )
        prompt = f"""
Analyze this question and provide detailed insights:

Question: {request}
Schema: {schema}

Please provide:
1. INTENT: What is the user trying to find out?
2. KEY_ENTITIES: What nodes/relationships are involved?
3. FILTERS: What conditions need to be applied?
4. OUTPUT_FORMAT: How should results be presented?
5. COMPLEXITY_NOTES: Any special considerations?
6. SUGGESTED_APPROACH: Recommended query structure
7. Respect given values, do not change them, values may take more than 1 word, use the complete given value

Format your response as JSON with these keys.
"""
        span = start_span(self.langfuse, "expand", {"request": request})
        response = self.llm.invoke([
            ("system", system_message),
            ("user", prompt),
        ])
        expanded = response.content if hasattr(response, "content") else str(response)
        finish_span(span, {"expanded": expanded})
        return expanded
