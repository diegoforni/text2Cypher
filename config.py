import os
from dotenv import load_dotenv
from typing import Optional

from langchain_openai import ChatOpenAI
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:  # pragma: no cover - optional dependency
    ChatGoogleGenerativeAI = None  # type: ignore

load_dotenv()

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
NEO4J_DB = os.getenv("NEO4J_DB", "neo4j")

LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")


def get_llm() -> Optional[object]:
    """Return a chat model instance based on environment configuration."""
    if MODEL_PROVIDER == "openai" and OPENAI_API_KEY:
        return ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0)
    if MODEL_PROVIDER == "gemini" and GEMINI_API_KEY and ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY, model="gemini-1.5-pro")
    return None
