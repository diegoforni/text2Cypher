"""Configuration helpers for environment variables and LLM selection.

This module now attempts to locate a top-level `.env` file reliably using
python-dotenv's finder and explicitly strips surrounding single/double
quotes from API key values (a common source of "not configured" errors
when keys are written like OPENAI_API_KEY='sk-...').
"""

import logging
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from typing import Optional, List, Tuple, Any
import threading
import json as _json
import requests

from langchain_openai import ChatOpenAI
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:  # pragma: no cover - optional dependency
    ChatGoogleGenerativeAI = None  # type: ignore

logger = logging.getLogger(__name__)

# Try to find a .env file automatically (project root, parent dirs, etc.)
_dotenv_path = find_dotenv()  # returns '' if not found
if _dotenv_path:
    load_dotenv(_dotenv_path)
    logger.debug("Loaded environment from %s", _dotenv_path)
else:
    # If find_dotenv didn't locate a file (e.g., process cwd differs), try
    # loading a .env file next to this config module (project root).
    _project_env = Path(__file__).resolve().parent / ".env"
    if _project_env.exists():
        load_dotenv(_project_env)
        logger.debug("Loaded environment from project .env at %s", _project_env)
    else:
        # Fallback to default behaviour (attempt to load from cwd/.env)
        load_dotenv()
        logger.debug("No .env found by find_dotenv(); called load_dotenv() fallback")


def _strip_quotes(value: str | None) -> str | None:
    """Remove surrounding single or double quotes from a string value."""
    if not isinstance(value, str):
        return value
    v = value.strip()
    if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
        return v[1:-1]
    return v

MODEL_PROVIDER = _strip_quotes(os.getenv("MODEL_PROVIDER", "openai")).lower()
OPENAI_API_KEY = _strip_quotes(os.getenv("OPENAI_API_KEY"))
GEMINI_API_KEY = _strip_quotes(os.getenv("GEMINI_API_KEY"))
# Optional: multiple Gemini keys, comma/semicolon/newline separated or JSON array
_GEMINI_KEYS_RAW = os.getenv("GEMINI_API_KEYS", "").strip()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_0")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
NEO4J_DB = os.getenv("NEO4J_DB", "neo4j")

LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")

# Helpful warnings when config looks mis-specified
if MODEL_PROVIDER == "gemini" and ChatGoogleGenerativeAI is None:
    logger.warning(
        "MODEL_PROVIDER=gemini but langchain_google_genai is not importable. "
        "Install 'langchain-google-genai' and its dependencies or set MODEL_PROVIDER=openai."
    )
if MODEL_PROVIDER == "openai" and not OPENAI_API_KEY:
    logger.warning("MODEL_PROVIDER=openai but OPENAI_API_KEY is not set.")


def _parse_keys(value: str) -> List[str]:
    if not value:
        return []
    value = value.strip()
    # Try JSON array first
    try:
        data = _json.loads(value)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass
    # Fallback to splitting by common delimiters
    parts: List[str] = []
    for sep in [",", "\n", ";", " "]:
        if sep in value:
            parts = [p.strip() for p in value.split(sep)]
            break
    if not parts:
        parts = [value]
    # Remove inline comments and empties
    cleaned: List[str] = []
    for p in parts:
        if "#" in p:
            p = p.split("#", 1)[0].strip()
        if p:
            cleaned.append(p)
    return cleaned


_GEMINI_KEYS: List[str] = _parse_keys(_GEMINI_KEYS_RAW)


def get_llm() -> Optional[object]:
    """Return a chat model instance based on environment configuration.

    When MODEL_PROVIDER == 'gemini', supports either a single GEMINI_API_KEY
    or multiple keys via GEMINI_API_KEYS (round-robin, no back-to-back reuse).
    """
    if MODEL_PROVIDER == "openai" and OPENAI_API_KEY:
        # Include reasoning controls for GPT models on every call
        return ChatOpenAI(
            api_key=OPENAI_API_KEY,
            temperature=0,
            model='gpt-5-nano',
            model_kwargs={
                "reasoning": {"effort": "minimal"},
            },
        )
    if MODEL_PROVIDER == "gemini" and ChatGoogleGenerativeAI:
        # Prefer multiple-key rotation when provided
        keys: List[str] = _GEMINI_KEYS[:] if _GEMINI_KEYS else ([GEMINI_API_KEY] if GEMINI_API_KEY else [])
        keys = [k for k in keys if k]
        if not keys:
            return None
        if len(keys) == 1:
            return ChatGoogleGenerativeAI(api_key=keys[0], model="gemini-2.5-flash")
        return _RoundRobinGeminiLLM(keys, model="gemini-2.5-flash")
    if MODEL_PROVIDER == "ollama":
        return _ChatOllamaCompat(OLLAMA_HOST, OLLAMA_MODEL, temperature=0)
    return None


class _OllamaResponse:
    def __init__(self, content: str, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.content = content
        # Match LangChain-style metadata expected by TokenCountingLLM
        self.response_metadata = {
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        }


class _ChatOllamaCompat:
    """Minimal compatibility wrapper for Ollama's /api/chat.

    Provides an `.invoke(messages)` API returning an object with `.content`
    and `.response_metadata.token_usage` for token counting.
    """

    def __init__(self, host: str, model: str, temperature: float = 0.0):
        self.host = host
        self.model = model
        self.temperature = temperature
        # Expose common attributes used elsewhere
        self.model_name = model

    def invoke(self, messages: Any) -> _OllamaResponse:
        # Normalize messages into list of {role, content}
        payload_msgs = []
        if isinstance(messages, list):
            for m in messages:
                if isinstance(m, tuple) and len(m) == 2:
                    role, content = m
                    payload_msgs.append({"role": str(role), "content": str(content)})
                elif isinstance(m, dict) and "role" in m and "content" in m:
                    payload_msgs.append({"role": str(m["role"]), "content": str(m["content"])})
                else:
                    # Fallback: best-effort mapping
                    payload_msgs.append({"role": "user", "content": str(m)})
        else:
            payload_msgs = [{"role": "user", "content": str(messages)}]

        body = {
            "model": self.model,
            "messages": payload_msgs,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        url = f"{self.host}/api/chat"
        try:
            resp = requests.post(url, json=body, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            # Provide a consistent failure mode
            return _OllamaResponse(content=f"[ollama error] {e}", prompt_tokens=0, completion_tokens=0)

        msg = (data or {}).get("message") or {}
        content = msg.get("content", "")
        prompt_tokens = int((data or {}).get("prompt_eval_count") or 0)
        completion_tokens = int((data or {}).get("eval_count") or 0)
        return _OllamaResponse(content=content, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


class _RoundRobinGeminiLLM:
    """Thread-safe round-robin wrapper for Gemini with multiple API keys.

    - Ensures consecutive calls do not use the same key when 2+ keys provided.
    - Exposes `.invoke(messages)` and minimal metadata for compatibility.
    - Stores `last_api_key` used for TokenCountingLLM to compute token usage.
    """

    def __init__(self, api_keys: List[str], model: str):
        self._keys = api_keys[:]
        self._n = len(self._keys)
        self._idx = -1
        self._last_key: Optional[str] = None
        self._lock = threading.Lock()
        self.model = model
        self.model_name = model
        # For external access (TokenCountingLLM)
        self.last_api_key: Optional[str] = None

    def _next_key(self) -> str:
        with self._lock:
            # Standard round-robin
            self._idx = (self._idx + 1) % self._n
            candidate = self._keys[self._idx]
            # If two or more keys and candidate equals last, advance once more
            if self._n > 1 and self._last_key and candidate == self._last_key:
                self._idx = (self._idx + 1) % self._n
                candidate = self._keys[self._idx]
            self._last_key = candidate
            return candidate

    def invoke(self, messages: Any):
        if not ChatGoogleGenerativeAI:
            raise RuntimeError("langchain-google-genai is not installed")
        key = self._next_key()
        self.last_api_key = key
        llm = ChatGoogleGenerativeAI(api_key=key, model=self.model)
        return llm.invoke(messages)
