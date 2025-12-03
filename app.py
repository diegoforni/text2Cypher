"""Flask web interface for text2Cypher."""
from __future__ import annotations

# Ensure the project's .env is loaded as early as possible so a Flask/Werkzeug
# process that imports this module (possibly with a different cwd) gets the
# correct MODEL_PROVIDER and API key environment variables.
import logging
from pathlib import Path
from dotenv import load_dotenv
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
    logging.getLogger(__name__).debug("Loaded .env from %s", _env_path)

import atexit
import json
import logging
import queue
import threading
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from neo4j import Driver, GraphDatabase

from config import NEO4J_DB, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from main import run as run_pipeline

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _first_present(row: Dict[str, object], keys: Iterable[str]) -> object:
    """Return the first non-empty value found for any key."""
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _clean_label(label: str | List[str] | Tuple[str, ...]) -> str:
    """Normalize Neo4j label/type identifiers for display."""
    if isinstance(label, (list, tuple)):
        parts = [_clean_label(x) for x in label if x]
        return ":".join(part for part in parts if part)
    if not isinstance(label, str):
        return str(label or "").strip(":`")
    label = label.strip()
    if label.startswith(":"):
        label = label[1:]
    return label.strip("`")


def _format_type(types: List[str] | str | None) -> str:
    if not types:
        return "ANY"
    if isinstance(types, str):
        return types
    # Deduplicate while preserving order
    seen = []
    for t in types:
        if t not in seen and t is not None:
            seen.append(t)
    return " | ".join(seen) if seen else "ANY"


def _connect_driver() -> Driver:
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _safe_session_kwargs() -> Dict[str, str]:
    if NEO4J_DB:
        return {"database": NEO4J_DB}
    return {}


def _gather_schema_text(driver) -> str:
    """Fetch a descriptive schema string from Neo4j."""
    session_kwargs = _safe_session_kwargs()
    with driver.session(**session_kwargs) as session:
        node_rows = session.run(
            "CALL db.schema.nodeTypeProperties()"
        ).data()
        rel_rows = session.run(
            "CALL db.schema.relTypeProperties()"
        ).data()

    node_props: Dict[str, Dict[str, str]] = {}
    for row in node_rows:
        label = _clean_label(
            _first_present(
                row,
                (
                    "nodeType",
                    "nodeLabel",
                    "nodeLabels",
                    "label",
                    "labels",
                ),
            )
        )
        if not label:
            continue
        props = node_props.setdefault(label, {})
        prop_name = _first_present(row, ("propertyName", "propertyKey", "property"))
        if not prop_name:
            continue
        prop_type = _format_type(row.get("propertyTypes") or row.get("propertyType"))
        props[str(prop_name)] = prop_type

    rel_props: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for row in rel_rows:
        start = _clean_label(
            _first_present(
                row,
                (
                    "sourceNodeType",
                    "startNodeType",
                    "sourceNodeLabels",
                    "startNodeLabels",
                    "sourceNodeLabel",
                    "startNodeLabel",
                    "sourceNode",
                    "startNode",
                    "fromNode",
                    "from",
                ),
            )
        )
        end = _clean_label(
            _first_present(
                row,
                (
                    "targetNodeType",
                    "endNodeType",
                    "targetNodeLabels",
                    "endNodeLabels",
                    "targetNodeLabel",
                    "endNodeLabel",
                    "targetNode",
                    "endNode",
                    "toNode",
                    "to",
                ),
            )
        )
        rel = _clean_label(
            _first_present(
                row,
                (
                    "relType",
                    "relationshipType",
                    "relationshipTypeName",
                    "relationship",
                    "type",
                ),
            )
        )
        if not (start and end and rel):
            continue
        key = (start, rel, end)
        props = rel_props.setdefault(key, {})
        prop_name = row.get("propertyName") or row.get("propertyKey")
        if prop_name:
            prop_type = _format_type(row.get("propertyTypes") or row.get("propertyType"))
            props[str(prop_name)] = prop_type

    node_lines = []
    for label, props in sorted(node_props.items()):
        if props:
            prop_parts = ", ".join(f"{name}: {ptype}" for name, ptype in sorted(props.items()))
            node_lines.append(f"(:{label} {{{prop_parts}}})")
        else:
            node_lines.append(f"(:{label})")

    rel_lines = []
    for (start, rel, end), props in sorted(rel_props.items()):
        if props:
            prop_parts = ", ".join(f"{name}: {ptype}" for name, ptype in sorted(props.items()))
            rel_lines.append(f"(:{start})-[:{rel} {{{prop_parts}}}]->(:{end})")
        else:
            rel_lines.append(f"(:{start})-[:{rel}]->(:{end})")

    if not node_lines or not rel_lines:
        logger.warning(
            "Schema introspection returned incomplete results; falling back to default"
        )
        return _default_schema()

    schema_sections = ["Nodes:"]
    schema_sections.extend(node_lines)
    schema_sections.append("Relationships:")
    schema_sections.extend(rel_lines)
    return "\n".join(schema_sections)


def _default_schema() -> str:
    return (
        "Nodes: (:IP {address, botnet, pulse_ids}), (:Country {name}), (:ASN {number})\n"
        "Relationships: (ip:IP)-[:ATTACKS {technique, protocol, technology, industry, pulse_id, "
        "documented_create_date, documented_modified_date}]->(country:Country)\n"
        "Additional: (ip:IP)-[:FROM]->(country:Country), (ip:IP)-[:BELONGS_TO]->(asn:ASN)"
    )


@lru_cache(maxsize=1)
def get_schema() -> str:
    """Return schema description, caching after first successful fetch."""
    driver = _connect_driver()
    try:
        schema_text = _gather_schema_text(driver)
        logger.info("Loaded schema description from Neo4j")
        return schema_text
    except Exception as exc:  # pragma: no cover - network/environmental failures
        logger.warning("Falling back to default schema: %s", exc)
        return _default_schema()
    finally:
        driver.close()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


def _format_sse(event: str, data: Dict[str, object] | None = None) -> str:
    payload = "" if data is None else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@app.route("/query", methods=["POST"])
def query():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    schema_text = get_schema()

    def event_stream():
        event_queue: "queue.Queue[tuple[str, Dict[str, object] | None]]" = queue.Queue()

        def enqueue(event: str, data: Dict[str, object] | None = None) -> None:
            event_queue.put((event, data))

        def progress(phase: str, status: str, details: Dict[str, object] | None = None) -> None:
            message: Dict[str, object] = {"phase": phase, "status": status}
            if details:
                message.update(details)
            enqueue("progress", message)

        def run_workflow():
            try:
                result = run_pipeline(question, schema_text, progress_callback=progress)
                enqueue(
                    "complete",
                    {
                        "query": result.get("final_query"),
                        "results": result.get("results") or [],
                        "explanation": result.get("explanation"),
                        "error": result.get("error"),
                    },
                )
            except Exception as exc:  # pragma: no cover - runtime errors bubble to UI
                logger.exception("Pipeline execution failed")
                enqueue("error", {"error": str(exc)})
            finally:
                enqueue("end", None)

        worker = threading.Thread(target=run_workflow, name="query-worker", daemon=True)
        worker.start()

        enqueue("progress", {"phase": "submit", "status": "start"})

        while True:
            event, data = event_queue.get()
            if event == "end":
                break
            yield _format_sse(event, data)

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }

    return Response(stream_with_context(event_stream()), headers=headers)


@atexit.register
def _cleanup_driver() -> None:
    # Clear cached schema so the driver is re-created next time the process starts.
    get_schema.cache_clear()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
