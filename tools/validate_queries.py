"""
Validate all Cypher queries in queries.json against the configured Neo4j DB.

Usage:
  python tools/validate_queries.py [--db NAME]
"""

import json
import os
from pathlib import Path
import argparse
import time

# Ensure project root is importable when running from tools/
import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

# Load .env early so config picks it up
load_dotenv()

from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD  # type: ignore


def _pick_database(driver, candidates: list[str]) -> str | None:
    for db in [c for c in candidates if c]:
        try:
            with driver.session(database=db) as s:
                s.run("RETURN 1 AS ok").consume()
            return db
        except Exception as e:
            print(f"Database '{db}' not usable: {e}")
            continue
    return None


def main() -> int:
    # .env already loaded at import time
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--db", dest="db", default=None, help="Neo4j database name override")
    parser.add_argument("--wait-seconds", dest="wait", type=int, default=600, help="Max seconds to wait for DB ready")
    args = parser.parse_args()

    queries_path = Path("queries.json")
    if not queries_path.exists():
        print("queries.json not found")
        return 1
    data = json.loads(queries_path.read_text())

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
        max_connection_pool_size=5,
        connection_timeout=10.0,
    )

    env_db = os.getenv("NEO4J_DB")
    candidates = []
    if args.db:
        candidates.append(args.db)
    if env_db and env_db not in candidates:
        candidates.append(env_db)
    if "neo4j" not in candidates:
        candidates.append("neo4j")

    # Poll until a usable database is found or timeout
    db = None
    deadline = time.time() + max(0, args.wait)
    attempt = 0
    while time.time() < deadline and db is None:
        attempt += 1
        print(f"Probing databases (attempt {attempt})...")
        db = _pick_database(driver, candidates)
        if db is None:
            time.sleep(5)
    if db is None:
        print("Could not connect to any candidate database within wait window. Use --db to set explicitly.")
        return 3
    print(f"Using database: {db}")

    ok_count = 0
    for idx, item in enumerate(data, start=1):
        prompt = item.get("prompt")
        cypher = item.get("cypher", "").strip()
        category = item.get("category")
        print(f"\n[{idx}] Category={category} Prompt={prompt}")
        try:
            with driver.session(database=db) as session:
                result = session.run(cypher)
                rows = [r.data() for r in result]
            ok_count += 1
            print(f"  -> OK. Rows returned: {len(rows)}")
        except Exception as e:
            print(f"  -> FAIL. Error: {e}")

    print(f"\nValidated {ok_count}/{len(data)} successfully.")
    return 0 if ok_count == len(data) else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
