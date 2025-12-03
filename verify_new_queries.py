#!/usr/bin/env python3
"""Verify three new queries:
16. Primary target country for Data Destruction (T1485)
17. Countries receiving attacks most regularly (by active days)
18. Top 10 source IPs with most persistent activity (distinct active days + span)
"""
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DB

queries = [
    (
        "Q16 - Primary target of Data Destruction (T1485)",
        """
MATCH (:IP)-[a:ATTACKS]->(c:Country)
WHERE (a.technique STARTS WITH 'T1485') OR toLower(a.technique) CONTAINS 'data destruct'
RETURN c.name AS country, count(*) AS attacks
ORDER BY attacks DESC
LIMIT 1
        """,
    ),
    (
        "Q17 - Countries receiving attacks most regularly (active days)",
        """
MATCH (:IP)-[a:ATTACKS]->(c:Country)
WHERE a.documented_create_date IS NOT NULL
WITH c.name AS country, count(DISTINCT date(a.documented_create_date)) AS active_days
RETURN country, active_days
ORDER BY active_days DESC
LIMIT 10
        """,
    ),
    (
        "Q18 - Top 10 source IPs with most persistent activity",
        """
MATCH (ip:IP)-[a:ATTACKS]->(:Country)
WHERE a.documented_create_date IS NOT NULL
WITH ip.address AS ip, count(DISTINCT date(a.documented_create_date)) AS active_days,
     min(a.documented_create_date) AS first, max(a.documented_create_date) AS last
WITH ip, active_days, first, last, duration.between(first, last) AS span
RETURN ip, active_days, first, last, span.days AS span_days
ORDER BY active_days DESC, span_days DESC
LIMIT 10
        """,
    ),
]


def run():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session(database=NEO4J_DB) as session:
        print("=" * 80)
        print("Running verification for new queries")
        print("=" * 80)
        for name, q in queries:
            print(f"\n{name}")
            print("-" * 80)
            try:
                result = session.run(q.strip())
                records = list(result)
                if records:
                    for r in records:
                        print(dict(r))
                else:
                    print("No results")
            except Exception as e:
                print("ERROR:", e)
    driver.close()

if __name__ == '__main__':
    run()
