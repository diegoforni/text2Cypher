from neo4j import GraphDatabase
import os
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DB

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def check_date_format():
    with driver.session(database=NEO4J_DB) as session:
        # Check the type and structure of documented_create_date
        result = session.run("""
            MATCH ()-[a:ATTACKS]->()
            WHERE a.documented_create_date IS NOT NULL
            RETURN a.documented_create_date AS date,
                   toString(a.documented_create_date) AS date_str
            LIMIT 5
        """)

        print("Sample documented_create_date values:")
        for record in result:
            print(f"Raw: {record['date']}")
            print(f"String: {record['date_str']}")
            print("---")

        # Check if we can access date properties directly
        result = session.run("""
            MATCH ()-[a:ATTACKS]->()
            WHERE a.documented_create_date IS NOT NULL
            RETURN a.documented_create_date.year AS year,
                   a.documented_create_date.month AS month,
                   a.documented_create_date.day AS day,
                   a.documented_create_date.hour AS hour
            LIMIT 3
        """)

        print("\nDirect date property access:")
        for record in result:
            print(f"Year: {record['year']}, Month: {record['month']}, Day: {record['day']}, Hour: {record['hour']}")
            print("---")

        # Test if we can use array access (expected format)
        try:
            result = session.run("""
                MATCH ()-[a:ATTACKS]->()
                WHERE a.documented_create_date IS NOT NULL
                RETURN a.documented_create_date[0] AS first_date
                LIMIT 1
            """)
            print("\nArray access result:")
            for record in result:
                print(f"First element: {record['first_date']}")
        except Exception as e:
            print(f"\nArray access failed: {e}")

        # Test the date.truncate function
        result = session.run("""
            MATCH ()-[a:ATTACKS]->()
            WHERE a.documented_create_date IS NOT NULL
            RETURN date.truncate('month', a.documented_create_date) AS year_month
            LIMIT 3
        """)

        print("\nYear-month using date.truncate:")
        for record in result:
            print(f"Year-Month: {record['year_month']}")

        # Test substring method (from expected queries)
        try:
            result = session.run("""
                MATCH ()-[a:ATTACKS]->()
                WHERE a.documented_create_date IS NOT NULL
                RETURN substring(toString(a.documented_create_date),0,7) AS year_month
                LIMIT 3
            """)
            print("\nYear-month using substring:")
            for record in result:
                print(f"Year-Month: {record['year_month']}")
        except Exception as e:
            print(f"Substring method failed: {e}")

if __name__ == "__main__":
    check_date_format()
    driver.close()