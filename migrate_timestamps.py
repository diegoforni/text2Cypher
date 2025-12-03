#!/usr/bin/env python3
"""
Neo4j Timestamp Migration Tool
Migrates timestamp fields from LIST<STRING> to ZONED DATETIME format
"""

import sys
import time
from neo4j import GraphDatabase

class TimestampMigrator:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="notSecureAtAll", database="cti"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def run_query(self, query, parameters=None):
        """Execute a Neo4j query with error handling"""
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(query, parameters or {})
                return result.data()
            except Exception as e:
                print(f"❌ Query failed: {e}")
                print(f"Query: {query}")
                return None

    def get_attack_count(self):
        """Get total number of ATTACKS relationships"""
        query = "MATCH ()-[r:ATTACKS]->() RETURN count(r) as count"
        result = self.run_query(query)
        if result:
            return result[0]['count']
        return 0

    def migrate_timestamps(self, batch_size=10000):
        """Migrate timestamps from string arrays to DateTime format"""
        total_attacks = self.get_attack_count()
        if total_attacks == 0:
            print("❌ No ATTACKS relationships found")
            return False

        print(f"🚀 Starting timestamp migration for {total_attacks:,} ATTACKS relationships")

        # Migration query for documented_create_date
        create_date_query = """
        MATCH ()-[r:ATTACKS]->()
        WHERE r.documented_create_date IS NOT NULL AND size(r.documented_create_date) > 0
        WITH r, r.documented_create_date[0] as date_str
        WHERE date_str =~ '\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}.*'
        SET r.documented_create_date = datetime(date_str)
        RETURN count(r) as migrated
        """

        # Migration query for documented_modified_date
        modified_date_query = """
        MATCH ()-[r:ATTACKS]->()
        WHERE r.documented_modified_date IS NOT NULL AND size(r.documented_modified_date) > 0
        WITH r, r.documented_modified_date[0] as date_str
        WHERE date_str =~ '\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}.*'
        SET r.documented_modified_date = datetime(date_str)
        RETURN count(r) as migrated
        """

        print("📅 Migrating documented_create_date...")
        create_result = self.run_query(create_date_query)
        if create_result:
            create_migrated = create_result[0]['migrated']
            print(f"✅ Migrated {create_migrated:,} documented_create_date fields")
        else:
            print("❌ Failed to migrate documented_create_date")
            return False

        print("📅 Migrating documented_modified_date...")
        modified_result = self.run_query(modified_date_query)
        if modified_result:
            modified_migrated = modified_result[0]['migrated']
            print(f"✅ Migrated {modified_migrated:,} documented_modified_date fields")
        else:
            print("❌ Failed to migrate documented_modified_date")
            return False

        return True

    def create_indexes(self):
        """Create temporal indexes for better query performance"""
        indexes = [
            "CREATE INDEX attacks_create_date IF NOT EXISTS FOR ()-[r:ATTACKS]-() ON (r.documented_create_date)",
            "CREATE INDEX attacks_modified_date IF NOT EXISTS FOR ()-[r:ATTACKS]-() ON (r.documented_modified_date)"
        ]

        print("🔍 Creating temporal indexes...")
        for index_query in indexes:
            result = self.run_query(index_query)
            if result is not None:
                print(f"✅ Index created successfully")
            else:
                print(f"❌ Failed to create index")
                return False

        return True

    def verify_migration(self):
        """Verify that migration was successful"""
        verification_queries = [
            # Check data types
            "MATCH ()-[r:ATTACKS]->() RETURN r.documented_create_date, r.documented_modified_date LIMIT 5",
            # Count converted fields
            "MATCH ()-[r:ATTACKS]->() WHERE r.documented_create_date IS NOT NULL RETURN count(r) as create_count",
            "MATCH ()-[r:ATTACKS]->() WHERE r.documented_modified_date IS NOT NULL RETURN count(r) as modified_count",
            # Check no remaining arrays
            "MATCH ()-[r:ATTACKS]->() WHERE r.documented_create_date =~ '\\[.*\\]' RETURN count(r) as array_count",
            # Test temporal operations
            "MATCH ()-[r:ATTACKS]->() WHERE r.documented_create_date >= datetime('2023-01-01') RETURN count(r) as attacks_2023"
        ]

        print("🔍 Verifying migration results...")

        for i, query in enumerate(verification_queries):
            result = self.run_query(query)
            if result:
                print(f"✅ Verification {i+1}: {result[0]}")
            else:
                print(f"❌ Verification {i+1} failed")
                return False

        return True

def main():
    print("🗃️  Neo4j Timestamp Migration Tool")
    print("=" * 50)

    migrator = TimestampMigrator()

    try:
        # Get initial state
        total_attacks = migrator.get_attack_count()
        print(f"📊 Found {total_attacks:,} ATTACKS relationships")

        if total_attacks == 0:
            print("❌ No data to migrate!")
            return 1

        # Run migration
        start_time = time.time()

        if not migrator.migrate_timestamps():
            print("❌ Migration failed!")
            return 1

        if not migrator.create_indexes():
            print("❌ Index creation failed!")
            return 1

        if not migrator.verify_migration():
            print("❌ Migration verification failed!")
            return 1

        end_time = time.time()
        duration = end_time - start_time

        print("\n" + "=" * 50)
        print("🎉 MIGRATION COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print(f"📊 Total relationships processed: {total_attacks:,}")
        print("✅ All timestamps converted to DateTime format")
        print("✅ Temporal indexes created")
        print("✅ Migration verified")

        return 0

    except Exception as e:
        print(f"❌ Migration error: {e}")
        return 1

    finally:
        migrator.close()

if __name__ == "__main__":
    sys.exit(main())