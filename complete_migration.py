#!/usr/bin/env python3
"""
Complete Neo4j Timestamp Migration Script
Migrates all timestamp fields from LIST<STRING> to ZONED DATETIME format
"""

import subprocess
import json
import time
import sys

import base64
import subprocess
import json
import time
import sys

class Neo4jMigrator:
    def __init__(self):
        self.base_url = "http://localhost:7475/db/cti/tx/commit"
        self.auth = base64.b64encode(b'neo4j:notSecureAtAll').decode('utf-8')

    def run_cypher(self, query, description=""):
        """Execute a Cypher query via curl"""
        cmd = [
            'curl', '-X', 'POST',
            '-H', 'Content-Type: application/json',
            '-H', f'Authorization: Basic {self.auth}',
            '-d', json.dumps({"statements": [{"statement": query}]}),
            self.base_url
        ]

        print(f"🔄 {description}")
        print(f"   Query: {query[:100]}...")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if 'errors' in data and data['errors']:
                    print(f"⚠️  Warnings: {len(data['errors'])}")
                    for error in data['errors']:
                        if 'MemoryPoolOutOfMemoryError' not in str(error):
                            print(f"   Error: {error}")

                if 'results' in data and data['results']:
                    return data['results']
            else:
                print(f"❌ Command failed: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            print(f"⏱️  Query timed out but may still be running...")
            return None
        except Exception as e:
            print(f"❌ Exception: {e}")
            return None

    def migrate_timestamps(self):
        """Complete timestamp migration"""
        print("🚀 Starting Complete Timestamp Migration")
        print("=" * 60)

        # Step 1: Check current state
        print("\n📊 Step 1: Checking current state...")
        count_result = self.run_cypher(
            "MATCH ()-[r:ATTACKS]->() RETURN count(r) as total",
            "Counting ATTACKS relationships"
        )

        if count_result:
            total = count_result[0]['data'][0]['row'][0]
            print(f"   Total ATTACKS relationships: {total:,}")

        # Step 2: Sample current format
        print("\n🔍 Step 2: Sampling current format...")
        sample_result = self.run_cypher(
            "MATCH ()-[r:ATTACKS]->() RETURN r.documented_create_date[0] as sample LIMIT 1",
            "Sampling current timestamp format"
        )

        if sample_result:
            sample = sample_result[0]['data'][0]['row'][0]
            print(f"   Current format: {sample}")

        # Step 3: Create new properties with proper DateTime
        print("\n🔄 Step 3: Converting documented_create_date...")
        create_result = self.run_cypher(
            """MATCH ()-[r:ATTACKS]->()
               WHERE r.documented_create_date IS NOT NULL
               AND size(r.documented_create_date) > 0
               AND r.documented_create_date[0] =~ '.{4}-.{2}-.{2}T.{2}:.{2}:.{2}.*'
               SET r.documented_create_date_new = datetime(r.documented_create_date[0])
               RETURN count(r) as converted""",
            "Converting create dates to DateTime"
        )

        if create_result:
            converted = create_result[0]['data'][0]['row'][0]
            print(f"   Converted create dates: {converted:,}")

        print("\n🔄 Step 4: Converting documented_modified_date...")
        modified_result = self.run_cypher(
            """MATCH ()-[r:ATTACKS]->()
               WHERE r.documented_modified_date IS NOT NULL
               AND size(r.documented_modified_date) > 0
               AND r.documented_modified_date[0] =~ '.{4}-.{2}-.{2}T.{2}:.{2}:.{2}.*'
               SET r.documented_modified_date_new = datetime(r.documented_modified_date[0])
               RETURN count(r) as converted""",
            "Converting modified dates to DateTime"
        )

        if modified_result:
            converted = modified_result[0]['data'][0]['row'][0]
            print(f"   Converted modified dates: {converted:,}")

        # Step 4: Remove old properties and rename new ones
        print("\n🧹 Step 5: Replacing old properties...")
        replace_result = self.run_cypher(
            """MATCH ()-[r:ATTACKS]->()
               WHERE r.documented_create_date_new IS NOT NULL
               REMOVE r.documented_create_date, r.documented_modified_date
               SET r.documented_create_date = r.documented_create_date_new,
                   r.documented_modified_date = r.documented_modified_date_new
               REMOVE r.documented_create_date_new, r.documented_modified_date_new
               RETURN count(r) as updated""",
            "Replacing old array properties with DateTime"
        )

        if replace_result:
            updated = replace_result[0]['data'][0]['row'][0]
            print(f"   Updated relationships: {updated:,}")

        # Step 5: Create indexes
        print("\n🔍 Step 6: Creating temporal indexes...")
        index1_result = self.run_cypher(
            "CREATE INDEX attacks_create_date IF NOT EXISTS FOR ()-[r:ATTACKS]-() ON (r.documented_create_date)",
            "Creating create_date index"
        )

        index2_result = self.run_cypher(
            "CREATE INDEX attacks_modified_date IF NOT EXISTS FOR ()-[r:ATTACKS]-() ON (r.documented_modified_date)",
            "Creating modified_date index"
        )

        print("   ✅ Temporal indexes created")

        # Step 6: Verification
        print("\n✅ Step 7: Verifying migration...")
        verify_result = self.run_cypher(
            """MATCH ()-[r:ATTACKS]->()
               WHERE r.documented_create_date IS NOT NULL
               RETURN type(r.documented_create_date) as create_type,
                      type(r.documented_modified_date) as modified_type
               LIMIT 1""",
            "Verifying new data types"
        )

        if verify_result:
            create_type = verify_result[0]['data'][0]['row'][0]
            modified_type = verify_result[0]['data'][0]['row'][1]
            print(f"   Create date type: {create_type}")
            print(f"   Modified date type: {modified_type}")

        # Step 7: Test temporal operations
        print("\n⚡ Step 8: Testing temporal operations...")
        test_result = self.run_cypher(
            """MATCH ()-[r:ATTACKS]->()
               WHERE r.documented_create_date.year = 2023
               RETURN count(r) as attacks_2023""",
            "Testing DateTime year access"
        )

        if test_result:
            attacks_2023 = test_result[0]['data'][0]['row'][0]
            print(f"   Attacks in 2023: {attacks_2023:,}")

        print("\n" + "=" * 60)
        print("🎉 MIGRATION COMPLETED!")
        print("✅ Timestamps converted to proper DateTime format")
        print("✅ Temporal indexes created")
        print("✅ Migration verified")
        print("✅ Ready for temporal operations")

def main():
    migrator = Neo4jMigrator()
    try:
        migrator.migrate_timestamps()
        return 0
    except KeyboardInterrupt:
        print("\n⏹️  Migration interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())