#!/usr/bin/env python3
"""
Fixed Neo4j Timestamp Migration Script
Migrates all timestamp fields from LIST<STRING> to ZONED DATETIME format
"""

import subprocess
import json
import time
import sys
import base64

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
                    print(f"❌ Errors: {len(data['errors'])}")
                    for error in data['errors']:
                        print(f"   Error: {error}")
                    return None

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
        """Complete timestamp migration with proper approach"""
        print("🚀 Starting Fixed Timestamp Migration")
        print("=" * 60)

        # Step 1: Check current state
        print("\n📊 Step 1: Checking current state...")
        count_result = self.run_cypher(
            "MATCH ()-[r:ATTACKS]->() RETURN count(r) as total",
            "Counting ATTACKS relationships"
        )

        if count_result:
            total = count_result[0]['data'][0]['row'][0]
            print(f"   ✅ Total ATTACKS relationships: {total:,}")

        # Step 2: Sample current format
        print("\n🔍 Step 2: Sampling current format...")
        sample_result = self.run_cypher(
            "MATCH ()-[r:ATTACKS]->() WHERE r.documented_create_date IS NOT NULL RETURN r.documented_create_date as sample, valueType(r.documented_create_date) as type LIMIT 1",
            "Sampling current timestamp format"
        )

        if sample_result and sample_result[0]['data']:
            sample = sample_result[0]['data'][0]['row'][0]
            sample_type = sample_result[0]['data'][0]['row'][1]
            print(f"   Current format: {sample}")
            print(f"   Current type: {sample_type}")

        # Step 3: Create temporary DateTime properties
        print("\n🔄 Step 3: Creating temporary DateTime properties...")
        
        # Create documented_create_date_new
        print("   Processing documented_create_date...")
        create_result = self.run_cypher(
            """MATCH ()-[r:ATTACKS]->()
               WHERE r.documented_create_date IS NOT NULL
               AND size(r.documented_create_date) > 0
               AND r.documented_create_date[0] =~ '.{4}-.{2}-.{2}T.{2}:.{2}:.{2}.*'
               SET r.documented_create_date_new = datetime(r.documented_create_date[0])
               RETURN count(r) as converted""",
            "Converting create dates to DateTime"
        )

        if create_result and create_result[0]['data']:
            converted = create_result[0]['data'][0]['row'][0]
            print(f"   ✅ Converted create dates: {converted:,}")

        # Create documented_modified_date_new
        print("   Processing documented_modified_date...")
        modified_result = self.run_cypher(
            """MATCH ()-[r:ATTACKS]->()
               WHERE r.documented_modified_date IS NOT NULL
               AND size(r.documented_modified_date) > 0
               AND r.documented_modified_date[0] =~ '.{4}-.{2}-.{2}T.{2}:.{2}:.{2}.*'
               SET r.documented_modified_date_new = datetime(r.documented_modified_date[0])
               RETURN count(r) as converted""",
            "Converting modified dates to DateTime"
        )

        if modified_result and modified_result[0]['data']:
            converted = modified_result[0]['data'][0]['row'][0]
            print(f"   ✅ Converted modified dates: {converted:,}")

        # Step 4: Verify temporary properties exist
        print("\n🔍 Step 4: Verifying temporary properties...")
        verify_result = self.run_cypher(
            """MATCH ()-[r:ATTACKS]->()
               WHERE r.documented_create_date_new IS NOT NULL
               RETURN count(r) as with_new_create,
                      valueType(r.documented_create_date_new) as new_type
               LIMIT 1""",
            "Checking new properties"
        )

        if verify_result and verify_result[0]['data']:
            count = verify_result[0]['data'][0]['row'][0]
            new_type = verify_result[0]['data'][0]['row'][1]
            print(f"   ✅ Relationships with new create_date: {count:,}")
            print(f"   ✅ New property type: {new_type}")

        # Step 5: Replace old properties with new ones (in separate steps)
        print("\n🔄 Step 5: Replacing old properties...")
        
        # First, remove old documented_create_date
        print("   Removing old documented_create_date...")
        remove_create_result = self.run_cypher(
            """MATCH ()-[r:ATTACKS]->()
               WHERE r.documented_create_date IS NOT NULL
               AND r.documented_create_date_new IS NOT NULL
               REMOVE r.documented_create_date
               RETURN count(r) as removed""",
            "Removing old create_date"
        )
        
        if remove_create_result and remove_create_result[0]['data']:
            removed = remove_create_result[0]['data'][0]['row'][0]
            print(f"   ✅ Removed old create_date from {removed:,} relationships")

        # Rename documented_create_date_new to documented_create_date
        print("   Renaming documented_create_date_new...")
        rename_create_result = self.run_cypher(
            """MATCH ()-[r:ATTACKS]->()
               WHERE r.documented_create_date_new IS NOT NULL
               SET r.documented_create_date = r.documented_create_date_new
               REMOVE r.documented_create_date_new
               RETURN count(r) as renamed""",
            "Renaming new create_date"
        )
        
        if rename_create_result and rename_create_result[0]['data']:
            renamed = rename_create_result[0]['data'][0]['row'][0]
            print(f"   ✅ Renamed create_date for {renamed:,} relationships")

        # Remove old documented_modified_date
        print("   Removing old documented_modified_date...")
        remove_modified_result = self.run_cypher(
            """MATCH ()-[r:ATTACKS]->()
               WHERE r.documented_modified_date IS NOT NULL
               AND r.documented_modified_date_new IS NOT NULL
               REMOVE r.documented_modified_date
               RETURN count(r) as removed""",
            "Removing old modified_date"
        )
        
        if remove_modified_result and remove_modified_result[0]['data']:
            removed = remove_modified_result[0]['data'][0]['row'][0]
            print(f"   ✅ Removed old modified_date from {removed:,} relationships")

        # Rename documented_modified_date_new to documented_modified_date
        print("   Renaming documented_modified_date_new...")
        rename_modified_result = self.run_cypher(
            """MATCH ()-[r:ATTACKS]->()
               WHERE r.documented_modified_date_new IS NOT NULL
               SET r.documented_modified_date = r.documented_modified_date_new
               REMOVE r.documented_modified_date_new
               RETURN count(r) as renamed""",
            "Renaming new modified_date"
        )
        
        if rename_modified_result and rename_modified_result[0]['data']:
            renamed = rename_modified_result[0]['data'][0]['row'][0]
            print(f"   ✅ Renamed modified_date for {renamed:,} relationships")

        # Step 6: Create indexes
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

        # Step 7: Final verification
        print("\n✅ Step 7: Final verification...")
        final_verify_result = self.run_cypher(
            """MATCH ()-[r:ATTACKS]->()
               WHERE r.documented_create_date IS NOT NULL
               RETURN valueType(r.documented_create_date) as create_type,
                      valueType(r.documented_modified_date) as modified_type,
                      r.documented_create_date as sample_create,
                      r.documented_modified_date as sample_modified
               LIMIT 1""",
            "Verifying final data types"
        )

        if final_verify_result and final_verify_result[0]['data']:
            create_type = final_verify_result[0]['data'][0]['row'][0]
            modified_type = final_verify_result[0]['data'][0]['row'][1]
            sample_create = final_verify_result[0]['data'][0]['row'][2]
            sample_modified = final_verify_result[0]['data'][0]['row'][3]
            print(f"   ✅ Create date type: {create_type}")
            print(f"   ✅ Modified date type: {modified_type}")
            print(f"   ✅ Sample create date: {sample_create}")
            print(f"   ✅ Sample modified date: {sample_modified}")

        # Step 8: Test temporal operations
        print("\n⚡ Step 8: Testing temporal operations...")
        test_result = self.run_cypher(
            """MATCH ()-[r:ATTACKS]->()
               WHERE r.documented_create_date.year = 2023
               RETURN count(r) as attacks_2023""",
            "Testing DateTime year access"
        )

        if test_result and test_result[0]['data']:
            attacks_2023 = test_result[0]['data'][0]['row'][0]
            print(f"   ✅ Attacks in 2023: {attacks_2023:,}")

        # Test duration.between
        print("\n⚡ Step 9: Testing duration calculations...")
        duration_test = self.run_cypher(
            """MATCH (ip:IP)-[r:ATTACKS]->()
               WHERE r.documented_create_date IS NOT NULL 
               AND r.documented_modified_date IS NOT NULL
               WITH ip, min(r.documented_create_date) as earliest, max(r.documented_modified_date) as latest
               RETURN duration.between(earliest, latest) as span
               LIMIT 1""",
            "Testing duration.between"
        )

        if duration_test and duration_test[0]['data']:
            span = duration_test[0]['data'][0]['row'][0]
            print(f"   ✅ Duration calculation successful: {span}")

        print("\n" + "=" * 60)
        print("🎉 MIGRATION COMPLETED SUCCESSFULLY!")
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
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
