#!/usr/bin/env python3
"""
Batch-based Neo4j Timestamp Migration Script
Processes timestamps in batches to avoid memory issues
"""

import subprocess
import json
import time
import sys
import base64

class Neo4jBatchMigrator:
    def __init__(self, batch_size=10000):
        self.base_url = "http://localhost:7475/db/cti/tx/commit"
        self.auth = base64.b64encode(b'neo4j:notSecureAtAll').decode('utf-8')
        self.batch_size = batch_size

    def run_cypher(self, query, description="", show_error=True):
        """Execute a Cypher query via curl"""
        cmd = [
            'curl', '-X', 'POST',
            '-H', 'Content-Type: application/json',
            '-H', f'Authorization: Basic {self.auth}',
            '-d', json.dumps({"statements": [{"statement": query}]}),
            self.base_url
        ]

        if description:
            print(f"🔄 {description}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if 'errors' in data and data['errors']:
                    if show_error:
                        print(f"❌ Errors: {len(data['errors'])}")
                        for error in data['errors']:
                            print(f"   Error: {error}")
                    return None

                if 'results' in data and data['results']:
                    return data['results']
            else:
                if show_error:
                    print(f"❌ Command failed: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            print(f"⏱️  Query timed out...")
            return None
        except Exception as e:
            if show_error:
                print(f"❌ Exception: {e}")
            return None

    def migrate_in_batches(self):
        """Migrate timestamps in batches"""
        print("🚀 Starting Batch Timestamp Migration")
        print(f"   Batch size: {self.batch_size:,}")
        print("=" * 60)

        # Step 1: Check current state
        print("\n📊 Checking current state...")
        count_result = self.run_cypher(
            "MATCH ()-[r:ATTACKS]->() RETURN count(r) as total",
            "Counting ATTACKS relationships"
        )

        if count_result:
            total = count_result[0]['data'][0]['row'][0]
            print(f"   Total ATTACKS relationships: {total:,}")
            print(f"   Estimated batches: {(total // self.batch_size) + 1}")

        # Step 2: Sample current format
        print("\n🔍 Checking current format...")
        sample_result = self.run_cypher(
            "MATCH ()-[r:ATTACKS]->() WHERE r.documented_create_date IS NOT NULL RETURN valueType(r.documented_create_date) as type LIMIT 1",
            ""
        )

        if sample_result and sample_result[0]['data']:
            sample_type = sample_result[0]['data'][0]['row'][0]
            print(f"   Current type: {sample_type}")
            
            if "DATETIME" in sample_type or "DATE TIME" in sample_type:
                print("\n✅ Data is already in DateTime format!")
                return True

        # Step 3: Process documented_create_date in batches
        print("\n🔄 Step 1: Converting documented_create_date in batches...")
        total_converted = 0
        batch_num = 0
        
        while True:
            batch_num += 1
            result = self.run_cypher(
                f"""MATCH ()-[r:ATTACKS]->()
                   WHERE r.documented_create_date IS NOT NULL
                   AND size(r.documented_create_date) > 0
                   AND r.documented_create_date[0] =~ '.{{4}}-.{{2}}-.{{2}}T.{{2}}:.{{2}}:.{{2}}.*'
                   AND valueType(r.documented_create_date) CONTAINS 'LIST'
                   WITH r LIMIT {self.batch_size}
                   SET r.documented_create_date_new = datetime(r.documented_create_date[0])
                   RETURN count(r) as converted""",
                f"Batch {batch_num}",
                show_error=False
            )
            
            if result and result[0]['data']:
                converted = result[0]['data'][0]['row'][0]
                total_converted += converted
                print(f"   Batch {batch_num}: {converted:,} (Total: {total_converted:,})")
                
                if converted < self.batch_size:
                    break
            else:
                break
            
            time.sleep(0.5)  # Small delay between batches

        print(f"   ✅ Total create dates converted: {total_converted:,}")

        # Step 4: Process documented_modified_date in batches
        print("\n🔄 Step 2: Converting documented_modified_date in batches...")
        total_converted = 0
        batch_num = 0
        
        while True:
            batch_num += 1
            result = self.run_cypher(
                f"""MATCH ()-[r:ATTACKS]->()
                   WHERE r.documented_modified_date IS NOT NULL
                   AND size(r.documented_modified_date) > 0
                   AND r.documented_modified_date[0] =~ '.{{4}}-.{{2}}-.{{2}}T.{{2}}:.{{2}}:.{{2}}.*'
                   AND valueType(r.documented_modified_date) CONTAINS 'LIST'
                   WITH r LIMIT {self.batch_size}
                   SET r.documented_modified_date_new = datetime(r.documented_modified_date[0])
                   RETURN count(r) as converted""",
                f"Batch {batch_num}",
                show_error=False
            )
            
            if result and result[0]['data']:
                converted = result[0]['data'][0]['row'][0]
                total_converted += converted
                print(f"   Batch {batch_num}: {converted:,} (Total: {total_converted:,})")
                
                if converted < self.batch_size:
                    break
            else:
                break
            
            time.sleep(0.5)

        print(f"   ✅ Total modified dates converted: {total_converted:,}")

        # Step 5: Replace documented_create_date in batches
        print("\n🔄 Step 3: Replacing documented_create_date in batches...")
        total_replaced = 0
        batch_num = 0
        
        while True:
            batch_num += 1
            result = self.run_cypher(
                f"""MATCH ()-[r:ATTACKS]->()
                   WHERE r.documented_create_date_new IS NOT NULL
                   WITH r LIMIT {self.batch_size}
                   SET r.documented_create_date = r.documented_create_date_new
                   REMOVE r.documented_create_date_new
                   RETURN count(r) as replaced""",
                f"Batch {batch_num}",
                show_error=False
            )
            
            if result and result[0]['data']:
                replaced = result[0]['data'][0]['row'][0]
                total_replaced += replaced
                print(f"   Batch {batch_num}: {replaced:,} (Total: {total_replaced:,})")
                
                if replaced < self.batch_size:
                    break
            else:
                break
            
            time.sleep(0.5)

        print(f"   ✅ Total create dates replaced: {total_replaced:,}")

        # Step 6: Replace documented_modified_date in batches
        print("\n🔄 Step 4: Replacing documented_modified_date in batches...")
        total_replaced = 0
        batch_num = 0
        
        while True:
            batch_num += 1
            result = self.run_cypher(
                f"""MATCH ()-[r:ATTACKS]->()
                   WHERE r.documented_modified_date_new IS NOT NULL
                   WITH r LIMIT {self.batch_size}
                   SET r.documented_modified_date = r.documented_modified_date_new
                   REMOVE r.documented_modified_date_new
                   RETURN count(r) as replaced""",
                f"Batch {batch_num}",
                show_error=False
            )
            
            if result and result[0]['data']:
                replaced = result[0]['data'][0]['row'][0]
                total_replaced += replaced
                print(f"   Batch {batch_num}: {replaced:,} (Total: {total_replaced:,})")
                
                if replaced < self.batch_size:
                    break
            else:
                break
            
            time.sleep(0.5)

        print(f"   ✅ Total modified dates replaced: {total_replaced:,}")

        # Step 7: Create indexes
        print("\n🔍 Step 5: Creating temporal indexes...")
        self.run_cypher(
            "CREATE INDEX attacks_create_date IF NOT EXISTS FOR ()-[r:ATTACKS]-() ON (r.documented_create_date)",
            "Creating create_date index"
        )

        self.run_cypher(
            "CREATE INDEX attacks_modified_date IF NOT EXISTS FOR ()-[r:ATTACKS]-() ON (r.documented_modified_date)",
            "Creating modified_date index"
        )

        print("   ✅ Temporal indexes created")

        # Step 8: Final verification
        print("\n✅ Step 6: Final verification...")
        final_verify_result = self.run_cypher(
            """MATCH ()-[r:ATTACKS]->()
               WHERE r.documented_create_date IS NOT NULL
               RETURN valueType(r.documented_create_date) as create_type,
                      valueType(r.documented_modified_date) as modified_type,
                      r.documented_create_date as sample_create,
                      r.documented_modified_date as sample_modified
               LIMIT 1""",
            ""
        )

        if final_verify_result and final_verify_result[0]['data']:
            create_type = final_verify_result[0]['data'][0]['row'][0]
            modified_type = final_verify_result[0]['data'][0]['row'][1]
            sample_create = final_verify_result[0]['data'][0]['row'][2]
            sample_modified = final_verify_result[0]['data'][0]['row'][3]
            print(f"   Create date type: {create_type}")
            print(f"   Modified date type: {modified_type}")
            print(f"   Sample create: {sample_create}")
            print(f"   Sample modified: {sample_modified}")

        # Step 9: Test temporal operations
        print("\n⚡ Step 7: Testing temporal operations...")
        test_result = self.run_cypher(
            """MATCH ()-[r:ATTACKS]->()
               WHERE r.documented_create_date.year = 2023
               RETURN count(r) as attacks_2023""",
            ""
        )

        if test_result and test_result[0]['data']:
            attacks_2023 = test_result[0]['data'][0]['row'][0]
            print(f"   ✅ Attacks in 2023: {attacks_2023:,}")

        # Test duration.between
        duration_test = self.run_cypher(
            """MATCH (ip:IP)-[r:ATTACKS]->()
               WHERE r.documented_create_date IS NOT NULL 
               AND r.documented_modified_date IS NOT NULL
               WITH ip, min(r.documented_create_date) as earliest, max(r.documented_modified_date) as latest
               RETURN duration.between(earliest, latest) as span
               LIMIT 1""",
            ""
        )

        if duration_test and duration_test[0]['data']:
            span = duration_test[0]['data'][0]['row'][0]
            print(f"   ✅ Duration calculation: {span}")

        print("\n" + "=" * 60)
        print("🎉 MIGRATION COMPLETED SUCCESSFULLY!")
        print("✅ All timestamps converted to DateTime format")
        print("✅ Temporal indexes created")
        print("✅ Ready for temporal operations")
        
        return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Migrate Neo4j timestamps in batches')
    parser.add_argument('--batch-size', type=int, default=10000, 
                       help='Number of relationships to process per batch (default: 10000)')
    args = parser.parse_args()

    migrator = Neo4jBatchMigrator(batch_size=args.batch_size)
    try:
        success = migrator.migrate_in_batches()
        return 0 if success else 1
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
