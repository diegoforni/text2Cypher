#!/usr/bin/env python3
"""
Migrate timestamp properties from string arrays to Neo4j DateTime objects.

This script safely converts documented_create_date and documented_modified_date
properties on :ATTACKS relationships from their current format:
    ['2022-10-11T13:14:18.676000']
to Neo4j DateTime objects for better performance and native temporal operations.

Usage:
    python migrate_timestamps.py --preview          # Show sample conversions
    python migrate_timestamps.py --convert         # Run the full migration
    python migrate_timestamps.py --verify          # Check migration status
    python migrate_timestamps.py --create-indexes  # Create temporal indexes
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path to import config
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DB


def get_session_kwargs():
    """Return session kwargs with database if configured."""
    return {"database": NEO4J_DB} if NEO4J_DB else {}


def preview_conversion(driver):
    """Show sample conversions without making changes."""
    print("\n" + "=" * 80)
    print("PREVIEW: Sample Timestamp Conversions")
    print("=" * 80)
    
    with driver.session(**get_session_kwargs()) as session:
        result = session.run("""
            MATCH ()-[a:ATTACKS]->()
            WHERE a.documented_create_date IS NOT NULL
            WITH a LIMIT 10
            RETURN 
                a.documented_create_date[0] AS old_create_string,
                datetime(a.documented_create_date[0]) AS new_create_datetime,
                a.documented_modified_date[0] AS old_modified_string,
                datetime(a.documented_modified_date[0]) AS new_modified_datetime
        """)
        
        print("\nSample conversions:")
        print("-" * 80)
        for i, record in enumerate(result, 1):
            print(f"\nRecord {i}:")
            print(f"  CREATE DATE:")
            print(f"    Old: {record['old_create_string']}")
            print(f"    New: {record['new_create_datetime']}")
            print(f"  MODIFIED DATE:")
            print(f"    Old: {record['old_modified_string']}")
            print(f"    New: {record['new_modified_datetime']}")
    
    print("\n" + "=" * 80)
    print("✓ Preview complete. No changes made to database.")
    print("=" * 80)


def check_conversion_count(driver):
    """Check how many relationships need conversion."""
    print("\n" + "=" * 80)
    print("CHECKING: Relationships to Convert")
    print("=" * 80)
    
    with driver.session(**get_session_kwargs()) as session:
        # Total count
        total_result = session.run("MATCH ()-[a:ATTACKS]->() RETURN count(a) AS total")
        total = total_result.single()["total"]
        
        # Simple approach: try to convert one and see if it's a list or DateTime
        test_result = session.run("""
            MATCH ()-[a:ATTACKS]->()
            WHERE a.documented_create_date IS NOT NULL
            WITH a LIMIT 1
            RETURN valueType(a.documented_create_date) AS type_name
        """)
        test_rec = test_result.single()
        is_list = "LIST" in test_rec["type_name"]
        
        if is_list:
            # Not yet converted - count total
            needs_conversion_result = session.run("""
                MATCH ()-[a:ATTACKS]->()
                WHERE a.documented_create_date IS NOT NULL
                RETURN count(a) AS needs_conversion
            """)
            needs_conversion = needs_conversion_result.single()["needs_conversion"]
            converted = 0
        else:
            # Already converted
            converted_result = session.run("""
                MATCH ()-[a:ATTACKS]->()
                WHERE a.documented_create_date IS NOT NULL
                RETURN count(a) AS converted
            """)
            converted = converted_result.single()["converted"]
            needs_conversion = 0
        
        print(f"\nTotal :ATTACKS relationships: {total:,}")
        print(f"Already converted (DateTime): {converted:,}")
        print(f"Needs conversion (List): {needs_conversion:,}")
        print(f"Progress: {(converted/total*100):.1f}%")
        
        return needs_conversion


def check_apoc_available(driver):
    """Check if APOC procedures are available."""
    try:
        with driver.session(**get_session_kwargs()) as session:
            result = session.run("RETURN apoc.version() AS version")
            version = result.single()["version"]
            print(f"\n✓ APOC is available (version: {version})")
            return True
    except Exception as e:
        print(f"\n✗ APOC is not available: {e}")
        print("  Will use manual batching instead.")
        return False


def convert_with_apoc(driver, batch_size=10000):
    """Convert timestamps using APOC periodic iterate."""
    print("\n" + "=" * 80)
    print(f"CONVERTING: Using APOC (batch size: {batch_size:,})")
    print("=" * 80)
    
    with driver.session(**get_session_kwargs()) as session:
        # Convert documented_create_date
        print("\nConverting documented_create_date...")
        result1 = session.run("""
            CALL apoc.periodic.iterate(
                "MATCH ()-[a:ATTACKS]->() WHERE a.documented_create_date IS NOT NULL AND valueType(a.documented_create_date) CONTAINS 'LIST' RETURN a",
                "SET a.documented_create_date = datetime(a.documented_create_date[0])",
                {batchSize: $batch_size, parallel: false}
            )
            YIELD batches, total, errorMessages
            RETURN batches, total, errorMessages
        """, batch_size=batch_size)
        
        record1 = result1.single()
        print(f"  Batches: {record1['batches']}")
        print(f"  Total: {record1['total']:,}")
        print(f"  Errors: {record1['errorMessages']}")
        
        # Convert documented_modified_date
        print("\nConverting documented_modified_date...")
        result2 = session.run("""
            CALL apoc.periodic.iterate(
                "MATCH ()-[a:ATTACKS]->() WHERE a.documented_modified_date IS NOT NULL AND valueType(a.documented_modified_date) CONTAINS 'LIST' RETURN a",
                "SET a.documented_modified_date = datetime(a.documented_modified_date[0])",
                {batchSize: $batch_size, parallel: false}
            )
            YIELD batches, total, errorMessages
            RETURN batches, total, errorMessages
        """, batch_size=batch_size)
        
        record2 = result2.single()
        print(f"  Batches: {record2['batches']}")
        print(f"  Total: {record2['total']:,}")
        print(f"  Errors: {record2['errorMessages']}")
    
    print("\n✓ Conversion complete!")


def convert_manual_batch(driver, batch_size=100000):
    """Convert timestamps using manual batching (no APOC)."""
    print("\n" + "=" * 80)
    print(f"CONVERTING: Manual batching (batch size: {batch_size:,})")
    print("=" * 80)
    print("\nThis may take several minutes for large databases...")
    
    with driver.session(**get_session_kwargs()) as session:
        # Convert documented_create_date
        print("\nConverting documented_create_date...")
        total_converted = 0
        batch_num = 0
        
        while True:
            batch_num += 1
            result = session.run("""
                MATCH ()-[a:ATTACKS]->()
                WHERE a.documented_create_date IS NOT NULL 
                  AND valueType(a.documented_create_date) CONTAINS 'LIST'
                WITH a LIMIT $batch_size
                SET a.documented_create_date = datetime(a.documented_create_date[0])
                RETURN count(a) AS converted
            """, batch_size=batch_size)
            
            converted = result.single()["converted"]
            total_converted += converted
            print(f"  Batch {batch_num}: {converted:,} records converted (total: {total_converted:,})")
            
            if converted < batch_size:
                break
        
        # Convert documented_modified_date
        print("\nConverting documented_modified_date...")
        total_converted = 0
        batch_num = 0
        
        while True:
            batch_num += 1
            result = session.run("""
                MATCH ()-[a:ATTACKS]->()
                WHERE a.documented_modified_date IS NOT NULL 
                  AND valueType(a.documented_modified_date) CONTAINS 'LIST'
                WITH a LIMIT $batch_size
                SET a.documented_modified_date = datetime(a.documented_modified_date[0])
                RETURN count(a) AS converted
            """, batch_size=batch_size)
            
            converted = result.single()["converted"]
            total_converted += converted
            print(f"  Batch {batch_num}: {converted:,} records converted (total: {total_converted:,})")
            
            if converted < batch_size:
                break
    
    print("\n✓ Conversion complete!")


def verify_conversion(driver):
    """Verify the conversion was successful."""
    print("\n" + "=" * 80)
    print("VERIFYING: Conversion Results")
    print("=" * 80)
    
    with driver.session(**get_session_kwargs()) as session:
        # Check sample converted values
        print("\nSample converted values:")
        print("-" * 80)
        result = session.run("""
            MATCH ()-[a:ATTACKS]->()
            WHERE a.documented_create_date IS NOT NULL
              AND NOT valueType(a.documented_create_date) CONTAINS 'LIST'
            WITH a LIMIT 5
            RETURN 
                a.documented_create_date AS create_date,
                a.documented_modified_date AS modified_date,
                a.documented_create_date.year AS create_year,
                a.documented_create_date.month AS create_month,
                a.documented_create_date.day AS create_day
        """)
        
        for i, record in enumerate(result, 1):
            print(f"\nRecord {i}:")
            print(f"  Create Date: {record['create_date']}")
            print(f"  Modified Date: {record['modified_date']}")
            print(f"  Year: {record['create_year']}, Month: {record['create_month']}, Day: {record['create_day']}")
        
        # Count remaining unconverted
        remaining = session.run("""
            MATCH ()-[a:ATTACKS]->()
            WHERE a.documented_create_date IS NOT NULL 
              AND valueType(a.documented_create_date) CONTAINS 'LIST'
            RETURN count(a) AS remaining
        """).single()["remaining"]
        
        print("\n" + "-" * 80)
        if remaining == 0:
            print("✓ All relationships successfully converted to DateTime!")
        else:
            print(f"⚠ Warning: {remaining:,} relationships still need conversion")
    
    print("=" * 80)


def create_indexes(driver):
    """Create temporal indexes on the DateTime properties."""
    print("\n" + "=" * 80)
    print("CREATING: Temporal Indexes")
    print("=" * 80)
    
    with driver.session(**get_session_kwargs()) as session:
        # Create index on documented_create_date
        print("\nCreating index on documented_create_date...")
        try:
            session.run("""
                CREATE INDEX attacks_create_date IF NOT EXISTS
                FOR ()-[a:ATTACKS]-()
                ON (a.documented_create_date)
            """)
            print("✓ Index created: attacks_create_date")
        except Exception as e:
            print(f"✗ Error creating index: {e}")
        
        # Create index on documented_modified_date
        print("\nCreating index on documented_modified_date...")
        try:
            session.run("""
                CREATE INDEX attacks_modified_date IF NOT EXISTS
                FOR ()-[a:ATTACKS]-()
                ON (a.documented_modified_date)
            """)
            print("✓ Index created: attacks_modified_date")
        except Exception as e:
            print(f"✗ Error creating index: {e}")
        
        # Show all indexes
        print("\nCurrent indexes:")
        print("-" * 80)
        result = session.run("SHOW INDEXES")
        for record in result:
            print(f"  {record.get('name', 'N/A')}: {record.get('type', 'N/A')} on {record.get('labelsOrTypes', 'N/A')}")
    
    print("\n" + "=" * 80)
    print("✓ Index creation complete!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Neo4j timestamp properties from string arrays to DateTime objects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Preview the conversion without making changes
    python migrate_timestamps.py --preview
    
    # Check how many relationships need conversion
    python migrate_timestamps.py --check
    
    # Run the full migration
    python migrate_timestamps.py --convert
    
    # Verify conversion was successful
    python migrate_timestamps.py --verify
    
    # Create temporal indexes after migration
    python migrate_timestamps.py --create-indexes
        """
    )
    
    parser.add_argument("--preview", action="store_true", 
                       help="Preview sample conversions without making changes")
    parser.add_argument("--check", action="store_true",
                       help="Check how many relationships need conversion")
    parser.add_argument("--convert", action="store_true",
                       help="Run the full migration")
    parser.add_argument("--verify", action="store_true",
                       help="Verify the conversion was successful")
    parser.add_argument("--create-indexes", action="store_true",
                       help="Create temporal indexes on DateTime properties")
    parser.add_argument("--batch-size", type=int, default=10000,
                       help="Batch size for conversion (default: 10000)")
    
    args = parser.parse_args()
    
    if not any([args.preview, args.check, args.convert, args.verify, args.create_indexes]):
        parser.print_help()
        sys.exit(1)
    
    # Connect to Neo4j
    print(f"\nConnecting to Neo4j at {NEO4J_URI}...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        # Verify connection
        with driver.session(**get_session_kwargs()) as session:
            session.run("RETURN 1")
        print("✓ Connected successfully!")
        
        if args.preview:
            preview_conversion(driver)
        
        if args.check:
            check_conversion_count(driver)
        
        if args.convert:
            # Check count first
            needs_conversion = check_conversion_count(driver)
            
            if needs_conversion == 0:
                print("\n✓ No conversion needed - all relationships already use DateTime!")
            else:
                # Confirm before proceeding
                print("\n" + "=" * 80)
                print("⚠  WARNING: This will modify your database!")
                print("=" * 80)
                response = input("\nDo you want to proceed with the conversion? (yes/no): ")
                
                if response.lower() in ['yes', 'y']:
                    # Check if APOC is available
                    apoc_available = check_apoc_available(driver)
                    
                    if apoc_available:
                        convert_with_apoc(driver, args.batch_size)
                    else:
                        convert_manual_batch(driver, args.batch_size)
                    
                    # Auto-verify after conversion
                    verify_conversion(driver)
                else:
                    print("\n✗ Conversion cancelled.")
        
        if args.verify:
            verify_conversion(driver)
        
        if args.create_indexes:
            create_indexes(driver)
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        driver.close()
        print("\n✓ Connection closed.")


if __name__ == "__main__":
    main()
