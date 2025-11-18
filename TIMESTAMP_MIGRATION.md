# Neo4j Timestamp Format Migration Guide

## Current State Analysis

### Timestamp Properties Location
- **Relationship**: `:ATTACKS`
- **Properties**: 
  - `documented_create_date`
  - `documented_modified_date`
- **Total Relationships**: ~2,968,825

### Current Format Issues
```
Current Format: ['2022-10-11T13:14:18.676000']
- Type: Array of strings
- Pattern: YYYY-MM-DDTHH:MM:SS.mmmmmm
- Issues:
  ✗ Stored as string array instead of native DateTime
  ✗ Cannot use Neo4j temporal functions directly
  ✗ Requires string manipulation (substring) for operations
  ✗ No built-in date comparison or arithmetic
  ✗ Inefficient filtering and indexing
```

### Code Evidence
From `agents/generation_agent.py`:
```python
"TEMPORAL RULES: Many date properties (e.g., a.documented_create_date) are arrays of strings. "
"To derive calendar year-month (YYYY-MM), always use substring(toString(<prop>[0]),0,7). "
"Do NOT call datetime() on an array value and do NOT use left(); prefer substring()+toString() as shown. "
```

---

## Recommended Solution: Convert to Neo4j DateTime

### Benefits
- ✓ Native temporal type with full Neo4j support
- ✓ Direct use of temporal functions (date(), datetime(), duration())
- ✓ Efficient date range filtering and comparisons
- ✓ Better query performance with temporal indexes
- ✓ Preserves full timestamp precision (date + time + microseconds)
- ✓ Timezone support if needed

---

## Migration Cypher Queries

### Option A: Full DateTime Conversion (RECOMMENDED)

This converts the string arrays to native Neo4j DateTime objects, preserving full timestamp precision.

```cypher
// Step 1: Preview the conversion (test on 10 records)
MATCH ()-[a:ATTACKS]->()
WHERE a.documented_create_date IS NOT NULL
WITH a LIMIT 10
RETURN 
  a.documented_create_date[0] AS old_create_string,
  datetime(a.documented_create_date[0]) AS new_create_datetime,
  a.documented_modified_date[0] AS old_modified_string,
  datetime(a.documented_modified_date[0]) AS new_modified_datetime;

// Step 2: Backup check - count records to be updated
MATCH ()-[a:ATTACKS]->()
WHERE a.documented_create_date IS NOT NULL
RETURN count(a) AS total_to_update;

// Step 3: Convert documented_create_date (in batches for safety)
// Process 10,000 relationships at a time
CALL apoc.periodic.iterate(
  "MATCH ()-[a:ATTACKS]->() WHERE a.documented_create_date IS NOT NULL RETURN a",
  "SET a.documented_create_date = datetime(a.documented_create_date[0])",
  {batchSize: 10000, parallel: false}
)
YIELD batches, total, errorMessages
RETURN batches, total, errorMessages;

// Step 4: Convert documented_modified_date (in batches for safety)
CALL apoc.periodic.iterate(
  "MATCH ()-[a:ATTACKS]->() WHERE a.documented_modified_date IS NOT NULL RETURN a",
  "SET a.documented_modified_date = datetime(a.documented_modified_date[0])",
  {batchSize: 10000, parallel: false}
)
YIELD batches, total, errorMessages
RETURN batches, total, errorMessages;

// Step 5: Verify the conversion
MATCH ()-[a:ATTACKS]->()
WITH a LIMIT 5
RETURN 
  a.documented_create_date AS create_date,
  a.documented_modified_date AS modified_date,
  a.documented_create_date.year AS create_year,
  a.documented_create_date.month AS create_month,
  a.documented_create_date.day AS create_day;
```

### Option A (Without APOC) - Manual Batching

If APOC is not available, process in manual batches:

```cypher
// Convert documented_create_date - Batch 1 (first 100k)
MATCH ()-[a:ATTACKS]->()
WHERE a.documented_create_date IS NOT NULL
WITH a LIMIT 100000
SET a.documented_create_date = datetime(a.documented_create_date[0])
RETURN count(a) AS updated;

// Convert documented_modified_date - Batch 1 (first 100k)
MATCH ()-[a:ATTACKS]->()
WHERE a.documented_modified_date IS NOT NULL
WITH a LIMIT 100000
SET a.documented_modified_date = datetime(a.documented_modified_date[0])
RETURN count(a) AS updated;

// Repeat the above queries until all records are updated
// Check progress with:
MATCH ()-[a:ATTACKS]->()
WHERE a.documented_create_date IS NOT NULL 
  AND NOT a.documented_create_date:List
RETURN count(a) AS converted_count;
```

---

### Option B: Date Only Conversion (No Time Component)

If you only need the date portion without time:

```cypher
// Preview
MATCH ()-[a:ATTACKS]->()
WHERE a.documented_create_date IS NOT NULL
WITH a LIMIT 10
RETURN 
  a.documented_create_date[0] AS old_string,
  date(a.documented_create_date[0]) AS new_date;

// Convert to Date type (with APOC)
CALL apoc.periodic.iterate(
  "MATCH ()-[a:ATTACKS]->() WHERE a.documented_create_date IS NOT NULL RETURN a",
  "SET a.documented_create_date = date(a.documented_create_date[0])",
  {batchSize: 10000, parallel: false}
)
YIELD batches, total, errorMessages
RETURN batches, total, errorMessages;

CALL apoc.periodic.iterate(
  "MATCH ()-[a:ATTACKS]->() WHERE a.documented_modified_date IS NOT NULL RETURN a",
  "SET a.documented_modified_date = date(a.documented_modified_date[0])",
  {batchSize: 10000, parallel: false}
)
YIELD batches, total, errorMessages
RETURN batches, total, errorMessages;
```

---

### Option C: Flatten to Simple String (Not Recommended)

Only removes the array wrapper, keeping string format:

```cypher
// Convert array to single string
CALL apoc.periodic.iterate(
  "MATCH ()-[a:ATTACKS]->() WHERE a.documented_create_date IS NOT NULL RETURN a",
  "SET a.documented_create_date = a.documented_create_date[0]",
  {batchSize: 10000, parallel: false}
)
YIELD batches, total, errorMessages
RETURN batches, total, errorMessages;

CALL apoc.periodic.iterate(
  "MATCH ()-[a:ATTACKS]->() WHERE a.documented_modified_date IS NOT NULL RETURN a",
  "SET a.documented_modified_date = a.documented_modified_date[0]",
  {batchSize: 10000, parallel: false}
)
YIELD batches, total, errorMessages
RETURN batches, total, errorMessages;
```

---

## Post-Migration Changes

### 1. Update Agent Code

After migration to DateTime, update `agents/generation_agent.py`:

**OLD (current):**
```python
"TEMPORAL RULES: Many date properties (e.g., a.documented_create_date) are arrays of strings. "
"To derive calendar year-month (YYYY-MM), always use substring(toString(<prop>[0]),0,7). "
```

**NEW (after DateTime migration):**
```python
"TEMPORAL RULES: Date properties (documented_create_date, documented_modified_date) are Neo4j DateTime types. "
"Use native temporal functions: a.documented_create_date.year, a.documented_create_date.month. "
"For year-month grouping: use date.truncate('month', a.documented_create_date). "
"For date ranges: use a.documented_create_date >= datetime('2023-01-01') AND a.documented_create_date < datetime('2024-01-01'). "
```

### 2. Query Examples After Migration

```cypher
// Get attacks by year-month (NEW way)
MATCH (ip:IP)-[a:ATTACKS]->(c:Country)
WITH date.truncate('month', a.documented_create_date) AS year_month, count(*) AS attacks
RETURN year_month, attacks
ORDER BY year_month DESC;

// Filter by date range (NEW way)
MATCH (ip:IP)-[a:ATTACKS]->(c:Country)
WHERE a.documented_create_date >= datetime('2023-01-01')
  AND a.documented_create_date < datetime('2024-01-01')
RETURN ip.address, c.name, a.documented_create_date
LIMIT 10;

// Get attacks from last 30 days (NEW way)
MATCH (ip:IP)-[a:ATTACKS]->(c:Country)
WHERE a.documented_create_date >= datetime() - duration({days: 30})
RETURN count(*) AS attacks_last_30_days;

// Compare dates (NEW way)
MATCH (ip:IP)-[a1:ATTACKS]->(c1:Country)
MATCH (ip)-[a2:ATTACKS]->(c2:Country)
WHERE a1.documented_create_date < a2.documented_create_date
RETURN ip.address, c1.name, c2.name, a1.documented_create_date, a2.documented_create_date
LIMIT 10;
```

### 3. Create Temporal Indexes for Performance

```cypher
// Create indexes on the DateTime properties
CREATE INDEX attacks_create_date IF NOT EXISTS
FOR ()-[a:ATTACKS]-()
ON (a.documented_create_date);

CREATE INDEX attacks_modified_date IF NOT EXISTS
FOR ()-[a:ATTACKS]-()
ON (a.documented_modified_date);

// Verify indexes
SHOW INDEXES;
```

---

## Migration Checklist

- [ ] **Backup Database** - Create a full backup before migration
- [ ] **Test on Sample** - Run preview queries on 10-100 records
- [ ] **Check APOC Availability** - Verify if APOC procedures are installed
- [ ] **Choose Migration Option** - Select Option A (DateTime) recommended
- [ ] **Run Conversion Queries** - Execute migration in batches
- [ ] **Verify Conversion** - Check sample records after migration
- [ ] **Update Application Code** - Modify `generation_agent.py` system message
- [ ] **Create Indexes** - Add temporal indexes for performance
- [ ] **Test Queries** - Verify text2Cypher generates correct queries
- [ ] **Monitor Performance** - Check query execution times improve

---

## Rollback Strategy

If issues occur, you can rollback by re-importing from your backup. There's no direct rollback since we're overwriting the property values.

**Prevention**: Test on a small subset first!

```cypher
// Create a test property first (safer approach)
MATCH ()-[a:ATTACKS]->()
WHERE a.documented_create_date IS NOT NULL
WITH a LIMIT 1000
SET a.documented_create_date_new = datetime(a.documented_create_date[0])
RETURN count(a) AS test_count;

// Verify the test property looks good
MATCH ()-[a:ATTACKS]->()
WHERE a.documented_create_date_new IS NOT NULL
RETURN a.documented_create_date[0] AS old, a.documented_create_date_new AS new
LIMIT 10;

// If satisfied, then do the full migration to overwrite the original property
```

---

## Performance Considerations

**Estimated Time**: 
- With APOC (10k batch): ~5-10 minutes for ~3M relationships
- Without APOC (manual 100k batches): ~30-60 minutes total

**Memory**: Batching prevents memory issues

**Downtime**: Consider maintenance window for production databases

---

## Summary

**Current State**: 
- Format: `['2022-10-11T13:14:18.676000']` (string array)
- Location: `:ATTACKS` relationship properties

**Recommended Migration**: 
- Target: Neo4j DateTime native type
- Method: `datetime(property[0])`
- Batch size: 10,000 records

**Benefits**:
- ✅ Use native temporal functions
- ✅ Better query performance
- ✅ Cleaner, more maintainable queries
- ✅ Standard Neo4j best practices
