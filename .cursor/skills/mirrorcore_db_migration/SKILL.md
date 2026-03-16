---
name: mirrorcore-db-migration
description: Create safe, idempotent SQLite schema migrations for the Mirrorcore project that preserve existing data and keep old sessions readable. Use when adding or evolving database schema for Mirrorcore’s SQLite backend.
---

# Mirrorcore DB Migration

## Purpose

This skill guides the agent through designing and applying **safe, additive SQLite migrations** for Mirrorcore, aligned with the project’s local-first and backward-compatibility requirements.

Use this skill whenever:
- **Schema changes** are needed for Mirrorcore’s SQLite database.
- A **new feature or phase** requires additional persistence.
- You must **audit or adjust** existing tables without losing historical data or breaking prior sessions.

## Core Rules

1. **Preserve existing data**
   - Do not drop tables, columns, or indexes unless the user explicitly approves a destructive change.
   - Never rewrite or delete existing rows as part of a default migration path.
   - Assume user data is critical and must remain readable after every migration.

2. **Make migrations idempotent**
   - Each migration must be safe to run multiple times without error.
   - Guard all `ALTER TABLE`, `CREATE TABLE`, and `CREATE INDEX` statements with existence checks or equivalent logic in Python migration code.
   - Prefer explicit versioning or migration-tracking tables so the same step is not applied twice.

3. **Prefer additive schema changes**
   - Add **new nullable columns** or **new tables** that reference existing keys.
   - Avoid changing column types, constraints, or primary keys on existing tables.
   - Default new fields to `NULL` or safe defaults so historical rows remain valid.

4. **Avoid destructive changes by default**
   - Only perform destructive operations (dropping columns/tables, tightening constraints that invalidate rows) when:
     - The user explicitly requests or approves them, and
     - There is a clear, documented migration path.
   - When in doubt, keep the old structure and layer new structures beside it.

5. **Keep old sessions readable**
   - All queries and models must continue to work with pre-existing rows.
   - New code must handle `NULL` or missing values for any newly added fields.
   - Do not introduce `NOT NULL` constraints on existing columns without guaranteed safe defaults for all existing data.

## Migration Workflow

Follow this workflow for every Mirrorcore DB migration:

### 1. Analyze current schema

1. Locate the SQLite database models and migration utilities under the `db/` layer.
2. Inspect the current schema by:
   - Reading existing migration scripts and models, and/or
   - Using `PRAGMA table_info(<table>)` and `PRAGMA foreign_key_list(<table>)` where appropriate.
3. Identify:
   - Existing tables, primary keys, and foreign-key relationships.
   - Any migration/versioning mechanism already in place (e.g., migrations table, schema version field).

### 2. Identify required additions

1. From the requested feature or phase, list the **new data** that must be stored.
2. Decide whether each new requirement fits best as:
   - A **new nullable column** on an existing table, or
   - A **new table** keyed by existing IDs (e.g., session, investigation, hypothesis).
3. Prefer:
   - Keeping existing tables stable and adding new, focused tables for new concepts.
   - Using `INTEGER` and `TEXT` types with simple constraints to avoid brittle migrations.

### 3. Design minimal ALTER / CREATE changes

1. For each table to change, design the **smallest** additive change set:
   - `ALTER TABLE <table> ADD COLUMN <name> <type> NULL` (or with a safe default).
   - `CREATE TABLE IF NOT EXISTS <new_table>(...)` with appropriate foreign keys.
   - `CREATE INDEX IF NOT EXISTS <index_name> ON <table>(<column>)`.
2. Avoid:
   - Renaming tables or columns unless absolutely necessary.
   - Changing existing column types, nullability, or default values.
3. Document how each change maps to the new behavior (which agent/feature uses it).

### 4. Implement idempotent migration code

1. Use the existing Mirrorcore migration mechanism (Python scripts or migration manager) to:
   - Check for table/column/index existence before applying a change.
   - Record migration application state (e.g., in a migrations table or schema version).
2. Ensure that:
   - Running the migration on a **fresh** database produces the intended final schema.
   - Running the migration on an **already-migrated** database is a no-op and does not error.

### 5. Ensure old sessions remain readable

1. Update data access code to:
   - Treat new fields as optional and handle `NULL` gracefully.
   - Fallback to existing behavior when new-phase data is missing.
2. Confirm that:
   - Core workflows (CLI commands and investigations) still succeed using only pre-migration data.
   - Any new logic is gated on the presence of new schema elements or data.

### 6. Validate against real data

1. If possible, run migrations against a copy of an existing database with real or representative data.
2. Validate:
   - No errors during migration.
   - No loss of rows or key relationships.
   - Old investigations/sessions can still be opened, read, and summarized.

## Checklist

Before finalizing a Mirrorcore DB migration, ensure:

- [ ] All changes are **additive** (new tables/columns/indexes) unless explicitly approved otherwise.
- [ ] Existing data is preserved; no rows are dropped or rewritten.
- [ ] Migrations are **idempotent** and safe to re-run.
- [ ] New columns are nullable or have safe defaults for old rows.
- [ ] Old sessions and investigations remain readable and usable.
- [ ] Any destructive changes were explicitly requested and are clearly documented.

## Output Expectations

At the end of a migration task, summarize:

- **Schema changes**
  - Briefly list the tables and columns added or modified.
- **Backward compatibility**
  - Explain how existing data and sessions remain readable.
- **Idempotency guarantees**
  - Note how the migration avoids double-application and handles already-migrated databases.

