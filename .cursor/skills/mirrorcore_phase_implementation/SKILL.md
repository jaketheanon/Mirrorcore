---
name: mirrorcore-phase-implementation
description: Implements new development phases for the Mirrorcore project with minimal safe, in-place extensions. Use when adding or evolving a phase that must integrate with existing CLI commands, investigation workflows, and database schema without redesigning architecture or causing regressions.
---

# Mirrorcore Phase Implementation

## Purpose

This skill guides the agent through implementing a **new development phase** for the Mirrorcore project while **extending the existing system in place** with **minimal, safe changes**. It preserves current architecture, CLI behavior, and investigation workflows while integrating new phase logic.

Use this skill whenever:
- The user asks to add or evolve a "phase" in Mirrorcore.
- A new phase must hook into investigation or analysis flows.
- Changes may affect `analyze-log`, `analyze-followup`, or the investigation lifecycle.

## Core Rules

1. **Extend in place**
   - Work inside the existing project; do not create parallel architectures.
   - Prefer reusing existing abstractions (agents, models, repositories, workflows) over introducing new patterns.

2. **No architecture redesign**
   - Do not change core agent responsibilities or boundaries (`intake`, `memory`, `persona`, `terminal`, `decision`, `llm`, `db`) unless explicitly requested.
   - Avoid large-scale refactors; keep changes tightly scoped to the new phase.

3. **Preserve public surfaces**
   - Do **not** rename or remove:
     - Modules, packages, or agent directories
     - CLI commands or subcommands
     - Database tables or primary keys
     - Public APIs used across agents
   - Only introduce new names or parameters when strictly necessary, and keep them backwards compatible.

4. **Preserve existing workflows**
   - Keep the behavior of existing, working flows unchanged unless explicitly modifying them for the new phase.
   - In particular, preserve:
     - Existing CLI commands
     - Investigation workflows and lifecycle transitions
     - Prior phase behaviors that have already been validated

## Integration Requirements

When implementing a new phase, ensure it integrates cleanly with:

- **`analyze-log`**: 
  - New phase logic must be discoverable from log analysis where appropriate.
  - Reuse or extend existing analysis helpers rather than duplicating logic.

- **`analyze-followup`**:
  - Ensure follow-up analysis can see and respect the new phase state.
  - Extend follow-up routing so that, when relevant, it can advance or refine the new phase.

- **Investigation lifecycle**:
  - Respect existing lifecycle states and transitions.
  - Add new states or transitions only when necessary, and default to existing behavior if the new phase data is absent.

- **Hypothesis tracking**:
  - Make sure the new phase can:
    - Attach to existing hypotheses, or
    - Create additional hypotheses in a way that does not break current consumers.
  - Do not change the semantics of existing hypothesis records.

- **Stall detection**:
  - Ensure stall detection can see the new phase’s contribution (e.g., additional steps, retries, or loops).
  - Avoid creating unbounded loops or re-entrant flows that bypass current stall safeguards.

- **Strategy switching**:
  - Integrate with existing strategy-switch logic rather than inventing new switching mechanisms.
  - New phase-specific strategies should plug into the same selection / evaluation paths as existing ones.

- **Session resolution**:
  - Make sure the new phase does not block or regress session resolution.
  - If the new phase adds optional work, it should degrade gracefully when skipped or absent.

## Database and Schema Rules

When a new phase requires persistence:

1. **Minimal schema changes**
   - Prefer:
     - Adding nullable columns to existing tables, or
     - Adding new tables that reference existing keys.
   - Avoid heavy re-modeling of existing schemas.

2. **Backward compatible migrations**
   - All migrations must be safe to apply on existing databases.
   - Keep defaults and nullability such that old data remains valid without transformation.
   - Do not drop columns, tables, or indexes that may still be used.

3. **Existing data remains readable**
   - All existing queries and models must continue to work against pre-existing data.
   - New logic must handle cases where phase-specific fields are missing or null.
   - Any new constraints must not invalidate old rows.

4. **Migration workflow**
   - Add or update migration scripts in the existing migration system.
   - Run migrations in development before relying on new fields or tables.

## Implementation Workflow

Follow this workflow whenever applying this skill:

### 1. Inspect the existing codebase

1. Locate:
   - Existing phase implementations (if any).
   - Investigation lifecycle models and state machines.
   - CLI entrypoints for `analyze-log`, `analyze-followup`, and investigation-related commands.
   - Hypothesis tracking, stall detection, strategy switching, and session resolution code.
2. Read enough to understand:
   - Where new phase hooks belong.
   - How current logic is wired (function calls, events, or state transitions).

### 2. Identify minimal safe extension points

1. Prefer:
   - Adding new functions, methods, or small helpers.
   - Extending existing enums or small state machines with additional values and guarded handling.
   - Adding optional parameters or configuration flags with safe defaults.
2. Avoid:
   - Large rewrites of existing functions.
   - Cross-cutting refactors that touch many unrelated modules.

### 3. Modify only necessary files

1. Determine the **smallest set of files** that must change to:
   - Represent the new phase concept (models / types).
   - Integrate with CLI commands.
   - Plug into investigation, hypotheses, stall detection, strategy switching, and resolution.
2. Keep each change narrowly focused on the new behavior.

### 4. Preserve earlier phase behavior

1. At every branch where the new phase is considered:
   - Maintain the existing default path when the new phase data is absent.
   - Ensure existing tests and workflows still pass conceptually (even if tests are not run).
2. If branching on phase, use patterns like:
   - "If new phase -> do extra work; else -> current behavior".

### 5. Integrate with core flows

For each of these, explicitly consider and implement the needed hooks:

- `analyze-log`:
  - Decide how the new phase influences log analysis.
  - Extend analysis functions or routing to include the phase when present.

- `analyze-followup`:
  - Ensure follow-up paths can inspect and update new phase-related state.

- Investigation lifecycle:
  - Extend lifecycle definitions minimally (new state, flag, or transition).
  - Keep old transitions valid and unchanged by default.

- Hypothesis tracking:
  - Add phase-aware fields or relationships without changing old semantics.

- Stall detection:
  - Ensure the new phase’s steps are observable by existing stall metrics/timers.

- Strategy switching:
  - Integrate new strategies or phase-aware behavior by plugging into existing switch points.

- Session resolution:
  - Confirm that sessions can still resolve successfully even if the new phase is unused.

## Database Change Checklist

Before finalizing any DB work:

- [ ] New columns are nullable or have safe defaults.
- [ ] No existing columns, tables, or indexes are dropped.
- [ ] Foreign keys from new tables reference existing keys without breaking existing rows.
- [ ] Migrations run successfully on a database with existing data.
- [ ] Application code handles missing or null new-phase fields gracefully.

## Output Expectations

At the end of a phase implementation session, always produce a concise summary with:

- **Files changed**
  - List the files (with minimal paths) that were modified or added.

- **Summary of modifications**
  - 1–3 bullet points describing the new phase behavior and where it hooks into the system.

- **Regression assessment**
  - An explicit statement confirming that:
    - Existing CLI commands are preserved.
    - Investigation workflows still follow their prior lifecycle by default.
    - Schema changes (if any) are minimal and backward compatible.
    - Existing data remains readable and compatible with new code paths.

Example output structure:

```markdown
### Files changed
- `path/to/file_a.py`
- `path/to/file_b.py`

### Summary
- Added [PhaseName] phase integration to investigation lifecycle with minimal branching.
- Extended `analyze-log` and `analyze-followup` to surface and update [PhaseName]-related state.
- Introduced backward-compatible DB changes for storing [PhaseName] metadata.

### Regression risk
**No regression risk expected.** Existing phases, CLI commands, investigation workflows, and data remain compatible; new behavior is additive and gated on new-phase state.
```

