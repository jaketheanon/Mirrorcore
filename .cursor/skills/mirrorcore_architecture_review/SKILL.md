---
name: mirrorcore-architecture-review
description: Review proposed changes against the Mirrorcore architecture. Use when evaluating modifications for module boundaries, local-first guarantees, SQLite storage, CLI workflows, and agent responsibility separation to prevent architectural regressions.
---

# Mirrorcore Architecture Review

## Purpose

This skill guides the agent in **reviewing proposed changes** to the Mirrorcore codebase against its **established architecture and constraints**. It helps keep:

- **Module boundaries** clean and intentional
- **Local-first** guarantees intact
- **SQLite** storage assumptions preserved
- **CLI workflows** and investigation lifecycle behavior stable
- **Agent responsibilities** clearly separated

Use this skill whenever:
- The user proposes structural changes, new subsystems, or significant refactors.
- Changes touch agent modules (`intake`, `memory`, `persona`, `terminal`, `decision`, `llm`, `db`).
- Storage logic, background processing, or external integrations are being modified or added.

## Core Architectural Principles

When reviewing changes, enforce these principles:

1. **Respect module boundaries**
   - Each agent (`intake`, `memory`, `persona`, `terminal`, `decision`, `llm`) and the `db` layer should remain **loosely coupled** with clear responsibilities.
   - Cross-agent calls should go through existing interfaces where possible; avoid ad hoc backdoors or circular dependencies.
   - Shared utilities belong in common modules, not duplicated across agents.

2. **Preserve local-first design**
   - All core functionality must work **fully offline** using local SQLite storage and local processing.
   - Cloud or remote dependencies must remain **optional**, explicitly gated, and require user consent.
   - No background network calls, telemetry, or external APIs should be introduced silently.

3. **Maintain SQLite storage assumptions**
   - Persistence must continue to use the existing SQLite-based `db` layer.
   - New storage concepts should integrate via existing data access patterns and migration mechanisms.
   - Avoid introducing additional primary datastores (e.g., Postgres, Redis, cloud DBs) unless explicitly justified and clearly optional.

4. **Preserve CLI workflows and investigation lifecycle**
   - Existing CLI commands, flags, and workflows are treated as **stable** unless the task explicitly changes them.
   - Investigation lifecycle behavior (creation, analysis, follow-up, resolution) must remain intact by default.
   - New behavior should be **additive and gated**, not replace or silently change existing flows.

5. **Keep agent responsibilities separated**
   - Do not merge or blur agent roles (e.g., decision logic inside `terminal`, persona modeling in `db`).
   - New capabilities should be attached to the most appropriate existing agent rather than introducing overlapping agents.
   - If a new agent or module is truly required, it must have a **narrow, well-defined responsibility** and integrate through clear interfaces.

## Review Workflow

Follow this workflow when applying this skill.

### 1. Understand the proposed change

1. Identify:
   - Which files, agents, and modules are being changed.
   - Whether new directories or subsystems are introduced.
   - Which CLI commands, workflows, or database tables are impacted.
2. Summarize in your own words:
   - The **goal** of the change.
   - The **scope** (which parts of the system it touches).
   - Any potential cross-cutting effects (e.g., new background tasks, new data flows).

### 2. Check module boundaries

For each change:
- Verify that logic added to an agent/module matches its documented responsibilities (see `AGENTS.md` and `.cursor/rules`).
- Look for:
  - New imports that create circular or wide-reaching dependencies.
  - Direct database access from agents that should use repository abstractions.
  - Large, cross-cutting helper modules that start to centralize unrelated logic.
- Prefer:
  - Small, focused helpers or extensions within the appropriate agent.
  - Reuse of existing patterns for communication between agents.

### 3. Check local-first and cloud dependencies

Inspect any code that:
- Performs network I/O, calls external APIs, or shells out to cloud CLIs.
- Starts background threads, async tasks, or daemons that may rely on connectivity.

Ensure that:
- Core workflows run without any network access.
- Any optional cloud integrations are:
  - Clearly labeled and documented as optional.
  - Explicitly configured or enabled by the user.
  - Not invoked silently during normal CLI commands.

### 4. Check SQLite storage and DB usage

When changes touch persistence:
- Confirm the system still uses the existing SQLite `db` layer.
- Ensure new tables, columns, or models:
  - Are minimal and backward compatible.
  - Integrate with existing repositories and transaction patterns.
- Reject or strongly question:
  - New primary datastores (cloud DBs, external caches) as requirements.
  - Architectural shifts that bypass the `db` layer for core data.

### 5. Check CLI workflow and investigation lifecycle

For any changes that impact CLI or investigations:
- Verify that:
  - Existing commands, flags, and options remain available and behave as before.
  - Default investigation lifecycle remains unchanged unless explicitly updated.
  - New flows are clearly additive and opt-in where possible.
- Pay special attention to:
  - `analyze-log`, `analyze-followup`, and related investigation commands.
  - Session resolution, stall detection, and strategy switching behaviors.

### 6. Check agent responsibility separation

Validate that:
- Each agent keeps its domain focus:
  - `intake`: initial assessment and profiling
  - `memory`: episodic memory and retrieval
  - `persona`: voice, values, decision style
  - `terminal`: CLI parsing and troubleshooting
  - `decision`: reasoning and tradeoff analysis
  - `llm`: language model abstraction
- Data access and persistence continue to flow through the `db` layer.
- New capabilities do not cause one agent to depend deeply on another’s internals.

## Rejection Criteria

Flag and reject (or request redesign of) changes that:

- **Introduce unnecessary new subsystems**
  - New orchestration layers, microservices, message queues, or background daemons without a strong, explicit need.
  - Alternative databases or caches as hard dependencies for core workflows.

- **Blur module responsibilities**
  - Agents taking on responsibilities clearly belonging to another agent.
  - Large “god modules” or utility packages that start mixing concerns across agents.

- **Break investigation lifecycle behavior**
  - Changes that alter default investigation transitions without clear justification.
  - New flows that can leave sessions in inconsistent or unrecoverable states.
  - Removal or silent change of core CLI behavior.

- **Introduce silent cloud dependencies**
  - Any network calls in default workflows without user opt-in.
  - Required configuration for remote services just to run the core CLI.
  - Telemetry, logging, or analytics that send data off-device by default.

When you detect any of the above, explicitly call them out in your review and propose safer, in-place alternatives that respect the existing architecture.

## Output Expectations

At the end of an architecture review, produce a succinct result with:

- **Scope summary**
  - One short paragraph describing what the change is trying to do and which parts of the system it touches.

- **Architecture assessment**
  - 3–7 bullets noting where the change:
    - Aligns well with Mirrorcore’s architecture, and
    - Risks or violates the principles above.

- **Decision**
  - One of:
    - **Approved**: Fits existing architecture and constraints.
    - **Approved with notes**: Acceptable but with clear recommendations.
    - **Rejected**: Violates core architectural principles (list the blocking issues).

Example structure:

```markdown
### Scope
Proposed change adds a new troubleshooting helper in the `terminal` agent and extends SQLite models for storing additional session metadata.

### Architecture assessment
- Aligns with terminal agent responsibility for CLI troubleshooting.
- Reuses existing `db` repositories and keeps schema changes minimal and backward compatible.
- Preserves CLI commands and investigation lifecycle; new behavior is additive and gated.

### Decision
**Approved with notes.** Consider extracting shared parsing logic into a small utility module if usage grows, but current design respects module boundaries and local-first constraints.
```

