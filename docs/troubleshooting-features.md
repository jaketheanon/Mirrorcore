# Troubleshooting features

Mirrorcore’s terminal workflow is **local** and **deterministic** at its core: log text is parsed, incident-like patterns are scored, and historical fix outcomes influence ranking. This page summarizes what the CLI exposes; implementation lives under `src/mirrorcore/terminal/`, investigation/analysis modules, and `src/mirrorcore/db/`.

## Commands

| Command | Role |
|---------|------|
| `mirrorcore debug` | Guided debugging from a problem description (or stdin with `--stdin`). |
| `mirrorcore analyze-log` | Paste multiline terminal output; extracts error signals and may classify an incident type. |
| `mirrorcore analyze-followup` | Attach new evidence to a stored analysis session (iterative investigation). |
| `mirrorcore debug-outcome` | Record what you tried and whether it worked, tied to recent incidents. |
| `mirrorcore debug-history` | Show ranked fix history, optionally filtered by `incident_type`. |

## Log analysis (high level)

- **Command context** — The parser looks at early lines for common tool families (e.g. package managers, containers, git) to steer incident detection.
- **Error signals** — Lines are classified with pattern-based rules (errors, tracebacks, permissions, timeouts, etc.).
- **Incident detection** — Candidate incident types are scored with subsystem and ambiguity rules; low-confidence or mixed-signal cases may avoid a single “winner” and fall back to broader guidance.
- **History reuse** — When an incident type matches and outcomes exist, suggestions can reflect past success/failure counts (deterministic ranking).

## Follow-up and investigation sessions

Multi-step flows store enough state to:

- Suggest **next diagnostic checks** (displayed for you to run; nothing is executed automatically).
- Apply **follow-up evidence** so hypotheses or emphasis can shift deterministically.
- Track **investigation progress** (e.g. completed vs pending steps) where implemented for the session type.

Exact fields and transitions are defined in code and tests (search for analysis session / investigation usage under `src/mirrorcore/` and `tests/`).

## Outcomes and learning

`debug-outcome` records structured outcomes. Those records feed **ranked fix history** and can adjust future guidance for the same or similar incident labels. All storage is in your local SQLite database.

## What this document does not claim

- No guarantee that every possible log shape gets a correct incident label.
- No automatic execution of shell commands recommended in output.
- Optional LLM paths, if present in your build, are separate from the deterministic baseline described here.
