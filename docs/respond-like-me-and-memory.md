# Respond-like-me, feedback, and memory

This document describes the **respond-like-me** flow and how it ties into stored decision/style memory, structured feedback, and short-term situation carryover. Implementation: `src/mirrorcore/persona/respond.py`, feedback helpers, and `src/mirrorcore/db/store.py`.

## Commands

| Command | Purpose |
|---------|---------|
| `mirrorcore respond-like-me [scenario]` | Builds a “likely you” answer from recent decision and style rows, with confidence and brief reasoning. |
| `mirrorcore respond-like-me --no-feedback` | Same, but skips the interactive rating prompt after the answer. |
| `mirrorcore respond-like-me --debug-carryover` | Prints JSON diagnostics for short-term situation carryover (intended for debugging). |
| `mirrorcore respond-feedback --recent N` | Lists recent saved ratings from respond-like-me. |

## How an answer is grounded

- The prompt is **normalized** and routed to an **effective decision family** (and related gates) using existing ontology/ranking code.
- **Decision** and **style** memories are retrieved with deterministic scoring; evidence multipliers and feedback-derived weights can adjust ranking.
- Output includes **likely answer**, **short reasoning**, **confidence**, and optional **memory basis** lines.

## User feedback

After a response (when stdin is interactive and `--no-feedback` is not set), you may be prompted to rate the answer and optionally provide **replacement wording** for “wording off” style feedback. Feedback is stored in SQLite and can influence:

- Future **retrieval scores** for similar prompts.
- **Short-term situation** stance snippets when wording-off feedback rewrites the stored line.
- **Promoted examples** when repeated corrections meet the promotion rules (see code: response example promotion from feedback).

## Short-term carryover (situation continuation)

Recent **unresolved** situation rows (short-term memory, not long-term traits) can affect continuity wording, retrieval, and escalation-related behavior when the new prompt is judged to continue the same thread. Deterministic thresholds and gates live in `src/mirrorcore/decision/situation_carryover.py` and related respond-side checks.

With `--debug-carryover`, JSON may include fields such as carryover pick diagnostics, Phase 45 escalation flags, and **feedback replacement** diagnostics (whether a user-approved replacement was available, considered on-thread, selected for the surface answer, or blocked with a reason).

## Privacy

All of the above uses **local** SQLite data unless you have explicitly configured optional external or LLM features elsewhere.

## Further reading

- [memory_model.md](memory_model.md) — conceptual memory layers (schema details: DB layer in code).
- [cli_commands.md](cli_commands.md) — exact CLI flags.
