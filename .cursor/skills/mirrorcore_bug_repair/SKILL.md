---
name: mirrorcore-bug-repair
description: Perform minimal, targeted bug repairs in existing Mirrorcore codebases. Use when fixing a specific failing behavior in Mirrorcore while preserving CLI behavior, database/session compatibility, and earlier phase workflows.
---

# Mirrorcore Bug Repair

## Instructions

Follow this workflow whenever repairing a bug in an existing Mirrorcore codebase:

1. **Identify failing behavior**
   - Reproduce the issue using the existing CLI commands and workflows.
   - Capture the exact command, arguments, environment, and observed vs expected behavior.
   - Prefer using current tests or adding a focused regression test over ad-hoc scripts when possible.

2. **Trace the responsible code path**
   - Start from the failing CLI command or user-visible behavior and walk inward through the agents (terminal, decision, memory, persona, etc.) and shared layers.
   - Use logs, stack traces, and code search to find the minimal function(s) directly responsible for the incorrect behavior.
   - Confirm the root cause before proposing any changes (data assumption, control flow, edge case, integration between agents, schema mismatch, etc.).

3. **Modify the smallest safe section**
   - Change only the minimal code necessary to correct the root cause.
   - Do not refactor unrelated modules, rename public interfaces, or restructure agents.
   - Preserve existing CLI output format, command names, flags, prompts, and return codes.
   - Keep changes local to the identified function, method, or narrow module boundary whenever possible.

4. **Ensure database and session compatibility**
   - Treat the current SQLite schema and stored data as production; avoid breaking changes.
   - Prefer additive or backward-compatible adjustments over destructive migrations.
   - When touching persistence, confirm that existing sessions, memories, and phases continue to load and behave correctly.

5. **Verify earlier phases remain unaffected**
   - Re-run relevant Mirrorcore workflows that use earlier phases (intake, memory, persona, decision, terminal, etc.) that might depend on the changed code.
   - Check that phase transitions, stored state, and retrieval logic still behave as before, except for the specific bug fix.
   - Add or update targeted tests to cover both the repaired behavior and a representative earlier-phase path.

6. **Bias toward targeted fixes**
   - Prefer small, surgical changes over structural redesigns.
   - If a larger refactor seems desirable, document it separately and do not mix it into the bug-fix change set.

## Examples

- Fixing an incorrect CLI error message:
  - Reproduce the failing command.
  - Trace the CLI handler and the specific agent method that produces the message.
  - Update only the message construction or condition without changing command signatures or exit codes.

- Repairing a memory retrieval edge case:
  - Reproduce the scenario that fails to retrieve or misorders memories.
  - Inspect the query and sorting logic in the memory agent and DB layer.
  - Adjust the query or in-memory filtering minimally, keeping the schema and public APIs unchanged.

