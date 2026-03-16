# Phase-Based Development Rules

Mirrorcore is developed in bounded phases. Each completed and verified phase is treated as stable behavior unless the current task explicitly requires changing it.

Rules:
- do not redesign architecture when implementing a new phase
- do not refactor previously working logic unless required for the current phase or a confirmed bug fix
- prefer minimal, targeted edits
- extend the existing implementation in place
- preserve working commands, workflows, and storage behavior unless explicitly changed by the task
- treat the latest verified stable phase as the source of truth

Stable behavior that must not regress unless explicitly required:
- analyze-log
- analyze-followup
- debug-outcome
- investigation tracking
- hypothesis ranking
- hypothesis elimination
- session continuity across follow-ups
- stall detection
- strategy switching
- resolution/session closure behavior once implemented

Database rules:
- schema changes must be minimal
- migrations must be backwards-compatible
- existing data should remain readable unless the task explicitly authorizes a breaking migration

Implementation workflow:
1. inspect the current implementation
2. identify the smallest safe extension point
3. modify only what is necessary
4. verify the new phase behavior
5. confirm earlier phase behavior still works