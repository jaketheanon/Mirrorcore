# Mirrorcore Coding Standards

Mirrorcore favors clarity, determinism, and minimal safe changes over cleverness.

Rules:
- prefer small, targeted edits over broad refactors
- do not rewrite working code without a concrete reason
- preserve public command behavior unless the current task explicitly changes it
- prefer explicit data structures and readable control flow
- avoid introducing heavy dependencies
- keep logic local-first and lightweight
- make schema/storage changes minimal and backwards-compatible
- preserve compatibility with existing session/history data whenever possible

Code style expectations:
- use descriptive names
- keep functions focused on one job
- avoid hidden side effects
- prefer straightforward branching over overly abstract patterns
- keep CLI output consistent and readable
- document behavior changes in README when relevant

Testing expectations:
- verify the modified command or workflow directly
- do not assume passing syntax means behavior is correct
- protect earlier verified phases from regression