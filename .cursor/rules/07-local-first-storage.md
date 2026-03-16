# Local-First Storage Rules

Mirrorcore uses local SQLite-backed persistence as a core architectural constraint.

Rules:
- prefer extending existing tables/fields before creating large new persistence subsystems
- keep migrations simple and idempotent
- preserve read compatibility with existing stored session data when possible
- keep stored structures explicit and explainable
- avoid hidden state transitions
- ensure session lookup logic matches the intended lifecycle rules

Important storage expectations:
- active and stalled investigations may be reused for follow-up
- resolved and abandoned investigations must remain queryable for history, but not reused for follow-up
- evidence history, hypothesis history, and investigation state must remain coherent across repeated follow-ups