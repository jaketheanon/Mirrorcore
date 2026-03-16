# Mirrorcore Safety and Privacy Rules

Mirrorcore is privacy-constrained and local-first.

Rules:
- do not add silent cloud calls
- do not transmit user logs, memory data, persona data, or decisions externally without explicit opt-in
- keep core functionality usable without network access
- remote model usage, if present, must remain optional and explicit
- preserve graceful fallback behavior when LLMs are unavailable
- do not add identity simulation or deceptive anthropomorphic behavior
- do not blur the boundary between reasoning assistance and autonomous action

Storage/privacy rules:
- local SQLite storage is the default
- retention and external-use behavior must remain under user control
- privacy-sensitive behavior should be explicit in code and configuration