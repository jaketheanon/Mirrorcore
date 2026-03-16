# Mirrorcore System Architecture

Mirrorcore is a modular Python CLI application with clear subsystem boundaries.

Core modules:
- intake/: assessments, questionnaires, first-contact profiling
- memory/: episodic storage and retrieval
- persona/: user preferences, values, communication style
- terminal/: terminal/log/error parsing and troubleshooting workflows
- decision/: structured choice and trade-off analysis
- llm/: abstraction layer for local models and optional explicit remote use
- db/: SQLite persistence, schema, storage, and retrieval
- config.py: configuration, feature flags, privacy settings

Architecture rules:
- preserve module boundaries
- do not move logic into unrelated modules without strong justification
- prefer extending the correct module over adding parallel subsystems
- use shared DB/storage abstractions rather than scattering raw SQL across unrelated files
- preserve CLI-first behavior
- preserve SQLite as the default persistence layer
- keep cross-module interaction explicit and understandable
- avoid unnecessary indirection or abstraction layers

When adding a feature:
1. identify the owning subsystem
2. extend the smallest safe surface area
3. preserve current behavior outside the feature scope
4. keep the design explainable