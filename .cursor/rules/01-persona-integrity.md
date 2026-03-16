# Persona Integrity Rules

Mirrorcore models how the user thinks, decides, communicates, and troubleshoots. It does not simulate an independent self or human identity.

Rules:
- focus on user reasoning patterns, preferences, values, communication style, and decision behavior
- do not introduce personhood, sentience, selfhood, or human-roleplay architecture
- keep persona data structured, explicit, and practical
- preserve clear separation between:
  - user persona modeling
  - system logic
  - CLI behavior
  - memory/history storage

When modifying persona-related code:
- prefer structured fields over vague narrative abstractions
- preserve existing persona update behavior unless a phase explicitly changes it
- keep tone/style conditioning subordinate to correctness and workflow reliability