# Mirrorcore documentation

Start with the repository **[README.md](../README.md)** for install, quick start, and command overview.

## Guides

| File | Purpose |
|------|---------|
| [cli_commands.md](cli_commands.md) | Subcommands and global options (matches current `main.py` parsers) |
| [architecture.md](architecture.md) | Agent modules and design principles |
| [troubleshooting-features.md](troubleshooting-features.md) | Log analysis, incidents, follow-up sessions, outcomes |
| [respond-like-me-and-memory.md](respond-like-me-and-memory.md) | Personal responses, feedback, short-term carryover |
| [memory_model.md](memory_model.md) | Memory concepts (see also SQLite code under `src/mirrorcore/db/`) |
| [product_spec.md](product_spec.md) | Product-level specification |
| [persona_assessment_design.md](persona_assessment_design.md) | Assessment flow design |

## Source of truth

- CLI behavior: `src/mirrorcore/main.py` and `mirrorcore --help`
- Persistence: `src/mirrorcore/db/`
- Tests: `tests/`
