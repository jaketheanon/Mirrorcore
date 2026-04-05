# CLI command reference

This file tracks subcommands **as registered in** `src/mirrorcore/main.py`. If anything disagrees with your tree, treat **`mirrorcore --help`** and **`mirrorcore <cmd> --help`** as authoritative.

## Global options

These apply before the subcommand:

| Option | Purpose |
|--------|---------|
| `--version` | Print version and exit |
| `--config PATH` | Configuration file path |
| `--verbose` | Verbose output |
| `--quiet` | Suppress non-error output |

## Subcommands

| Command | Help text (from parser) |
|---------|-------------------------|
| `start` | Guided onboarding and quick access to main features |
| `interactive` | Start interactive session |
| `debug` | Debug terminal issues — args: `problem` string; `--stdin` |
| `debug-outcome` | Record outcome of debugging attempt |
| `debug-history` | View ranked fix history — optional `incident_type` |
| `analyze-log` | Analyze raw terminal output, logs, or stack traces |
| `analyze-followup` | Update previous analysis with new evidence |
| `decide` | Get decision assistance — arg: `decision` |
| `profile` | Manage persona profile — `--analyze`, `--update` |
| `assess` | Run initial assessment — `--continue` |
| `init` | Initialize persona assessment |
| `memory` | Memory management — required `action`: `search`, `stats`, `export`, `import`, `drift` |
| `interview` | Run a structured decision interview |
| `calibrate-style` | Run a short style/persona calibration session |
| `respond-like-me` | Generate a likely-you response from saved decision and style memory — optional `scenario`; `--no-feedback`, `--debug-carryover` |
| `respond-feedback` | Review recent respond-like-me ratings — `--recent N` |
| `ask` | Route one plain-language request into the right MirrorCore flow — optional `query` words |
| `status` | Show system status |
| `config` | Configuration management — `--get`, `--set KEY VALUE` |

## Default command

If you run `mirrorcore` with **no** subcommand, the CLI defaults to **`interactive`** (see `main()` in `main.py`).

## Related docs

- [troubleshooting-features.md](troubleshooting-features.md) — behavior behind debug / analyze / follow-up
- [respond-like-me-and-memory.md](respond-like-me-and-memory.md) — respond-like-me and feedback
