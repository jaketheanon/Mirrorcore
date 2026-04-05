# Mirrorcore

Local-first Python CLI that stores decisions, troubleshooting history, and persona-style signals in **SQLite** on your machine, then reuses that data for grounded answers—without sending your content off-device by default.

## What it does

Mirrorcore is for people who want a **private, searchable trail** of how they troubleshoot and decide. New questions are matched against what you already saved so suggestions and phrasing stay tied to **your** patterns, not a generic playbook.

## Current status

Under **active development**. Implemented behavior is defined by the code and tests; this file is an overview only. For flags and edge cases, use `mirrorcore --help` and the docs under `docs/`.

## Key features

- **Persona / assessment** — Onboarding and profile work (`init`, `assess`, `interview`, `calibrate-style`, `profile`, `start`).
- **Terminal troubleshooting** — Debug commands, paste logs/stack traces, record outcomes, inspect fix history (`debug`, `analyze-log`, `analyze-followup`, `debug-outcome`, `debug-history`).
- **Decisions** — Decision assistance using stored memory (`decide`).
- **Ask / respond** — Routed plain-language input and “respond like me” with optional feedback (`ask`, `respond-like-me`, `respond-feedback`).
- **Housekeeping** — Status, memory subcommands, config (`status`, `memory`, `config`).

Investigation and ranking paths used in troubleshooting are largely **deterministic** (rule-based). Optional LLM use is not required for the baseline CLI flows.

## Requirements

- **Python** 3.10 or newer (`requires-python` in `pyproject.toml`)
- **Runtime dependency:** `prompt-toolkit>=0.5.0` (interactive editing / prompts)
- **Storage:** writable path for the SQLite database (after `init`)

## Quick start

```bash
git clone https://github.com/jaketheanon/Mirrorcore.git
cd Mirrorcore

python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e .

mirrorcore init
mirrorcore --help
```

Run from a checkout **without** installing the package:

```bash
cd Mirrorcore
PYTHONPATH=src python -m mirrorcore --help
```

## Main commands

| Area | Commands |
|------|----------|
| First run | `mirrorcore init`, `mirrorcore start` |
| Interactive | `mirrorcore` (no subcommand → interactive), `mirrorcore interactive` |
| Terminal | `mirrorcore debug "<problem>"`, `mirrorcore analyze-log`, `mirrorcore analyze-followup` |
| Outcomes | `mirrorcore debug-outcome`, `mirrorcore debug-history [incident_type]` |
| Decisions | `mirrorcore decide "<question>"` |
| Voice / phrasing | `mirrorcore respond-like-me "<scenario>"`, `mirrorcore respond-feedback` |
| Router | `mirrorcore ask …` |
| Profile / memory | `mirrorcore profile`, `mirrorcore memory` with `search`, `stats`, `export`, `import`, or `drift` |
| System | `mirrorcore status`, `mirrorcore config [--get KEY \| --set KEY VALUE]` |

Examples: `mirrorcore respond-like-me --debug-carryover "…"`, `mirrorcore profile --analyze`.

Authoritative syntax: **`mirrorcore <command> --help`**.

## Project structure

| Path | Role |
|------|------|
| `src/mirrorcore/` | Application code (CLI, agents, DB, workflows) |
| `tests/` | Pytest suite |
| `docs/` | Longer documentation (see [docs/README.md](docs/README.md)) |
| `pyproject.toml` | Package metadata, dependencies, console script `mirrorcore` |
| `.cursor/rules/`, `AGENTS.md` | Contributor / agent guardrails |

## Development workflow

- Use **short-lived branches** for changes; merge through your usual review process (this repo has used `main`, `master`, and many `phase*-` integration branches—check `git branch` locally).
- Run tests with `pytest`; a typical invocation is:

  ```bash
  PYTHONPATH=src python -m pytest tests/ -q
  ```

## Documentation

| Document | Contents |
|----------|----------|
| [docs/README.md](docs/README.md) | Index of all docs |
| [docs/cli_commands.md](docs/cli_commands.md) | CLI reference (aligned to current parsers) |
| [docs/architecture.md](docs/architecture.md) | High-level modules and agents |
| [docs/troubleshooting-features.md](docs/troubleshooting-features.md) | Log analysis, follow-up, outcomes (deep dive) |
| [docs/respond-like-me-and-memory.md](docs/respond-like-me-and-memory.md) | Respond-like-me, feedback, carryover |
| [docs/memory_model.md](docs/memory_model.md) | Memory design notes |
| [docs/product_spec.md](docs/product_spec.md), [docs/persona_assessment_design.md](docs/persona_assessment_design.md) | Product / assessment design |

## Privacy

Data stays **local** by default. Any optional external or LLM use depends on your configuration and explicit choices.

## License

MIT — see `LICENSE`.
