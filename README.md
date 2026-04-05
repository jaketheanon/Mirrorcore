# Mirrorcore

Mirrorcore is a **local-first Python CLI** that stores your answers, decisions, and troubleshooting history in a **SQLite** database on your machine. It uses that data to give grounded, personalized guidance—terminal debugging, decision-style support, and phrasing help—without sending your content off-device by default.

## What problem it solves

Command-line work and repeated social or work decisions rarely leave a durable, searchable trail *you* control. Mirrorcore keeps that trail local, ties new questions to what you already saved, and surfaces likely-next steps or wording based on your own patterns—not a generic playbook.

## What it can do today

- **Intake / profile** — Structured assessment and ongoing calibration of how you decide and communicate (`init`, `assess`, `interview`, `calibrate-style`, `profile`).
- **Terminal troubleshooting** — Debug failed commands, analyze pasted logs/stack traces, record outcomes, and reuse ranked fix history (`debug`, `analyze-log`, `analyze-followup`, `debug-outcome`, `debug-history`).
- **Decisions** — Decision assistance grounded in stored memory (`decide`).
- **Ask / respond** — Scenario-style help and “respond like me” flows with feedback hooks (`ask`, `respond-like-me`, `respond-feedback`).
- **Operations** — Status, memory inspection, configuration (`status`, `memory`, `config`).

Core ranking and investigation logic is **deterministic** (rule-based); optional LLM integration is not required for the baseline paths described here.

## Requirements

- **Python 3.10+**
- **Dependencies**: `prompt-toolkit` (see `pyproject.toml`)
- **Data**: a writable SQLite database path (created on first use after `init`).

## Quick start

```bash
cd mirrorcore
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .

mirrorcore init
mirrorcore --help
```

Run from a checkout without installing:

```bash
cd mirrorcore
PYTHONPATH=src python -m mirrorcore --help
```

## Common commands

| Area | Examples |
|------|----------|
| Setup | `mirrorcore init` |
| Interactive | `mirrorcore` (no subcommand defaults to interactive), or `mirrorcore interactive` |
| Terminal | `mirrorcore debug "<command or error>"`, `mirrorcore analyze-log` |
| Follow-up | `mirrorcore analyze-followup` |
| Learning | `mirrorcore debug-outcome`, `mirrorcore debug-history` |
| Decisions | `mirrorcore decide "<question>"` |
| Persona / voice | `mirrorcore respond-like-me "<scenario>"`, `mirrorcore respond-feedback` |
| Ask flow | `mirrorcore ask` |
| Inspection | `mirrorcore status`, `mirrorcore profile`, `mirrorcore memory` |

Use `mirrorcore <command> --help` for options (e.g. `respond-like-me --debug-carryover`).

## Documentation

Deeper detail lives under **`docs/`** (architecture, CLI notes, memory model, product spec). The README stays a landing page; it does not duplicate those documents.

## Development

- **Layout**: application code under `src/mirrorcore/`; tests under `tests/`.
- **Rules**: Cursor/agent guidance in `.cursor/rules/`; high-level agent notes in **`AGENTS.md`**.
- **Workflow**: day-to-day work typically uses short-lived **topic branches** merged into a stable baseline branch (exact names depend on your fork; check `git branch` and recent history).
- **Tests**: `pytest` from the repo root; many setups use `PYTHONPATH=src`:

  ```bash
  PYTHONPATH=src python -m pytest tests/ -q
  ```

## Privacy

Data is stored **locally**. Optional cloud or LLM use, if enabled in your configuration, is explicit and off by default for the core flows described above.

## License

MIT — see `LICENSE`.

## Current status

The project is **under active development**. Behavior and tests are the source of truth for what is implemented; this README is kept short and factual and may lag a specific command flag—prefer `mirrorcore --help` and `docs/` for exhaustive detail.
