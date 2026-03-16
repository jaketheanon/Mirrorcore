---
name: mirrorcore-investigation-test
description: Simulates realistic troubleshooting investigations for the Mirrorcore CLI by generating synthetic terminal commands, logs, stack traces, repeated evidence patterns, stalls, strategy switches, and resolution outcomes. Use when testing investigation workflows, hypothesis ranking, stall detection, strategy switching, and resolution capture.
---

# Mirrorcore Investigation Test

## Purpose

Use this skill to **simulate a realistic troubleshooting investigation** for Mirrorcore so that the project's **investigation workflows, heuristics, and agents** can be tested safely without requiring real failures.

The goal is to generate:
- **Terminal commands and logs**
- **Error and stack traces** (when appropriate)
- **Repeated evidence patterns** (e.g., recurring error lines)
- **Investigation stalls** (e.g., uninformative loops or dead ends)
- **Resolution scenarios** (e.g., configuration fix, dependency install, or command adjustment)

The synthetic evidence MUST be rich enough to exercise:
- **Hypothesis ranking**
- **Stall detection**
- **Strategy switching**
- **Resolution capture and postmortem**

Keep simulations **plausible for a local-first Python CLI on Linux** using Mirrorcore-style workflows.

---

## When to Use This Skill

Use this skill when:
- The user asks to **test or exercise Mirrorcore's investigation logic** without touching real systems.
- You need to **generate example investigations** for:
  - hypothesis ranking or strategy switching,
  - stall detection and recovery,
  - resolution confirmation and capture.
- You are asked to **produce synthetic terminal logs or stack traces** for Mirrorcore or Python CLI behavior.

Do **not** use this skill:
- For actual live debugging of the user's real terminal commands.
- To fabricate logs that are meant to be presented as *real* history; clearly treat them as **simulated evidence**.

---

## Investigation Simulation Checklist

When simulating an investigation, follow this checklist:

1. **Define the scenario**
   - Briefly describe:
     - **User goal** (e.g., "run `mirrorcore run-intake` after fresh install").
     - **Environment** (Linux, Python version, local SQLite, offline/online).
     - **Initial symptom** (error message, hang, wrong behavior).

2. **Lay out investigation phases**
   - Structure as a sequence of **phases**, such as:
     - Phase 1: Initial failure & quick checks
     - Phase 2: Deeper inspection & hypothesis branching
     - Phase 3: Stall & strategy switch
     - Phase 4: Resolution and verification

3. **Produce step-by-step evidence**
   - For each phase, output a sequence of **steps**.
   - Each step should usually contain:
     - A **shell command** (or user action).
     - The **terminal output/logs**.
     - Optionally a **short reasoning note** (what this evidence suggests).

4. **Include rich error and stack traces**
   - When a Python error occurs, include:
     - `Traceback (most recent call last):`
     - 2–6 stack frames with plausible file names and line numbers.
     - The final error type and message (e.g., `sqlite3.OperationalError`, `FileNotFoundError`, `ClickException`).
   - Make at least one trace **repeat** or **echo** across commands to simulate recurring evidence.

5. **Simulate repeated evidence patterns**
   - Ensure some log lines or error summaries appear **multiple times**, such as:
     - The same `OperationalError` when re-running a command.
     - A repeated `"database is locked"` or `"unable to open database file"` line.
     - A recurring warning about configuration or environment variables.
   - This repetition should be explicit enough that a pattern-mining or hypothesis-ranking system can detect it.

6. **Simulate investigation stalls**
   - Include at least one segment where:
     - The user runs similar commands repeatedly with **no new information**, or
     - The logs are noisy but **no new hypothesis** is advanced, or
     - Progress stops because a **missing prerequisite** is not yet identified.
   - Mark this clearly (e.g., "Investigation appears stalled here: same error despite multiple attempts").

7. **Trigger strategy switching**
   - After the stall, introduce a **clear strategy change**, for example:
     - Switch from re-running the CLI to **inspecting log files or config**.
     - Switch from guessing to **checking docs or `--help`**.
     - Switch from focusing on Mirrorcore to **verifying Python/SQLite or file permissions**.
   - Show before-and-after evidence that motivates the switch (e.g., "repeated `database is locked` suggests a lingering process, so we check running processes").

8. **Show resolution and capture**
   - Present a plausible **fix action**, like:
     - Killing a rogue process holding the DB lock.
     - Upgrading/installing a missing Python package.
     - Creating a directory/file Mirrorcore expects.
     - Adjusting a CLI flag that was misused.
   - Follow with:
     - Re-run of the originally failing command.
     - Confirmed **success output**.
     - A short **resolution summary** that:
       - Describes root cause at a practical level.
       - Mentions the key evidence that supported the final hypothesis.

---

## Output Format Template

Use this **structured template** when generating a simulated investigation. You may adapt names and details, but keep the overall structure so Mirrorcore can parse and test its workflows.

```markdown
## Scenario
- Goal: [short goal description]
- Environment: [OS, Python version, offline/online, relevant context]
- Initial Symptom: [brief description]

## Phase 1: Initial Failure

### Step 1
**Command**
```bash
[user@host]$ [command here]
```

**Output**
```text
[terminal output, including any errors or traces]
```

**Notes**
- [What this suggests about possible causes]

### Step 2
...

## Phase 2: Hypothesis Branching
...

## Phase 3: Stall and Strategy Switch

### Stall Segment
**Commands / Outputs**
```bash
[user@host]$ [repeated/variant command]
...
```

```text
[mostly repeated error / no new evidence]
```

**Stall Observation**
- Investigation appears stalled because [...]

### Strategy Switch
- Previous strategy: [...]
- New strategy: [...]
- Rationale: [...]

## Phase 4: Resolution and Verification

### Fix
**Action**
- [Describe concrete fix applied]

**Evidence**
```bash
[user@host]$ [command confirming fix]
```

```text
[successful output]
```

### Resolution Summary
- Root cause: [...]
- Key evidence: [...]
- Final confirmation: [...]
```

When generating content, **fill in each section concretely**, using realistic Mirrorcore-style commands and plausible Python/SQLite/log output.

---

## Error and Trace Generation Guidelines

- Prefer **Python-style tracebacks** consistent with a local CLI:
  - Use realistic module paths like `mirrorcore/terminal/agent.py`, `mirrorcore/db/session.py`.
  - Example errors: `sqlite3.OperationalError`, `RuntimeError`, `ValueError`, `click.exceptions.ClickException`.
- Keep traces **short but meaningful**:
  - Typically 3–6 frames.
  - At least one frame pointing into Mirrorcore code, one into the database or config layer.
- Use **stable, repeated error messages** when testing hypothesis ranking:
  - The exact same final error line should appear multiple times.
  - Vary only the surrounding context if you need to show different attempts.

Example trace pattern:

```text
Traceback (most recent call last):
  File "/home/kali/Desktop/mirrorcore/mirrorcore/cli.py", line 42, in main
    app()
  File "/home/kali/.venv/lib/python3.11/site-packages/click/core.py", line 1130, in __call__
    return self.main(*args, **kwargs)
  File "/home/kali/Desktop/mirrorcore/mirrorcore/db/session.py", line 97, in get_connection
    conn = sqlite3.connect(db_path)
sqlite3.OperationalError: database is locked
```

Reuse the same `OperationalError: database is locked` line in multiple steps when simulating repeated evidence.

---

## Stall Simulation Patterns

When simulating stalls, prefer patterns like:

- **Re-running the same failing CLI command** 2–4 times with identical errors.
- **Trying nearby options** (`--verbose`, `--debug`) that add noise but not new clues.
- **Inspecting irrelevant files** whose contents don't change the hypothesis set.

Clearly mark in prose that:
- The investigation is **not gaining new information**.
- A higher-level system **should consider switching strategies** or asking for new information.

Example stall snippet:

```bash
[user@host]$ mirrorcore run-terminal-session
```

```text
sqlite3.OperationalError: database is locked
...
[same traceback as before]
```

```bash
[user@host]$ mirrorcore run-terminal-session --verbose
```

```text
[DEBUG] Opening database at /home/kali/.mirrorcore/mirrorcore.db
sqlite3.OperationalError: database is locked
...
```

Stall annotation:
- Despite toggling verbosity and re-running, the **core error did not change**, indicating a stall.

---

## Strategy Switching Patterns

After recognizing a stall, introduce a **clear strategy shift**, such as:

- From **re-running CLI** → to **checking file locks / processes** (`lsof`, `ps`, `fuser`).
- From **guessing config** → to **reading `mirrorcore --help` or configuration docs**.
- From **treating it as a Mirrorcore bug** → to **verifying environment dependencies** (Python version, virtualenv, SQLite file permissions).

Clearly document:
- **Old strategy name** (e.g., "rerun CLI with more flags").
- **New strategy name** (e.g., "inspect database lock holders").
- **Reason for switch** (e.g., "repeated `database is locked` strongly suggests an external lock holder").

---

## Resolution Scenarios

When you need to provide a **resolution path**, choose from realistic classes of fixes:

- **Environment fixes**
  - Installing missing dependencies.
  - Activating the correct virtual environment.
  - Fixing `$PATH` or setting `MIRRORCORE_DB_PATH`.

- **Database fixes**
  - Closing or killing stale processes.
  - Removing corrupted temporary files (with caution).
  - Allowing Mirrorcore to recreate a local SQLite file.

- **Configuration fixes**
  - Creating/editing a config file under the expected directory.
  - Correcting a mis-typed profile or phase name.

- **Usage fixes**
  - Adjusting the command to a supported one.
  - Adding a required flag (e.g., selecting the correct phase or target).

For each resolution:
- Show the **fix action**.
- Show the **post-fix command and successful output**.
- Provide a brief **resolution summary** suitable for later capture into Mirrorcore's memory system.

---

## Minimal Example (Short Scenario)

Use this example as a pattern when you need a **compact** investigation that still triggers ranking, stall, strategy switch, and resolution.

```markdown
## Scenario
- Goal: Run Mirrorcore terminal investigation phase
- Environment: Linux, Python 3.11, local SQLite, offline
- Initial Symptom: `mirrorcore run-terminal-session` hangs then fails with `database is locked`

## Phase 1: Initial Failure

### Step 1
**Command**
```bash
[user@host]$ mirrorcore run-terminal-session
```

**Output**
```text
Starting terminal investigation...
Traceback (most recent call last):
  File "/home/kali/Desktop/mirrorcore/mirrorcore/cli.py", line 42, in main
    app()
  File "/home/kali/Desktop/mirrorcore/mirrorcore/db/session.py", line 97, in get_connection
    conn = sqlite3.connect("/home/kali/.mirrorcore/mirrorcore.db")
sqlite3.OperationalError: database is locked
```

**Notes**
- Suggests contention on the local SQLite database.

## Phase 2: Hypothesis Branching

### Step 2
**Command**
```bash
[user@host]$ mirrorcore run-terminal-session --verbose
```

**Output**
```text
[DEBUG] Opening database at /home/kali/.mirrorcore/mirrorcore.db
sqlite3.OperationalError: database is locked
```

**Notes**
- Confirms the same `database is locked` condition; likely another process.

## Phase 3: Stall and Strategy Switch

### Stall Segment
**Commands / Outputs**
```bash
[user@host]$ mirrorcore run-terminal-session --retry 3
```

```text
Attempt 1/3...
sqlite3.OperationalError: database is locked
Attempt 2/3...
sqlite3.OperationalError: database is locked
Attempt 3/3...
sqlite3.OperationalError: database is locked
```

**Stall Observation**
- Multiple retries yield identical `sqlite3.OperationalError: database is locked`. No new evidence: investigation is stalled.

### Strategy Switch
- Previous strategy: Re-run Mirrorcore with retries and verbosity.
- New strategy: Inspect which process holds the database file lock.
- Rationale: Repeated identical SQLite error suggests an external lock holder.

### Step 3
**Command**
```bash
[user@host]$ lsof /home/kali/.mirrorcore/mirrorcore.db
```

```text
python3  9321 kali  txt REG  252,0  32768 /home/kali/.mirrorcore/mirrorcore.db
```

**Notes**
- Shows an old Python process still holding the database file open.

## Phase 4: Resolution and Verification

### Fix
**Action**
- Terminate the stale Python process holding the database lock.

**Evidence**
```bash
[user@host]$ kill 9321
[user@host]$ mirrorcore run-terminal-session
```

```text
Starting terminal investigation...
Session initialized.
Ready for terminal troubleshooting.
```

### Resolution Summary
- Root cause: Stale Python process holding a lock on the Mirrorcore SQLite database.
- Key evidence: Repeated `sqlite3.OperationalError: database is locked` and `lsof` output showing PID 9321 on `mirrorcore.db`.
- Final confirmation: After killing PID 9321, `mirrorcore run-terminal-session` completes successfully and initializes the investigation session.
```

Use this example as a template; extend or vary it to test different error classes, stalls, strategies, and resolutions.

