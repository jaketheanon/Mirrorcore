# CLI and Investigation Workflow Rules

Mirrorcore is a Python CLI project. Command behavior is part of the system contract.

Core command flows to preserve:
- analyze-log
- analyze-followup
- debug-outcome
- resolve-investigation, if present
- related history/review commands

Rules:
- do not rename existing commands unless explicitly required
- do not silently repurpose commands to do different jobs
- preserve investigation lifecycle continuity:
  - session creation
  - follow-up evidence updates
  - investigation step tracking
  - hypothesis updates
  - stall detection
  - strategy switching
  - resolution capture and session closure
- preserve readable, deterministic CLI output
- prefer extending existing workflows over creating duplicate commands with overlapping purpose

When modifying CLI workflows:
- keep prompts and output sections structurally consistent
- ensure resolved/abandoned sessions are not reused for follow-up
- ensure active/stalled investigations remain discoverable when appropriate
- update README examples when command behavior changes