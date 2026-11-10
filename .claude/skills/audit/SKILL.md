---
name: audit
description: Run the SECOM auditor - tests, deterministic audit, drift vs accepted baseline, diary cross-check, bug and reasoning-gap hunt. Use after any change to code, data or results, or before quoting numbers. Args - optional focus (e.g. "D-035", "pipeline", "deep pipeline").
context: fork
agent: secom-auditor
---

Audit the SECOM project now, following your full procedure.

Focus (may be empty, meaning a general audit of recent changes): $ARGUMENTS

If the focus names a diary ID, cross-check every number in that entry against its artifacts.
If it starts with "deep", run deep mode for the named stage only.
Return the report in your standard format.
