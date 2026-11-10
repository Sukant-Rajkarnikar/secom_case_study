# SECOM project instructions

## Testing and audit protocol (central)

All testing goes through the **`secom-auditor`** agent (`.claude/agents/secom-auditor.md`,
also available as `/audit [focus]`). It is read-only and reports. It never fixes.

- **After any change** to `src/`, `tests/`, `notebooks/` or anything in `data/processed/` or
  `reports/`: run `/audit` (or the agent) before saying the work is done.
- **Before quoting a number** in the diary, notebook or slides: run `/audit D-NNN` for the
  entry that quotes it.
- Fast check without the agent: `.venv/bin/python -m pytest -q` and
  `.venv/bin/python -m secom_analysis.audit` (exit 1 on CRITICAL/HIGH).
- **Baseline:** `docs/audit_baseline.json` holds the accepted results. Drift is measured
  against it. Only the user accepts a new baseline
  (`python -m secom_analysis.audit --accept-baseline`), after reviewing the drift. Commit
  it and log the acceptance in the diary.
- New checks that can be made deterministic go into `src/secom_analysis/audit.py` with a
  test in `tests/test_audit.py`; judgement checks go into the agent prompt.

## Standing rules

- Log every change in `docs/scientific_diary.md` (append-only, next free ID, update the index
  and open items).
- The test set (`data/processed/test_holdout.csv`, and `secom_raw.csv`, which contains it) is
  sealed until final evaluation.
- Python: `.venv/bin/python`.
