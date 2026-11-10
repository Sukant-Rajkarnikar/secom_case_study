from secom_analysis.audit import check_diary, compare, parse_diary

DIARY = """# Diary

## Decision index

| ID | Date | Phase | Type | Title | Status |
|---|---|---|---|---|---|
| D-000 | 2026-11-09 | X | Finding | Start | Done |
| D-001 | 2026-11-09 | X | Decision | Plan A | Superseded by D-002 |
| D-002 | 2026-11-09 | X | Decision | Plan B | Done |

## Open items

| From | Item | Status |
|---|---|---|
| D-001 | do a thing | Closed by D-002 |

## Log

### D-000 · Start
- **Date:** 2026-11-09
- **Type:** Finding
- **Status:** Done

**What happened** — fine.

### D-001 · Plan A
- **Date:** 2026-11-09
- **Type:** Decision
- **Status:** Superseded by D-002

### D-002 · Plan B
- **Date:** 2026-11-09
- **Type:** Decision
- **Status:** Done (replaces D-001)
"""


def _checks(findings, severity=None):
    return {f.check for f in findings if severity is None or f.severity == severity}


def test_parse_diary_reads_index_entries_and_open_items() -> None:
    parsed = parse_diary(DIARY)
    assert set(parsed["index"]) == {"D-000", "D-001", "D-002"}
    assert parsed["entries"]["D-002"][0]["status"].startswith("Done")
    assert parsed["open_items"][0]["from"] == "D-001"


def test_consistent_diary_has_only_info() -> None:
    findings = check_diary(DIARY)
    assert {f.severity for f in findings} == {"INFO"}


def test_diary_detects_missing_entry_dangling_reference_and_status_mismatch() -> None:
    broken = DIARY.replace("| D-002 | 2026-11-09 | X | Decision | Plan B | Done |",
                           "| D-002 | 2026-11-09 | X | Decision | Plan B | Proposed |\n"
                           "| D-003 | 2026-11-09 | X | Decision | Ghost | Done |")
    broken = broken.replace("Closed by D-002", "Closed by D-009")
    checks = _checks(check_diary(broken))
    assert {"index_without_entry", "status_mismatch", "open_item_closure"} <= checks


def test_compare_flags_material_metric_drift_and_feature_change() -> None:
    base = {"state": {"kept_columns": 450}, "selected_features": ["a", "b"], "hashes": {},
            "metrics": {"m::x::pr_auc": 0.20, "m::x::roc_auc": 0.70, "m::x::cost": 6600.0}}
    same = compare(base, base)
    assert _checks(same) == {"none"}

    cur = {**base, "selected_features": ["a", "c"],
           "metrics": {"m::x::pr_auc": 0.17, "m::x::roc_auc": 0.7000001, "m::x::cost": 6610.0}}
    findings = compare(base, cur)
    assert "selected_features" in _checks(findings, "HIGH")
    material = next(f for f in findings if f.check == "metrics_material")
    assert any("pr_auc" in e for e in material.evidence)
    assert not any("cost" in e for e in material.evidence)  # 0.15% relative change on a count
    assert "metrics_noise" in _checks(findings, "LOW")


def test_compare_treats_preparation_state_change_as_critical() -> None:
    base = {"state": {"kept_columns": 450}, "metrics": {}, "hashes": {}}
    cur = {"state": {"kept_columns": 449}, "metrics": {}, "hashes": {}}
    assert "preparation_state" in _checks(compare(base, cur), "CRITICAL")
