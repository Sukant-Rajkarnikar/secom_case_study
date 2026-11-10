"""Project audit: invariants, diary consistency, staleness and result drift.

Deterministic checks that the ``secom-auditor`` agent runs first, before it reasons about
anything. Nothing here modifies data, code or the diary; it only reads and reports.

    python -m secom_analysis.audit                    # run all checks, compare with baseline
    python -m secom_analysis.audit --accept-baseline  # after review: current results become baseline

Outputs
    reports/audit/latest.md         human-readable findings
    reports/audit/latest.json       machine-readable findings + snapshot
    reports/audit/history.csv       one row per run (drift over time)
    docs/audit_baseline.json        accepted snapshot (tracked in git), written only on --accept

Exit code 1 if any CRITICAL or HIGH finding, else 0.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = Path("data/processed")
REPORTS = Path("reports")
DIARY = Path("docs/scientific_diary.md")
BASELINE = Path("docs/audit_baseline.json")
AUDIT_DIR = REPORTS / "audit"

SEVERITIES = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO")
TOTAL_ROWS = 1567
TOTAL_FEATURES = 590

# Result tables whose numbers are tracked for drift: path -> key columns.
METRIC_TABLES: dict[str, list[str]] = {
    "reports/pipeline/cv_summary.csv": ["balancer"],
    "reports/pipeline/confusion_oof.csv": [],
    "reports/feature_selection/cv_summary.csv": ["variant"],
    "reports/imputation_cv/cv_summary.csv": ["model", "strategy"],
    "reports/imputation/comparison.csv": ["strategy"],
    "reports/split_study/split_summary.csv": ["model", "scheme"],
}

# Files whose content hash is tracked (any change is reported).
HASHED_FILES = [
    "data/processed/train_raw.csv",
    "data/processed/train_reduced.csv",
    "data/processed/train_prepared.csv",
    "data/processed/test_holdout.csv",
    "data/processed/three_sigma_bounds.csv",
    "data/processed/preparation_state.json",
    "data/processed/selected_features.json",
    *METRIC_TABLES,
]

# Producer module -> outputs it writes. Used for staleness checks.
PRODUCERS: dict[str, list[str]] = {
    "src/secom_analysis/health.py": ["reports/health"],
    "src/secom_analysis/impute.py": ["reports/imputation", "data/processed/imputed"],
    "src/secom_analysis/impute_cv.py": ["reports/imputation_cv"],
    "src/secom_analysis/split_study.py": ["reports/split_study"],
    "src/secom_analysis/selection.py": [
        "reports/feature_selection",
        "data/processed/selected_features.json",
    ],
    "src/secom_analysis/pipeline.py": ["reports/pipeline"],
    "src/secom_analysis/cli.py": ["reports/descriptive_report.md", "reports/eda_overview.png"],
}
# Modules every producer depends on: if these change, all outputs are potentially stale.
SHARED_MODULES = ["src/secom_analysis/preparation.py", "src/secom_analysis/data.py"]

# Course rules (see docs/build_plan.md): patterns that must not appear in src/.
FORBIDDEN_PATTERNS = {
    r"1\.5\s*\*\s*\(?\s*(iqr|IQR|q3|p75)|(iqr|IQR)\s*\*\s*1\.5": "IQR outlier rule (course mandates 3σ)",
    r"SimpleImputer\(\s*strategy\s*=\s*[\"']mean": "mean imputation (banned by the manual)",
}


@dataclass
class Finding:
    """One audit result. Header: severity, area, check name, message, optional evidence."""

    severity: str
    area: str
    check: str
    message: str
    evidence: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- helpers


def _p(rel: str | Path) -> Path:
    """Header: resolve a repo-relative path against the project root."""
    return ROOT / rel


def _sha256(path: Path) -> str:
    """Header: SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    """Header: run a git command in the repo; empty string on failure."""
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _newest_mtime(path: Path) -> float | None:
    """Header: newest mtime of a file, or of any file below a directory."""
    if path.is_file():
        return path.stat().st_mtime
    if path.is_dir():
        times = [f.stat().st_mtime for f in path.rglob("*") if f.is_file()]
        return max(times) if times else None
    return None


def _oldest_mtime(path: Path) -> float | None:
    """Header: oldest mtime of a file, or of any file below a directory."""
    if path.is_file():
        return path.stat().st_mtime
    if path.is_dir():
        times = [f.stat().st_mtime for f in path.rglob("*") if f.is_file()]
        return min(times) if times else None
    return None


def _feature_columns(path: Path) -> list[str]:
    """Header: feature column names of a split CSV (header only)."""
    header = pd.read_csv(path, nrows=0).columns
    return [c for c in header if c.startswith("feature_")]


# ---------------------------------------------------------------- invariant checks


def check_seal() -> list[Finding]:
    """Header: sealed test set matches its .sha256 file and preparation_state.json."""
    out: list[Finding] = []
    test = _p(PROCESSED / "test_holdout.csv")
    seal = _p(PROCESSED / "test_holdout.csv.sha256")
    state = _p(PROCESSED / "preparation_state.json")
    if not test.exists():
        return [Finding("CRITICAL", "test-seal", "test_exists", "test_holdout.csv is missing")]
    actual = _sha256(test)
    if seal.exists():
        sealed = seal.read_text().split()[0]
        if sealed != actual:
            out.append(
                Finding("CRITICAL", "test-seal", "seal_file",
                        "test_holdout.csv no longer matches its .sha256 seal",
                        [f"seal {sealed[:16]}", f"file {actual[:16]}"])
            )
    else:
        out.append(Finding("CRITICAL", "test-seal", "seal_file", ".sha256 seal file missing"))
    if state.exists():
        recorded = json.loads(state.read_text()).get("test_holdout_sha256")
        if recorded != actual:
            out.append(
                Finding("CRITICAL", "test-seal", "state_hash",
                        "preparation_state.json records a different test hash",
                        [f"state {str(recorded)[:16]}", f"file {actual[:16]}"])
            )
    if not out:
        out.append(Finding("INFO", "test-seal", "seal", f"test set sealed and intact ({actual[:16]})"))
    return out


def check_split_and_preparation() -> list[Finding]:
    """Header: row/column bookkeeping across raw, reduced, prepared splits and the 3σ bounds."""
    out: list[Finding] = []
    files = {n: _p(PROCESSED / f"{n}.csv") for n in
             ("train_raw", "test_holdout", "train_reduced", "train_prepared", "three_sigma_bounds")}
    missing = [n for n, f in files.items() if not f.exists()]
    if missing:
        return [Finding("HIGH", "preparation", "files", "processed files missing", missing)]

    train_ids = pd.read_csv(files["train_raw"], usecols=["source_line"])["source_line"]
    test_ids = pd.read_csv(files["test_holdout"], usecols=["source_line"])["source_line"]
    shared = set(train_ids) & set(test_ids)
    if shared:
        out.append(Finding("CRITICAL", "split", "overlap",
                           f"{len(shared)} wafers are in both train and test",
                           [str(s) for s in sorted(shared)[:10]]))
    if len(train_ids) + len(test_ids) != TOTAL_ROWS:
        out.append(Finding("HIGH", "split", "row_total",
                           f"train {len(train_ids)} + test {len(test_ids)} != {TOTAL_ROWS}"))
    if train_ids.duplicated().any() or test_ids.duplicated().any():
        out.append(Finding("HIGH", "split", "duplicate_ids", "duplicate source_line within a split"))

    state = json.loads(_p(PROCESSED / "preparation_state.json").read_text())
    zero, high = state.get("zero_variance_columns", []), state.get("high_missing_columns", [])
    kept = state.get("kept_columns")
    reduced_cols = _feature_columns(files["train_reduced"])
    prepared_cols = _feature_columns(files["train_prepared"])
    raw_cols = _feature_columns(files["train_raw"])

    if len(raw_cols) != TOTAL_FEATURES:
        out.append(Finding("HIGH", "preparation", "raw_columns",
                           f"train_raw has {len(raw_cols)} features, expected {TOTAL_FEATURES}"))
    n_zero = len(zero) if isinstance(zero, list) else zero
    n_high = len(high) if isinstance(high, list) else high
    if isinstance(kept, list):
        kept = len(kept)
    if n_zero + n_high + len(reduced_cols) != len(raw_cols):
        out.append(Finding("HIGH", "preparation", "column_accounting",
                           f"{n_zero} zero-var + {n_high} high-missing + {len(reduced_cols)} kept "
                           f"!= {len(raw_cols)} raw"))
    if kept is not None and kept != len(reduced_cols):
        out.append(Finding("HIGH", "preparation", "kept_columns",
                           f"state says {kept} kept, train_reduced has {len(reduced_cols)}"))
    leaked = set(zero if isinstance(zero, list) else []) & set(reduced_cols)
    leaked |= set(high if isinstance(high, list) else []) & set(reduced_cols)
    if leaked:
        out.append(Finding("HIGH", "preparation", "dropped_columns_present",
                           "columns marked as dropped are still in train_reduced", sorted(leaked)[:10]))
    if prepared_cols != reduced_cols:
        out.append(Finding("MEDIUM", "preparation", "prepared_vs_reduced",
                           "train_prepared and train_reduced have different feature columns"))

    bounds = pd.read_csv(files["three_sigma_bounds"], index_col=0)
    if len(bounds) != len(reduced_cols) or set(bounds.index) != set(reduced_cols):
        out.append(Finding("MEDIUM", "preparation", "bounds_columns",
                           f"3σ bounds cover {len(bounds)} columns vs {len(reduced_cols)} kept"))
    else:
        prepared = pd.read_csv(files["train_prepared"], usecols=prepared_cols)
        lo, hi = bounds["lower"].reindex(prepared_cols), bounds["upper"].reindex(prepared_cols)
        tol = 1e-9 * (1 + np.maximum(lo.abs(), hi.abs()))
        below = (prepared.lt(lo - tol, axis=1)).sum().sum()
        above = (prepared.gt(hi + tol, axis=1)).sum().sum()
        if below or above:
            out.append(Finding("HIGH", "preparation", "capping",
                               f"train_prepared has {below} values below and {above} above the 3σ bounds"))
    if not out:
        out.append(Finding("INFO", "preparation", "bookkeeping",
                           f"split and column accounting consistent: {len(train_ids)}/{len(test_ids)} "
                           f"rows, {n_zero}+{n_high}+{len(reduced_cols)}={len(raw_cols)} columns"))
    return out


def check_selected_features() -> list[Finding]:
    """Header: selected feature list is valid and consistent with the pipeline's report."""
    out: list[Finding] = []
    path = _p(PROCESSED / "selected_features.json")
    if not path.exists():
        return out
    selected = json.loads(path.read_text()).get("selected_features", [])
    reduced = set(_feature_columns(_p(PROCESSED / "train_reduced.csv")))
    unknown = [f for f in selected if f not in reduced]
    if unknown:
        out.append(Finding("HIGH", "selection", "unknown_features",
                           "selected features not present in train_reduced", unknown))
    if len(set(selected)) != len(selected):
        out.append(Finding("MEDIUM", "selection", "duplicates", "duplicate names in selected_features"))
    if not 15 <= len(selected) <= 20:
        out.append(Finding("LOW", "selection", "count",
                           f"{len(selected)} stable features; the manual targets about 15–20"))
    checks = _p(REPORTS / "pipeline/pipeline_checks.json")
    if checks.exists():
        in_fold = json.loads(checks.read_text()).get("selected_features", [])
        if in_fold and set(in_fold) != set(selected):
            out.append(Finding(
                "INFO", "selection", "fixed_vs_in_fold",
                f"stable set ({len(selected)}) differs from the pipeline's in-fold Boruta set "
                f"({len(in_fold)}); expected if the pipeline refits Boruta per fold, but any "
                "report that quotes 'the selected features' must say which set it means",
                [f"only stable: {sorted(set(selected) - set(in_fold))}",
                 f"only in-fold: {sorted(set(in_fold) - set(selected))}"]))
    return out


def check_test_access() -> list[Finding]:
    """Header: flag code or notebooks that read the sealed test set (anything but a seal check)."""
    out: list[Finding] = []
    allowed = re.compile(r"verify_sealed|save_split\(.*seal=True|\.sha256|test_path\b|^\s*#|\"#")
    reader = re.compile(r"read_csv|load_split|open\(|np\.loadtxt|read_sql|COPY ")
    # secom_raw.csv holds the full dataset, test rows included (D-030): reading it bypasses the seal.
    full_copy = re.compile(r"secom_raw\.csv")
    paths = list(_p("src").rglob("*.py")) + list(_p("notebooks").glob("*.ipynb"))
    for path in paths:
        if path.name == "audit.py":
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if full_copy.search(line) and reader.search(line):
                out.append(Finding("HIGH", "test-seal", "full_copy_read",
                                   f"reads secom_raw.csv (contains test rows) in "
                                   f"{path.relative_to(ROOT)}:{number}", [line.strip()[:160]]))
                continue
            if "test_holdout" not in line or allowed.search(line):
                continue
            if reader.search(line):
                severity = "HIGH"
            elif path.suffix == ".py":
                severity = "LOW"
            else:
                continue  # notebook prose / printed text
            out.append(Finding(severity, "test-seal", "test_access",
                               f"reference to the sealed test set in {path.relative_to(ROOT)}:{number}",
                               [line.strip()[:160]]))
    return out


def check_course_rules() -> list[Finding]:
    """Header: grep src/ for patterns the course forbids (IQR outliers, mean imputation)."""
    out: list[Finding] = []
    for path in _p("src").rglob("*.py"):
        if path.name == "audit.py":
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for pattern, what in FORBIDDEN_PATTERNS.items():
                if re.search(pattern, line) and not line.strip().startswith("#"):
                    out.append(Finding("MEDIUM", "course-rules", "forbidden_pattern",
                                       f"{what} in {path.relative_to(ROOT)}:{number}",
                                       [line.strip()[:160]]))
    state_path = _p(PROCESSED / "preparation_state.json")
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("missing_threshold_pct") not in (60, 60.0):
            out.append(Finding("HIGH", "course-rules", "missing_threshold",
                               f"missing threshold is {state.get('missing_threshold_pct')}, course sets 60%"))
        if "3 std" not in str(state.get("outlier_rule", "")):
            out.append(Finding("HIGH", "course-rules", "outlier_rule",
                               f"outlier rule is '{state.get('outlier_rule')}', course sets 3σ"))
    return out


# ---------------------------------------------------------------- diary


INDEX_ROW = re.compile(r"^\|\s*(D-\d{3})\s*\|\s*([^|]*)\|\s*([^|]*)\|\s*([^|]*)\|\s*([^|]*)\|\s*([^|]*)\|")
ENTRY_HEAD = re.compile(r"^### (D-\d{3}) · (.+)$")
STATUS_LINE = re.compile(r"^- \*\*Status:\*\*\s*(.+)$")
DATE_LINE = re.compile(r"^- \*\*Date:\*\*\s*(\S+)")
ID_REF = re.compile(r"D-\d{3}")


def parse_diary(text: str) -> dict:
    """Header: split the diary into index rows, log entries and open items."""
    lines = text.splitlines()
    section, index, entries, open_items, current = None, {}, {}, [], None
    for line in lines:
        if line.startswith("## "):
            section = line[3:].strip().lower()
            current = None
            continue
        if section == "decision index" and (m := INDEX_ROW.match(line)):
            index.setdefault(m[1], []).append(
                {"date": m[2].strip(), "title": m[5].strip(), "status": m[6].strip()})
        elif section == "open items" and line.startswith("| D-"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            open_items.append({"from": cells[0], "item": cells[1], "status": cells[-1]})
        elif section == "log":
            if m := ENTRY_HEAD.match(line):
                current = m[1]
                entries.setdefault(current, []).append({"title": m[2].strip(), "status": None,
                                                        "date": None, "body": []})
            elif current:
                entry = entries[current][-1]
                entry["body"].append(line)
                if (s := STATUS_LINE.match(line)) and entry["status"] is None:
                    entry["status"] = s[1].strip()
                if (d := DATE_LINE.match(line)) and entry["date"] is None:
                    entry["date"] = d[1].strip()
    return {"index": index, "entries": entries, "open_items": open_items}


def check_diary(text: str | None = None) -> list[Finding]:
    """Header: diary index, log and open items agree with each other."""
    if text is None:
        if not _p(DIARY).exists():
            return [Finding("HIGH", "diary", "exists", "docs/scientific_diary.md missing")]
        text = _p(DIARY).read_text(encoding="utf-8")
    d = parse_diary(text)
    index, entries, items = d["index"], d["entries"], d["open_items"]
    out: list[Finding] = []
    for kind, table in (("index", index), ("log", entries)):
        dupes = sorted(k for k, v in table.items() if len(v) > 1)
        if dupes:
            out.append(Finding("HIGH", "diary", f"duplicate_{kind}_ids", f"IDs repeated in {kind}", dupes))
    only_index = sorted(set(index) - set(entries))
    only_log = sorted(set(entries) - set(index))
    if only_index:
        out.append(Finding("HIGH", "diary", "index_without_entry", "index rows with no log entry", only_index))
    if only_log:
        out.append(Finding("HIGH", "diary", "entry_without_index", "log entries missing from index", only_log))

    numbers = sorted(int(k[2:]) for k in set(index) | set(entries))
    if numbers:
        gaps = sorted(set(range(numbers[0], numbers[-1] + 1)) - set(numbers))
        if gaps:
            out.append(Finding("MEDIUM", "diary", "id_gaps", "missing IDs", [f"D-{g:03d}" for g in gaps]))
    order = [int(k[2:]) for k in entries]
    if order != sorted(order):
        swaps = [f"D-{a:03d} before D-{b:03d}" for a, b in pairwise(order) if a > b]
        out.append(Finding("LOW", "diary", "log_order", "log entries out of ID order", swaps))

    known = set(index) | set(entries)
    today = datetime.now().date().isoformat()
    for key in sorted(set(index) & set(entries)):
        row, entry = index[key][0], entries[key][0]
        if _status_word(row["status"]) != _status_word(entry["status"] or ""):
            out.append(Finding("MEDIUM", "diary", "status_mismatch",
                               f"{key}: index says '{row['status']}', entry says '{entry['status']}'"))
        if row["date"] != entry["date"]:
            out.append(Finding("LOW", "diary", "date_mismatch",
                               f"{key}: index date {row['date']}, entry date {entry['date']}"))
        if entry["date"] and entry["date"] > today:
            out.append(Finding("MEDIUM", "diary", "future_date", f"{key} is dated {entry['date']}"))
        a, b = _norm(row["title"]), _norm(entry["title"])
        if not (a.startswith(b) or b.startswith(a)):
            out.append(Finding("LOW", "diary", "title_mismatch",
                               f"{key}: '{row['title']}' vs '{entry['title']}'"))
        for ref in ID_REF.findall(row["status"]):
            if ref not in known:
                out.append(Finding("HIGH", "diary", "dangling_supersede", f"{key} superseded by unknown {ref}"))
        body = "\n".join(entry["body"])
        for ref in set(ID_REF.findall(body)) - known - {key}:
            out.append(Finding("MEDIUM", "diary", "dangling_reference", f"{key} cites unknown {ref}"))
        finished = _status_word(entry["status"] or "") in {"done", "accepted"}
        if finished and "What happened" not in body and entry_type(entry) in {"experiment", "finding"}:
            out.append(Finding("LOW", "diary", "no_results",
                               f"{key} is a {entry_type(entry)} marked {entry['status']} "
                               "but has no 'What happened' section"))

    for item in items:
        if item["from"] not in known:
            out.append(Finding("MEDIUM", "diary", "open_item_source",
                               f"open item cites unknown {item['from']}", [item["item"][:120]]))
        for ref in ID_REF.findall(item["status"]):
            if ref not in known:
                out.append(Finding("MEDIUM", "diary", "open_item_closure",
                                   f"open item closed by unknown {ref}", [item["item"][:120]]))
    n_open = sum(1 for i in items if i["status"].lower().startswith("open"))
    out.append(Finding("INFO", "diary", "summary",
                       f"{len(entries)} entries, {len(index)} index rows, {len(items)} open-item rows "
                       f"({n_open} still Open); latest {max(entries, default='-')}"))
    return out


def entry_type(entry: dict) -> str:
    """Header: the **Type:** value of a parsed diary entry, lower-case."""
    for line in entry["body"]:
        if line.startswith("- **Type:**"):
            return line.split("**Type:**", 1)[1].strip().lower()
    return ""


def _status_word(status: str) -> str:
    """Header: first word of a status ('Done (closes ...)' -> 'done'; 'Superseded by D-1' kept whole)."""
    status = status.strip().lower()
    if status.startswith("superseded"):
        return " ".join(status.split()[:3])
    return status.split(" ")[0].split("(")[0] if status else ""


def _norm(title: str) -> str:
    """Header: normalise a title for loose comparison (case, backticks, punctuation)."""
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def check_diary_freshness() -> list[Finding]:
    """Header: code or results changed after the last diary edit (possible unlogged change)."""
    out: list[Finding] = []
    diary = _p(DIARY)
    if not diary.exists():
        return out
    diary_time = diary.stat().st_mtime
    dirty = _git("status", "--porcelain", "--", "src", "tests", "notebooks")
    diary_dirty = bool(_git("status", "--porcelain", "--", str(DIARY)))
    if dirty and not diary_dirty:
        out.append(Finding("MEDIUM", "diary", "uncommitted_without_diary",
                           "uncommitted changes in src/tests/notebooks but the diary is unchanged",
                           dirty.splitlines()[:15]))
    newer = []
    for rel in [*PRODUCERS, *SHARED_MODULES]:
        path = _p(rel)
        if path.exists() and path.stat().st_mtime > diary_time + 60:
            newer.append(rel)
    for rel in {o for outs in PRODUCERS.values() for o in outs}:
        stamp = _newest_mtime(_p(rel))
        if stamp and stamp > diary_time + 60:
            newer.append(rel)
    if newer:
        out.append(Finding("LOW", "diary", "changed_after_diary",
                           "files modified after the last diary edit; check they are logged",
                           sorted(newer)))
    return out


# ---------------------------------------------------------------- staleness


def check_staleness() -> list[Finding]:
    """Header: outputs older than the code that produces them or the data they are built from."""
    out: list[Finding] = []
    shared = max((_p(m).stat().st_mtime for m in SHARED_MODULES if _p(m).exists()), default=0)
    upstream = _p(PROCESSED / "train_reduced.csv")
    upstream_time = upstream.stat().st_mtime if upstream.exists() else 0
    for module, outputs in PRODUCERS.items():
        code = _p(module)
        if not code.exists():
            continue
        code_time = max(code.stat().st_mtime, shared)
        for rel in outputs:
            oldest = _oldest_mtime(_p(rel))
            if oldest is None:
                out.append(Finding("LOW", "staleness", "missing_output", f"{rel} not produced yet"))
                continue
            if oldest + 1 < code_time:
                out.append(Finding("MEDIUM", "staleness", "code_newer",
                                   f"{rel} is older than {module} (or a shared module); rerun before quoting it",
                                   [f"output {_ts(oldest)}", f"code {_ts(code_time)}"]))
            if rel.startswith("reports") and oldest + 1 < upstream_time:
                out.append(Finding("HIGH", "staleness", "data_newer",
                                   f"{rel} predates the current train_reduced.csv",
                                   [f"output {_ts(oldest)}", f"data {_ts(upstream_time)}"]))
    return out


def _ts(stamp: float) -> str:
    """Header: format a unix time as local ISO minutes."""
    return datetime.fromtimestamp(stamp).isoformat(timespec="minutes")


# ---------------------------------------------------------------- snapshot and drift


def snapshot() -> dict:
    """Header: everything tracked for drift: hashes, metrics, preparation state, selection."""
    snap: dict = {"created": datetime.now().isoformat(timespec="seconds"),
                  "git_head": _git("rev-parse", "--short", "HEAD"),
                  "hashes": {}, "metrics": {}, "state": {}, "selected_features": None,
                  "pipeline_checks": None}
    for rel in HASHED_FILES:
        path = _p(rel)
        snap["hashes"][rel] = _sha256(path) if path.exists() else None
    for rel, keys in METRIC_TABLES.items():
        path = _p(rel)
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        keys = keys or [c for c in frame.columns if frame[c].dtype == object]
        numeric = [c for c in frame.select_dtypes("number").columns if c not in keys]
        for _, row in frame.iterrows():
            key = "/".join(str(row[k]) for k in keys) if keys else str(_)
            for col in numeric:
                value = row[col]
                snap["metrics"][f"{rel}::{key}::{col}"] = None if pd.isna(value) else float(value)
    state = _p(PROCESSED / "preparation_state.json")
    if state.exists():
        raw = json.loads(state.read_text())
        snap["state"] = {k: (len(v) if isinstance(v, list) else v) for k, v in raw.items()}
    selected = _p(PROCESSED / "selected_features.json")
    if selected.exists():
        snap["selected_features"] = sorted(json.loads(selected.read_text()).get("selected_features", []))
    checks = _p(REPORTS / "pipeline/pipeline_checks.json")
    if checks.exists():
        snap["pipeline_checks"] = json.loads(checks.read_text())
    return snap


def compare(baseline: dict, current: dict, tolerance: float = 1e-9,
            material: float = 0.01) -> list[Finding]:
    """Header: classify differences between an accepted baseline and the current snapshot.

    tolerance: smallest metric change reported at all. material: absolute change that counts
    as a real drift (HIGH) rather than numeric noise (LOW).
    """
    out: list[Finding] = []
    base_state, cur_state = baseline.get("state", {}), current.get("state", {})
    for key in sorted(set(base_state) | set(cur_state)):
        if base_state.get(key) != cur_state.get(key):
            out.append(Finding("CRITICAL", "drift", "preparation_state",
                               f"preparation_state.{key} changed",
                               [f"baseline {base_state.get(key)}", f"current {cur_state.get(key)}"]))
    if baseline.get("selected_features") != current.get("selected_features"):
        b, c = set(baseline.get("selected_features") or []), set(current.get("selected_features") or [])
        out.append(Finding("HIGH", "drift", "selected_features", "stable Boruta feature set changed",
                           [f"removed {sorted(b - c)}", f"added {sorted(c - b)}"]))

    changed_hashes = []
    for rel in sorted(set(baseline.get("hashes", {})) | set(current.get("hashes", {}))):
        b, c = baseline.get("hashes", {}).get(rel), current.get("hashes", {}).get(rel)
        if b != c:
            changed_hashes.append(rel)
            if rel.endswith("test_holdout.csv"):
                out.append(Finding("CRITICAL", "drift", "test_hash", "sealed test file changed"))
    data_changed = [r for r in changed_hashes if r.startswith("data/")]
    if data_changed:
        out.append(Finding("HIGH", "drift", "data_files",
                           "processed data files changed since baseline", data_changed))

    bm, cm = baseline.get("metrics", {}), current.get("metrics", {})
    gone = sorted(set(bm) - set(cm))
    new = sorted(set(cm) - set(bm))
    if gone:
        out.append(Finding("MEDIUM", "drift", "metrics_removed",
                           f"{len(gone)} tracked metrics no longer present", gone[:20]))
    if new:
        out.append(Finding("LOW", "drift", "metrics_added", f"{len(new)} new metrics", new[:20]))
    noise, real = [], []
    for key in sorted(set(bm) & set(cm)):
        b, c = bm[key], cm[key]
        if b is None or c is None:
            if b != c:
                real.append(f"{key}: {b} -> {c}")
            continue
        if any(s in key for s in ("seconds", "dataset_sha256", "fit_seconds")):
            continue
        diff = c - b
        scale = 1.0 if abs(b) <= 1.5 else abs(b)  # rates vs counts/costs
        if abs(diff) / scale > material:
            real.append(f"{key}: {b:.4g} -> {c:.4g} ({diff:+.4g})")
        elif abs(diff) > tolerance:
            noise.append(f"{key}: {b:.6g} -> {c:.6g}")
    if real:
        out.append(Finding("HIGH", "drift", "metrics_material",
                           f"{len(real)} metrics moved by more than {material:.0%} (relative for counts)",
                           real[:40]))
    if noise:
        out.append(Finding("LOW", "drift", "metrics_noise",
                           f"{len(noise)} metrics changed slightly (non-determinism?)", noise[:20]))
    if not out:
        out.append(Finding("INFO", "drift", "none",
                           f"no drift vs baseline from {baseline.get('created')} ({baseline.get('git_head')})"))
    return out


# ---------------------------------------------------------------- reporting


def run_all(include_drift: bool = True) -> tuple[list[Finding], dict]:
    """Header: run every check; return findings sorted by severity plus the current snapshot."""
    findings: list[Finding] = []
    for check in (check_seal, check_split_and_preparation, check_selected_features,
                  check_test_access, check_course_rules, check_diary, check_diary_freshness,
                  check_staleness):
        try:
            findings.extend(check())
        except Exception as error:  # noqa: BLE001 - an audit must report, never crash
            findings.append(Finding("HIGH", "audit", check.__name__, f"check crashed: {error!r}"))
    current = snapshot()
    if include_drift:
        if _p(BASELINE).exists():
            findings.extend(compare(json.loads(_p(BASELINE).read_text()), current))
        else:
            findings.append(Finding("MEDIUM", "drift", "no_baseline",
                                    "no accepted baseline; review results, then run with --accept-baseline"))
    findings.sort(key=lambda f: SEVERITIES.index(f.severity))
    return findings, current


def to_markdown(findings: list[Finding], current: dict) -> str:
    """Header: render findings as a markdown report."""
    counts = {s: sum(f.severity == s for f in findings) for s in SEVERITIES}
    lines = [f"# Audit {current['created']} · HEAD {current['git_head']}", "",
             " · ".join(f"**{s}** {n}" for s, n in counts.items()), ""]
    for f in findings:
        lines.append(f"- **{f.severity}** `{f.area}/{f.check}`: {f.message}")
        lines.extend(f"    - {e}" for e in f.evidence)
    return "\n".join(lines) + "\n"


def write_outputs(findings: list[Finding], current: dict) -> Path:
    """Header: write latest.md / latest.json and append a row to history.csv."""
    out_dir = _p(AUDIT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latest.md").write_text(to_markdown(findings, current), encoding="utf-8")
    (out_dir / "latest.json").write_text(
        json.dumps({"findings": [asdict(f) for f in findings], "snapshot": current},
                   indent=2, default=str) + "\n", encoding="utf-8")
    history = out_dir / "history.csv"
    new = not history.exists()
    with history.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(["created", "git_head", *SEVERITIES])
        writer.writerow([current["created"], current["git_head"],
                         *(sum(f.severity == s for f in findings) for s in SEVERITIES)])
    return out_dir / "latest.md"


def main(argv: list[str] | None = None) -> int:
    """Header: CLI entry point; exit 1 on CRITICAL/HIGH findings."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--accept-baseline", action="store_true",
                        help="store the current snapshot as the accepted baseline (after review)")
    parser.add_argument("--no-drift", action="store_true", help="skip the baseline comparison")
    parser.add_argument("--quiet", action="store_true", help="print only the summary line")
    args = parser.parse_args(argv)

    findings, current = run_all(include_drift=not args.no_drift)
    report = write_outputs(findings, current)
    text = to_markdown(findings, current)
    print(text.splitlines()[2] if args.quiet else text)
    print(f"report: {report.relative_to(ROOT)}")
    if args.accept_baseline:
        _p(BASELINE).write_text(json.dumps(current, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"baseline accepted: {BASELINE} (commit it and log the acceptance in the diary)")
    return int(any(f.severity in {"CRITICAL", "HIGH"} for f in findings))


if __name__ == "__main__":
    raise SystemExit(main())
