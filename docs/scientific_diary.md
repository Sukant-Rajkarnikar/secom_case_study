# SECOM — Scientific Diary

A chronological log of every technical change and decision in this project: **what we
changed, why we made that call, what happened, and what comes next.** Entries are
append-only. If a decision is later reversed, add a new entry that *supersedes* the old
one instead of editing history (only the old entry's **Status** line changes).

---

## How to write an entry

Copy this template to the end of the log and give it the next ID.

```markdown
### D-NNN · <short title>
- **Date:** YYYY-MM-DD
- **CRISP-DM phase:** Business Understanding | Data Understanding | Data Preparation |
  Modeling | Evaluation | Deployment | Infrastructure
- **Type:** Decision | Change | Finding | Fix | Experiment
- **Status:** Proposed | Accepted | Done | Superseded by D-NNN | Rejected

**Context** — what situation or question triggered this.
**Decision / change** — what we did, stated precisely.
**Rationale** — why this option. **Alternatives considered** — what else, and why not.
**What happened** — results, with numbers and the command that produced them.
**Artifacts** — files, tables, reports created or changed.
**Next steps** — concrete follow-ups (mirrored in the Open items list).
```

Conventions:

- Put numbers in the entry, not "it improved". Say which split they come from
  (full / train / test).
- Decisions about preprocessing and models cite **training-split** evidence only. The test
  split is touched once, at final evaluation.
- Link related entries by ID (e.g. "follows D-004").

---

## Decision index

| ID | Date | Phase | Type | Title | Status |
|---|---|---|---|---|---|
| D-000 | 2026-11-09 | Business Understanding | Finding | Starting point: project scaffold | Done |
| D-001 | 2026-11-09 | Data Understanding | Decision | Build a data-health and profiling module | Done |
| D-002 | 2026-11-09 | Data Understanding | Change | Loader finds the data sub-folder and keeps timestamps | Done |
| D-003 | 2026-11-09 | Data Understanding | Fix | Duplicate-column check counted constant features | Done |
| D-004 | 2026-11-09 | Data Understanding | Finding | Health-check results on the raw data | Done |
| D-005 | 2026-11-09 | Data Understanding | Change | Standalone HTML descriptive report | Done |
| D-006 | 2026-11-09 | Data Understanding | Fix | Timestamp-order check missed reversals next to unparsed values | Done |
| D-007 | 2026-11-09 | Infrastructure | Decision | Project-local PostgreSQL 16 cluster | Done |
| D-008 | 2026-11-09 | Data Preparation | Decision | `secom_raw_staging` schema design and load | Done |
| D-009 | 2026-11-09 | Data Understanding | Finding | Missingness structure on the training split | Done |
| D-010 | 2026-11-09 | Data Preparation | Decision | Imputation strategy candidates and evaluation protocol | Superseded by D-021 |
| D-011 | 2026-11-09 | Infrastructure | Decision | Keep this scientific diary | Accepted |
| D-012 | 2026-11-09 | Data Understanding | Change | Notebook 01 and the SIP-table convention | Done |
| D-013 | 2026-11-09 | Data Understanding | Finding | Dataset background, target coding, class imbalance | Done |
| D-014 | 2026-11-09 | Data Understanding | Finding | Missing-value audit: every missing value is flagged | Done |
| D-015 | 2026-11-09 | Data Preparation | Decision | Stratified 80/20 split persisted; test set sealed | Done |
| D-016 | 2026-11-09 | Data Preparation | Decision | Remove the 116 zero-variance columns | Done |
| D-017 | 2026-11-09 | Data Preparation | Decision | Missing-value threshold 60% (474 → 450 columns) | Done |
| D-018 | 2026-11-09 | Data Preparation | Decision | Correlation: visualise only, defer removal | Done |
| D-019 | 2026-11-09 | Data Preparation | Decision | Outliers: 3σ rule, capped at the boundary | Done |
| D-020 | 2026-11-09 | Data Understanding | Change | Health module switched from IQR to the 3σ rule | Done |
| D-021 | 2026-11-09 | Business Understanding | Decision | Literature summary reconciled; revised imputation and evaluation plan | Proposed |
| D-022 | 2026-11-09 | Business Understanding | Finding | Both papers checked: summary corrected; leakage in the published pipelines | Done |
| D-023 | 2026-11-09 | Data Understanding | Change | EDA script: readable missingness plot and one-page overview | Done |
| D-024 | 2026-11-09 | Data Understanding | Change | EDA script: per-column descriptive report (dtypes, nulls, unique values, shape) | Done |
| D-025 | 2026-11-09 | Data Preparation | Change | Five imputed training datasets built, compared, and loaded into `secom_production` | Done (selection pending) |
| D-026 | 2026-11-09 | Data Preparation | Finding | Split method reviewed: stratified hold-out kept; defect rate drifts strongly over time | Done |
| D-027 | 2026-11-09 | Data Preparation | Change | Visual comparison of the imputation strategies | Done |
| D-028 | 2026-11-09 | Data Preparation | Experiment | Split schemes compared empirically: one hold-out is noisy, time order costs about 0.07 PR-AUC | Done |
| D-029 | 2026-11-09 | Data Preparation | Decision | Downstream CV: no imputation strategy beats median; recommend median + missing flags | Proposed (awaiting team confirmation) |
| D-030 | 2026-11-09 | Data Understanding | Change | Raw data exported to a single CSV | Done |
| D-031 | 2026-11-10 | Business Understanding | Decision | Students Manual read; build plan derived (docs/build_plan.md) | Accepted |
| D-032 | 2026-11-10 | Data Preparation | Decision | Official imputer: k-NN (k = 5, scaled); median / MICE kept as tested alternatives | Accepted |
| D-033 | 2026-11-10 | Data Preparation | Experiment | Duplicate check on train, first random forest, Boruta stability selection (450 → 13) | Done |
| D-034 | 2026-11-10 | Infrastructure | Change | Scaffold overview document (docs/scaffold.md) | Done |
| D-035 | 2026-11-10 | Data Preparation | Experiment | Full pipeline: switchable 3σ step, normality, min-max, balancers, first confusion matrices | Done |
| D-036 | 2026-11-10 | Infrastructure | Change | Audit agent and drift checks: `secom-auditor`, `/audit`, `secom_analysis.audit` | Done |

---

## Open items

| From | Item | Status |
|---|---|---|
| D-004 | Investigate the 65 rows sharing a timestamp (batch logging, or real duplicates?) | Open |
| D-004 | Decide what to do with the 10 near-constant and 126 extremely skewed features | Open |
| D-007 | Server does not start automatically after a reboot; consider a systemd user unit | Open |
| D-008 | Add an integration test for the staging loader (needs a running database) | Open |
| D-008 | Build the next layer from staging: typed and joined views, train/test membership in the database | Open |
| D-021 | Implement `impute.py` and notebook 02: the revised candidate set (incl. tree-based MICE), statistical-consistency checks, cross-validated comparison on train | Partly done by D-025: candidates and label-free comparison built; downstream CV comparison still open |
| D-021 | Verify the claims attributed to Study A / Study B against the original papers before citing them in the report | Closed by D-022 |
| D-022 | Report: present the papers' Table I as an optimistic upper reference, and explain the gap (leakage, unstratified split, duplicated minority) | Open |
| D-022 | Add Paper B's deterministic correlation rule as an option row in the notebook 01 correlation SIP table | Open |
| D-010 | Choose the missingness drop threshold (50 / 60 / 80%) by cross-validation, not by convention | Closed by D-017: the course sets 60%, and every threshold from 49.1% to 61.3% gives the same result |
| D-009 | Confirm the 4-feature informative block (072/073/345/346) holds up inside cross-validation folds | Open |
| D-015 | At final evaluation, apply the stored preparation (D-016/D-017 drop lists, D-019 bounds) to the test set without refitting, and check `verify_sealed` first | Open |
| D-019 | 3σ bounds on skewed columns flag mostly one tail. Revisit after imputation: compare capping with a transform-then-cap variant | Open |
| D-018 | Model-based redundancy handling (Boruta, week 4) for the 608 pairs with \|r\| > 0.8 | Closed by D-033 |
| D-025 | Downstream comparison of the five candidates: repeated stratified 5-fold CV on train, imputer fitted inside each fold (D-026), PR-AUC / recall / MCC; then `python -m secom_analysis.production --select <name>` | Comparison done (D-029); selection awaits team confirmation |
| D-029 | Re-run `impute_cv` if the week-4 pipeline adds distance-based steps (SMOTE, k-NN), where fill shape may matter more | Open |
| D-024 | Six features are non-zero on exactly one wafer (source_line 1459); decide whether to treat them as constant | Open |
| D-026 | Time-ordered sensitivity check inside the training split (earlier months → later months) to measure the effect of the defect-rate drift | Closed by D-028 (time-ordered 0.10–0.16 vs about 0.20 in random CV) and D-029 |
| D-028 | Final test-set evaluation: report PR-AUC/recall with a bootstrap CI (a single 80/20 hold-out varies by about ±0.07 PR-AUC) | Open |
| D-031 | Script must download SECOM at runtime (ucimlrepo id=179 or UCI URL) and auto-install packages; current loader reads `data/raw` | Open |
| D-031 | Team decision: official imputer kNN (manual) vs median + flags (D-029) | Closed by D-032 |
| D-031 | Rerun the duplicate-column check on X_train and add its SIP row (manual §4.3) | Closed by D-033 (0 duplicates); SIP row still to add to notebook |
| D-031 | Confirm the manual conflicts: cost 100/400 vs €200/€5000; scaler/selector order; Boruta required | Open |
| D-033 | Grid search (phase J) must refit Boruta inside each fold and include an "all features" option: the honest in-fold Boruta loses 0.044 PR-AUC vs all 450 | Open |
| D-035 | Phase K: choose the decision threshold on CV predictions (cost 100/400 implies about 0.2 for calibrated probabilities); at 0.5 no option beats always-pass on cost | Open |
| D-035 | Notebook 02: normality, min-max, balancing SIP tables and Q&A 10.6 | Open |
| D-033 | Add notebook 02 sections: duplicate SIP row, first model, Boruta theory (shadow features), SIP 7.7 | Open |
| D-036 | Rerun `secom_analysis.pipeline` (report predates the last `pipeline.py` edit) and `impute_cv`; then review and accept the first audit baseline | Open |
| D-036 | Rerun the health report (`reports/health` predates the current `train_reduced.csv`) | Open |
| D-036 | D-030 entry has no Date/Type/Status block; add it via a superseding note or a metadata fix | Open |
| D-036 | Reports must say which feature set they mean: stable Boruta 13 vs in-fold Boruta (18 in the first fold) | Open |

---

## Environment (as of 2026-11-09)

Python 3.12.3 · pandas 3.1.1 · scikit-learn 1.9.1 · scipy 1.18.1 · psycopg 3.3.8 ·
PostgreSQL 16.13 (project-local, port 5433). Raw data: UCI SECOM, in `data/raw/secom (1)/`.

SHA-256 of the source files (also recorded in `secom_raw_staging.source_files`):

| file | sha256 (prefix) | bytes | lines |
|---|---|---|---|
| secom.data | `20f0e7ee434f` | 5,389,983 | 1,567 |
| secom_labels.data | `126884cf4537` | 40,638 | 1,567 |
| secom.names | `6d91b0b46cde` | 4,223 | 98 |

---

## Log

### D-000 · Starting point: project scaffold
- **Date:** 2026-11-09
- **CRISP-DM phase:** Business Understanding
- **Type:** Finding
- **Status:** Done

**Context** — First diary entry; it records what already existed so later entries have a
baseline.

**State before this diary**
- `src/secom_analysis/data.py`: `load_secom()` (features + 0/1 target) and `stratified_split()` (80/20, `random_state=42`).
- `src/secom_analysis/eda.py`: missingness %, variance, constant-feature detection, a drop rule for features over 60% missing, and a missingness histogram.
- `src/secom_analysis/cli.py`: writes `descriptive_statistics.json` and a plot.
- Two unit tests. A README with a 4-week plan: (1) descriptive statistics → (2) missingness, constants, split, outliers → (3) imputation and a Random Forest baseline → (4) Boruta, class balancing, and evaluation focused on catching defects.
- Ground rule from the README: **the test set is never used to choose thresholds, features, imputers or parameters.**

**Business objective** — Find which of the 590 sensor signals drive in-house test
failures (yield excursions), so engineers can monitor fewer signals and react sooner.

**Next steps** — Before any preparation work, check that the raw data can be trusted (D-001).

---

### D-001 · Build a data-health and profiling module
- **Date:** 2026-11-09
- **CRISP-DM phase:** Data Understanding
- **Type:** Decision
- **Status:** Done

**Context** — The scaffold computed a few summary numbers but never checked that the
files were structurally sound. It also never profiled the train and test splits
separately, or checked the split for leakage.

**Decision** — Add `secom_analysis.health`. It profiles and checks **every dataset**:
- the raw files,
- the labels and timestamps,
- the full feature matrix,
- the train and test splits,
- cross-split checks: leakage and drift.

Each check returns PASS / WARN / FAIL. The two failing levels mean different things:
- **FAIL** means a structural problem: the data can't be trusted as loaded. Examples: wrong field counts, tokens that won't parse, labels outside −1/1, overlap between train and test.
- **WARN** means a quality issue that preparation must handle: missingness, constant features, class imbalance, drift.

The command exits non-zero on any FAIL; `--fail-on-warn` also fails on WARN.

**Rationale**
- *Two severities:* the SECOM data is expected to have many quality WARNs. Keeping them separate from FAILs means real breakage never gets lost among known issues.
- *Raw-file checks before pandas:* `read_csv` would quietly turn a malformed line into NaNs. Counting fields in the text first catches that.
- *Per-split profiles:* train and test are what preparation and evaluation actually use, so we need to see both.
- *Per-class means in the profile:* they give a cheap first look at which features differ between passing and failing units.

**Alternatives considered**
- *Great Expectations / pandera:* heavyweight for one fixed dataset; a later option if the pipeline grows.
- *ydata-profiling:* a generic report that doesn't include our project-specific checks (split leakage, label domain).

**What happened** — The run takes about 5 seconds. Results are in D-004.

**Artifacts** — `src/secom_analysis/health.py`. Outputs in `reports/health/`: `health_checks.csv`,
`health_summary.json`, `health_report.md`, `{features,train,test}_profile.csv`,
`split_drift_profile.csv`. `scipy` added as an explicit dependency (used for the KS tests).
Run with: `python -m secom_analysis.health --data-dir data/raw --output-dir reports/health`.

**Next steps** — D-002 (loader), D-004 (read the results).

---

### D-002 · Loader finds the data sub-folder and keeps timestamps
- **Date:** 2026-11-09
- **CRISP-DM phase:** Data Understanding
- **Type:** Change
- **Status:** Done

**Context**
- The UCI download unpacked into `data/raw/secom (1)/`, but `load_secom()` expected the files directly in `data/raw/`.
- `load_secom()` also threw away the timestamps, which the time-based checks need.

**Change**
- `resolve_data_dir()` now looks one folder level down. If it finds the files in more than one sub-folder, it raises an error rather than picking one.
- New `load_labels()` returns the raw label, the raw timestamp text, and the timestamp parsed with format `%d/%m/%Y %H:%M:%S`. `load_secom()` uses it, and its signature and outputs are unchanged.

**Rationale** — The user didn't have to rename folders. Refusing to guess between several
sub-folders means we never silently load the wrong copy of the data.

**What happened** — Both existing tests still pass, and the CLI works with `--data-dir data/raw`.

**Artifacts** — `src/secom_analysis/data.py`.

---

### D-003 · Duplicate-column check counted constant features
- **Date:** 2026-11-09
- **CRISP-DM phase:** Data Understanding
- **Type:** Fix
- **Status:** Done

**Context** — The first run reported **104 duplicate columns (17.6%)**. That looked
suspicious next to 116 constant columns.

**What happened** — Investigation showed that among the 474 features that actually vary,
there are **0** duplicates. Every one of the 104 was a constant column that matched
another constant column. The same feature was being flagged by two checks.

**Change** — The duplicate check now runs only on features that vary. Each check now
reports one distinct problem.

**Rationale** — Counting the same problem twice would have overstated the redundancy in
the data. It could also have led us to drop columns twice or argue for dropping them on
the wrong grounds.

**Artifacts** — `check_features()` in `health.py`. The test `test_profile_and_feature_checks_flag_quality_issues` covers it.

---

### D-004 · Health-check results on the raw data
- **Date:** 2026-11-09
- **CRISP-DM phase:** Data Understanding
- **Type:** Finding
- **Status:** Done

**What happened** — Overall: **WARN**, with 45 PASS, 17 WARN and 0 FAIL.

*Structural checks (all PASS)*
- 1,567 lines, each with exactly 590 fields.
- Every token is a number or `NaN`.
- Labels are only −1/1; there are 1,567 of them, the same as the measurement rows.
- All timestamps parse and are in time order.
- Window: 2008-07-19 to 2008-10-17, 86 production days, median 15.5 units per day.

*Quality checks on the full data (WARN)*

| Issue | Value |
|---|---|
| Class imbalance | 104 fails / 1,463 passes, so **6.64% defective** |
| Missing cells | 41,951 (**4.54%**); **0 complete rows**; 538 of 590 features have at least one gap |
| Features over 50% missing | 28 |
| Constant features | **116** (19.7%) |
| Near-constant (one value in ≥95% of rows) | 10 |
| Extreme skew (\|skew\| > 10) | 126 |
| Rows sharing a timestamp | 65 |
| Duplicate rows / duplicate varying columns | 0 / 0 |

*Split checks (all PASS)*
- No row appears in both train and test.
- Train has 1,253 rows and test 314, which covers all 1,567.
- Defect rate: train 6.62% vs test 6.69%, a gap of 0.06 percentage points.
- KS test (p < 0.01): **0** features differ in distribution between train and test.
- No feature's missing percentage differs by more than 10 points between train and test.

**What this means**
- *Imbalance:* plain accuracy is useless as a metric. Evaluate on recall for failing units, PR-AUC and balanced accuracy.
- *No complete rows:* imputation can't be avoided (deleting every incomplete row would delete everything).
- *Constants:* drop the 116 constant features.
- *Skew:* prefer robust statistics, such as the median.
- *Split:* the stratified split is sound.

**Next steps** — Look at how values go missing before choosing an imputer (D-009). Open
items: the 65 shared timestamps, and what to do with near-constant and skewed features.

---

### D-005 · Standalone HTML descriptive report
- **Date:** 2026-11-09
- **CRISP-DM phase:** Data Understanding
- **Type:** Change
- **Status:** Done

**Context** — The health outputs were CSV, JSON and Markdown files. That's fine for a
machine, but slow for a person to read and hard to share with reviewers.

**Change** — `secom_analysis.report` is generated on every health run and writes
`reports/health/descriptive_report.html`. It contains:
- headline numbers and key findings, built from the data rather than hard-coded;
- charts: units tested per week split by pass/fail, a histogram of feature missingness, counts of features flagged by issue, and the 15 features whose means differ most between passing and failing units;
- the dataset overview;
- every health check, with a toggle to hide the passing ones;
- a sortable, searchable table of statistics for all 590 features.

**Rationale**
- *Self-contained:* inline SVG and CSS with no external libraries, so it opens offline and is easy to archive with each run.
- *Generated by the pipeline:* the report can't drift from the data it describes.
- *Caveat in the report:* the pass/fail mean-shift chart is labelled as a screening signal for a single variable at a time. It is not evidence of cause, and it is not a substitute for feature selection on the training split.

**What happened** — Rendered in headless Chromium to check the layout. One fix followed:
the y-axis ticks fell on awkward values (62.5, 188), and now use round steps
(0/50/…/250).

**Artifacts** — `src/secom_analysis/report.py`, and the call from `health._write_reports`.

---

### D-006 · Timestamp-order check missed reversals next to unparsed values
- **Date:** 2026-11-09
- **CRISP-DM phase:** Data Understanding
- **Type:** Fix
- **Status:** Done

**Context** — A new unit test built a label sequence with a timestamp that couldn't be
parsed (NaT) sitting between two rows in the wrong order.

**What happened** — The order check returned PASS. `diff()` across a NaT gives NaN, so
the backwards step between the two valid timestamps was never compared.

**Change** — The check now drops unparsed timestamps first (`ts.dropna().diff()`).
Unparsed timestamps are still caught by their own check, `timestamp_parseable`.

**Rationale** — An unparseable value must not hide an ordering problem. This didn't
affect the real SECOM data, where every timestamp parses.

**Artifacts** — `check_labels()` in `health.py`, and `tests/test_health.py` (3 new tests; the suite has 5, all passing).

---

### D-007 · Project-local PostgreSQL 16 cluster
- **Date:** 2026-11-09
- **CRISP-DM phase:** Infrastructure
- **Type:** Decision
- **Status:** Done

**Context** — We wanted a local database to hold the raw data as a staging layer. The
system PostgreSQL 16 on port 5432 is already running, but it has no role for user
`bnam`, and creating one needs sudo, which wasn't available to the assistant.

**Decision** — Use a dedicated cluster for this project:
- created with `initdb` from the installed PostgreSQL 16 binaries;
- data stored in `./.pgdata/` (excluded from git);
- listening on **localhost:5433**;
- one database, `secom`.

**Authentication**
- *Unix socket:* peer authentication. Your OS user is the cluster superuser, so no password is needed.
- *TCP on localhost:* `scram-sha-256` with a random 48-character password, stored in `.env` as `SECOM_DATABASE_URL` (excluded from git; file mode 600).

**Rationale**
- No sudo needed, and it doesn't touch the system server.
- Same major version as the system server.
- The whole database can be deleted by removing one folder.

**Alternatives considered**
- *System cluster:* needs sudo to create a role.
- *Docker `postgres:16`:* works, but needs an image pull and container lifecycle management, for no benefit over local binaries.
- *Trust authentication:* rejected, because any local user could connect.

**What happened** — The cluster initialised and the `secom` database was created
successfully.

**Artifacts** — `scripts/db.sh` (`init | start | stop | status | psql`), `.gitignore` entries, and a README section.

**Next steps** — The server doesn't start on boot; run `scripts/db.sh start` after a
restart (open item: a systemd user unit).

---

### D-008 · `secom_raw_staging` schema design and load
- **Date:** 2026-11-09
- **CRISP-DM phase:** Data Preparation
- **Type:** Decision
- **Status:** Done

**Context** — The request was to push all raw data to "SECOM_raw_staging". We read this
as a **schema** called `secom_raw_staging` inside the `secom` database. It is lowercase
because mixed-case names in PostgreSQL have to be quoted in every query.

**Decision**

*Tables*
- `measurements`: `source_line` (primary key) plus `feature_000`…`feature_589` as double precision.
- `labels`: `source_line` (primary key), the raw `label` (constrained to −1/1), `timestamp_raw` exactly as in the file, and the parsed `test_timestamp`.
- `source_files`: file name, SHA-256, size and line count, plus the full text of `secom.names`.
- `load_log`: one row per load run, with a foreign key to it from every row.

*Rules*
- **Raw means untransformed.** We keep the original −1/1 labels (no remapping to 0/1) and the timestamp text as written. The only conversion is the file's `NaN` → SQL `NULL`.
- **Join key:** `source_line`, the 1-based line number shared by both files.
- **Idempotent load:** a single transaction that empties the tables, inserts the load-log row, loads with `COPY`, then runs `ANALYZE`.
- **Validate before loading:** check field counts, numeric parsing, label domain and timestamp format before touching the database.

**Rationale**
- *NULL instead of float NaN:* PostgreSQL aggregates propagate NaN, so `avg()` would return NaN, while SQL's NULL handling is what downstream queries expect.
- *Checksums and line counts:* they make lineage auditable. Any later layer can prove which file version it came from.
- *Wide table:* 590 columns is well under PostgreSQL's 1,600-column limit and mirrors the source exactly. A long or tidy layout belongs in the next layer, not in raw staging.

**What happened**
- 1,567 measurement rows, 1,567 label rows and 3 source files loaded in about 3 seconds.
- **41,951 NULL cells**, which matches the health report's missing-cell count.
- Every label row has a matching measurement row.
- A cell-by-cell comparison of the database against pandas reading the files found identical values, an identical NULL pattern and identical labels.
- A second run reloaded cleanly, confirming the load can be repeated.
- The database is 22 MB.

**Artifacts** — `src/secom_analysis/staging.py` (`python -m secom_analysis.staging`) and a README section. `psycopg[binary]` added as a dependency.

**Next steps** — An integration test for the loader. Then a cleaned or typed layer built
from staging, with train/test membership stored so SQL and Python agree on the split.

---

### D-009 · Missingness structure on the training split
- **Date:** 2026-11-09
- **CRISP-DM phase:** Data Understanding
- **Type:** Finding
- **Status:** Done

**Context** — Before choosing an imputer, we needed to know *how* values go missing.
Following the README's rule, this analysis uses the **training split only** (1,253 rows,
474 features that vary).

**What happened**

| Finding | Value |
|---|---|
| Missingness per feature | 84 complete · 276 at 0–1% · 62 at 1–5% · 20 at 5–20% · 8 at 20–50% · 16 at 50–80% · 8 at 80–100% |
| Features missing together | 390 features with gaps share only **34 distinct patterns** (largest groups: 44, 38, 36, 32 features) |
| Does how much a row is missing predict failure? | Point-biserial r = −0.038, p = 0.18: **no** |
| Exception: features where being missing predicts the outcome (Fisher p < 0.01) | 4 features, `072, 073, 345, 346` (49% missing): units with them missing fail **4.7%** of the time vs **8.5%** when recorded |
| Missingness over time | 4.7–6.5% of cells missing every month, so no outage period |
| Heavy-missing features in time order | They switch between missing and present about 200–250 times: intermittent, not one contiguous block |
| Skew | 185 of 338 lightly-missing features have \|skew\| > 2 |
| Redundancy | 355 feature pairs with \|r\| > 0.9 |

**Interpretation**
- Missingness comes in groups of sensors that drop out together, and in general it doesn't depend on the outcome. The 4-feature group is the exception: there, being missing carries information about the outcome.
- Skew argues for robust fills such as the median.
- High correlation means model-based imputers can borrow strength from related features.
- Grouped dropouts weaken KNN, because the neighbour distances are computed without the whole missing group.

**Next steps** — D-010.

---

### D-010 · Imputation strategy candidates and evaluation protocol
- **Date:** 2026-11-09
- **CRISP-DM phase:** Data Preparation → Modeling → Evaluation
- **Type:** Decision
- **Status:** Superseded by D-021

**Context** — Week 3 needs an imputer. The findings in D-004 and D-009 rule out a
one-size-fits-all choice.

**Decision (proposed)**

*Step 1: reduce, fitted on train only*
- Drop the 116 constant features.
- Drop features above a missingness threshold. Treat 50/60/80% as options to compare rather than fixing the README's 60%.
- **Keep the informative 4-feature group** even if it crosses the threshold.

*Step 2: the candidate imputers*

| ID | Strategy | Role |
|---|---|---|
| A | Median + one missing-flag per missingness pattern (at most 34 flags) | Baseline |
| B | KNN imputer (k = 5–15, scaled first) | Candidate |
| C | Iterative imputer / MICE (BayesianRidge, `n_nearest_features≈50`), optionally after a Yeo-Johnson transform | Candidate |
| D | MissForest-style (iterative with ExtraTrees) | Only if C clearly beats A |
| E | No imputation: HistGradientBoosting, which handles missing values itself | Reference: does imputation help at all? |
| F | Tiered hybrid: median under 5% missing, MICE for 5–50%, flag only or drop above 50% | Candidate |

*Step 3: evaluation protocol*
- *Pipeline:* every strategy runs inside one `Pipeline` (drop, impute, scale, SMOTE, model). Score with repeated stratified 5-fold cross-validation on the **training split**. SMOTE comes after imputation, inside each fold.
- *Main measure:* PR-AUC, recall on failing units and balanced accuracy, using the week-3 Random Forest.
- *Tie-breaker:* hide about 10% of the known values using the real 34-pattern structure, then measure normalised RMSE of recovery and how much variance and correlation are flattened.
- *Stability:* overlap between Boruta feature selections under each imputer (week 4).
- *Deployment:* the fitted pipeline is saved as one object and applied to the test split exactly once. A missingness pattern never seen in training raises a warning.

**Rationale** — The decision rests on how well the model catches failing units
downstream. A plausible-looking imputation that doesn't help the model is worthless. The
fidelity and stability checks protect against an imputer that wins by chance on 104
failing units.

**Expectation (to be tested, not assumed)** — A or C wins, given the skew, the grouped
dropouts and the small number of failing units.

**Next steps** — Implement `impute.py` with the candidate pipelines and a comparison
harness. Record the cross-validation results as a new entry, and accept or revise this one.

---

### D-011 · Keep this scientific diary
- **Date:** 2026-11-09
- **CRISP-DM phase:** Infrastructure
- **Type:** Decision
- **Status:** Accepted

**Context** — Decisions so far lived in chat history and commit-less files. That makes
it hard to reconstruct *why* the data looks the way it does at any later stage.

**Decision** — Keep `docs/scientific_diary.md` as the single structured, append-only log:
- one entry per change, decision, finding, fix or experiment;
- every entry follows the template above;
- the decision index and the open-items list are updated together with each entry.

D-000 to D-010 were written after the fact on the same day, from the session record.

**Next steps** — Every future change gets an entry in the same commit, or the same
working session, as the change itself.

---

### D-012 · Notebook 01 and the SIP-table convention
- **Date:** 2026-11-09
- **CRISP-DM phase:** Data Understanding → Data Preparation
- **Type:** Change
- **Status:** Done

**Context** — The course brief asks for the next steps as a notebook: dataset background,
descriptive statistics, a missing-value audit, a stratified split, and rough feature
reduction (constant features, many-N/A features, correlation, 3σ outliers). After every
preparation step it requires a **Step Impact Protocol (SIP) table** with four columns:
options, pro and con, decision, consequence.

**Change**
- New notebook `notebooks/01_data_understanding_and_preparation.ipynb` (41 cells, fully executed).
- The logic lives in a tested module, `src/secom_analysis/preparation.py`, so the notebook stays narrative:
  - `zero_variance_features`, `missing_percentage`, `missing_threshold_pareto`;
  - `three_sigma_bounds`, `flag_outliers`, `cap_outliers`, `remove_outliers`;
  - `save_split` / `load_split` / `verify_sealed`;
  - `sip_table`, `save_state`.
- SIP tables are **generated from computed values** (f-strings), so the documented
  consequences can't drift from the data.
- Figures are written to `reports/figures/01…07_*.png`.

**Rationale** — Functions outside the notebook can be unit-tested (4 new tests; the suite
has 9, all passing) and re-applied to the test set later without copying notebook cells.

**Artifacts**
- Notebook and `preparation.py`.
- `tests/test_preparation.py`.
- Dev dependencies `ipykernel`, `nbformat`, `nbclient` and `jinja2` (needed for styled tables); the Jupyter kernel `secom` is registered.

---

### D-013 · Dataset background, target coding, class imbalance
- **Date:** 2026-11-09
- **CRISP-DM phase:** Business / Data Understanding
- **Type:** Finding
- **Status:** Done

**What happened**
- **Background:** semiconductor line monitoring (UCI SECOM, McCann and Johnston, 2008).
- **Rows:** 1,567 production entities (wafers), each tested in house.
- **Columns:** 590 anonymised sensor or process signals, with no names or units. The `.names` file's "591 attributes" means 590 measurements plus the label.
- **Target coding:** −1 = pass (good) → `defective = 0`; 1 = fail → `defective = 1`.
- **Class balance:** 1,463 good vs **104 defective (6.64%)**, a ratio of 14.1 : 1. A model that always predicts "pass" would score 93.36% accuracy.
- **Measurements:**
  - standard deviations range from 1.0e-4 to 6,554, so features must be scaled;
  - 269 columns have \|skew\| > 2;
  - no row is complete.

**Interpretation**
- The data is **imbalanced because the process is under statistical process control**. Defects are rare by design, so this is not a sampling artefact.
- Consequences:
  - use stratified splits;
  - evaluate with recall, PR-AUC and balanced accuracy;
  - balance classes on training data only, later.

---

### D-014 · Missing-value audit: every missing value is flagged
- **Date:** 2026-11-09
- **CRISP-DM phase:** Data Understanding
- **Type:** Finding
- **Status:** Done

**Context** — A missing value that isn't recognised as missing would distort means,
variances, the 3σ bounds and the imputer.

**What happened** — The audit was run against the **raw text file**:
- 924,530 tokens, which equals 1,567 × 590;
- 41,951 `NaN` tokens, equal to the 41,951 NaN cells in the DataFrame;
- 0 other non-numeric tokens and 0 ±∞.

Placeholder scan for −99999, −9999, −999, −99, 99, 999, 9999 and 99999:
- 9 hits, all inside their column's 1st–99th percentile range (\|z\| ≤ 1.3).
- The only repeated hit, `99` in `feature_135` (24×), is in a column of whole numbers with 133 distinct values, where common values occur 24–31 times. It is an ordinary reading.

**Decision** — `NaN` is the only missing-value marker, and it is kept as `np.nan`
throughout. Every later statistic is NaN-aware, so missing values are never counted as
zeros.

---

### D-015 · Stratified 80/20 split persisted; test set sealed
- **Date:** 2026-11-09
- **CRISP-DM phase:** Data Preparation
- **Type:** Decision
- **Status:** Done

**Context** — The split already existed in code (`stratified_split`, D-000), but it was
recomputed on every run and never stored. The brief requires the test set to be put
aside and never touched.

**Decision**
- `train_test_split(test_size=0.2, stratify=y, random_state=42)`.
- The test set is written once to `data/processed/test_holdout.csv` with a SHA-256 fingerprint file, then **deleted from notebook memory**.
- All later steps load only `train_*.csv`.

**Rationale** — 1,000 *non-stratified* random 80/20 splits put between 9 and 34
defective wafers in the test set (95% range: 13–29). That means a test defect rate of
anywhere from 4.1% to 9.2%, which is too unstable for evaluation. Stratification fixes
it at 21.

**What happened**

| Dataset | Wafers | Defective | Defective % |
|---|---|---|---|
| Original | 1,567 | 104 | 6.637 |
| Train | 1,253 | 83 | 6.624 (−0.013 pp) |
| Test | 314 | 21 | 6.688 (+0.051 pp) |

The remaining gap can't be avoided, because 104 × 0.2 = 20.8 wafers. There is no overlap
between train and test.
- Test file SHA-256 prefix: `f2a9d0248b2cc8ee`.
- `verify_sealed` returned True at the end of the notebook.

**Artifacts** — `data/processed/train_raw.csv`, `test_holdout.csv` and `test_holdout.csv.sha256`.

**Next steps** — At final evaluation: verify the fingerprint, then apply the stored
preparation without refitting (open item).

---

### D-016 · Remove the 116 zero-variance columns
- **Date:** 2026-11-09
- **CRISP-DM phase:** Data Preparation
- **Type:** Decision
- **Status:** Done

**Decision** — Compute the per-column variance on the **training set** (NaN-aware) and
remove the columns whose variance is **exactly 0**. Near-constant columns are kept.

**What happened**
- 116 columns have variance = 0, and none have undefined variance.
- The smallest non-zero variance is 1.1e-08, and that column is kept.
- **590 → 474 columns.**

**SIP summary**

| Option | Decision | Consequence |
|---|---|---|
| Remove variance exactly 0 | **Chosen** | 590 → 474 (116 removed, 19.7%) |
| Also remove near-constant columns | Not chosen | The cut-off would be arbitrary and depend on scale; a rare deviation may be the defect signal |
| Keep all | Not chosen | Constant columns add cost and noise only |

---

### D-017 · Missing-value threshold 60% (474 → 450 columns)
- **Date:** 2026-11-09
- **CRISP-DM phase:** Data Preparation
- **Type:** Decision
- **Status:** Done (closes the threshold open item from D-010)

**Context** — The papers use 55%. The course says to start at 60% so as not to lose too
many features.

**What happened**
- Diagram 1 (missing % per column, sorted) shows **plateaus**: groups of columns that go missing together, at 45.3% (4 columns), 49.1% (4), 61.3% (4), 66.3% (12), 85.6% (4) and 90.7% (4).
- Diagram 2 (Pareto: columns removed vs. threshold) is a staircase that is **flat from 49.1% to 61.3%**. So **55% and 60% both remove the same 24 columns**.

**Decision** — Threshold of **60%**: remove the columns with missing % > 60.

**Rationale**
- The threshold sits in the natural gap, so it isn't sensitive.
- It keeps the 45–49% groups, including the group whose missingness is associated with defects (D-009).
- Going above 61.3% would mean imputing more than 60% of a column's values.

**Consequence**
- **474 → 450 columns** (24 removed). Cumulative: 590 → 450 (140 removed).
- Missing cells drop from 33,091 to **11,211** (from 5.57% to 1.99% of cells).

**Artifacts** — `reports/figures/04_missing_values_threshold.png` and `05_missing_values_histogram.png`.

---

### D-018 · Correlation: visualise only, defer removal
- **Date:** 2026-11-09
- **CRISP-DM phase:** Data Preparation
- **Type:** Decision
- **Status:** Done

**Context** — Papers often drop one feature of each pair with \|r\| > 0.8. Which feature
of a pair is the valuable one is unknown, so the drop would be a random choice.

**What happened** — On the 450 training features (pairwise Pearson correlation):
- 101,001 pairs in total;
- **608 pairs with \|r\| > 0.8**, 371 with \|r\| > 0.9, and 26 practically identical (\|r\| > 0.9999);
- 335 features take part in at least one \|r\| > 0.8 pair;
- median \|r\| = 0.024.

The clustered heatmap shows a handful of large redundant sensor blocks against a mostly
uncorrelated background.

**Decision** — Visualise only; **remove nothing**. The redundancy will be handled by
model-based selection (Boruta, week 4).

**Consequence** — 450 features remain. The correlation structure is documented, and it
will help model-based imputation (D-010, candidate C).

**Artifacts** — `reports/figures/06_correlation_heatmap.png`.

---

### D-019 · Outliers: 3σ rule, capped at the boundary
- **Date:** 2026-11-09
- **CRISP-DM phase:** Data Preparation
- **Type:** Decision
- **Status:** Done

**Context**
- The brief prescribes the 3σ rule (mean ± 3·std, on training data). An IQR approach is explicitly **not** to be used.
- Three options for treatment: do nothing, substitute by the 3σ boundary, or remove and leave gaps.

**What happened (training set, 450 features)**
- **4,872 outlier values**, 0.88% of the 552,639 observed values.
- They sit in 412 columns, with at most 61 in one column (`feature_038`).
- 1,008 of 1,253 wafers (80.4%) have at least one.
- **The defect rate rises with the number of outliers per wafer:**

| Outliers per wafer | Defect rate |
|---|---|
| 0–1 | 4.90% |
| 2 | 5.88% |
| 3–5 | 7.74% |
| 6–52 | 8.50% |

Consequences of each option:

| Option | Values changed | Missing cells afterwards | Median change in column std |
|---|---|---|---|
| Do nothing | 0 | 11,211 | 0% |
| **Cap at 3σ** | 4,872 | 11,211 | −12.9% |
| Remove (NaN) | 4,872 | 16,083 (+43%) | −20.0% |

**Decision** — **Cap at the 3σ boundary.** The bounds are fitted once on the training set
and saved (`three_sigma_bounds.csv`) so they can be re-applied unchanged to the test set.

**Rationale**
- Extremeness is associated with defects. Capping keeps "this wafer is extreme in this sensor" on the correct side.
- Removing would add 4,872 gaps for the imputer to guess, undoing part of D-017.
- Doing nothing would leave values up to \|z\| = 35. Those distort scaling, KNN/MICE imputation and SMOTE.

**Caveats**
- The 3σ rule assumes a symmetric distribution; skewed columns get mostly one-sided caps (open item).
- Capping is applied once; re-estimating after capping would shrink the bounds indefinitely.

**Artifacts**
- `data/processed/train_reduced.csv` (after D-016 and D-017).
- `train_prepared.csv` (after capping).
- `three_sigma_bounds.csv`, `preparation_state.json`.
- `reports/figures/07_outliers_3sigma.png`.

**Next steps** — Imputation (D-010) on `train_prepared.csv`: 450 columns, 11,211
missing cells.

---

### D-020 · Health module switched from IQR to the 3σ rule
- **Date:** 2026-11-09
- **CRISP-DM phase:** Data Understanding
- **Type:** Change
- **Status:** Done

**Context** — D-001 counted outliers with Tukey 1.5×IQR fences, for information only.
The course brief rules out IQR-based outlier identification.

**Change**
- The profile column `iqr_outliers` / `iqr_outlier_pct` became `sigma3_outliers` / `sigma3_outlier_pct`.
- The health check `iqr_outliers` became `sigma3_outliers`.
- The threshold setting `iqr_multiplier` became `outlier_sigma = 3.0`.
- The report chart and table labels were updated to match.

The quartile columns (`p25`, `p75`, `iqr`) remain in the profile as descriptive
statistics only.

**Rationale** — One outlier definition across the project, so the health report and the
preparation notebook can't disagree.

**What happened** — All health checks were re-run; the result is unchanged (45 PASS,
17 WARN, 0 FAIL).

---

### D-021 · Literature summary reconciled; revised imputation and evaluation plan
- **Date:** 2026-11-09
- **CRISP-DM phase:** Business Understanding → Data Preparation / Evaluation
- **Type:** Decision
- **Status:** Proposed (supersedes D-010)

**Context** — The team supplied a paper-comparison and workflow summary, saved as
`docs/literature/paper_comparison_summary.md`. It covers:
- **Study A:** feature selection plus boosting under imbalance; KNN, decision trees, AdaBoost/XGBoost.
- **Study B (Ranthinam & Ramadoss):** basic classifiers and an adaptive Mahalanobis distance.
- A reference pipeline, and two checks for validating an imputer.

The claims about the papers come from the summary. The original papers have not been
checked by us yet (open item).

**How our pipeline maps onto the reference pipeline**

| Reference step | Our pipeline | Status |
|---|---|---|
| 1. Cleaning & EDA; find imbalance and missingness | Health checks (D-001/D-004), notebook 01 §§1–3 (D-013/D-014) | Done; consistent |
| 2. Stratified 80/20 split; test set untouched | D-015: sealed with SHA-256 and deleted from memory | Done; consistent |
| (not in the reference) Rough reduction before imputation | D-016 variance = 0, D-017 > 60% missing, D-019 3σ capping | Done; an extra step required by the course; it reduces what the imputer has to fill |
| 3A. Advanced imputation (MICE with decision trees) | D-010 had tree-based MICE only as a conditional candidate | **Revised below** |
| 3B. Feature selection: filter → wrapper/embedded | Week 4 Boruta (wrapper around a Random Forest); D-018 deferred correlation filtering | Planned; consistent |
| 3C. Training plus balancing (boosting/oversampling) | SMOTE inside the CV folds; Random Forest baseline | Planned; consistent |
| 4. Final testing: precision, recall, F1 | D-004 named recall, PR-AUC and balanced accuracy | **Metric set widened below** |

**Where we deliberately deviate from the summary**
1. **Imputation is validated on training data, not on the test set.** Section 4 of the
   summary says to compare imputers by downstream metrics "on the unseen validation/test
   set". Choosing an imputer on the test set is leakage, and it contradicts the summary's
   own step 2. We compare imputers with **stratified k-fold CV inside the training set**,
   and the test set is used once, for the final model only.
2. **Missingness thresholds are decided on training data.** EDA on the full data is
   descriptive only. Every threshold and boundary is fitted on train (D-016 to D-019).
3. **"High raw accuracy" is not evidence.** At 6.6% defects, a majority-class model scores
   93.4% accuracy (D-013). Model comparisons use defect-focused metrics.
4. **Imbalance is defined by the data, not by the summary's example numbers** ("95%+ vs
   below 5%"). SECOM is 93.4% / 6.6%: still severe, a ratio of 14 : 1.

**Revised decision: imputation candidates** (replaces the table in D-010)

| ID | Strategy | Role |
|---|---|---|
| A | Median + one missing flag per missingness pattern | Baseline ("standard median filling" in the summary) |
| B | KNN imputer (scaled first) | Candidate |
| C | MICE with BayesianRidge (linear) | Candidate |
| **D** | **MICE with decision trees / ExtraTrees (MissForest-style)** | **Promoted to a primary candidate** (the reference method in the summary; handles skew and non-linear relationships) |
| E | No imputation: HistGradientBoosting handles missing values natively | Reference only |

**Revised decision: validation protocol** (the summary's two checks, made leakage-free)
1. **Statistical consistency, on training folds:** compare mean, variance and distribution
   shape (KS distance) of each column before and after imputation, and measure variance
   shrinkage. Also hide about 10% of the known values in the real missingness patterns and
   measure how well each imputer recovers them (NRMSE).
2. **Downstream performance, repeated stratified 5-fold CV on train:**
   - the imputer is fitted inside each fold;
   - SMOTE is applied after imputation, inside the fold;
   - metrics: **precision, recall, F1** (from the summary) plus PR-AUC, ROC-AUC and balanced accuracy;
   - the main ranking measure is PR-AUC / recall on defective wafers;
   - every candidate is compared against baseline A.
3. Tie-breaker: how stable the Boruta feature selection is across imputers (week 4).

**Rationale** — We keep the summary's sound structure (split → impute → select →
balance → test) and its validation idea, and remove the one step that would leak test
information.

**Next steps**
- Implement `impute.py` and notebook 02 according to this plan.
- Verify the paper claims before citing them.

---

### D-022 · Both papers checked: summary corrected; leakage in the published pipelines
- **Date:** 2026-11-09
- **CRISP-DM phase:** Business Understanding
- **Type:** Finding
- **Status:** Done (closes the D-021 open item on verifying the papers)

**Context** — The course papers were provided in `data/raw/`:
- **Paper A:** Kerdprasop & Kerdprasop, 2011.
- **Paper B:** Munirathinam & Ramadoss, 2016, IJET 8(4), DOI 10.7763/IJET.2016.V8.898.

Both were read in full and compared with the team summary (D-021). The full review is in
`docs/literature/paper_review.md`.

**What happened**
1. **The summary has 5 substantive errors:**
   - the author name ("Ranthinam" should be Munirathinam);
   - "AdaBoost/XGBoost": Paper A's "boosting" is **duplicating the fail cases**, and it uses no ensemble at all;
   - "filter → wrapper/embedded": Paper A uses filters and PCA only;
   - adaptive Mahalanobis attributed to Study B: it appears only in Paper A's related work;
   - "MICE with decision trees" presented as coming from the papers: neither paper uses MICE.
2. **Both papers fit their preprocessing on all 1,567 rows before splitting.**
   Paper B fills missing values with the **class mean**, which is target leakage.
3. **Their split isn't stratified**, although they say it is: test 59/468 = 12.6% fail,
   train 45/1,099 = 4.1% fail. The minority class is duplicated about 24× and then 1-NN
   is used.
4. **Paper B repeats Paper A's experiment.** The split, the confusion matrices, the
   metric table and several sentences are identical, without citing Paper A. For the
   classifier results the two papers are one evidence source, not two.
5. **The cleaning can be reproduced closely.** "Single value out + more than 55% missing
   out" on the full data gives 140 removed → 450 features. The papers report 137 → 454.
   Our training-split result is also 450 (D-017), so our threshold choice of 60% vs 55%
   makes no difference.
6. **A cross-check.** Paper A's decision tree splits first on S511. If the papers number
   sensors from 1, that is our `feature_510`, the 3rd-largest pass/fail mean shift in our
   report (D-005).

**Decisions**
- **Class-mean imputation is rejected** as an imputation option, because of target
  leakage. It will appear as a rejected row in the imputation SIP table.
- The papers' best results (DT: TP 1.0, FP 0.161, precision 0.472, F 0.641) are treated
  as an **optimistic upper reference, not a target**. A leakage-free pipeline should
  expect lower recall and precision, and the report will explain why.
- Evaluation metrics also include **MCC, ROC area and PRC area**, following Paper B
  (in addition to D-021's set).
- Paper B's deterministic correlation rule (of a pair with \|r\| ≥ 0.8, drop the member
  with the larger mean \|r\| to all other features) qualifies the course note that
  correlation pruning "must rely on a random choice". It's recorded as an alternative.
  Removal stays deferred (D-018).

**Artifacts**
- `docs/literature/paper_review.md` (new).
- `docs/literature/paper_comparison_summary.md`: a header now warns about the errors and points to the review.

**Next steps**
- Imputation (notebook 02) per D-021, with class-mean imputation as a rejected option.
- Add the correlation-rule option row to notebook 01.

---

### D-023 · EDA script: readable missingness plot and one-page overview
- **Date:** 2026-11-09
- **CRISP-DM phase:** Data Understanding
- **Type:** Change
- **Status:** Done

**Context** — We ran the original EDA script
(`python -m secom_analysis.cli --data-dir data/raw --output-dir reports`). Its numbers
match the notebook and the health checks:
- 1,567 × 590, 41,951 missing cells;
- 1,463 good / 104 defective (6.64%);
- train 1,253 (6.62%) / test 314 (6.69%);
- 116 constant and 24 over-60%-missing features → 450.

But its only chart, `missingness_histogram.png`, used a linear y-axis. The roughly 530
low-missing features flattened the 45–90% groups that drive the threshold decision,
making them nearly invisible.

**Change** — In `eda.py` / `cli.py`:
- **Missingness histogram:** log-scale y-axis, 5% bins, the threshold drawn as a line, and bars above the threshold in orange. It also now follows `--missing-threshold`.
- **New `save_eda_overview` → `reports/eda_overview.png`:** four panels:
  - (a) defect rate for full / train / test, with counts (imbalance and stratification);
  - (b) the missingness histogram;
  - (c) log10 variance of the 474 non-constant features, with the 116 zero-variance count noted;
  - (d) the reduction funnel 590 → 474 → 450.
- Panels (b)–(d) use the **training split**, which is what the reduction is fitted on. The seaborn dependency was dropped from `eda.py`.

**What happened** — Rendered and checked visually. A label in panel (c) overlapped a
bar, so headroom was added. Lint and the tests (9) pass.

**Artifacts** — `reports/eda_overview.png`, `reports/missingness_histogram.png`, `reports/descriptive_statistics.json`.

---

### D-024 · EDA script: per-column descriptive report
- **Date:** 2026-11-09
- **CRISP-DM phase:** Data Understanding
- **Type:** Change
- **Status:** Done

**Context** — The EDA script produced only aggregate numbers and charts. The team asked
for a standard descriptive table per column: null %, unique values, data types, shape and size.

**Change** — `eda.py`:
- `column_summary()`: per column: dtype, non-null, null count and %, unique count and %,
  constant/binary flags, `top_value_pct` (share of the most frequent value), mean, std,
  min, quartiles, max, % zeros.
- `dataset_overview_table()`: per dataset: rows, columns, cells, memory, dtype mix, null
  cells, columns/rows with nulls, complete rows, duplicate rows, constant columns.
- `write_descriptive_report()` → Markdown.

The CLI writes `reports/column_summary.csv`, `reports/dataset_overview.csv`,
`reports/descriptive_report.md`. One test added.

**What happened**
- All 590 features are `float64`; labels file: `int64` code, raw text and parsed timestamp.
  7.4 MB in memory.
- **No wafer is complete**: every one of the 1,567 rows has at least one null (538 columns
  have nulls). After reduction, 192 of the 1,253 training wafers are complete.
- Null share per feature: 52 none, 486 at most 5%, 20 at 5–20%, 4 at 20–50%,
  4 at 50–60% (the informative block 072/073/345/346 at 50.7% on the full data),
  24 above 60%.
- Unique values: 116 constant, 6 binary, 1 with 3–10 values, 286 with more than 500.
- **New finding:** the 6 "binary" features (074, 206, 209, 342, 347, 478) are zero
  on every wafer except **source_line 1459** (a good wafer). That one wafer explains why the
  test split shows 122 constant columns versus 116 in train: it sits in train.
- 30 duplicate rows in the labels file = identical (label, timestamp) pairs, consistent
  with the 65 shared timestamps (D-004).

**Next steps** — Decide on the six single-wafer features (open item).

---

### D-025 · Five imputed training datasets built, compared, and loaded into `secom_production`
- **Date:** 2026-11-09
- **CRISP-DM phase:** Data Preparation
- **Type:** Change
- **Status:** Done (selection pending)

**Context** — Per D-021, the team wants one dataset per imputation strategy so they can be
compared against each other and against the un-imputed data. Raw data stays in
`secom_raw_staging`; imputed data goes to a production layer; later only one strategy goes
forward.

**Change**
- `src/secom_analysis/impute.py` (`python -m secom_analysis.impute`). Input
  `data/processed/train_prepared.csv` (1,253 × 450, 11,211 NaN, 3σ-capped). Strategies, all
  fitted on the training split only:
  - `median`; `median_flags` (median + one 0/1 flag per block of always-co-missing
    columns, blocks ≥ 1% of wafers → **8 flags**);
  - `knn` (k = 5, distance-weighted, standardised);
  - `mice_bayesridge` (IterativeImputer, 40 nearest columns, 10 rounds);
  - `mice_extratrees` (30 trees, min leaf 3, 40 nearest columns, 5 rounds).
  Model-based fills are clipped to the observed training range; observed cells are checked
  to be unchanged.
- Label-free comparison: **pattern-masked reconstruction** (hide observed cells using
  another wafer's real missing pattern: 7,993 cells; NRMSE in std units), variance ratio,
  KS of filled vs observed, mean shift (over the 28 columns with ≥ 5% missing), and
  correlation distortion.
- Outputs: `data/processed/imputed/train_<strategy>.csv`, fitted imputers in
  `models/imputers/<strategy>.joblib` (for the test set later; git-ignored),
  `reports/imputation/` (comparison CSV, report, two figures, per-column CSVs).
- `src/secom_analysis/production.py`: schema **`secom_production`** with
  `train_prepared` (NULL = missing), `train_imputed_<strategy>` ×5, registry
  `imputation_strategy` (params, metrics, hashes, status `candidate`/`selected`/`rejected`),
  and `--select <name>` → view `train_selected`. Load verified: row counts, NULL counts,
  every `source_line` joins to `secom_raw_staging.measurements`.
- 3 tests added (13 total pass).

**What happened**

| strategy | NRMSE (≥5% cols) | variance kept | KS filled vs observed | corr. distortion | fit |
|---|---|---|---|---|---|
| median | 0.998 | 0.840 | 0.500 | 0.0017 | 0.1 s |
| median_flags | 0.998 | 0.840 | 0.500 | 0.0017 | 0.1 s |
| knn | 0.963 | **0.856** | **0.249** | 0.0022 | 1 s |
| mice_bayesridge | **0.910** | 0.821 | 0.468 | **0.0014** | 23 s |
| mice_extratrees | 0.950 | 0.828 | 0.530 | 0.0015 | 120 s |

- **No strategy reconstructs hidden values much better than the mean** (best 0.91).
  Values go missing in whole sensor blocks, so the correlated sensors that could predict
  them are usually missing too.
- All strategies shrink the variance of high-missing columns (about 0.82–0.86); none
  restores the spread.
- k-NN gives the most realistic fills (lowest KS); BayesianRidge the lowest error.
- The label-free metrics do **not** single out a winner. The decision must come from
  the downstream model comparison (cross-validated, label-aware), as planned in D-021.
  The missing flags look identical here by construction; their value can only show up
  downstream.

**Next steps** — Downstream comparison with repeated stratified CV, the imputer fitted
inside each fold (D-026). Then select one strategy with
`python -m secom_analysis.production --select <name>`.

---

### D-026 · Split method reviewed: stratified hold-out kept; defect rate drifts over time
- **Date:** 2026-11-09
- **CRISP-DM phase:** Data Preparation / Evaluation design
- **Type:** Finding
- **Status:** Done

**Context** — The team asked how the split was made and whether it is the right trade-off.

**What we do** — A single stratified random hold-out: `train_test_split(test_size=0.2,
stratify=y, random_state=42)` (D-012). 1,253 train (83 defective) / 314 test (21 defective),
sealed by SHA-256.

**Options considered**

| Option | Pro | Con |
|---|---|---|
| Stratified random hold-out 80/20 (current) | Course requirement; same defect rate in both parts; simple; test set untouched | Only 21 defective wafers in test, so one wafer moves recall by 4.8 points; one split = one draw |
| Unstratified random split | Simplest | 1,000 simulations: 9–34 defective in test. Unreliable |
| Stratified k-fold CV only (no hold-out) | Uses all data for evaluation | Nothing left that model selection never saw |
| Time-ordered split (train on earlier, test on later) | Mirrors deployment | Defect rate differs by month (below); against the course's stratification requirement |
| **Hold-out + repeated stratified k-fold inside train** | Clean final test; stable model selection | More compute |

**Finding:** defect rate by month: **Jul 22.2% (14/63), Aug 9.2%, Sep 2.9%, Oct 6.1%.**
The random stratified split spreads every month over both sets (test share 18–25% per
month), so train and test are comparable. But a model judged on a random split may
look better than it would on future production.

**Decision** — Keep the sealed stratified hold-out. All selection (imputation, features,
balancing, hyperparameters) uses **repeated stratified 5-fold CV inside the training
split**, with every fitted step (imputer, scaler, SMOTE) inside the fold. Report a
time-ordered sensitivity check as a secondary result.

**Next steps** — Use this protocol for the D-025 downstream comparison.

---

### D-027 · Visual comparison of the imputation strategies
- **Date:** 2026-11-09
- **CRISP-DM phase:** Data Preparation
- **Type:** Change
- **Status:** Done

**Context** — The team asked to see the imputation strategies (D-025) visually before the
downstream comparison.

**Change**
- New `src/secom_analysis/impute_viz.py`, called at the end of `impute.run`, or on its own
  with `python -m secom_analysis.impute_viz` (no refit).
- `masked_reconstruction` now saves the hidden cells (true vs filled) as
  `reports/imputation/masked_cells_<strategy>.csv`.
- Five figures in `reports/imputation/figures/`:
  - distributions (4 block columns × strategy);
  - relationship scatter (feature_072 vs feature_066);
  - masked true-vs-filled;
  - variance-kept heatmap (28 columns);
  - per-column error vs missing %.
- Gallery: `reports/imputation/imputation_visuals.md`. The re-run reproduced the D-025 numbers exactly.

**What happened**
- The median spikes. The model fills are narrower than real readings, and on two-peaked
  columns (546, 562) they land between the peaks.
- MICE fills follow the 072–066 relationship; median ignores it.
- Masked cells: corr(true, filled) is 0.01 for median, 0.24 for k-NN, 0.32 for BayesianRidge and 0.21 for ExtraTrees.
- The 45–49% missing columns keep about 0.51–0.62 of their variance, which is what a mean fill
  gives. The exception is 072 under MICE (0.78/0.74).
- The gains are concentrated in the low-missing columns: ExtraTrees has 31/102 columns with
  NRMSE < 0.8, BayesianRidge 28, median 25, k-NN 24.

**Interpretation** — Model-based imputation helps only where an observed, correlated sensor
exists. For whole missing blocks, every method is close to a mean fill, so the missing flags
may matter more than the fill method. No winner yet; the downstream CV decides (D-025, D-026).

**Next steps** — The downstream comparison.

---

### D-028 · Split schemes compared empirically: one hold-out is noisy, time order costs about 0.07 PR-AUC
- **Date:** 2026-11-09
- **CRISP-DM phase:** Data Preparation / Evaluation design
- **Type:** Experiment
- **Status:** Done (closes the D-026 open item on the time-ordered check)

**Context** — The team asked for an analysis of how the train/test split is done, compared with
other methods. D-026 compared them on paper; this entry measures them.

**Change**
- New `src/secom_analysis/split_study.py` (`python -m secom_analysis.split_study`, about 25 min).
  It runs one fixed pipeline under 8 split schemes on the **training split only**, so the
  sealed test set is not read. The pipeline is a `FoldPreparation` step (D-016/D-017/D-019
  refitted per fold), then median imputation → scaling → LR (C = 0.01) or RF (300 trees).
- 4 tests in `tests/test_split_study.py`: preparation learns only from the fit rows,
  stratification fixes the defect count, time schemes never train on the future,
  leave-one-month-out covers every row once.
- Outputs: `reports/split_study/` (`split_study.md`, `split_summary.csv`, `split_scores.csv`,
  `monthly_defect_rate.csv`, `split_comparison.png`).
- The first run was killed by the OOM killer: the RF with `n_jobs=-1` started 12 workers
  while about 9 GB was free. It is now capped at 4.

**What happened** (PR-AUC, training split; chance about 0.066)

| Scheme | LR mean (SD) | RF mean (SD) |
|---|---|---|
| Stratified 80/20 hold-out, 100 seeds (current method) | 0.208 (0.078) | 0.214 (0.069) |
| Random 80/20 hold-out, 100 seeds (8–27 defective in test) | 0.201 (0.071) | 0.215 (0.078) |
| Stratified 5-fold CV, 20 repeats | 0.194 (0.016) | 0.207 (0.015) |
| Time-ordered hold-out (latest 20% = 2–17 Oct) | 0.157 | 0.102 |
| Forward chaining, 5 expanding folds | 0.103 | 0.123 |
| Leave one month out | 0.143 | 0.136 |
| Leaky: preparation fitted before CV | 0.196 (0.017) | 0.206 (0.017) |
| Leaky: fail class duplicated 24× before split (the papers) | 0.951 | 1.000 |

- Across seeds a single hold-out ranges from **0.09 to 0.43** for the same pipeline. Seed 42
  gives 0.136 (LR) / 0.154 (RF), around the 16th–17th percentile. Repeated CV is about 5× more stable.
- Time-ordered schemes are about 0.05–0.10 below random splits. ROC-AUC falls too (about 0.70 →
  0.52–0.69), so this is not only a prevalence effect. Defect rate by month (train): Jul 17.0%,
  Aug 9.0%, Sep 2.9%, Oct 7.3%. This is consistent with D-029's October check.
- Fitting the preparation before splitting changes PR-AUC by at most 0.002.
- Duplicating the fail class before splitting puts copies of about 82 of the 83 defective wafers in
  both parts, which yields near-perfect scores. (Chance there is about 0.63, because the test set is 63% defective.)

**Interpretation** — The split method does not change what the pipeline *can* do. It changes
how noisy (single hold-out) or how optimistic (random vs time order; oversampling before the
split) the measured number is. The papers' high scores are explained by oversampling before
the split, not by fitting the preparation before the split.

**Decision** — D-026 stands, now with evidence: a sealed stratified hold-out for the single final
number, reported with a bootstrap CI; repeated stratified CV for all selection; forward chaining /
leave-one-month-out reported as the deployment-like figure. Oversampling or SMOTE only inside the training fold.

**Next steps**
- Add a bootstrap CI to the final test-set evaluation (open item).
- Report section on evaluation design: use `split_comparison.png` and connect it to D-022.

---

### D-029 · Downstream CV: no imputation strategy beats median
- **Date:** 2026-11-09
- **CRISP-DM phase:** Data Preparation (feeds Modeling)
- **Type:** Decision
- **Status:** Proposed (awaiting team confirmation)
- **Note:** D-028 is used by the parallel split-scheme study (`split_study.py`).

**Context** — The label-free checks (D-025, D-027) did not pick a winner. Per D-026, the
decision comes from leakage-free, cross-validated model performance on the training split.

**Change**
- New `src/secom_analysis/impute_cv.py` (`python -m secom_analysis.impute_cv`, about 23 min):
  3×5-fold repeated stratified CV with the imputer, scaler and model fitted inside every
  fold. Two models: logistic regression (C = 0.05, balanced) and random forest (400 trees,
  balanced_subsample). Metrics: PR-AUC (primary), ROC-AUC, recall / precision / MCC at 10% FPR.
  Paired comparison against median with the Nadeau-Bengio corrected t-test. Plus a
  time-ordered check (Jul–Sep → Oct).
- `production.py --record-cv` stores the CV means in `secom_production.imputation_strategy`
  (`metrics -> 'cv'`). Two tests added (15 total).
- Outputs: `reports/imputation_cv/` (`cv_report.md`, `cv_summary.csv`, `cv_folds.csv`,
  `temporal_check.csv`, `cv_pr_auc.png`).

**What happened** (PR-AUC, mean over 15 folds; no skill = 0.066)

| strategy | Logistic regression | Random forest | Δ vs median (LR / RF) | corrected p (LR / RF) | Oct check (LR / RF) |
|---|---|---|---|---|---|
| median | 0.172 | 0.194 | — | — | 0.116 / 0.104 |
| median_flags | 0.171 | 0.194 | −0.001 / −0.000 | 0.91 / 1.00 | **0.118 / 0.106** |
| knn | 0.178 | 0.190 | +0.006 / −0.004 | 0.52 / 0.84 | 0.113 / 0.097 |
| mice_bayesridge | 0.169 | 0.197 | −0.003 / +0.003 | 0.62 / 0.91 | 0.111 / 0.093 |
| mice_extratrees | 0.171 | 0.195 | −0.001 / +0.001 | 0.80 / 0.96 | 0.116 / 0.101 |

- **No difference is significant:** at most 0.006 PR-AUC, against a fold-to-fold SD of about 0.06.
  This is what D-027 predicted: for whole missing blocks, every method is close to a mean fill.
- Both models are clearly above chance (about 2.6–2.9× the no-skill PR-AUC), but the signal is weak.
- **The time-ordered check drops PR-AUC to about 0.10–0.12 for every strategy.** The temporal
  drift costs far more than the choice of imputer.
- Fit cost per fold: median under 0.1 s, k-NN about 1 s, BayesianRidge about 20 s, ExtraTrees about 140 s.

**SIP: imputation**

| Options | Pro / con | Decision | Consequence |
|---|---|---|---|
| Median | + simplest, deterministic, robust to skew and outliers; trivial on the test set. − spike at one value; variance shrinks (D-027) | Not chosen, but equivalent | Same performance as the recommended option |
| **Median + missing flags (8 block flags)** | + same fill as median, and keeps "was missing" visible to feature selection and models (the informative block 072/073/345/346, D-009); best on the October check (within noise). − 8 extra columns | **Recommended** | 458 columns; Boruta (week 4) can judge whether the flags carry signal |
| k-NN (k = 5) | + most realistic fill shape. − needs scaling, slower, sensitive to the 15-order scale range; no gain | Rejected | — |
| MICE · BayesianRidge | + follows sensor relationships, lowest masked error. − 20 s per fit, extrapolation risk; no gain | Rejected | — |
| MICE · ExtraTrees | + non-linear. − about 140 s per fit (about 1,400× median); no gain | Rejected | — |
| Class-mean imputation (Paper B) | Uses the label to fill features: target leakage, not deployable | Rejected (D-022) | — |
| HistGradientBoosting native NaN | No imputation needed; model-specific | Deferred | Can be tried as a model in week 4 |

**Interpretation** — When strategies are statistically equivalent, parsimony decides: the
simplest fill, plus flags so that the missingness information is not lost.

**Next steps**
- Team confirms → `python -m secom_analysis.production --select median_flags` (creates
  `secom_production.train_selected`).
- Re-run `impute_cv` if the week-4 pipeline adds SMOTE or k-NN.
- Combine the drift finding with the split study (D-028).


### D-030 · Raw data exported to a single CSV (2026-11-09)

**What** — `secom.data` + `secom_labels.data` merged into `data/processed/secom_raw.csv`
(1,567 rows × 592 columns: `timestamp`, `label`, `feature_000`–`feature_589`).

**How** — Read with the project loaders (`load_secom`, `load_labels`); no cleaning, imputation
or splitting. Labels keep the original coding (-1 pass / 1 fail); timestamps written as ISO
`YYYY-MM-DD HH:MM:SS`; missing values left as empty cells.

**Check** — Round-trip read: 41,951 missing cells (same as raw), 1,463 pass / 104 fail.

**Note** — Full dataset, test rows included: convenience copy only; the sealed test set rule
still applies to anything built from it.

---

### D-031 · Students Manual read; build plan derived
- **Date:** 2026-11-10
- **CRISP-DM phase:** Business Understanding
- **Type:** Decision
- **Status:** Accepted

**Context** — The course Students Manual (`data/raw/Linda_SECOM Case Study - Students Manual (1).docx`)
was added. It defines the deliverables, the weekly to-dos (weeks 1–7), the rules, and 17
implementation sections, each with a checklist, decision points and Q&A tables.

**Decision / change** — `docs/build_plan.md` maps every manual step to a build step (phases A–M),
marks it Done / Partial / To do against the existing diary entries, and lists the manual's
internal conflicts with the option the plan uses.

**What happened** — Phases A–D (understanding, split, rough reduction, 3σ capping) are largely
done (D-013 to D-019). Gaps that the manual makes mandatory:
- the analysis script must download the data at runtime and install packages itself;
- the duplicate check must run on X_train with its own SIP row;
- kNN imputation is the manual's required method, which conflicts with the median + flags
  proposal in D-029;
- Boruta (about 15–20 features), min-max scaling, SMOTE in an `imblearn` pipeline, stratified
  CV grid search (outlier treatment × k × balancer × RF parameters), and cost-based selection
  (Type I = 100, Type II = 400) are still to be built;
- deliverables: one end-to-end script, the Google Docs notebook with Q&A tables, and the slides.

**Manual conflicts** — cost 100/400 (weeks 6–7, §17.3) vs €200/€5000 (§14.2); scaler before
selector (§10.3) vs after (§17.1); Boruta optional (§7.3) vs required (week 4, §17.1); outlier
treatment outside the pipeline (§10.2) vs part of the grid search (week 7).

**Artifacts** — `docs/build_plan.md`.

**Next steps** — See the D-031 rows in Open items; next build step is phase E4 (imputer decision),
then F (Boruta).

### D-034 · Scaffold overview document
- **Date:** 2026-11-10
- **CRISP-DM phase:** Infrastructure
- **Type:** Change
- **Status:** Done

**Context** — A single-page summary of the scaffold was requested.
**Decision / change** — Added `docs/scaffold.md`. It covers the purpose, fixed rules, directory layout, module table,
data flow, commands, key findings, phase status and open team decisions. It summarises the README, `build_plan.md`
and this diary, and introduces no new decisions.
**What happened** — `python -m pytest -q`: 19 passed.
**Artifacts** — `docs/scaffold.md`.
**Next steps** — Update it when a phase changes status.

*Note: first written as D-032, at the same time as the imputer entry D-032; renumbered to D-034 to keep IDs unique.*

---

### D-032 · Official imputer: k-NN (k = 5, scaled)
- **Date:** 2026-11-10
- **CRISP-DM phase:** Data Preparation
- **Type:** Decision
- **Status:** Accepted (supersedes the D-029 proposal)

**Context** — D-029 proposed median + missing flags. The Students Manual (§6, D-031) requires
k-NN imputation with an adjustable k and scaled distances, and lists median and MICE only as
optional comparisons.

**Decision** — k-NN, k = 5, distance-weighted, on standardised features (`ScaledImputer`,
`impute.py`). k stays adjustable for the grid search.

**Rationale** — The manual requires it, and D-029 found no significant difference between the five
strategies (PR-AUC 0.169–0.178 for logistic regression, 0.190–0.197 for the random forest; fold SD
≈ 0.06). Choosing k-NN therefore costs nothing measurable.

**Alternatives considered** — median, median + flags, MICE BayesianRidge, MICE ExtraTrees (D-025,
D-029). All go into the imputation SIP table as tested alternatives.

**SIP: imputation**

| Options / approaches | Pro and con | Decision made | Consequence |
|---|---|---|---|
| Mean | Simple / collapses variance, ignores class-related missingness; banned by the course | Rejected | — |
| Median, median + flags | Fast, robust / single value per column | Tested alternative | PR-AUC within 0.006 of k-NN (D-029) |
| k-NN, k = 5, scaled | Case-specific fills, manual's method / needs scaling, slower (≈1 s per fit) | **Chosen** | 11,211 NaN → 0 on the 1,253 × 450 train split |
| MICE (BayesianRidge, ExtraTrees) | Models each column / slow (20–140 s per fold), no gain | Tested alternative | No significant difference (D-029) |

---

### D-033 · Duplicate check on train, first random forest, Boruta stability selection
- **Date:** 2026-11-10
- **CRISP-DM phase:** Data Preparation (feeds Modeling)
- **Type:** Experiment
- **Status:** Done

**Context** — Build plan phases C2, E5 and F (manual §4.3, week 3, §7). Training split only
(`train_prepared.csv`: 1,253 rows × 450 features, 83 defective = 6.62%); k-NN imputer per D-032.

**Change** — New `src/secom_analysis/selection.py` (`python -m secom_analysis.selection`,
≈ 13 min) with `BorutaSelector` (a pipeline-ready BorutaPy wrapper), `ColumnSubset`,
`duplicate_features`, `stability_selection`. 4 tests in `tests/test_selection.py` (23 pass).
`boruta` 0.4 and `imbalanced-learn` 0.14 added to `pyproject.toml`.

**What happened** (all numbers: training split)
- **Duplicates (§4.3):** 104 among the raw 590 columns, all of them constant columns (as in D-003);
  **0** among the 474 varying columns and 0 among the 450 kept. Nothing removed.
- **First model (week 3):** default random forest (500 trees). On its own training rows:
  accuracy 1.000, recall 1.000. Under stratified 5-fold CV: **accuracy 0.934, recall 0.000**
  (0 of 83 defects caught; ROC-AUC 0.705, PR-AUC 0.146). The CV accuracy equals the
  always-pass baseline (1,170 / 1,253 = 0.934), which shows why accuracy and training-set results
  cannot be trusted here and why balancing / a lower threshold is needed.
- **Boruta:** BorutaPy, depth-5 class-weighted forest, 100 iterations, 10 seeds; keep a feature
  when it is confirmed or tentative in ≥ 50% of runs. **450 → 13 features:** 031, 033, 059, 064, 065,
  103, 130, 213, 351, 477, 510 (confirmed in 10/10), 205 (8 confirmed + 2 tentative),
  280 (1 + 6, tentative most of the time). Next best: 021 (4/10). The set sits just under the
  manual's 15–20 target. Lowering BorutaPy's `perc` to 90 gives 76 features, so the result was
  not forced into the target range.
- **Cross-check:** 12 of the 13 are in the random-forest importance top 13, but only 7 in the
  ANOVA F-test top 13. Part of the signal is non-linear or comes from interactions.
  The D-009 block (072/073/345/346) was not selected.
- **§7.6 comparison** (3×5-fold, class-weighted RF, PR-AUC; no skill = 0.066):

| Feature set | Features | PR-AUC (SD) | ROC-AUC | Recall @10% FPR | Δ PR-AUC vs all, corrected p |
|---|---|---|---|---|---|
| All 450 | 450 | 0.205 (0.069) | 0.714 | 0.304 | — |
| Boruta set fixed on all train rows (manual §7.6) | 13 | 0.224 (0.063) | 0.727 | 0.320 | +0.019, p = 0.69 |
| Boruta refitted inside each fold | 10–25 (mean 15.6) | 0.162 (0.052) | 0.681 | 0.236 | −0.044, p = 0.18; worse in 13/15 folds |

**Interpretation** — The apparent gain of the fixed 13 is selection bias: those features were chosen
using the validation rows too (the stricter standard in manual §7.5). Measured honestly, Boruta
costs about 0.04 PR-AUC against all features. The difference is not significant, but it is
consistent across folds. The 13-feature model is far more parsimonious and interpretable,
which is the course's aim, so it is kept as the working set. The grid search must refit Boruta
inside each fold, and an "all features" option should stay in the comparison.

**SIP: feature selection**

| Options / approaches | Pro and con | Decision made | Consequence |
|---|---|---|---|
| Filter (ANOVA F) | Fast, model-free / univariate, linear only | Cross-check only | 7/13 overlap with Boruta |
| Embedded (RF importance) | One fit / tied to RF, biased to high-cardinality | Cross-check only | 12/13 overlap |
| Wrapper: Boruta (shadow features) | Tests each feature against its own shuffled copy, all-relevant / slower (≈ 26 s per run), seed-dependent | **Chosen**, 10-seed stability rule, tentatives kept at ≥ 50% | 450 → 13 features |
| Number of features | Manual 15–20 / Boruta finds 13 stable ones; perc=90 gives 76 | 13 (data-driven, not padded) | In-fold honest PR-AUC 0.162 vs 0.205 for all 450 |

**Artifacts** — `reports/feature_selection/` (`report.md`, `boruta_stability.png`,
`cv_feature_sets.png`, `boruta_runs.csv`, `boruta_stability.csv`, `ranking_cross_check.csv`,
`first_model.csv`, `duplicates.csv`, `cv_folds.csv`, `cv_summary.csv`);
`data/processed/selected_features.json`; run log `reports/selection_run.log`.

**Next steps** — Phase G/H: min-max scaling and SMOTE vs class weighting, inside an `imblearn`
pipeline with Boruta refitted per fold. Notebook 02 sections for this entry.

---

### D-035 · Full pipeline: switchable 3σ step, normality, min-max, balancers, first confusion matrices
- **Date:** 2026-11-10
- **CRISP-DM phase:** Data Preparation (feeds Modeling / Evaluation)
- **Type:** Experiment
- **Status:** Done

**Context** — Build plan D3, G, H, I and the first part of K (manual §5, §8, §9, §10, §13). Training
split only, starting from `train_reduced.csv` (1,253 × 450, 83 defective, **not yet capped**), so
the 3σ step is refitted inside every fold.

**Change** — New `src/secom_analysis/pipeline.py` (`python -m secom_analysis.pipeline`, ≈ 10 min)
and `tests/test_pipeline.py` (10 tests; 33 pass in total). `build_pipeline(outlier, sigma,
n_neighbors, selector, balancer, n_estimators)` returns an `imblearn` Pipeline:

`OutlierTreatment (cap | remove→NaN | none, kσ) → k-NN imputer (k=5, scaled) → Boruta | all →
MinMaxScaler → balancer → RandomForest (500)`

**Pipeline order** — The manual conflicts with itself: §10.3 puts the scaler before selection, and §17.1 puts it after.
We use **selector before scaler** (§17.1): the k-NN imputer already scales internally for its
distances, Boruta's forest is scale-invariant, and min-max on the final features keeps the
scaled set small and readable. Samplers act only during `fit`, so no synthetic row reaches a
validation fold.

**What happened** (all numbers: training split)
- **Normality (§8):** 445 of the 450 imputed features reject normality (Shapiro-Wilk p < 0.05), and 223 have
  |skew| > 1. Yeo-Johnson brings the 13 Boruta features to |skew| ≤ 0.31, but Shapiro still rejects
  all of them. Random forests split on thresholds, so **no normality transform** is applied. It stays an
  option for logistic regression / SVM.
- **Min-max (§8):** fitted on the fold's training rows after selection; range on train exactly [0, 1];
  column names kept.
- **Class counts (§9):** 1,170 good / 83 defective; always-pass baseline accuracy 0.934 (recall 0).
  After resampling: ROSE-style and SMOTE 1,170 / 1,170 (2,340 rows); ADASYN 1,170 / 1,187.
  ROSE itself is R-only; `RandomOverSampler(shrinkage=1.0)` (smoothed bootstrap) is the
  Python equivalent.
- **Pipeline checks (§10.5, first fold):** 1,002 train / 251 validation rows, 0 shared; shapes
  450 → 450 → 450 → 18 (Boruta) → 18 → 1,872 rows after SMOTE (fit only); 251 predictions returned;
  identical predictions on refit; test set seal intact.
- **Balancers** (3×5-fold CV, Boruta refitted per fold: 10–26 features; threshold 0.5):

| Balancer | Sensitivity | Specificity | Precision | F2 | PR-AUC | ROC-AUC | Cost per fold (SD) |
|---|---|---|---|---|---|---|---|
| None | 0.008 | 0.998 | 0.089 | 0.010 | 0.172 | 0.675 | 6,627 (205) |
| Class weights | 0.076 | 0.975 | 0.213 | 0.086 | 0.159 | 0.677 | 6,720 (487) |
| ROSE-style | **0.289** | 0.895 | 0.166 | **0.249** | **0.192** | 0.697 | 7,180 (722) |
| SMOTE | 0.145 | 0.943 | 0.157 | 0.144 | 0.153 | 0.692 | 7,013 (628) |
| ADASYN | 0.145 | 0.938 | 0.144 | 0.142 | 0.157 | 0.688 | 7,140 (738) |

- **Initial confusion matrices** (out-of-fold, CV repeat 1, all 1,253 rows, threshold 0.5):

| Balancer | TP | FN | FP | TN | Accuracy | Cost (100·FP + 400·FN) |
|---|---|---|---|---|---|---|
| None | 1 | 82 | 1 | 1,169 | 0.934 | 32,900 |
| Class weights | 8 | 75 | 20 | 1,150 | 0.924 | 32,000 |
| ROSE-style | 24 | 59 | 113 | 1,057 | 0.863 | 34,900 |
| SMOTE | 12 | 71 | 50 | 1,120 | 0.903 | 33,400 |
| ADASYN | 12 | 71 | 57 | 1,113 | 0.898 | 34,100 |

  Always-pass costs 83 × 400 = **33,200**.

**Interpretation** — At threshold 0.5, **no option beats the always-pass classifier on cost**: each
extra defect caught costs about 4–5 false alarms. Balancing moves the operating point (ROSE-style triples
sensitivity vs SMOTE and has the best PR-AUC), but ranking quality is close for all options (ROC-AUC
0.68–0.70, fold SD ≈ 0.065), and none of these differences is clearly beyond fold noise. The
threshold, not the balancer, decides the cost. With costs of 100/400 the expected-cost threshold for calibrated
probabilities is 100 / (100 + 400) = 0.2, so phase K must pick the threshold on CV predictions.
ROSE-style and SMOTE both go into the grid search; SMOTE stays the default because the manual names it.

**SIP: normality and scaling**

| Options / approaches | Pro and con | Decision made | Consequence |
|---|---|---|---|
| No transform | Simple; RF is invariant to monotone transforms / LR, SVM may suffer | **Chosen for RF** | 445/450 features non-normal, left as is |
| Yeo-Johnson | Fixes skew (13 Boruta features to \|skew\| ≤ 0.31) / not normal anyway (Shapiro), extra fitted step | Option for LR/SVM | — |
| Min-max (0–1) | Manual's choice, bounded, keeps shape / sensitive to extremes (handled by 3σ cap first) | **Chosen**, after Boruta | Train range [0, 1] |
| Standardisation | Mean 0, SD 1 / unbounded | Used inside the k-NN imputer only | — |

**SIP: class balancing**

| Options / approaches | Pro and con | Decision made | Consequence |
|---|---|---|---|
| None | No synthetic data / model ignores defects | Baseline | Sensitivity 0.008 |
| Class weights | No new rows / weak effect with RF at 0.5 | Grid option | Sensitivity 0.076 |
| ROSE-style (smoothed bootstrap) | Simple, manual's ROSE idea / jitter can blur classes | Grid option | Sensitivity 0.289, PR-AUC 0.192 |
| SMOTE | Manual's default, interpolates neighbours / can create noise in overlap | **Default** | 83 → 1,170 defects in fit; sensitivity 0.145 |
| ADASYN | Focuses on hard cases / more noise | Grid option | Same as SMOTE, slightly higher cost |

**Artifacts** — `reports/pipeline/` (`report.md`, `normality.png`, `confusion_matrices.png`,
`normality_all_features.csv`, `resampling_counts.csv`, `cv_folds.csv`, `cv_summary.csv`,
`confusion_oof.csv`, `pipeline_checks.json`); run log `reports/pipeline_run.log`.

**Next steps** — Phase J grid search over outlier × k × selector (Boruta / all) × balancer × RF
parameters; phase K threshold choice on CV predictions with cost 100/400; notebook 02.

### D-036 · Audit agent and drift checks: `secom-auditor`, `/audit`, `secom_analysis.audit`
- **Date:** 2026-11-10
- **CRISP-DM phase:** Infrastructure
- **Type:** Change
- **Status:** Done

**Context** — Results, diary, notebooks and code are now spread over about 15 modules and 36 entries.
Numbers are quoted in several places, and nothing checked that they still agree or that results
have not silently moved after a code change.

**Change**
- `src/secom_analysis/audit.py` (`python -m secom_analysis.audit`): deterministic checks for the test
  seal, the train/test overlap and row total, column accounting (zero-variance + high-missing + kept =
  590), 3σ capping of `train_prepared`, selected-feature validity, reads of the sealed test set
  (and of `secom_raw.csv`, which contains it) in code, course-rule patterns (IQR outlier rule, mean
  imputation), diary structure (index ↔ log ↔ open items, statuses, dangling IDs), staleness
  (outputs older than their code or input data), and **drift** against an accepted snapshot
  `docs/audit_baseline.json`: hashes, every numeric cell of the CV summary tables, preparation
  state, stable feature set. A metric moving by > 0.01 (absolute for rates, relative for counts) is
  HIGH; smaller non-zero changes are flagged as possible non-determinism. Output goes to
  `reports/audit/latest.{md,json}` and `history.csv`. Exit 1 on CRITICAL/HIGH.
- `tests/test_audit.py` (5 tests; **38 pass in total**).
- `.claude/agents/secom-auditor.md`: read-only agent that runs the tests and the audit, then
  cross-checks diary numbers against artifacts, code constants against the diary, and notebooks
  against reports, and hunts for leakage and reasoning gaps. It returns a ranked alert list and
  never fixes or accepts a baseline. `/audit [focus]` runs it. `CLAUDE.md` makes it the
  central testing step.

**Rationale** — Mechanical checks (hashes, counts, ID cross-references) belong in a script that
gives the same answer every time. Judgement checks (is a 0.02 PR-AUC difference a real win? does
an entry cite train-only evidence?) belong in the agent. The baseline is tracked in git because
`reports/` is git-ignored, so accepted numbers are versioned next to the diary.

**What happened** (first run, HEAD 4f0b329): CRITICAL 0 · HIGH 1 · MEDIUM 5 · LOW 7.
- Test seal intact; 1,253 / 314 rows, no overlap; 116 + 24 + 450 = 590 columns; `train_prepared`
  fully within its 3σ bounds.
- HIGH: `reports/health` predates the current `train_reduced.csv`.
- `reports/pipeline` (D-035 numbers) was written at 14:33, before the last `pipeline.py` edit at
  14:43, and `reports/imputation_cv` predates `impute_cv.py`. Their numbers are not yet confirmed to
  come from the current code.
- Stable Boruta set (13, D-033) ≠ the pipeline's in-fold set in its first fold (18): 3 only in the
  stable set, 8 only in-fold. Expected (Boruta is refitted per fold), but text must say which.
- Diary: D-030 has no Date/Type/Status block; D-034 is logged before D-032; D-026 title differs
  between index and entry.
- No baseline accepted yet. On purpose: accepting now would freeze possibly stale pipeline
  numbers.

**Artifacts** — `src/secom_analysis/audit.py`, `tests/test_audit.py`,
`.claude/agents/secom-auditor.md`, `.claude/skills/audit/SKILL.md`, `CLAUDE.md`,
`reports/audit/`.

**Next steps** — Rerun `pipeline` and `impute_cv`, review, then
`python -m secom_analysis.audit --accept-baseline` and commit the baseline; fix the D-030
metadata; run `/audit` after every change.
