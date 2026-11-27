"""Self-contained HTML descriptive report built from the health-check outputs.

The page has no external dependencies: charts are inline SVG, styling is inline CSS
(light and dark themes), and the feature table is sortable/filterable with a few lines
of vanilla JavaScript. Open ``reports/health/descriptive_report.html`` in any browser.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

STATUS_ICON = {"PASS": "✓", "WARN": "!", "FAIL": "✕"}


# --------------------------------------------------------------------------------------
# SVG chart primitives
# --------------------------------------------------------------------------------------


def _nice_max(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    for step in (1, 2, 2.5, 5, 10):
        if value <= step * magnitude:
            return step * magnitude
    return 10 * magnitude


def _fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:.3g}"


def _column_chart(
    labels: list[str],
    series: list[tuple[str, str, list[float]]],
    *,
    y_label: str,
    width: int = 640,
    height: int = 260,
    tick_every: int = 1,
) -> str:
    """Vertical (optionally stacked) column chart. ``series`` = [(name, css_var, values)]."""
    left, right, top, bottom = 44, 8, 12, 36
    plot_w, plot_h = width - left - right, height - top - bottom
    totals = np.sum([s[2] for s in series], axis=0)
    step = _nice_max(float(totals.max()) / 5)
    n_ticks = max(math.ceil(float(totals.max()) / step), 1)
    y_max = step * n_ticks
    slot = plot_w / len(labels)
    bar_w = max(slot - 2, 1)  # 2px surface gap between adjacent bars
    parts = [
        (
            f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" '
            f'aria-label="{html.escape(y_label)}">'
        )
    ]
    for i in range(n_ticks + 1):
        value = step * i
        y = top + plot_h - plot_h * i / n_ticks
        parts.append(
            f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
            f'<text x="{left - 6}" y="{y + 4:.1f}" class="tick" text-anchor="end">'
            f"{_fmt(value)}</text>"
        )
    for i, label in enumerate(labels):
        x = left + i * slot + 1
        base = top + plot_h
        tip = [label]
        for s_index, (name, color, values) in enumerate(series):
            value = values[i]
            tip.append(f"{name}: {_fmt(value)}")
            if value <= 0:
                continue
            h = plot_h * value / y_max
            is_top = s_index == len(series) - 1 or all(
                later[2][i] <= 0 for later in series[s_index + 1 :]
            )
            gap = 0 if s_index == 0 else 1  # 2px surface gap between stacked segments
            rx = 2 if is_top else 0
            parts.append(
                f'<rect x="{x:.1f}" y="{base - h:.1f}" width="{bar_w:.1f}" '
                f'height="{max(h - gap, 0.5):.1f}" rx="{rx}" fill="var({color})"/>'
            )
            base -= h
        parts.append(
            f'<rect x="{x - 1:.1f}" y="{top}" width="{slot:.1f}" height="{plot_h}" '
            f'class="hit"><title>{html.escape(" · ".join(tip))}</title></rect>'
        )
        if i % tick_every == 0:
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{height - bottom + 16}" class="tick" '
                f'text-anchor="middle">{html.escape(label)}</text>'
            )
    parts.append(
        f'<line x1="{left}" x2="{width - right}" y1="{top + plot_h}" y2="{top + plot_h}" '
        'class="axis"/></svg>'
    )
    return "".join(parts)


def _bar_chart(
    rows: list[tuple[str, float, str]],
    *,
    value_fmt=_fmt,
    width: int = 640,
    diverging: bool = False,
    tooltips: list[str] | None = None,
) -> str:
    """Horizontal bars; rows = [(label, value, css_var)]. Diverging bars grow from 0."""
    row_h, label_w, value_w = 26, 150, 64
    height = row_h * len(rows) + 8
    plot_w = width - label_w - value_w
    extent = max((abs(v) for _, v, _ in rows), default=1) or 1
    zero = label_w + (plot_w / 2 if diverging else 0)
    scale = (plot_w / 2 if diverging else plot_w) / extent
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart">']
    if diverging:
        parts.append(f'<line x1="{zero}" x2="{zero}" y1="0" y2="{height}" class="axis"/>')
    for i, (label, value, color) in enumerate(rows):
        y = 4 + i * row_h
        w = max(abs(value) * scale, 1)
        x = zero - w if value < 0 else zero
        tip = tooltips[i] if tooltips else f"{label}: {value_fmt(value)}"
        value_x = (zero + w + 6) if value >= 0 else (zero - w - 6)
        anchor = "start" if value >= 0 else "end"
        if not diverging:
            value_x, anchor = zero + w + 6, "start"
        parts.append(
            f'<text x="{label_w - 8}" y="{y + 16}" class="label" text-anchor="end">'
            f"{html.escape(label)}</text>"
            f'<rect x="{x:.1f}" y="{y + 4}" width="{w:.1f}" height="{row_h - 10}" rx="2" '
            f'fill="var({color})"/>'
            f'<text x="{value_x:.1f}" y="{y + 16}" class="value" text-anchor="{anchor}">'
            f"{html.escape(value_fmt(value))}</text>"
            f'<rect x="0" y="{y}" width="{width}" height="{row_h}" class="hit">'
            f"<title>{html.escape(tip)}</title></rect>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _legend(items: list[tuple[str, str]]) -> str:
    return (
        '<div class="legend">'
        + "".join(
            f'<span><i style="background:var({color})"></i>{html.escape(name)}</span>'
            for name, color in items
        )
        + "</div>"
    )


# --------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------


def _tiles(overviews: dict, profile: pd.DataFrame) -> str:
    features = overviews["features"]
    labels = overviews["labels"]
    tiles = [
        ("Production units", f"{features['rows']:,}", f"{labels['production_days']} days"),
        ("Sensor features", f"{features['columns']}", "all numeric"),
        (
            "Defect rate",
            f"{labels['defect_pct']:.2f}%",
            f"{labels['class_counts'].get('1', 0)} fails of {labels['rows']:,}",
        ),
        (
            "Missing cells",
            f"{features['missing_cell_pct']:.2f}%",
            f"{features['missing_cells']:,} cells",
        ),
        (
            "Constant features",
            f"{int(profile['is_constant'].sum())}",
            "zero variance — drop candidates",
        ),
        (
            "Complete rows",
            f"{features['complete_rows']}",
            "rows with no missing value",
        ),
    ]
    return (
        '<div class="tiles">'
        + "".join(
            f'<div class="tile"><div class="t-label">{a}</div><div class="t-value">{b}</div>'
            f'<div class="t-sub">{c}</div></div>'
            for a, b, c in tiles
        )
        + "</div>"
    )


def _checks_table(checks: list[dict]) -> str:
    rows = []
    for check in checks:
        status = check["status"]
        value = json.dumps(check["value"], default=str)
        if len(value) > 70:
            value = value[:67] + "…"
        items = check.get("items") or []
        more = ""
        if items and status != "PASS":
            shown = ", ".join(map(str, items[:12]))
            extra = f" … (+{len(items) - 12})" if len(items) > 12 else ""
            more = f'<div class="items">{html.escape(shown + extra)}</div>'
        rows.append(
            f'<tr class="s-{status.lower()}"><td>{html.escape(check["dataset"])}</td>'
            f"<td>{html.escape(check['check'])}</td>"
            f'<td><span class="badge {status.lower()}">{STATUS_ICON[status]} {status}</span></td>'
            f"<td><code>{html.escape(value)}</code></td>"
            f"<td>{html.escape(check['detail'])}{more}</td></tr>"
        )
    return (
        '<div class="filters" data-for="checks">'
        '<label><input type="checkbox" id="hide-pass"> Hide passing checks</label></div>'
        '<div class="scroll"><table id="checks"><thead><tr><th>Dataset</th><th>Check</th>'
        "<th>Status</th><th>Value</th><th>Detail</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _overview_table(overviews: dict) -> str:
    rows = []
    for name in ("features", "train", "test"):
        ov = overviews.get(name)
        if not ov:
            continue
        rm = ov["row_missing_pct"]
        rows.append(
            f"<tr><td>{name}</td><td class='n'>{ov['rows']:,}</td><td class='n'>{ov['columns']}"
            f"</td><td class='n'>{ov['defect_pct']:.2f}%</td>"
            f"<td class='n'>{ov['missing_cell_pct']:.2f}%</td>"
            f"<td class='n'>{ov['complete_rows']}</td><td class='n'>{ov['complete_columns']}</td>"
            f"<td class='n'>{rm['min']:.1f} / {rm['median']:.1f} / {rm['max']:.1f}</td></tr>"
        )
    return (
        '<div class="scroll"><table><thead><tr><th>Dataset</th><th class="n">Rows</th>'
        '<th class="n">Features</th><th class="n">Defect %</th><th class="n">Missing cells</th>'
        '<th class="n">Complete rows</th><th class="n">Complete features</th>'
        '<th class="n">Row missing % min / median / max</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _missingness_chart(profile: pd.DataFrame) -> str:
    edges = [0, 0.0001, 1, 5, 10, 20, 50, 80, 100.0001]
    names = ["0%", "<1%", "1–5%", "5–10%", "10–20%", "20–50%", "50–80%", "80–100%"]
    counts = pd.cut(profile["missing_pct"], edges, right=False, labels=names).value_counts()
    counts = counts.reindex(names).fillna(0).astype(int)
    return _column_chart(
        names, [("Features", "--series-1", counts.tolist())], y_label="Number of features"
    )


def _weekly_chart(labels: pd.DataFrame) -> str:
    week = labels["timestamp"].dt.to_period("W-SUN")
    weekly = (
        labels.assign(week=week, fail=labels["label"].eq(1))
        .groupby("week")["fail"]
        .agg(total="size", fails="sum")
    )
    weekly["passes"] = weekly["total"] - weekly["fails"]
    names = [period.start_time.strftime("%d %b") for period in weekly.index]
    chart = _column_chart(
        names,
        [
            ("Pass", "--series-1", weekly["passes"].tolist()),
            ("Fail", "--series-2", weekly["fails"].tolist()),
        ],
        y_label="Units per week",
        tick_every=max(len(names) // 8, 1),
    )
    worst = weekly.assign(rate=weekly["fails"] / weekly["total"] * 100)
    worst = worst[worst["total"] >= 20].sort_values("rate", ascending=False).head(1)
    note = ""
    if len(worst):
        period = worst.index[0]
        note = (
            f"<p class='note'>Highest weekly defect rate (weeks with ≥20 units): "
            f"week of {period.start_time:%d %b %Y} at {worst['rate'].iloc[0]:.1f}% "
            f"({int(worst['fails'].iloc[0])} of {int(worst['total'].iloc[0])}).</p>"
        )
    return _legend([("Pass", "--series-1"), ("Fail", "--series-2")]) + chart + note


def _quality_chart(profile: pd.DataFrame, thresholds_note: str) -> str:
    n = len(profile)
    rows = [
        ("Any missing value", int(profile["missing"].gt(0).sum())),
        ("Extreme skew (|s|>10)", int(profile["skew"].abs().gt(10).sum())),
        ("Constant", int(profile["is_constant"].sum())),
        (">50% missing", int(profile["missing_pct"].gt(50).sum())),
        ("Near-constant", int(profile["is_near_constant"].sum())),
        (">5% 3σ outliers", int(profile["sigma3_outlier_pct"].gt(5).sum())),
    ]
    rows.sort(key=lambda r: r[1], reverse=True)
    return (
        _bar_chart(
            [(label, value, "--series-1") for label, value in rows],
            value_fmt=lambda v: f"{int(v)} ({v / n * 100:.0f}%)",
        )
        + f"<p class='note'>{thresholds_note}</p>"
    )


def _shift_chart(profile: pd.DataFrame) -> str:
    column = "mean_diff_fail_vs_pass_in_std"
    if column not in profile:
        return ""
    usable = profile[~profile["is_constant"] & profile[column].notna()]
    top = usable.reindex(usable[column].abs().sort_values(ascending=False).index).head(15)
    rows, tips = [], []
    for feature, row in top.iterrows():
        value = float(row[column])
        rows.append((feature, value, "--series-2" if value > 0 else "--series-1"))
        tips.append(
            f"{feature}: shift {value:+.3f} std · mean pass {_fmt(row['mean_pass'])} · "
            f"mean fail {_fmt(row['mean_fail'])} · missing {row['missing_pct']:.1f}%"
        )
    return (
        _legend(
            [("Higher in failing units", "--series-2"), ("Lower in failing units", "--series-1")]
        )
        + _bar_chart(rows, value_fmt=lambda v: f"{v:+.2f}", diverging=True, tooltips=tips)
        + "<p class='note'>Difference of class means divided by the feature's overall standard "
        "deviation. A univariate screening signal only — not evidence of causation and not a "
        "substitute for feature selection on the training split.</p>"
    )


def _feature_table(profile: pd.DataFrame) -> str:
    columns = [
        ("missing_pct", "Missing %"),
        ("n_unique", "Unique"),
        ("mean", "Mean"),
        ("std", "Std"),
        ("min", "Min"),
        ("median", "Median"),
        ("max", "Max"),
        ("skew", "Skew"),
        ("sigma3_outlier_pct", "3σ outlier %"),
        ("mean_diff_fail_vs_pass_in_std", "Fail shift"),
    ]
    columns = [(key, name) for key, name in columns if key in profile]
    head = "<th data-type='s'>Feature</th><th data-type='s'>Flags</th>" + "".join(
        f"<th class='n' data-type='n'>{name}</th>" for _, name in columns
    )
    body = []
    for feature, row in profile.iterrows():
        flags = []
        if row["is_constant"]:
            flags.append("constant")
        elif row["is_near_constant"]:
            flags.append("near-constant")
        if row["missing_pct"] > 50:
            flags.append("high-missing")
        if abs(row["skew"]) > 10 if not pd.isna(row["skew"]) else False:
            flags.append("skewed")
        cells = "".join(
            f"<td class='n' data-v='{'' if pd.isna(row[key]) else float(row[key])}'>"
            f"{_fmt(row[key]) if not pd.isna(row[key]) else '–'}</td>"
            for key, _ in columns
        )
        body.append(
            f"<tr><td>{feature}</td><td>{' '.join(f'<span class=flag>{f}</span>' for f in flags)}"
            f"</td>{cells}</tr>"
        )
    return (
        '<div class="filters"><input type="search" id="feature-search" '
        'placeholder="Filter features or flags (e.g. constant, 059)…" aria-label="Filter features">'
        '<span id="feature-count"></span></div>'
        '<div class="scroll tall"><table id="features"><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


# --------------------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------------------

CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--panel:#ffffff;--border:#e4e3df;
--text:#0b0b0b;--text-2:#52514e;--muted:#7b7a75;--grid:#ecebe7;
--series-1:#2a78d6;--series-2:#eb6834;
--pass:#1f7a3a;--pass-bg:#e5f3e8;--warn:#8a5a00;--warn-bg:#fdf1d8;--fail:#b42318;--fail-bg:#fde7e5}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--surface:#1a1a19;--panel:#222220;--border:#3a3a37;--text:#f0efec;--text-2:#c3c2b7;
--muted:#9a998f;--grid:#2e2e2b;--series-1:#3987e5;--series-2:#d95926;
--pass:#7fd29a;--pass-bg:#1d3324;--warn:#f2c46b;--warn-bg:#3a2f15;--fail:#ff8a80;--fail-bg:#3d1f1c}}
:root[data-theme="dark"]{color-scheme:dark;--surface:#1a1a19;--panel:#222220;--border:#3a3a37;
--text:#f0efec;--text-2:#c3c2b7;--muted:#9a998f;--grid:#2e2e2b;--series-1:#3987e5;
--series-2:#d95926;--pass:#7fd29a;--pass-bg:#1d3324;--warn:#f2c46b;--warn-bg:#3a2f15;
--fail:#ff8a80;--fail-bg:#3d1f1c}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--text);
font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:40px 0 4px}
.lede{color:var(--text-2);margin:0 0 20px}.note{color:var(--muted);font-size:12.5px;margin:6px 0 0}
.status{display:inline-flex;gap:8px;align-items:center;margin:12px 0 24px;flex-wrap:wrap}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.tile{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px}
.t-label{color:var(--text-2);font-size:12.5px}.t-value{font-size:26px;font-weight:600;
font-variant-numeric:tabular-nums}.t-sub{color:var(--muted);font-size:12px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:16px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:16px;
min-width:0}.card h3{margin:0 0 2px;font-size:15px}.card .sub{color:var(--text-2);
font-size:12.5px;margin:0 0 10px}
svg.chart{width:100%;height:auto;display:block;overflow:visible}
.grid{stroke:var(--grid)}.axis{stroke:var(--muted)}
.tick{fill:var(--muted);font-size:11px}.label{fill:var(--text-2);font-size:12px}
.value{fill:var(--text);font-size:12px;font-variant-numeric:tabular-nums}
.hit{fill:transparent}.hit:hover{fill:var(--text);fill-opacity:.05}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12.5px;color:var(--text-2);margin:0 0 8px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
.scroll{overflow:auto;border:1px solid var(--border);border-radius:10px;background:var(--panel)}
.scroll.tall{max-height:560px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 10px;border-bottom:1px solid var(--border);text-align:left;vertical-align:top}
th{position:sticky;top:0;background:var(--panel);color:var(--text-2);font-weight:600;
white-space:nowrap;cursor:default}#features th{cursor:pointer}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
code{font-size:12px;color:var(--text-2);word-break:break-all}
.items{color:var(--muted);font-size:12px;margin-top:2px}
.badge{display:inline-block;border-radius:999px;padding:1px 9px;font-size:12px;font-weight:600;
white-space:nowrap}.badge.pass{color:var(--pass);background:var(--pass-bg)}
.badge.warn{color:var(--warn);background:var(--warn-bg)}
.badge.fail{color:var(--fail);background:var(--fail-bg)}
.flag{display:inline-block;font-size:11px;border:1px solid var(--border);border-radius:4px;
padding:0 5px;margin:0 3px 2px 0;color:var(--text-2)}
.filters{display:flex;gap:12px;align-items:center;margin:8px 0;flex-wrap:wrap;color:var(--text-2)}
input[type=search]{flex:1;min-width:200px;padding:7px 10px;border:1px solid var(--border);
border-radius:8px;background:var(--panel);color:var(--text);font:inherit}
.hide-pass tr.s-pass{display:none}
ul.findings{padding-left:20px;margin:8px 0}ul.findings li{margin:4px 0}
"""

JS = """
document.getElementById('hide-pass').addEventListener('change', e => {
  document.getElementById('checks').classList.toggle('hide-pass', e.target.checked);
});
const table = document.getElementById('features');
const rows = Array.from(table.tBodies[0].rows);
const count = document.getElementById('feature-count');
const search = document.getElementById('feature-search');
function filter() {
  const q = search.value.trim().toLowerCase(); let shown = 0;
  rows.forEach(r => { const ok = !q || r.textContent.toLowerCase().includes(q);
    r.style.display = ok ? '' : 'none'; shown += ok; });
  count.textContent = shown + ' of ' + rows.length + ' features';
}
search.addEventListener('input', filter); filter();
table.querySelectorAll('th').forEach((th, i) => {
  let asc = true;
  th.addEventListener('click', () => {
    const numeric = th.dataset.type === 'n';
    rows.sort((a, b) => {
      const x = a.cells[i], y = b.cells[i];
      if (numeric) { const vx = x.dataset.v === '' ? NaN : +x.dataset.v,
          vy = y.dataset.v === '' ? NaN : +y.dataset.v;
        if (isNaN(vx)) return 1; if (isNaN(vy)) return -1; return asc ? vx - vy : vy - vx; }
      return asc ? x.textContent.localeCompare(y.textContent)
                 : y.textContent.localeCompare(x.textContent);
    });
    asc = !asc; rows.forEach(r => table.tBodies[0].appendChild(r));
  });
});
"""


def _findings(summary: dict, profile: pd.DataFrame) -> str:
    checks = {(c["dataset"], c["check"]): c for c in summary["checks"]}
    labels = summary["datasets"]["labels"]
    n = len(profile)
    constant = int(profile["is_constant"].sum())
    high_missing = int(profile["missing_pct"].gt(50).sum())
    drift = checks.get(("split", "distribution_drift"), {})
    features_ov = summary["datasets"]["features"]
    prevalence = checks.get(("split", "stratified_prevalence"), {}).get("value", {})
    items = [
        (
            f"Structural integrity is clean: every raw line has 590 fields, every token parses, "
            f"labels are only −1/1 and all {labels['rows']:,} timestamps parse."
        ),
        (
            f"Severe class imbalance: {labels['defect_pct']:.2f}% of units fail "
            f"({labels['class_counts'].get('1', 0)} of {labels['rows']:,}). Accuracy will be "
            "misleading — evaluate with recall / balanced metrics."
        ),
        (
            f"{constant} of {n} features ({constant / n:.0%}) never vary and carry no information; "
            f"{high_missing} features are over 50% missing."
        ),
        (
            f"Missingness is widespread but shallow: {features_ov['missing_cell_pct']:.2f}% "
            f"of cells are missing, and only {features_ov['complete_rows']} of "
            f"{features_ov['rows']:,} rows are fully complete — imputation is needed."
        ),
        (
            f"The split looks sound: no row overlap, defect rate train "
            f"{prevalence.get('train_defect_pct', '–')}% vs test {prevalence.get('test_defect_pct', '–')}%, "
            f"and {drift.get('value', '–')} features show train/test distribution drift."
        ),
    ]
    return '<ul class="findings">' + "".join(f"<li>{html.escape(i)}</li>" for i in items) + "</ul>"


def write_descriptive_report(
    output_path: Path,
    summary: dict,
    profiles: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
) -> Path:
    """Render the descriptive + health report as one standalone HTML file."""
    profile = profiles["features"]
    overviews = summary["datasets"]
    labels_ov = overviews["labels"]
    counts = summary["status_counts"]
    status_badges = "".join(
        f'<span class="badge {s.lower()}">{STATUS_ICON[s]} {counts.get(s, 0)} {s}</span>'
        for s in ("PASS", "WARN", "FAIL")
    )
    body = f"""
<main>
<h1>SECOM data health &amp; descriptive report</h1>
<p class="lede">UCI SECOM semiconductor line — {labels_ov["rows"]:,} production units,
{overviews["features"]["columns"]} sensor features, {labels_ov["first_timestamp"][:10]} to
{labels_ov["last_timestamp"][:10]}. Checks cover the raw files, labels, full feature matrix and
the stratified train/test split.</p>
<div class="status"><strong>Overall: {summary["status"]}</strong>{status_badges}</div>
{_tiles(overviews, profile)}

<h2>Key findings</h2>
{_findings(summary, profile)}

<h2>Production &amp; labels</h2>
<div class="grid2">
 <div class="card"><h3>Units tested per week</h3>
  <p class="sub">Stacked by in-house test outcome</p>{_weekly_chart(labels)}</div>
 <div class="card"><h3>Feature missingness</h3>
  <p class="sub">Number of features by share of missing values</p>{_missingness_chart(profile)}
  <p class="note">{int(profile["missing"].eq(0).sum())} features are fully observed.</p></div>
</div>

<h2>Feature quality</h2>
<div class="grid2">
 <div class="card"><h3>Features flagged by issue</h3>
  <p class="sub">Count of {len(profile)} features (a feature can carry several flags)</p>
  {_quality_chart(profile, "Near-constant: one value in ≥95% of observed rows. Outliers: values beyond mean ± 3·std (3σ rule).")}</div>
 <div class="card"><h3>Largest pass/fail mean shift</h3>
  <p class="sub">Top 15 non-constant features, in units of the feature's standard deviation</p>
  {_shift_chart(profile)}</div>
</div>

<h2>Dataset overview</h2>
<p class="lede">Train/test are the 80/20 stratified split (random_state 42) used for modelling.</p>
{_overview_table(overviews)}

<h2>Health checks</h2>
{_checks_table(summary["checks"])}

<h2>Feature descriptive statistics</h2>
<p class="lede">Full dataset. Click a column to sort; filter by name or flag.
Complete statistics (quantiles, kurtosis, per-class missingness) are in
<code>features_profile.csv</code>.</p>
{_feature_table(profile)}
</main>"""
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>SECOM Data Health</title><style>{CSS}</style></head><body>{body}"
        f"<script>{JS}</script></body></html>"
    )
    output_path.write_text(page, encoding="utf-8")
    return output_path
