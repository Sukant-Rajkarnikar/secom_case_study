"""Visual comparison of the imputation strategies (written by ``impute.run``).

Each figure answers one question:

1. ``01_distributions.png``      - Do the filled values look like the real ones?
                                   (most-missing column of the four largest missing blocks)
2. ``02_relationships.png``      - Do the fills respect how sensors move together?
                                   (most-missing column vs its most correlated, observed partner)
3. ``03_masked_true_vs_imputed.png`` - When we hide values we know, how close is the fill?
4. ``04_variance_kept.png``      - Which columns lose spread, per strategy?
5. ``05_error_by_column.png``    - Reconstruction error per column vs how much it is missing.

Standalone (re-uses the saved datasets, no refitting)::

    python -m secom_analysis.impute_viz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .impute import (
    ID_COLUMN,
    MATERIAL_MISSING_PCT,
    STRATEGIES,
    STRATEGY_BY_NAME,
    TARGET_COLUMN,
    load_prepared,
)

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#7b7a75", "#ecebe7"
OBSERVED_DOT = "#9ec5f4"  # sequential blue step 200: context, recedes behind the fills
# Sequential blue, light (variance kept) -> dark (variance lost)
SHRINK_CMAP = LinearSegmentedColormap.from_list(
    "shrink", ["#0d366b", "#256abf", "#6da7ec", "#cde2fb", "#f7f9fc"]
)


def _style(axis: plt.Axes, grid: str = "y") -> None:
    axis.spines[["top", "right"]].set_visible(False)
    if grid:
        axis.grid(axis=grid, color=GRID, lw=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors="#52514e", labelsize=8)


def _label(name: str) -> str:
    return STRATEGY_BY_NAME[name].label


def missing_blocks(features: pd.DataFrame) -> list[list[str]]:
    """Groups of columns with identical missing masks, largest missing share first."""
    mask = features.isna()
    blocks: dict[bytes, list[str]] = {}
    for column in features.columns[mask.any()]:
        blocks.setdefault(mask[column].to_numpy().tobytes(), []).append(column)
    return sorted(blocks.values(), key=lambda cols: -mask[cols[0]].mean())


# ---------------------------------------------------------------- figure 1


def save_distributions(
    features: pd.DataFrame, datasets: dict[str, pd.DataFrame], output_path: Path, n: int = 4
) -> None:
    """Rows = the most-missing column of each of the ``n`` largest blocks; columns = strategy."""
    columns = [block[0] for block in missing_blocks(features)[:n]]
    names = list(datasets)
    figure, axes = plt.subplots(
        len(columns), len(names), figsize=(3.2 * len(names), 2.3 * len(columns)), squeeze=False
    )
    for r, column in enumerate(columns):
        observed = features[column].dropna()
        missing = features[column].isna().to_numpy()
        bins = np.histogram_bin_edges(observed, bins=30)
        for c, name in enumerate(names):
            axis = axes[r, c]
            filled = datasets[name].loc[missing, column]
            axis.hist(observed, bins=bins, density=True, color=BLUE, label="observed")
            axis.hist(
                filled,
                bins=bins,
                density=True,
                histtype="step",
                lw=2,
                color=ORANGE,
                label="filled",
            )
            axis.set_yticks([])
            _style(axis, grid="")
            axis.spines["left"].set_visible(False)
            if r == 0:
                axis.set_title(_label(name), fontsize=10, loc="left", color=INK)
            if c == 0:
                axis.set_ylabel(
                    f"{column}\n{missing.mean():.0%} missing",
                    fontsize=9,
                    color=INK,
                    rotation=0,
                    ha="right",
                    va="center",
                )
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    figure.suptitle(
        "Do the filled values look like the real ones?  Blue = observed values, orange line = "
        "filled values (both scaled to the same area)",
        fontsize=11,
        x=0.01,
        ha="left",
    )
    figure.text(
        0.01,
        0.005,
        "Median stacks every fill on one value (a spike). Model-based fills spread out but "
        "stay narrower than the real readings.",
        fontsize=9,
        color=MUTED,
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.97))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------- figure 2


def relationship_pair(features: pd.DataFrame) -> tuple[str, str]:
    """Most-missing column, and its most correlated partner that is observed when it is not."""
    target = missing_blocks(features)[0][0]
    missing = features[target].isna()
    candidates = [
        c
        for c in features.columns
        if c != target and features.loc[missing, c].notna().mean() > 0.95 and features[c].std() > 0
    ]
    corr = features[candidates].corrwith(features[target]).abs()
    return target, corr.idxmax()


def save_relationships(
    features: pd.DataFrame, datasets: dict[str, pd.DataFrame], output_path: Path
) -> None:
    target, partner = relationship_pair(features)
    r = features[[target, partner]].corr().iloc[0, 1]
    missing = features[target].isna().to_numpy()
    names = list(datasets)
    figure, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.6), sharey=True)
    for axis, name in zip(axes, names):
        data = datasets[name]
        axis.scatter(
            data.loc[~missing, partner],
            data.loc[~missing, target],
            s=8,
            color=OBSERVED_DOT,
            lw=0,
            label="observed",
        )
        axis.scatter(
            data.loc[missing, partner],
            data.loc[missing, target],
            s=8,
            color=ORANGE,
            lw=0,
            alpha=0.8,
            label="filled",
        )
        axis.set_title(_label(name), fontsize=10, loc="left", color=INK)
        axis.set_xlabel(partner, fontsize=9)
        _style(axis, grid="both")
    axes[0].set_ylabel(target, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, markerscale=2)
    figure.suptitle(
        f"Do the fills follow the sensor relationships?  {target} ({missing.mean():.0%} "
        f"missing) against {partner} (observed; r = {r:+.2f} on complete rows)",
        fontsize=11,
        x=0.01,
        ha="left",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------- figure 3


def save_masked_scatter(
    cells: dict[str, pd.DataFrame], material: list[str], output_path: Path
) -> None:
    names = list(cells)
    figure, axes = plt.subplots(
        1, len(names), figsize=(3.3 * len(names), 3.6), sharex=True, sharey=True
    )
    lim = 4.0
    for axis, name in zip(np.atleast_1d(axes), names):
        frame = cells[name]
        frame = frame[frame["column"].isin(material)]
        x = frame["true_std_units"].clip(-lim, lim)
        y = frame["imputed_std_units"].clip(-lim, lim)
        axis.scatter(x, y, s=5, color=BLUE, alpha=0.35, lw=0)
        axis.plot([-lim, lim], [-lim, lim], color=INK, lw=1, ls="--")
        axis.axhline(0, color=MUTED, lw=0.6)
        error = frame["imputed_std_units"] - frame["true_std_units"]
        nrmse = float(np.sqrt(np.mean(error**2)))
        corr = float(np.corrcoef(frame["true_std_units"], frame["imputed_std_units"])[0, 1])
        axis.text(
            0.03,
            0.97,
            f"NRMSE {nrmse:.2f}\nr(true, filled) {corr:.2f}",
            transform=axis.transAxes,
            va="top",
            fontsize=8.5,
            color=INK,
        )
        axis.set(xlim=(-lim, lim), ylim=(-lim, lim))
        axis.set_title(_label(name), fontsize=10, loc="left", color=INK)
        axis.set_xlabel("true value (std units)", fontsize=9)
        _style(axis, grid="both")
    np.atleast_1d(axes)[0].set_ylabel("filled value (std units)", fontsize=9)
    count = len(cells[names[0]].query("column in @material"))
    figure.suptitle(
        f"Hide {count:,} known values (copying real missing patterns) in the "
        f"{len(material)} columns with ≥{MATERIAL_MISSING_PCT:g}% missing, then fill them. "
        "Dashed line = perfect fill; flat cloud at 0 = just the average",
        fontsize=11,
        x=0.01,
        ha="left",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------- figure 4


def save_variance_heatmap(consistency: dict[str, pd.DataFrame], output_path: Path) -> None:
    names = list(consistency)
    first = consistency[names[0]]
    material = first[first["missing_pct"] >= MATERIAL_MISSING_PCT].sort_values(
        "missing_pct", ascending=False
    )
    ratios = pd.DataFrame(
        {name: consistency[name].loc[material.index, "variance_ratio"] for name in names}
    )
    figure, axis = plt.subplots(figsize=(1.5 * len(names) + 3, 0.3 * len(ratios) + 1.8))
    image = axis.imshow(ratios.to_numpy(), cmap=SHRINK_CMAP, vmin=0.3, vmax=1.0, aspect="auto")
    for (i, j), value in np.ndenumerate(ratios.to_numpy()):
        axis.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=7.5,
            color="white" if value < 0.65 else INK,
        )
    axis.set_xticks(range(len(names)), [_label(n) for n in names], fontsize=8.5)
    axis.xaxis.tick_top()
    axis.set_yticks(
        range(len(ratios)),
        [f"{c}  ({material.loc[c, 'missing_pct']:.0f}% missing)" for c in ratios.index],
        fontsize=7.5,
    )
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    bar = figure.colorbar(image, ax=axis, fraction=0.04, pad=0.02)
    bar.set_label("variance after / before (1 = spread kept)", fontsize=8)
    bar.outline.set_visible(False)
    axis.set_title(
        "Which columns lose spread?  Variance of the imputed column ÷ variance of its observed "
        "values\n(columns with ≥5% missing, most-missing first; darker = more shrinkage)",
        fontsize=10,
        loc="left",
        pad=28,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------- figure 5


def save_error_by_column(
    cells: dict[str, pd.DataFrame], features: pd.DataFrame, output_path: Path
) -> None:
    """Small multiples: per-column NRMSE vs missing % (one panel per strategy)."""
    missing_pct = features.isna().mean() * 100
    names = list(cells)
    figure, axes = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.4), sharey=True)
    for axis, name in zip(np.atleast_1d(axes), names):
        frame = cells[name]
        per_column = (
            frame.assign(sq=(frame["imputed_std_units"] - frame["true_std_units"]) ** 2)
            .groupby("column")
            .agg(nrmse=("sq", lambda v: float(np.sqrt(v.mean()))), cells=("sq", "size"))
        )
        per_column = per_column[per_column["cells"] >= 10]
        x = missing_pct.loc[per_column.index]
        axis.scatter(x, per_column["nrmse"], s=10, color=BLUE, alpha=0.6, lw=0)
        axis.axhline(1.0, color=INK, ls="--", lw=1)
        axis.text(
            0.06,
            1.03,
            "guessing the mean",
            fontsize=7.5,
            color=INK,
            va="bottom",
            transform=axis.get_yaxis_transform(),
        )
        axis.set_xscale("log")
        axis.set_xlim(0.05, 100)
        axis.set_xticks([0.1, 1, 10, 50], ["0.1", "1", "10", "50"])
        axis.minorticks_off()
        axis.set_title(_label(name), fontsize=10, loc="left", color=INK)
        axis.set_xlabel("column missing (%)", fontsize=9)
        _style(axis, grid="both")
        better = int((per_column["nrmse"] < 0.8).sum())
        axis.text(
            0.03,
            0.03,
            f"{better} of {len(per_column)} columns\nwith NRMSE < 0.8",
            transform=axis.transAxes,
            fontsize=8,
            color=INK,
        )
    np.atleast_1d(axes)[0].set_ylabel("NRMSE on hidden cells", fontsize=9)
    figure.suptitle(
        "Reconstruction error per column (columns with ≥10 hidden cells). Points well below the "
        "dashed line = columns where the strategy really learns something",
        fontsize=11,
        x=0.01,
        ha="left",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------- driver


def save_all_figures(
    features: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
    comparison: pd.DataFrame,
    report_out: Path,
) -> list[Path]:
    figures = report_out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    names = list(datasets)
    material = list(features.columns[features.isna().mean() * 100 >= MATERIAL_MISSING_PCT])
    cells = {
        n: pd.read_csv(report_out / f"masked_cells_{n}.csv")
        for n in names
        if (report_out / f"masked_cells_{n}.csv").exists()
    }
    consistency = {n: pd.read_csv(report_out / f"consistency_{n}.csv", index_col=0) for n in names}
    # median_flags fills exactly like median; keep it out of value-level plots
    value_names = [n for n in names if n != "median_flags"]
    paths = [
        figures / "01_distributions.png",
        figures / "02_relationships.png",
        figures / "03_masked_true_vs_imputed.png",
        figures / "04_variance_kept.png",
        figures / "05_error_by_column.png",
    ]
    save_distributions(features, {n: datasets[n] for n in value_names}, paths[0])
    save_relationships(features, {n: datasets[n] for n in value_names}, paths[1])
    if cells:
        value_cells = {n: cells[n] for n in value_names if n in cells}
        save_masked_scatter(value_cells, material, paths[2])
        save_error_by_column(value_cells, features, paths[4])
    save_variance_heatmap({n: consistency[n] for n in value_names}, paths[3])
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw the imputation comparison figures.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/train_prepared.csv"))
    parser.add_argument("--data-out", type=Path, default=Path("data/processed/imputed"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/imputation"))
    args = parser.parse_args(argv)
    features, _, _ = load_prepared(args.input)
    datasets = {}
    for strategy in STRATEGIES:
        path = args.data_out / f"train_{strategy.name}.csv"
        if path.exists():
            datasets[strategy.name] = pd.read_csv(path).drop(columns=[ID_COLUMN, TARGET_COLUMN])
    comparison = pd.read_csv(args.report_out / "comparison.csv", index_col=0)
    for path in save_all_figures(features, datasets, comparison, args.report_out):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
