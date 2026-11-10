from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_section2_visualizations(X, y):
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    class_counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Pass (-1)", "Fail (1)"], [class_counts[-1], class_counts[1]], color=["steelblue", "indianred"], edgecolor="black")
    ax.set_title("Wafer Class Distribution (Pass vs. Fail)")
    ax.set_ylabel("Number of Wafers")
    fig.tight_layout()
    fig.savefig("results/figures/class_distribution.png", dpi=150)
    plt.close(fig)

    sample_features = ["sensor_0", "sensor_1", "sensor_2", "sensor_3"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, col in zip(axes.flat, sample_features):
        ax.hist(X[col].dropna(), bins=30, edgecolor="black")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel("Sensor Reading")
        ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig("results/figures/sample_sensor_distributions.png", dpi=150)
    plt.close(fig)

    missing_by_feature = (X.isna().mean() * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(missing_by_feature)), missing_by_feature.values, width=1.0)
    ax.set_title("Missing Value Percentage by Sensor (Sorted)")
    ax.set_xlabel("Sensors (Sorted by % Missing)")
    ax.set_ylabel("Missing Values (%)")
    fig.tight_layout()
    fig.savefig("results/figures/initial_missingness_by_feature.png", dpi=150)
    plt.close(fig)


def plot_low_vola(non_zero_var):
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(np.log10(non_zero_var), bins=30, edgecolor="black")
    axes[0].axvline(np.log10(0.01), color="red", linestyle="--", label="var = 0.01")
    axes[0].set_title("Histogram of Sensor Variances (log10 scale)")
    axes[0].set_xlabel("log10(Variance)")
    axes[0].set_ylabel("Number of Sensors")
    axes[0].legend()

    var_thresholds = np.logspace(-6, 0, 25)
    removed_if_cut = [(non_zero_var < t).sum() for t in var_thresholds]
    axes[1].plot(var_thresholds, removed_if_cut, marker="o")
    axes[1].set_xscale("log")
    axes[1].set_title("Pareto Chart: Sensors Removed vs. Variance Threshold")
    axes[1].set_xlabel("Variance Threshold (log scale)")
    axes[1].set_ylabel("Number of Sensors Removed")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("results/figures/low_vola_analysis.png", dpi=150)
    plt.close(fig)


def plot_missingness(missing_pct):
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(missing_pct, bins=20, edgecolor="black")
    axes[0].axvline(60, color="red", linestyle="--", label="60% Threshold")
    axes[0].set_title("Histogram of Missing Values Across Sensors")
    axes[0].set_xlabel("Missing Values (%)")
    axes[0].set_ylabel("Number of Sensors")
    axes[0].legend()

    thresholds = np.arange(0, 101, 5)
    removed_counts = [(missing_pct > t).sum() for t in thresholds]
    axes[1].plot(thresholds, removed_counts, marker="o")
    axes[1].axvline(60, color="red", linestyle="--", label="60% Threshold")
    axes[1].set_title("Pareto Chart: Sensors Removed vs. Missing Threshold")
    axes[1].set_xlabel("Missingness Threshold (%)")
    axes[1].set_ylabel("Number of Sensors Removed")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig("results/figures/missingness_analysis.png", dpi=150)
    plt.close(fig)




