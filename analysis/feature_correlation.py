"""
analysis/feature_correlation.py
Bhandari §4.4 — Correlation heatmap and redundant feature removal.

Run ONCE on Fold 1 training data. Output:
  outputs/figures/feature_correlation_heatmap.png
  outputs/feature_selection_log.txt

Usage:
    python analysis/feature_correlation.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


def compute_and_plot_heatmap(df_train: pd.DataFrame, feature_cols: list,
                              output_dir: str = "outputs/figures",
                              corr_threshold: float = None) -> list:
    """
    1. Compute Pearson correlation matrix on df_train[feature_cols].
    2. Plot and save a heatmap (Figure 4 equivalent from the paper).
    3. Identify and remove redundant features (|r| > corr_threshold).
    4. Return the reduced feature list.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training data with feature columns.
    feature_cols : list
        List of feature column names to analyze.
    output_dir : str
        Directory to save heatmap figure.
    corr_threshold : float
        Correlation threshold for dropping features. If None, uses config value.

    Returns
    -------
    list
        Reduced feature list after removing highly correlated features.
    """
    if corr_threshold is None:
        corr_threshold = config.FEATURE_CORR_THRESHOLD

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Ensure all feature columns exist
    available_cols = [c for c in feature_cols if c in df_train.columns]
    if len(available_cols) < len(feature_cols):
        missing = set(feature_cols) - set(available_cols)
        print(f"Warning: Missing columns {missing}, using available: {available_cols}")
    feature_cols = available_cols

    corr = df_train[feature_cols].corr(method="pearson")

    # ── Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, len(feature_cols) * 1.0),
                                    max(8, len(feature_cols) * 0.8)))
    mask = np.triu(np.ones_like(corr, dtype=bool))   # upper triangle mask
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 10},
    )
    ax.set_title(
        "Pearson Correlation Heatmap — Training Features\n"
        "(Bhandari et al. §4.4 — Fold 1 Training Data)",
        fontsize=13
    )
    plt.tight_layout()
    fig.savefig(f"{output_dir}/feature_correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"[Feature Selection] Heatmap saved -> {output_dir}/feature_correlation_heatmap.png")

    # ── Remove redundant features (|r| > threshold) ───────────────────
    # Greedy: traverse upper triangle; if |r_ij| > threshold, drop column j.
    to_drop = set()
    cols = list(corr.columns)
    for i in range(len(cols)):
        if cols[i] in to_drop:
            continue
        for j in range(i + 1, len(cols)):
            if cols[j] in to_drop:
                continue
            if abs(corr.iloc[i, j]) > corr_threshold:
                to_drop.add(cols[j])
                print(f"  Drop '{cols[j]}' — |r| = {corr.iloc[i, j]:.3f} "
                      f"with '{cols[i]}'")

    reduced = [c for c in feature_cols if c not in to_drop]

    # ── Log ───────────────────────────────────────────────────────────
    log_path = "outputs/feature_selection_log.txt"
    with open(log_path, "w") as f:
        f.write(f"Correlation threshold: {corr_threshold}\n")
        f.write(f"Original features ({len(feature_cols)}): {feature_cols}\n\n")
        f.write(f"Dropped features ({len(to_drop)}): {sorted(to_drop)}\n\n")
        f.write(f"Retained features ({len(reduced)}): {reduced}\n")
    print(f"[Feature Selection] {len(to_drop)} features dropped, "
          f"{len(reduced)} retained. Log -> {log_path}")

    # ── Print correlation matrix for thesis table ─────────────────────
    print("\n" + "=" * 60)
    print("CORRELATION MATRIX (for thesis Table)")
    print("=" * 60)
    print(corr.round(3).to_string())

    return reduced


def print_config_snippet(reduced_features: list):
    """Print Python code snippet to paste into config.py."""
    print("\n" + "=" * 60)
    print("Copy this list into config.py -> FEATURE_COLS_AFTER_SELECTION:")
    print("=" * 60)
    print(f"FEATURE_COLS_AFTER_SELECTION = {reduced_features}")


if __name__ == "__main__":
    # Load features data
    features_path = f'data/processed/features_{config.UNIVERSE_MODE}.csv'

    if not os.path.exists(features_path):
        print(f"Error: {features_path} not found.")
        print("Run the feature engineering pipeline first:")
        print("  python pipeline/features.py")
        sys.exit(1)

    df = pd.read_csv(features_path, parse_dates=['Date'])

    # Use first 500 trading days as a proxy for Fold 1 training data
    dates_sorted = sorted(df['Date'].unique())
    fold1_train_days = config.TRAIN_DAYS
    if len(dates_sorted) < fold1_train_days:
        print(f"Warning: Only {len(dates_sorted)} dates available, using all")
        fold1_dates = dates_sorted
    else:
        fold1_dates = dates_sorted[:fold1_train_days]

    df_fold1_train = df[df['Date'].isin(fold1_dates)]
    print(f"Fold 1 training data: {len(df_fold1_train)} rows, "
          f"{len(fold1_dates)} dates")

    # Run correlation analysis
    reduced_features = compute_and_plot_heatmap(
        df_fold1_train,
        config.ALL_FEATURE_COLS,
        output_dir="outputs/figures"
    )

    print_config_snippet(reduced_features)
