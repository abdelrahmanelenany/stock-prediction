# Figure F6 — Feature Correlation Heatmap
# Placed in: Section 3.5.4 (Feature Assignment by Model Family)
# Purpose: Justifies why highly correlated features are dropped from LR
# Output: outputs/figures/F6_feature_correlation_heatmap.png

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

DATA_PATH = "data/processed/features_large_cap.csv"
OUT_FIG   = "outputs/figures/F6_feature_correlation_heatmap.png"
OUT_CSV   = "outputs/figures/F6_correlation_matrix.csv"

LAYER1 = [
    "Return_1d", "Return_5d", "Return_21d",
    "RSI_14", "MACD", "ATR_14",
    "BB_PctB", "RealVol_20d", "Volume_Ratio", "SectorRelReturn",
]

LAYER2 = [
    "Market_Return_1d", "Market_Return_5d", "Market_Return_21d",
    "Market_Vol_20d", "Market_Vol_60d",
    "RelToMarket_1d", "RelToMarket_5d", "RelToMarket_21d",
    "Beta_60d",
]

ALL_FEATURES = LAYER1 + LAYER2
HIGH_CORR_THRESHOLD = 0.70


def main():
    # --- load data ---
    if not os.path.exists(DATA_PATH):
        print("ERROR: Run main.py first to generate features_large_cap.csv")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)

    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        print(f"ERROR: The following feature columns are missing from the CSV: {missing}")
        sys.exit(1)

    df = df[ALL_FEATURES].dropna()
    print(f"Rows used in correlation computation: {len(df):,}")

    # --- correlation matrix ---
    corr = df.corr(method="pearson")
    corr.to_csv(OUT_CSV, float_format="%.4f")
    print(f"Saved correlation matrix to {OUT_CSV}")

    # --- diagnostics: top-5 pairs and high-corr pairs ---
    mask_upper = np.triu(np.ones(corr.shape, dtype=bool))
    corr_flat = corr.where(~mask_upper).stack()
    corr_flat.index.names = ["Feature A", "Feature B"]
    corr_abs = corr_flat.abs().sort_values(ascending=False)

    print("\nTop-5 pairs by |r| (lower triangle, excl. diagonal):")
    for (fa, fb), val in corr_abs.head(5).items():
        print(f"  {fa:25s} × {fb:25s}  r = {corr_flat.loc[(fa, fb)]:+.4f}")

    high_pairs = corr_abs[corr_abs > HIGH_CORR_THRESHOLD]
    print(f"\nPairs with |r| > {HIGH_CORR_THRESHOLD}:")
    if high_pairs.empty:
        print("  None")
    else:
        for (fa, fb), val in high_pairs.items():
            print(f"  {fa:25s} × {fb:25s}  r = {corr_flat.loc[(fa, fb)]:+.4f}")

    # --- build figure ---
    n = len(ALL_FEATURES)
    mask_upper_tri = np.triu(np.ones((n, n), dtype=bool))  # mask upper triangle

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        corr,
        mask=mask_upper_tri,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".1f",
        annot_kws={"size": 7},
        linewidths=0.4,
        linecolor="white",
        square=True,
        cbar_kws={"label": "Pearson r", "shrink": 0.8},
        ax=ax,
    )

    ax.set_title(
        "Figure F6 — Feature Correlation Matrix: Layer 1 and Layer 2",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9, rotation=0)

    # --- highlight |r| > threshold in lower triangle (excl. diagonal) ---
    for i in range(n):
        for j in range(i):  # lower triangle, j < i
            val = corr.iloc[i, j]
            if abs(val) > HIGH_CORR_THRESHOLD:
                # seaborn places cell (i, j) at axes coords (j, i) with origin top-left
                rect = mpatches.FancyBboxPatch(
                    (j, i),
                    1, 1,
                    boxstyle="square,pad=0",
                    linewidth=1.5,
                    edgecolor="red",
                    facecolor="none",
                    transform=ax.transData,
                    clip_on=False,
                )
                ax.add_patch(rect)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure to {OUT_FIG}")


if __name__ == "__main__":
    main()
