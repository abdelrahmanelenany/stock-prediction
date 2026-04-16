"""
generate_feature_importance_plot.py
------------------------------------
Reads reports/{prefix}_feature_importances_avg.csv and produces:
  outputs/figures/feature_importance_comparison.png

Four panels (one per model):
  - LR       : absolute normalised coefficient magnitude
  - RF       : mean decrease in impurity (normalised)
  - XGBoost  : average gain (normalised)
  - LSTM-B   : permutation importance (AUC drop, normalised)

Error bars = ±1 std across walk-forward folds (from per-fold CSV).
Features sorted by mean importance across all models.

Usage:
    python generate_feature_importance_plot.py [--prefix large_cap]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT    = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "reports")
FIGURES = os.path.join(ROOT, "outputs", "figures")
os.makedirs(FIGURES, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", default="large_cap",
                    help="Report prefix matching the pipeline run (default: large_cap)")
args, _ = parser.parse_known_args()
prefix = args.prefix

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
avg_path     = os.path.join(REPORTS, f"{prefix}_feature_importances_avg.csv")
per_fold_path = os.path.join(REPORTS, f"{prefix}_feature_importances_per_fold.csv")

if not os.path.exists(avg_path):
    print(f"ERROR: {avg_path} not found.")
    print("Run the pipeline first (python main.py) to generate feature importances.")
    sys.exit(1)

fi_avg = pd.read_csv(avg_path)

DISPLAY_NAMES = {
    "Return_1d":       "Return 1d",
    "Return_5d":       "Return 5d",
    "Return_21d":      "Return 21d",
    "RSI_14":          "RSI (14)",
    "BB_PctB":         "BB %B",
    "RealVol_20d":     "Realised Vol 20d",
    "Volume_Ratio":    "Volume Ratio",
    "SectorRelReturn": "Sector Rel. Return",
}
fi_avg["Label"] = fi_avg["Feature"].map(DISPLAY_NAMES).fillna(fi_avg["Feature"])

# Sort by mean importance across all four models for a stable visual order
model_cols = ["LR_coef", "RF_importance", "XGB_gain", "LSTM_B_perm"]
present_cols = [c for c in model_cols if c in fi_avg.columns]
fi_avg["mean_imp"] = fi_avg[present_cols].mean(axis=1)
fi_avg = fi_avg.sort_values("mean_imp", ascending=True).reset_index(drop=True)

labels = fi_avg["Label"].tolist()
n_feat = len(labels)
y_pos  = np.arange(n_feat)

# ---------------------------------------------------------------------------
# Per-fold std for error bars
# ---------------------------------------------------------------------------
err = {col: None for col in model_cols}
if os.path.exists(per_fold_path):
    pf = pd.read_csv(per_fold_path)
    pf_std = (
        pf.groupby("Feature")[[c for c in model_cols if c in pf.columns]]
        .std()
        .reset_index()
        .set_index("Feature")
        .reindex(fi_avg["Feature"])
        .reset_index()
    )
    for col in model_cols:
        if col in pf_std.columns:
            err[col] = pf_std[col].fillna(0).values

# ---------------------------------------------------------------------------
# Plot — 4 panels
# ---------------------------------------------------------------------------
MODEL_CFG = [
    ("LR_coef",       "Logistic Regression\n(|coefficient|)",   "#4393c3"),
    ("RF_importance", "Random Forest\n(mean ↓ impurity)",        "#74c476"),
    ("XGB_gain",      "XGBoost\n(avg. gain)",                    "#fd8d3c"),
    ("LSTM_B_perm",   "LSTM-B\n(permutation importance)",        "#9e9ac8"),
]

fig, axes = plt.subplots(1, 4, figsize=(18, max(5, n_feat * 0.55)), sharey=True)

for ax, (col, title, color) in zip(axes, MODEL_CFG):
    if col not in fi_avg.columns:
        ax.set_visible(False)
        continue

    vals = fi_avg[col].fillna(0).values
    bars = ax.barh(y_pos, vals, height=0.65,
                   color=color, alpha=0.85, edgecolor="white", linewidth=0.4)

    if err[col] is not None:
        ax.errorbar(vals, y_pos, xerr=err[col],
                    fmt="none", ecolor="black", elinewidth=0.8, capsize=3)

    ax.set_xlim(left=0)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Normalised importance", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)

    for bar, v in zip(bars, vals):
        ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", ha="left", fontsize=7.5)

axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(labels, fontsize=9)

has_err = any(v is not None for v in err.values())
err_note = " — error bars = ±1 std across folds" if has_err else ""
fig.suptitle(
    f"Feature Importance — All Models — {prefix.replace('_', '-').title()} Universe{err_note}",
    fontsize=11, fontweight="bold", y=1.01,
)
fig.tight_layout()

out_path = os.path.join(FIGURES, "feature_importance_comparison.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out_path}")
