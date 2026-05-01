"""
figures/fig_f5_walkforward_gantt.py
Walk-Forward Validation Timeline (Figure F5)
Generates figures/fig_f5_walkforward_gantt.pdf and .png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter, YearLocator, MonthLocator

# ── Walk-forward parameters (from config) ─────────────────────────────────────
TRAIN_DAYS  = 252
VAL_DAYS    = 63
TEST_DAYS   = 63
STRIDE      = 63          # = TEST_DAYS
START_DATE  = "2019-01-02"
END_DATE    = "2024-12-31"

FOLD_WINDOW = TRAIN_DAYS + VAL_DAYS + TEST_DAYS  # 378 trading days

# ── Build trading-day index ────────────────────────────────────────────────────
trading_days = pd.bdate_range(start=START_DATE, end=END_DATE)
N = len(trading_days)

# ── Compute folds ──────────────────────────────────────────────────────────────
folds = []
start_idx = 0
while start_idx + FOLD_WINDOW <= N:
    train_start = start_idx
    train_end   = start_idx + TRAIN_DAYS
    val_end     = train_end + VAL_DAYS
    test_end    = val_end + TEST_DAYS

    folds.append({
        "train_start": trading_days[train_start],
        "train_end":   trading_days[train_end - 1],
        "val_start":   trading_days[train_end],
        "val_end":     trading_days[val_end - 1],
        "test_start":  trading_days[val_end],
        "test_end":    trading_days[test_end - 1],
    })
    start_idx += STRIDE

n_folds = len(folds)

# ── Diagnostics ────────────────────────────────────────────────────────────────
print(f"Number of folds computed : {n_folds}")
last = folds[-1]
print(f"Last fold  train : {last['train_start'].date()} → {last['train_end'].date()}")
print(f"Last fold  val   : {last['val_start'].date()}   → {last['val_end'].date()}")
print(f"Last fold  test  : {last['test_start'].date()}  → {last['test_end'].date()}")

# ── Colours ────────────────────────────────────────────────────────────────────
CLR_TRAIN = "#1a3a5c"
CLR_VAL   = "#4a7fb5"
CLR_TEST  = "#a8c8e8"

# ── Font ───────────────────────────────────────────────────────────────────────
try:
    plt.rcParams.update({
        "font.family": "serif",
        "text.usetex": False,   # keep False so no LaTeX installation needed
    })
except Exception:
    pass

# ── Figure ─────────────────────────────────────────────────────────────────────
fig_h = 0.45 * n_folds + 1.5
fig, ax = plt.subplots(figsize=(12, fig_h))

bar_height = 0.6

for i, fold in enumerate(folds):
    # y position: Fold 1 at top → row 0 = highest y
    y = n_folds - 1 - i

    def bar(start, end, color):
        duration = (end - start).days
        ax.barh(
            y, duration,
            left=start,
            height=bar_height,
            color=color,
            edgecolor="white",
            linewidth=0.4,
        )

    bar(fold["train_start"], fold["train_end"], CLR_TRAIN)
    bar(fold["val_start"],   fold["val_end"],   CLR_VAL)
    bar(fold["test_start"],  fold["test_end"],  CLR_TEST)

# ── Axes formatting ────────────────────────────────────────────────────────────
ax.set_yticks(range(n_folds))
ax.set_yticklabels([f"Fold {n_folds - i}" for i in range(n_folds)], fontsize=8)
ax.set_ylim(-0.6, n_folds - 0.4)

ax.xaxis_date()
ax.xaxis.set_major_locator(YearLocator())
ax.xaxis.set_minor_locator(MonthLocator(bymonth=[4, 7, 10]))
ax.xaxis.set_major_formatter(DateFormatter("%Y"))
ax.set_xlim(
    pd.Timestamp(START_DATE) - pd.offsets.Day(10),
    pd.Timestamp("2025-01-15"),
)

ax.set_xlabel("Date", fontsize=10)
ax.set_title("Walk-Forward Validation Timeline", fontsize=12, fontweight="bold", pad=10)

# Subtle vertical grid
ax.xaxis.grid(True, which="major", alpha=0.3, linestyle="--", linewidth=0.6)
ax.set_axisbelow(True)

# Despine
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=CLR_TRAIN, label=f"Training ({TRAIN_DAYS} d)"),
    mpatches.Patch(color=CLR_VAL,   label=f"Validation ({VAL_DAYS} d)"),
    mpatches.Patch(color=CLR_TEST,  label=f"Test ({TEST_DAYS} d)"),
]
ax.legend(
    handles=legend_patches,
    loc="lower right",
    fontsize=9,
    framealpha=0.85,
    edgecolor="0.7",
)

plt.tight_layout()

# ── Save ───────────────────────────────────────────────────────────────────────
out_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)

pdf_path = os.path.join(out_dir, "fig_f5_walkforward_gantt.pdf")
png_path = os.path.join(out_dir, "fig_f5_walkforward_gantt.png")

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")

print(f"Saved → {pdf_path}")
print(f"Saved → {png_path}")

plt.close(fig)
