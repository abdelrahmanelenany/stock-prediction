"""Generate Figure F9 — AMH Regime Comparison (Net Sharpe by Universe)."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT_DIR = os.path.dirname(__file__)
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

MODELS = ["LR", "RF", "XGBoost", "LSTM", "TCN", "Ensemble"]


def _load_sharpe(universe: str) -> list:
    path = os.path.join(REPORTS_DIR, f"{universe}_table_T5_net_returns_5bps.csv")
    df = pd.read_csv(path)
    df["Model"] = df["Model"].str.strip()
    sharpe_map = df.set_index("Model")["Sharpe Ratio"].to_dict()
    return [float(sharpe_map[m]) for m in MODELS]


LARGE_CAP = _load_sharpe("large_cap")
SMALL_CAP = _load_sharpe("small_cap")

COLOR_LARGE = "#1f77b4"
COLOR_SMALL = "#ff7f0e"

x = np.arange(len(MODELS))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))

bars_large = ax.bar(x - width / 2, LARGE_CAP, width,
                    label="Large-Cap (mean-reversion)",
                    color=COLOR_LARGE, edgecolor="white", linewidth=0.6)
bars_small = ax.bar(x + width / 2, SMALL_CAP, width,
                    label="Small-Cap (momentum)",
                    color=COLOR_SMALL, edgecolor="white", linewidth=0.6)

# Value labels on bars
for bar in bars_large:
    h = bar.get_height()
    va = "bottom" if h >= 0 else "top"
    offset = 0.015 if h >= 0 else -0.015
    ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
            f"{h:.3f}", ha="center", va=va, fontsize=7.5, color="black")

for bar in bars_small:
    h = bar.get_height()
    va = "bottom" if h >= 0 else "top"
    offset = 0.015 if h >= 0 else -0.015
    ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
            f"{h:.3f}", ha="center", va=va, fontsize=7.5, color="black")

ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(MODELS, fontsize=10)
ax.set_ylabel("Net Sharpe Ratio (annualised)", fontsize=10)
ax.set_title(
    "Figure F9 — AMH Regime Comparison: Net Sharpe by Universe\n",
    fontsize=10,
)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax.yaxis.grid(True, color="lightgray", linewidth=0.6)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(fontsize=9, frameon=True)

plt.tight_layout()

pdf_path = os.path.join(OUT_DIR, "F9_amh_regime_comparison.pdf")
png_path = os.path.join(OUT_DIR, "F9_amh_regime_comparison.png")

plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
plt.savefig(png_path, dpi=150, bbox_inches="tight")
print(f"Saved PDF → {pdf_path}")
print(f"Saved PNG → {png_path}")
