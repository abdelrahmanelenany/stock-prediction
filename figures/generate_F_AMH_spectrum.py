"""
F_AMH: AMH Efficiency Spectrum — minimal academic figure with gradient bar.
Standalone: only matplotlib + numpy required.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUTPUT = "F_AMH_spectrum.png"

COL_LEFT  = "#2c4a6e"
COL_RIGHT = "#b5651d"
COL_GREY  = "#555555"
COL_DASH  = "#aaaaaa"
SERIF     = "DejaVu Serif"

fig, ax = plt.subplots(figsize=(11, 3))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("white")

BAR_X0, BAR_X1 = 0.05, 0.95
BAR_Y0, BAR_Y1 = 0.44, 0.56
BAR_Y_MID      = 0.50
N              = 200

# ── 1. Gradient bar ───────────────────────────────────────────────────────────
cmap = LinearSegmentedColormap.from_list("amh", [COL_LEFT, COL_RIGHT])
xs   = np.linspace(BAR_X0, BAR_X1, N + 1)
for i in range(N):
    ax.fill_between([xs[i], xs[i + 1]], BAR_Y0, BAR_Y1,
                    color=cmap(i / (N - 1)), linewidth=0)

# ── 2. End labels (inside bar) ────────────────────────────────────────────────
ax.text(0.07, BAR_Y_MID, "More Efficient",
        ha="left", va="center", fontsize=8.5,
        fontfamily=SERIF, style="italic", color="white")
ax.text(0.93, BAR_Y_MID, "Less Efficient",
        ha="right", va="center", fontsize=8.5,
        fontfamily=SERIF, style="italic", color="white")

# ── 3. Centre label ───────────────────────────────────────────────────────────
ax.text(0.50, BAR_Y_MID, "AMH Efficiency Spectrum",
        ha="center", va="center", fontsize=9,
        fontfamily=SERIF, fontweight="bold", color="white", zorder=5)

# ── 4. Universe markers (white circle, dark border) ───────────────────────────
for x in (0.20, 0.80):
    ax.plot(x, BAR_Y_MID, "o",
            markerfacecolor="white", markeredgecolor="#222222",
            markeredgewidth=0.8, markersize=8, zorder=6)

# ── 5. Universe titles + dashed connectors ────────────────────────────────────
TITLE_Y = 0.72

for x, label, color, ha in (
    (0.20, "Large-Cap Universe", COL_LEFT,  "center"),
    (0.80, "Small-Cap Universe", COL_RIGHT, "center"),
):
    ax.text(x, TITLE_Y, label,
            ha=ha, va="bottom", fontsize=10,
            fontfamily=SERIF, fontweight="bold", color=color)
    ax.plot([x, x], [BAR_Y1, TITLE_Y - 0.02],
            color=COL_DASH, lw=0.5, linestyle="--", zorder=3)

# ── 6. Regime labels (below bar) ──────────────────────────────────────────────
REGIME_Y = 0.28

for x, label, ha in (
    (0.20, "Mean-Reversion Regime", "center"),
    (0.80, "Momentum Regime",       "center"),
):
    ax.text(x, REGIME_Y, label,
            ha=ha, va="top", fontsize=9,
            fontfamily=SERIF, style="italic", color=COL_GREY)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.savefig(OUTPUT, dpi=300, bbox_inches="tight",
            pad_inches=0.10, facecolor="white")
plt.close()

import os
print(os.path.abspath(OUTPUT))
