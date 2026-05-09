"""Generate Figure F13: Training Window Sensitivity Chart."""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
import os

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# --- Data ---
models = ["LR", "RF", "XGBoost", "LSTM", "TCN", "Ensemble"]
windows = ["252d", "500d", "756d"]

large_cap = {
    "LR":       [0.180,  0.075,  0.509],
    "RF":       [0.268,  0.227,  0.785],
    "XGBoost":  [0.080, -0.188,  0.706],
    "LSTM":     [0.364, -0.169, -0.084],
    "TCN":      [0.000, -0.069,  0.078],
    "Ensemble": [0.699,  0.267, -0.139],
}

small_cap = {
    "LR":       [ 0.172, -0.362, -0.502],
    "RF":       [-0.087, -0.564,  0.414],
    "XGBoost":  [-0.182, -0.281, -0.289],
    "LSTM":     [ 0.350, -0.632,  0.300],
    "TCN":      [ 0.412, -0.410, -0.909],
    "Ensemble": [ 0.031, -0.153,  0.276],
}

# --- Visual design ---
# 252d: solid dark blue (primary); 500d: muted teal; 756d: warm orange
colors = ["#2166ac", "#74add1", "#d6604d"]
edge_colors = ["#144d82", "#4a90b8", "#b03a24"]
hatches = ["", "//", ".."]

bar_width = 0.22
group_gap = 0.08
n_windows = len(windows)

x = np.arange(len(models))
offsets = np.array([-1, 0, 1]) * bar_width

# Shared y-axis limits
all_vals = (
    [v for d in large_cap.values() for v in d]
    + [v for d in small_cap.values() for v in d]
)
y_min = min(all_vals) - 0.12
y_max = max(all_vals) + 0.15
# round to nearest 0.2
y_min = np.floor(y_min / 0.2) * 0.2
y_max = np.ceil(y_max / 0.2) * 0.2

# --- Figure ---
fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.8), sharey=True)
fig.subplots_adjust(wspace=0.06, left=0.09, right=0.98, top=0.82, bottom=0.17)

panel_data = [("Large-Cap Universe", large_cap), ("Small-Cap Universe", small_cap)]

for ax, (title, data) in zip(axes, panel_data):
    for w_idx, (window, color, ec, hatch) in enumerate(
        zip(windows, colors, edge_colors, hatches)
    ):
        vals = [data[m][w_idx] for m in models]
        bars = ax.bar(
            x + offsets[w_idx],
            vals,
            width=bar_width,
            color=color,
            edgecolor=ec,
            linewidth=0.6,
            hatch=hatch,
            zorder=3,
            label=window if w_idx == 0 or ax == axes[0] else "_nolegend_",
        )

    # Zero reference line
    ax.axhline(0, color="black", linewidth=0.9, linestyle="-", zorder=4)

    # Axes formatting
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0)
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    # Horizontal grid lines only (y-axis)
    ax.yaxis.grid(True, which="major", color="#dddddd", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_linewidth(0.8)

    ax.set_title(title, fontweight="bold", pad=6)

    if ax == axes[0]:
        ax.set_ylabel("Net Sharpe Ratio\n(5 bps per half-turn)", labelpad=6)
    else:
        ax.tick_params(axis="y", left=False)

# --- Legend (shared, placed above both panels) ---
legend_handles = []
for w_idx, (window, color, ec, hatch) in enumerate(
    zip(windows, colors, edge_colors, hatches)
):
    label = f"{window}  ← Primary" if w_idx == 0 else window
    patch = mpatches.Patch(
        facecolor=color,
        edgecolor=ec,
        hatch=hatch,
        linewidth=0.6,
        label=label,
    )
    legend_handles.append(patch)

fig.legend(
    handles=legend_handles,
    title="Training Window",
    title_fontsize=8,
    loc="upper center",
    bbox_to_anchor=(0.535, 1.00),
    ncol=3,
    frameon=True,
    framealpha=0.95,
    edgecolor="#cccccc",
    handlelength=1.6,
    handleheight=1.0,
    columnspacing=1.4,
)

# --- Save ---
out_dir = os.path.join(
    os.path.dirname(__file__), "outputs", "figures"
)
os.makedirs(out_dir, exist_ok=True)

pdf_path = os.path.join(out_dir, "F13_training_window_sensitivity.pdf")
png_path = os.path.join(out_dir, "F13_training_window_sensitivity.png")

fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
fig.savefig(png_path, format="png", dpi=150, bbox_inches="tight")

print(f"Saved PDF: {pdf_path}")
print(f"Saved PNG: {png_path}")
plt.close(fig)
