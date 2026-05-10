"""
Generate Figure F12: Feature-layer ablation grouped bar chart.
Two side-by-side panels (large-cap / small-cap). X-axis = models;
bars within each group = feature-layer conditions. Run from project root.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load data ─────────────────────────────────────────────────────────────────

df_large = pd.read_csv("reports/ablation/large_cap_ablation_summary.csv")
df_small = pd.read_csv("reports/ablation/small_cap_ablation_summary.csv")

# ── Constants ─────────────────────────────────────────────────────────────────

MODELS = ["LR", "RF", "XGBoost", "LSTM", "TCN"]
CONDITIONS = ["L1_only", "L1_L2", "L1_L2_L3"]
CONDITION_LABELS = {
    "L1_only":   "Layer 1 only",
    "L1_L2":     "Layers 1–2",
    "L1_L2_L3":  "Layers 1–2–3",
}

# Colorblind-friendly palette (blue, orange, green)
CONDITION_COLORS = ["#4878d0", "#ee854a", "#6acc65"]

N_CONDS  = len(CONDITIONS)
N_MODELS = len(MODELS)

# Grouped-bar geometry
GROUP_WIDTH = 0.70
BAR_WIDTH   = GROUP_WIDTH / N_CONDS
X_CENTERS   = np.arange(N_MODELS, dtype=float)

# ── rcParams — sized for 14 cm print width ────────────────────────────────────

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         7.5,
    "axes.labelsize":    8.5,
    "axes.titlesize":    9.5,
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   7.5,
    "legend.fontsize":   7.5,
    "axes.linewidth":    0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

FIG_W_IN = 14 / 2.54   # 14 cm
FIG_H_IN = 10 / 2.54   # ~10 cm — taller so bars fill more of the axis

fig, axes = plt.subplots(
    1, 2,
    figsize=(FIG_W_IN, FIG_H_IN),
    sharey=True,
)

# ── Draw helper ───────────────────────────────────────────────────────────────

def draw_panel(ax, df, title, ylabel):
    lookup = {
        (row["condition"], row["model"]): (
            row["mean_test_net_sharpe"],
            row["std_test_net_sharpe"],
        )
        for _, row in df.iterrows()
    }

    bar_records = []   # (condition, model, bar_obj, expected_mean) for assertions
    handles     = []
    all_means   = []

    for ci, cond in enumerate(CONDITIONS):
        offset = (ci - (N_CONDS - 1) / 2.0) * BAR_WIDTH
        color  = CONDITION_COLORS[ci]

        xs     = []
        means  = []
        models_drawn = []

        for mi, model in enumerate(MODELS):
            key = (cond, model)
            if key not in lookup:
                continue
            mean, _std = lookup[key]
            xs.append(X_CENTERS[mi] + offset)
            means.append(mean)
            models_drawn.append(model)

        if not means:
            continue

        bars = ax.bar(
            xs,
            means,
            width=BAR_WIDTH * 0.88,
            color=color,
            label=CONDITION_LABELS[cond],
            zorder=3,
        )
        handles.append(bars[0])
        for bar, m, mn in zip(bars, models_drawn, means):
            bar_records.append((cond, m, bar, mn))
            all_means.append(mn)

    # y = 0 reference
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", zorder=2)

    # y-axis grid only
    ax.yaxis.grid(True, linewidth=0.35, linestyle=":", color="#cccccc", zorder=0)
    ax.set_axisbelow(True)

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Axis labels and title
    ax.set_xticks(X_CENTERS)
    ax.set_xticklabels(MODELS)
    ax.set_xlabel("Model", labelpad=3)
    if ylabel:
        ax.set_ylabel("Mean Net Sharpe (5 bps, 17 folds)", labelpad=3)
    ax.set_title(title, fontweight="bold", pad=5)

    return handles, bar_records, all_means

# ── Render ────────────────────────────────────────────────────────────────────

handles_large, records_large, means_large = draw_panel(
    axes[0], df_large, "Large-Cap Universe", ylabel=True
)
handles_small, records_small, means_small = draw_panel(
    axes[1], df_small, "Small-Cap Universe", ylabel=False
)

# Tighten shared y-axis to the actual data range with 20 % padding
all_means = means_large + means_small
data_min  = min(all_means)
data_max  = max(all_means)
span      = max(data_max - data_min, 0.2)
pad       = span * 0.20
axes[0].set_ylim(data_min - pad, data_max + pad)   # sharey propagates to axes[1]

# Shared legend below both panels
fig.legend(
    handles_large,
    [CONDITION_LABELS[c] for c in CONDITIONS],
    loc="lower center",
    bbox_to_anchor=(0.5, 0.0),
    ncol=3,
    frameon=True,
    edgecolor="#cccccc",
    framealpha=0.9,
)

fig.tight_layout(rect=[0, 0.13, 1, 1])
fig.subplots_adjust(wspace=0.06)

# ── Bar height assertions ─────────────────────────────────────────────────────

print("=== Bar height assertions ===")
for panel_name, records in [("LARGE_CAP", records_large), ("SMALL_CAP", records_small)]:
    for cond, model, bar, expected in records:
        actual = round(bar.get_height(), 4)
        exp_r  = round(expected, 4)
        if actual != exp_r:
            raise AssertionError(
                f"Bar mismatch: {panel_name} | {cond} | {model} "
                f"| expected {exp_r} got {actual}"
            )
        print(f"PASS  {panel_name:10s} | {cond:10s} | {model:8s} | {actual:+.4f}")
print("=== All bar assertions PASSED ===\n")

# ── Save ──────────────────────────────────────────────────────────────────────

os.makedirs("figures", exist_ok=True)

PDF_PATH = "figures/F12_ablation_feature_layer.pdf"
PNG_PATH = "figures/F12_ablation_feature_layer.png"

fig.savefig(PDF_PATH, bbox_inches="tight")
fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")

print(f"Saved: {PDF_PATH}  ({os.path.getsize(PDF_PATH) / 1024:.1f} KB)")
print(f"Saved: {PNG_PATH}  ({os.path.getsize(PNG_PATH) / 1024:.1f} KB)")
