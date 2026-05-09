import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Hardcoded data ──────────────────────────────────────────────────────────

LARGE_CAP = {
    "Recovery/Bull\n(2020 Q3–2021)": {
        "LR": 0.21, "RF": -0.05, "XGB": -0.11,
        "LSTM": 0.84, "TCN": 0.23, "Ensemble": 0.85
    },
    "2022 Bear": {
        "LR": 0.19, "RF": 0.38, "XGB": 0.70,
        "LSTM": -0.08, "TCN": 0.11, "Ensemble": 1.06
    },
    "2023–24 AI Rally": {
        "LR": 0.16, "RF": 0.42, "XGB": -0.22,
        "LSTM": 0.29, "TCN": -0.23, "Ensemble": 0.36
    },
}

SMALL_CAP = {
    "Recovery/Bull\n(2020 Q3–2021)": {
        "LR": 1.00, "RF": 0.03, "XGB": 0.34,
        "LSTM": 1.05, "TCN": 1.18, "Ensemble": 0.97
    },
    "2022 Bear": {
        "LR": 0.13, "RF": -0.34, "XGB": -0.65,
        "LSTM": 0.01, "TCN": 0.14, "Ensemble": -0.02
    },
    "2023–24 AI Rally": {
        "LR": -0.68, "RF": -0.02, "XGB": -0.43,
        "LSTM": -0.28, "TCN": -0.15, "Ensemble": -0.86
    },
}

MODELS = ["LR", "RF", "XGB", "LSTM", "TCN", "Ensemble"]
PERIODS = list(LARGE_CAP.keys())

# ── Validation printout ──────────────────────────────────────────────────────

print("=== Validation: hardcoded values ===")
for universe_name, universe_data in [("LARGE_CAP", LARGE_CAP), ("SMALL_CAP", SMALL_CAP)]:
    for period, model_vals in universe_data.items():
        for model, value in model_vals.items():
            print(f"{universe_name} | {period!r} | {model} | {value}")

# Cross-check: ensure keys match
for period in PERIODS:
    assert period in LARGE_CAP, f"Missing period in LARGE_CAP: {period}"
    assert period in SMALL_CAP, f"Missing period in SMALL_CAP: {period}"
    for model in MODELS:
        assert model in LARGE_CAP[period], f"Missing model {model} in LARGE_CAP[{period}]"
        assert model in SMALL_CAP[period], f"Missing model {model} in SMALL_CAP[{period}]"
print("=== Validation PASSED ===\n")

# ── Style ────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", font_scale=0.95)

palette = sns.color_palette("tab10", 6)
model_colors = dict(zip(MODELS, palette))

# ── Layout ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(12, 9))

bar_width = 0.12
x = np.arange(len(PERIODS))
offsets = np.linspace(-2.5, 2.5, len(MODELS)) * bar_width

# Track all bars for the final assertion
# Structure: {panel_idx: {(period_idx, model): bar_object}}
all_bars = {}

def draw_panel(ax, universe_data, title, panel_idx):
    all_bars[panel_idx] = {}
    bars_for_legend = []

    for j, model in enumerate(MODELS):
        values = [universe_data[period][model] for period in PERIODS]
        bar_positions = x + offsets[j]
        bars = ax.bar(
            bar_positions,
            values,
            width=bar_width,
            color=model_colors[model],
            label=model,
            zorder=3,
        )
        bars_for_legend.append(bars[0])

        for i, (bar, val) in enumerate(zip(bars, values)):
            all_bars[panel_idx][(i, model)] = (bar, val)
            # Value label
            pad = 0.02
            if val >= 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + pad,
                    f"{val:.2f}",
                    ha="center", va="bottom",
                    fontsize=7.5, color="black",
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val - pad,
                    f"{val:.2f}",
                    ha="center", va="top",
                    fontsize=7.5, color="black",
                )

    # Zero line
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", zorder=0)

    # Axes formatting
    ax.set_xticks(x)
    ax.set_xticklabels(PERIODS, fontsize=9)
    ax.set_ylabel("Gross Sharpe Ratio", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return bars_for_legend

legend_handles = draw_panel(
    axes[0], LARGE_CAP,
    "Large-Cap Universe  (invert_signals=True)",
    panel_idx=0,
)
draw_panel(
    axes[1], SMALL_CAP,
    "Small-Cap Universe  (invert_signals=False)",
    panel_idx=1,
)

# ── Overall figure title ─────────────────────────────────────────────────────

fig.suptitle(
    "Sub-Period Gross Sharpe Ratio by Model",
    fontsize=13, fontweight="bold",
)

# ── Layout: reserve bottom strip for legend + footnote ───────────────────────
# rect=[left, bottom, right, top] — subplots are fitted inside this rectangle

fig.tight_layout(rect=[0, 0.11, 1, 0.97])

# ── Shared legend below both panels ──────────────────────────────────────────

fig.legend(
    legend_handles,
    MODELS,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.075),   # figure-coords: sits in the reserved strip
    ncol=6,
    frameon=True,
    fontsize=9,
    title=None,
)

# ── Footnote below legend ─────────────────────────────────────────────────────

fig.text(
    0.5, 0.01,
    "Gross returns; Pre-COVID and COVID crash periods not shown (in training window)",
    ha="center", va="bottom",
    fontsize=8, color="gray", style="italic",
)

# ── Bar assertion ─────────────────────────────────────────────────────────────

print("=== Bar height assertions ===")
for panel_idx, universe_data, universe_name in [
    (0, LARGE_CAP, "LARGE_CAP"),
    (1, SMALL_CAP, "SMALL_CAP"),
]:
    for period_idx, period in enumerate(PERIODS):
        for model in MODELS:
            bar, expected = all_bars[panel_idx][(period_idx, model)]
            actual = round(bar.get_height(), 2)
            if actual != round(expected, 2):
                raise AssertionError(
                    f"Bar mismatch: {universe_name} | {period!r} | {model} "
                    f"| expected {round(expected, 2)} got {actual}"
                )
            print(f"PASS  {universe_name} | {period!r} | {model} | {actual}")

print("=== All bar assertions PASSED ===\n")

# ── Save ──────────────────────────────────────────────────────────────────────

os.makedirs("figures", exist_ok=True)

pdf_path = "figures/F10_subperiod_bar.pdf"
png_path = "figures/F10_subperiod_bar.png"

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=150, bbox_inches="tight")

pdf_kb = os.path.getsize(pdf_path) / 1024
png_kb = os.path.getsize(png_path) / 1024
print(f"Saved: {pdf_path}  ({pdf_kb:.1f} KB)")
print(f"Saved: {png_path}  ({png_kb:.1f} KB)")
