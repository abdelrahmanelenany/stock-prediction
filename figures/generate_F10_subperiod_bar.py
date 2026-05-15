import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load data from reports/ ──────────────────────────────────────────────────

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

# Period names as they appear in the T6 CSVs → display labels for the chart
PERIOD_MAP = {
    "Recovery/bull":    "Recovery/Bull\n(2020 Q3–2021)",
    "2022 bear":        "2022 Bear",
    "2023-24 AI rally": "2023–24 AI Rally",
}

# Model names as they appear in the T6 CSVs → display labels for the chart
MODEL_MAP = {
    "LR":       "LR",
    "RF":       "RF",
    "XGBoost":  "XGB",
    "LSTM":     "LSTM",
    "TCN":      "TCN",
    "Ensemble": "Ensemble",
}

MODELS = ["LR", "RF", "XGB", "LSTM", "TCN", "Ensemble"]
PERIODS = list(PERIOD_MAP.values())


def load_universe(csv_name: str) -> dict:
    """Return nested dict: display_period → display_model → Sharpe Ratio."""
    path = os.path.join(REPORTS_DIR, csv_name)
    df = pd.read_csv(path)
    result: dict = {period_label: {} for period_label in PERIODS}

    for _, row in df.iterrows():
        csv_period = row["Period"]
        csv_model  = row["Model"]
        if csv_period not in PERIOD_MAP or csv_model not in MODEL_MAP:
            continue
        period_label = PERIOD_MAP[csv_period]
        model_label  = MODEL_MAP[csv_model]
        sharpe = row["Sharpe Ratio"]
        if pd.isna(sharpe):
            raise ValueError(
                f"Missing Sharpe Ratio for {csv_model} / {csv_period} in {csv_name}"
            )
        result[period_label][model_label] = round(float(sharpe), 3)

    # Verify completeness
    for period_label in PERIODS:
        for model_label in MODELS:
            if model_label not in result[period_label]:
                raise KeyError(
                    f"No entry for model={model_label!r}, period={period_label!r} in {csv_name}"
                )
    return result


LARGE_CAP = load_universe("large_cap_table_T6_subperiod_performance.csv")
SMALL_CAP = load_universe("small_cap_table_T6_subperiod_performance.csv")


def compute_shared_ylim(*universes: dict) -> tuple[float, float]:
    values = [
        value
        for universe in universes
        for period_values in universe.values()
        for value in period_values.values()
    ]
    min_value = min(values)
    max_value = max(values)
    span = max_value - min_value
    pad = max(0.15, span * 0.12)
    lower = min_value - pad
    upper = max_value + pad
    return (lower, upper)

# ── Validation printout ──────────────────────────────────────────────────────

print("=== Loaded values from reports/ ===")
for universe_name, universe_data in [("LARGE_CAP", LARGE_CAP), ("SMALL_CAP", SMALL_CAP)]:
    for period, model_vals in universe_data.items():
        for model, value in model_vals.items():
            print(f"{universe_name} | {period!r} | {model} | {value}")
print()

# ── Style ────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", font_scale=0.95)

palette = sns.color_palette("tab10", 6)
model_colors = dict(zip(MODELS, palette))

# ── Layout ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(12, 9))

bar_width = 0.12
x = np.arange(len(PERIODS))
offsets = np.linspace(-2.5, 2.5, len(MODELS)) * bar_width

all_bars: dict = {}


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

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", zorder=0)
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

shared_ylim = compute_shared_ylim(LARGE_CAP, SMALL_CAP)
for ax in axes:
    ax.set_ylim(shared_ylim)

# ── Overall figure title ─────────────────────────────────────────────────────

fig.suptitle(
    "Sub-Period Gross Sharpe Ratio by Model",
    fontsize=13, fontweight="bold",
)

fig.tight_layout(rect=[0, 0.11, 1, 0.97])

fig.legend(
    legend_handles,
    MODELS,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.075),
    ncol=6,
    frameon=True,
    fontsize=9,
    title=None,
)

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
            actual = round(bar.get_height(), 3)
            if actual != round(expected, 3):
                raise AssertionError(
                    f"Bar mismatch: {universe_name} | {period!r} | {model} "
                    f"| expected {round(expected, 3)} got {actual}"
                )
            print(f"PASS  {universe_name} | {period!r} | {model} | {actual}")

print("=== All bar assertions PASSED ===\n")

# ── Save ──────────────────────────────────────────────────────────────────────

out_dir = os.path.join(os.path.dirname(__file__))
os.makedirs(out_dir, exist_ok=True)

pdf_path = os.path.join(out_dir, "F10_subperiod_bar.pdf")
png_path = os.path.join(out_dir, "F10_subperiod_bar.png")

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=150, bbox_inches="tight")

pdf_kb = os.path.getsize(pdf_path) / 1024
png_kb = os.path.getsize(png_path) / 1024
print(f"Saved: {pdf_path}  ({pdf_kb:.1f} KB)")
print(f"Saved: {png_path}  ({png_kb:.1f} KB)")
