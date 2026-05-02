"""Generate Figure F8 — Small-Cap Cumulative Returns with Drawdown Overlay."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

# ── Paths ─────────────────────────────────────────────────────────────────────
REPORTS = os.path.join(os.path.dirname(__file__), "..", "reports")
OUT_DIR = os.path.dirname(__file__)

GROSS_CSV = os.path.join(REPORTS, "small_cap_daily_returns_gross.csv")
NET_CSV   = os.path.join(REPORTS, "small_cap_daily_returns_net_5bps.csv")

COLORS = {
    "LR":       "#1f77b4",
    "RF":       "#ff7f0e",
    "XGBoost":  "#2ca02c",
    "LSTM":     "#d62728",
    "TCN":      "#9467bd",
    "Ensemble": "#8c564b",
}
MODELS = list(COLORS.keys())

# ── Load data ─────────────────────────────────────────────────────────────────
gross = pd.read_csv(GROSS_CSV, parse_dates=["Date"]).set_index("Date")
net   = pd.read_csv(NET_CSV,   parse_dates=["Date"]).set_index("Date")

# Cumulative return: (1+r).cumprod() - 1  → 0.40 = +40 %
cum_gross = (1 + gross).cumprod() - 1
cum_net   = (1 + net).cumprod()   - 1

# Running maximum drawdown from net wealth index
wealth_net = 1 + cum_net
drawdown   = wealth_net / wealth_net.cummax() - 1

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(12, 7), sharex=True,
    gridspec_kw={"height_ratios": [2, 1]},
)

# ── Top panel: Cumulative returns ─────────────────────────────────────────────
for model in MODELS:
    lw  = 2.0 if model == "Ensemble" else 1.2
    col = COLORS[model]
    ax_top.plot(cum_gross.index, cum_gross[model] * 100,
                color=col, lw=lw, linestyle="-",  label=model)
    ax_top.plot(cum_net.index,   cum_net[model]   * 100,
                color=col, lw=lw, linestyle="--")

ax_top.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.5)
ax_top.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"{x:+.0f}%"
))
ax_top.set_ylabel("Cumulative Return")
ax_top.set_title("Small-Cap Universe — Cumulative Returns (Gross and Net, 5 bps)")
ax_top.yaxis.grid(True, color="lightgray", linewidth=0.6)
ax_top.xaxis.grid(True, color="lightgray", linewidth=0.6)
ax_top.set_axisbelow(True)

legend = ax_top.legend(
    title="Model",
    bbox_to_anchor=(1.01, 1), loc="upper left",
    frameon=True, fontsize=9,
)
fig.text(
    1.012, 0.60,
    "Solid = gross\nDashed = net",
    transform=ax_top.transAxes,
    fontsize=8, va="top", ha="left", color="dimgray",
)

# ── Bottom panel: Running drawdown ────────────────────────────────────────────
for model in MODELS:
    col = COLORS[model]
    lw  = 2.0 if model == "Ensemble" else 1.2
    ax_bot.fill_between(drawdown.index, drawdown[model], 0,
                        color=col, alpha=0.3)
    ax_bot.plot(drawdown.index, drawdown[model], color=col, lw=lw)

ax_bot.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
ax_bot.set_ylabel("Drawdown")
ax_bot.set_title("Running Drawdown (Net)")
ax_bot.yaxis.grid(True, color="lightgray", linewidth=0.6)
ax_bot.xaxis.grid(True, color="lightgray", linewidth=0.6)
ax_bot.set_axisbelow(True)

# ── Shared x-axis formatting ──────────────────────────────────────────────────
ax_bot.xaxis.set_major_locator(mdates.YearLocator())
ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()

# ── Save ───────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
pdf_path = os.path.join(OUT_DIR, "F8_smallcap_cumulative_returns.pdf")
png_path = os.path.join(OUT_DIR, "F8_smallcap_cumulative_returns.png")

plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
plt.savefig(png_path, dpi=150, bbox_inches="tight")
print(f"Saved PDF → {pdf_path}")
print(f"Saved PNG → {png_path}")
