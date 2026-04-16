"""
generate_thesis_outputs.py
--------------------------
Generates the following outputs from existing CSV reports (no pipeline re-run):

  Task 1 — Fold-by-fold Sharpe (gross + net) for all models
           → reports/fold_sharpe_decomposition.csv
           → outputs/figures/fold_sharpe_barplot.png

  Task 2 — Sub-period performance table for all models
           → reports/large_cap_table_T6_subperiod_performance_full.csv

  Task 3 — Cumulative return plot (all models + SPY benchmark)
           → outputs/figures/cumulative_returns_large_cap.png

Usage:
    python generate_thesis_outputs.py
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "reports")
FIGURES = os.path.join(ROOT, "outputs", "figures")
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(os.path.join(REPORTS, "latex_tables"), exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
fold_gross = pd.read_csv(os.path.join(REPORTS, "large_cap_fold_sharpe_per_model.csv"))
net_ret    = pd.read_csv(os.path.join(REPORTS, "large_cap_daily_returns_net_5bps.csv"),
                         parse_dates=["Date"], index_col="Date")
gross_ret  = pd.read_csv(os.path.join(REPORTS, "large_cap_daily_returns_gross.csv"),
                         parse_dates=["Date"], index_col="Date")

MODELS = ["LR", "RF", "XGBoost", "LSTM-B", "Ensemble"]
RF_DAILY = 0.00015   # ~3.8% / 252


# ---------------------------------------------------------------------------
# Helper: compute Sharpe for a return series
# ---------------------------------------------------------------------------
def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 5:
        return np.nan
    excess = r - RF_DAILY
    std = excess.std()
    if std == 0:
        return 0.0
    return float((excess.mean() / std) * np.sqrt(252))


# ===========================================================================
# TASK 1 — Fold-by-fold Sharpe decomposition (gross + net, all models)
# ===========================================================================
print("=== Task 1: Fold Sharpe Decomposition ===")

# Build fold date map from existing gross CSV
fold_dates = (
    fold_gross[fold_gross["Model"] == "LR"][["Fold", "Start Date", "End Date"]]
    .reset_index(drop=True)
)
fold_dates["Start Date"] = pd.to_datetime(fold_dates["Start Date"])
fold_dates["End Date"]   = pd.to_datetime(fold_dates["End Date"])

rows = []
for _, frow in fold_dates.iterrows():
    fold_num = int(frow["Fold"])
    s, e = frow["Start Date"], frow["End Date"]

    for model in MODELS:
        # gross Sharpe
        g_series = gross_ret.loc[s:e, model].dropna()
        n_series = net_ret.loc[s:e, model].dropna()

        rows.append({
            "Model":             model,
            "Fold":              fold_num,
            "Start Date":        s.date(),
            "End Date":          e.date(),
            "N Days":            len(n_series),
            "Sharpe Gross":      round(sharpe(g_series), 3),
            "Sharpe Net":        round(sharpe(n_series), 3),
        })

fold_df = pd.DataFrame(rows)
out_csv = os.path.join(REPORTS, "fold_sharpe_decomposition.csv")
fold_df.to_csv(out_csv, index=False)
print(f"  Saved → {out_csv}")

# --- Bar plot ---
fig, axes = plt.subplots(len(MODELS), 1,
                         figsize=(14, 3.5 * len(MODELS)),
                         sharex=True)

palette = {
    "positive": "#2c7bb6",
    "negative": "#d7191c",
}

for ax, model in zip(axes, MODELS):
    sub = fold_df[fold_df["Model"] == model].sort_values("Fold")
    folds  = sub["Fold"].values
    sharpes = sub["Sharpe Net"].values

    colors = [palette["positive"] if v >= 0 else palette["negative"] for v in sharpes]
    bars = ax.bar(folds, sharpes, color=colors, edgecolor="white", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Net Sharpe", fontsize=9)
    ax.set_title(model, fontsize=10, fontweight="bold", loc="left")
    ax.set_xticks(folds)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[-1].set_xlabel("Walk-Forward Fold", fontsize=9)

# Add a single legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=palette["positive"], label="Positive Sharpe"),
    Patch(facecolor=palette["negative"], label="Negative Sharpe"),
]
fig.legend(handles=legend_elements, loc="upper right",
           bbox_to_anchor=(0.98, 0.98), fontsize=9, framealpha=0.8)

fig.suptitle("Fold-by-Fold Net Sharpe Ratio (5 bps TC) — Large-Cap Universe",
             fontsize=12, fontweight="bold", y=1.002)
fig.tight_layout()

out_fig = os.path.join(FIGURES, "fold_sharpe_barplot.png")
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {out_fig}")


# ===========================================================================
# TASK 2 — Sub-period performance table (all models, net returns)
# ===========================================================================
print("\n=== Task 2: Sub-Period Analysis Table ===")

SUBPERIODS = {
    # Backtest starts 2020-06-29, so Pre-COVID and COVID crash have no OOS data.
    # Recovery window trimmed to backtest start.
    "Recovery/bull (2020-21)": ("2020-06-29", "2021-12-31"),
    "Bear (2022)":             ("2022-01-01", "2022-12-31"),
    "AI rally (2023-24)":      ("2023-01-01", "2024-12-31"),
    "Full period":             ("2020-06-29", "2024-12-31"),
}

def compute_metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 5:
        return {"N Days": 0, "Ann. Return (%)": np.nan,
                "Sharpe": np.nan, "Sortino": np.nan, "MDD (%)": np.nan}
    mean_d = r.mean()
    std_d  = r.std()
    excess = r - RF_DAILY
    sh = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0

    downside = r[r < RF_DAILY]
    if len(downside) > 1 and downside.std() > 0:
        so = (mean_d - RF_DAILY) / downside.std() * np.sqrt(252)
    else:
        so = np.nan

    cum = (1 + r).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    ann = ((1 + mean_d) ** 252 - 1) * 100

    return {
        "N Days":         int(len(r)),
        "Ann. Return (%)": round(ann, 2),
        "Sharpe":          round(sh, 3),
        "Sortino":         round(so, 3) if not np.isnan(so) else np.nan,
        "MDD (%)":         round(mdd, 2),
    }

sub_rows = []
for model in MODELS:
    for period_name, (start, end) in SUBPERIODS.items():
        sub = net_ret.loc[start:end, model]
        m = compute_metrics(sub)
        sub_rows.append({"Model": model, "Period": period_name, **m})

sub_df = pd.DataFrame(sub_rows)
out_sub = os.path.join(REPORTS, "large_cap_table_T6_subperiod_performance_full.csv")
sub_df.to_csv(out_sub, index=False)
print(f"  Saved → {out_sub}")

# Print summary table
pivot = sub_df.pivot_table(
    index="Period", columns="Model", values="Sharpe", aggfunc="first"
)[MODELS]
print("\n  Net Sharpe by Sub-Period:")
print(pivot.to_string())


# ===========================================================================
# TASK 3 — Cumulative Return Plot (all models + SPY)
# ===========================================================================
print("\n=== Task 3: Cumulative Return Plot ===")

# --- Try to fetch SPY or load from cache ---
spy_cache = os.path.join(REPORTS, "spy_daily_returns.csv")

spy = None
if os.path.exists(spy_cache):
    try:
        spy_raw = pd.read_csv(spy_cache, parse_dates=["Date"], index_col="Date")
        spy = spy_raw["SPY"]
        print("  Loaded SPY from cache.")
    except Exception as e:
        print(f"  Warning: could not load SPY cache: {e}")

if spy is None:
    try:
        import yfinance as yf
        start_date = net_ret.index.min().strftime("%Y-%m-%d")
        end_date   = net_ret.index.max().strftime("%Y-%m-%d")
        spy_raw = yf.download("SPY", start=start_date, end=end_date,
                              auto_adjust=True, progress=False)
        close = spy_raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        spy = close.pct_change().dropna()
        spy.name = "SPY"
        # Align to backtest dates
        spy = spy.reindex(net_ret.index).fillna(0)
        # Cache for next time
        pd.DataFrame({"SPY": spy}).to_csv(spy_cache)
        print("  Downloaded SPY and cached.")
    except Exception as e:
        print(f"  Warning: could not fetch SPY ({e}). Plotting without benchmark.")
        spy = None

# --- Build cumulative return DataFrame ---
cum_data = {}
for model in MODELS:
    cum_data[model] = (1 + net_ret[model].dropna()).cumprod()

if spy is not None:
    # Align SPY to same date range
    spy_aligned = spy.reindex(net_ret.index).dropna()
    # Rescale so SPY starts at 1.0 on the same first date as backtest
    spy_cum = (1 + spy_aligned).cumprod()
    spy_cum = spy_cum / spy_cum.iloc[0]
    cum_data["SPY"] = spy_cum

cum_df = pd.DataFrame(cum_data)

# --- Plot ---
fig, ax = plt.subplots(figsize=(14, 6))

MODEL_STYLES = {
    "LR":       {"color": "#aec6cf", "lw": 1.2, "ls": "--"},
    "RF":       {"color": "#7fcdbb", "lw": 1.2, "ls": "--"},
    "XGBoost":  {"color": "#fdae61", "lw": 1.2, "ls": "--"},
    "LSTM-B":   {"color": "#2c7bb6", "lw": 2.0, "ls": "-"},
    "Ensemble": {"color": "#d7191c", "lw": 2.0, "ls": "-"},
    "SPY":      {"color": "#444444", "lw": 1.8, "ls": ":",  "alpha": 0.8},
}

for col in cum_df.columns:
    style = MODEL_STYLES.get(col, {"color": "grey", "lw": 1.2, "ls": "-"})
    ax.plot(cum_df.index, cum_df[col],
            label=col,
            color=style["color"],
            linewidth=style["lw"],
            linestyle=style["ls"],
            alpha=style.get("alpha", 1.0))

ax.axhline(1.0, color="black", linewidth=0.6, linestyle=":")

# Shade recessions / regimes lightly
ax.axvspan(pd.Timestamp("2020-02-20"), pd.Timestamp("2020-04-30"),
           alpha=0.08, color="red", label="_COVID")
ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"),
           alpha=0.06, color="orange", label="_Bear2022")

ax.set_xlabel("Date", fontsize=10)
ax.set_ylabel("Cumulative Return (starting at 1.0)", fontsize=10)
ax.set_title("Cumulative Net Returns — Large-Cap Universe (5 bps TC)",
             fontsize=12, fontweight="bold")

ax.legend(loc="upper left", fontsize=9, framealpha=0.85,
          bbox_to_anchor=(1.01, 1.0))
ax.grid(axis="y", alpha=0.3, linewidth=0.5)
ax.grid(axis="x", alpha=0.15, linewidth=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Format x-axis with year labels only
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

fig.tight_layout()
out_cum = os.path.join(FIGURES, "cumulative_returns_large_cap.png")
fig.savefig(out_cum, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {out_cum}")

print("\n=== All tasks complete ===")
