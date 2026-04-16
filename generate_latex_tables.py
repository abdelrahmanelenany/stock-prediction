"""
generate_latex_tables.py
------------------------
Generates LaTeX-formatted tables for the thesis:

  Table T5 — Risk-return metrics (gross + net) for all models
  Table T8 — Classification metrics for all models
  Table T6 — Sub-period net Sharpe pivot

Output: reports/latex_tables/*.tex
"""

import os
import pandas as pd
import numpy as np

ROOT    = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(ROOT, "reports")
LATEX   = os.path.join(REPORTS, "latex_tables")
os.makedirs(LATEX, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt(val, digits=2):
    if pd.isna(val):
        return "—"
    try:
        return f"{float(val):.{digits}f}"
    except (TypeError, ValueError):
        return str(val)


def pct(val, digits=2):
    """Format as percentage with % sign."""
    if pd.isna(val):
        return "—"
    try:
        return f"{float(val):.{digits}f}\\%"
    except (TypeError, ValueError):
        return str(val)


def bold_best(series, higher_is_better=True):
    """Return copy of series with best value flagged (for LaTeX \\textbf)."""
    s = series.copy().astype(str)
    try:
        nums = pd.to_numeric(series, errors="coerce")
        if higher_is_better:
            best_idx = nums.idxmax()
        else:
            best_idx = nums.idxmin()
        s[best_idx] = r"\textbf{" + s[best_idx] + "}"
    except Exception:
        pass
    return s


# ---------------------------------------------------------------------------
# TABLE T5 — Risk-return metrics
# ---------------------------------------------------------------------------
print("=== Table T5 ===")

net_raw   = pd.read_csv(os.path.join(REPORTS, "large_cap_table_T5_net_returns_5bps.csv"))
gross_raw = pd.read_csv(os.path.join(REPORTS, "large_cap_table_T5_gross_returns.csv"))

MODELS_ORDER = ["LR", "RF", "XGBoost", "LSTM-B", "Ensemble"]

# Ensure model column exists
for df in [net_raw, gross_raw]:
    if "Model" not in df.columns and len(df) == len(MODELS_ORDER):
        df["Model"] = MODELS_ORDER

net_raw   = net_raw.set_index("Model").loc[MODELS_ORDER]
gross_raw = gross_raw.set_index("Model").loc[MODELS_ORDER]

T5_COLS = {
    "Ann. Return (%)": ("Annualized Return (%)", True),
    "Std Dev (%)":     ("Annualized Std Dev (%)", False),
    "Sharpe":          ("Sharpe Ratio",           True),
    "Sortino":         ("Sortino Ratio",           True),
    "MDD (%)":         ("Max Drawdown (%)",        True),   # less negative = better
    "Calmar":          ("Calmar Ratio",            True),
    "Win Rate (%)":    ("Win Rate (%)",             True),
}


def build_t5_block(df, label):
    rows_tex = []
    rows_tex.append(r"\multicolumn{9}{l}{\textit{" + label + r"}} \\")
    rows_tex.append(r"\hline")
    for model in MODELS_ORDER:
        row = df.loc[model]
        cells = [model]
        for col_label, (src_col, higher) in T5_COLS.items():
            v = row.get(src_col, np.nan)
            if col_label in ("Ann. Return (%)", "Std Dev (%)", "MDD (%)", "Win Rate (%)"):
                cells.append(pct(v))
            else:
                cells.append(fmt(v, 3))
        rows_tex.append(" & ".join(cells) + r" \\")
    return rows_tex


header_cols = ["Model"] + list(T5_COLS.keys())

tex_t5 = []
tex_t5.append(r"\begin{table}[htbp]")
tex_t5.append(r"\centering")
tex_t5.append(r"\caption{Risk-Return Metrics — Large-Cap Universe}")
tex_t5.append(r"\label{tab:T5_large_cap}")
tex_t5.append(r"\small")
tex_t5.append(r"\begin{tabular}{l" + "r" * len(T5_COLS) + "}")
tex_t5.append(r"\hline\hline")
tex_t5.append(" & ".join([r"\textbf{" + c + "}" for c in header_cols]) + r" \\")
tex_t5.append(r"\hline")
tex_t5 += build_t5_block(gross_raw, "Panel A: Gross Returns (0 bps)")
tex_t5.append(r"\hline")
tex_t5 += build_t5_block(net_raw, r"Panel B: Net Returns (5 bps TC)")
tex_t5.append(r"\hline\hline")
tex_t5.append(r"\end{tabular}")
tex_t5.append(r"\begin{tablenotes}\small")
tex_t5.append(r"\item Backtest period: 2020-06-29 to 2024-09-30. "
              r"Risk-free rate: 3.8\%\ p.a. (0.015\% daily). "
              r"$k = 10$ long/short positions from a 70-stock large-cap universe.")
tex_t5.append(r"\end{tablenotes}")
tex_t5.append(r"\end{table}")

t5_path = os.path.join(LATEX, "table_T5.tex")
with open(t5_path, "w") as f:
    f.write("\n".join(tex_t5))
print(f"  Saved → {t5_path}")


# ---------------------------------------------------------------------------
# TABLE T8 — Classification metrics
# ---------------------------------------------------------------------------
print("=== Table T8 ===")

t8_raw = pd.read_csv(os.path.join(REPORTS, "large_cap_table_T8_classification_metrics.csv"))

if "Model" not in t8_raw.columns and len(t8_raw) == len(MODELS_ORDER):
    t8_raw["Model"] = MODELS_ORDER

# Check we have Ensemble; if not, add placeholder
if "Ensemble" not in t8_raw["Model"].values:
    # Try the backtest summary text values
    pass

t8_raw = t8_raw.set_index("Model")
if "Ensemble" not in t8_raw.index:
    t8_raw.loc["Ensemble"] = np.nan

t8_raw = t8_raw.reindex(MODELS_ORDER)

T8_COLS = {
    "Accuracy":      ("Accuracy (%)", True),
    "AUC-ROC":       ("AUC-ROC",      True),
    "F1 Score":      ("F1 Score",     True),
    "Daily AUC":     ("Daily AUC (mean)", True),
    "Daily AUC Std": ("Daily AUC (std)",  False),
}

tex_t8 = []
tex_t8.append(r"\begin{table}[htbp]")
tex_t8.append(r"\centering")
tex_t8.append(r"\caption{Classification Metrics — Large-Cap Universe}")
tex_t8.append(r"\label{tab:T8_large_cap}")
tex_t8.append(r"\small")
tex_t8.append(r"\begin{tabular}{l" + "r" * len(T8_COLS) + "}")
tex_t8.append(r"\hline\hline")
header_t8 = ["Model"] + list(T8_COLS.keys())
tex_t8.append(" & ".join([r"\textbf{" + c + "}" for c in header_t8]) + r" \\")
tex_t8.append(r"\hline")

for model in MODELS_ORDER:
    row = t8_raw.loc[model]
    cells = [model]
    for col_label, (src_col, higher) in T8_COLS.items():
        v = row.get(src_col, np.nan)
        if col_label == "Accuracy":
            cells.append(pct(v))
        else:
            cells.append(fmt(v, 4))
    tex_t8.append(" & ".join(cells) + r" \\")

tex_t8.append(r"\hline\hline")
tex_t8.append(r"\end{tabular}")
tex_t8.append(r"\begin{tablenotes}\small")
tex_t8.append(r"\item Metrics evaluated with \texttt{invert\_signals=True} "
              r"(probabilities inverted to match trading direction). "
              r"Daily AUC: mean within-day cross-sectional AUC over the backtest period.")
tex_t8.append(r"\end{tablenotes}")
tex_t8.append(r"\end{table}")

t8_path = os.path.join(LATEX, "table_T8.tex")
with open(t8_path, "w") as f:
    f.write("\n".join(tex_t8))
print(f"  Saved → {t8_path}")


# ---------------------------------------------------------------------------
# TABLE T6 — Sub-period performance (Net Sharpe pivot)
# ---------------------------------------------------------------------------
print("=== Table T6 ===")

t6_raw = pd.read_csv(os.path.join(REPORTS, "large_cap_table_T6_subperiod_performance_full.csv"))

PERIODS_ORDER = [
    "Recovery/bull (2020-21)",
    "Bear (2022)",
    "AI rally (2023-24)",
    "Full period",
]

# Build pivot of Sharpe, Ann. Return, MDD
metric_pairs = [
    ("Sharpe",          "Sharpe",           True),
    ("Ann. Return (%)", "Ann. Return (%)",  True),
    ("MDD (%)",         "MDD (%)",          True),
]

# MultiColumn header: each model spans 2 cols (Sharpe + Ann.Ret)
# Simpler: one wide table with Sharpe only, then separate table for full metrics

# --- Sharpe-only pivot ---
pivot_s = t6_raw.pivot_table(index="Period", columns="Model",
                              values="Sharpe", aggfunc="first")
pivot_s = pivot_s.reindex(index=PERIODS_ORDER)[MODELS_ORDER]

tex_t6 = []
tex_t6.append(r"\begin{table}[htbp]")
tex_t6.append(r"\centering")
tex_t6.append(r"\caption{Sub-Period Net Sharpe Ratios — Large-Cap Universe (5 bps TC)}")
tex_t6.append(r"\label{tab:T6_large_cap}")
tex_t6.append(r"\small")
tex_t6.append(r"\begin{tabular}{l" + "r" * len(MODELS_ORDER) + "}")
tex_t6.append(r"\hline\hline")
tex_t6.append(r"\textbf{Period} & " +
              " & ".join([r"\textbf{" + m + "}" for m in MODELS_ORDER]) + r" \\")
tex_t6.append(r"\hline")

for period in PERIODS_ORDER:
    if period not in pivot_s.index:
        continue
    row = pivot_s.loc[period]
    cells = [period]
    for model in MODELS_ORDER:
        v = row.get(model, np.nan)
        cells.append(fmt(v, 3))
    tex_t6.append(" & ".join(cells) + r" \\")

tex_t6.append(r"\hline\hline")
tex_t6.append(r"\end{tabular}")
tex_t6.append(r"\begin{tablenotes}\small")
tex_t6.append(r"\item Net Sharpe ratio per sub-period. "
              r"Backtest OOS window begins 2020-06-29; "
              r"Pre-COVID and COVID-crash sub-periods have no OOS data.")
tex_t6.append(r"\end{tablenotes}")
tex_t6.append(r"\end{table}")

t6_path = os.path.join(LATEX, "table_T6.tex")
with open(t6_path, "w") as f:
    f.write("\n".join(tex_t6))
print(f"  Saved → {t6_path}")

print("\n=== LaTeX tables complete ===")
print(f"  Output directory: {LATEX}")
