"""
Table T3 — Walk-forward fold calendar.

Outputs:
  figures/table_t3_walkforward_calendar.tex   (booktabs LaTeX snippet)
  figures/table_t3_walkforward_calendar.csv   (plain CSV)
  Console: tabulate github-fmt summary
"""

import os
import pandas as pd
from tabulate import tabulate

# ---------------------------------------------------------------------------
# Walk-forward parameters  (must match CLAUDE.md / fig_f5)
# ---------------------------------------------------------------------------
TRAIN_DAYS  = 252
VAL_DAYS    = 63
TEST_DAYS   = 63
STRIDE      = 63          # = TEST_DAYS
SAMPLE_START = "2019-01-02"   # first business day of 2019
SAMPLE_END   = "2024-12-31"

OUT_DIR = os.path.dirname(os.path.abspath(__file__))   # figures/

# ---------------------------------------------------------------------------
# Build the full business-day calendar once
# ---------------------------------------------------------------------------
all_bdays = pd.bdate_range(start=SAMPLE_START, end=SAMPLE_END)

def bday_offset(base_idx: int, n: int) -> int:
    """Return index into all_bdays that is n trading days after base_idx."""
    return base_idx + n


def fmt(idx: int) -> str:
    return all_bdays[idx].strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Generate folds
# ---------------------------------------------------------------------------
rows = []
fold = 1
train_start_idx = 0

while True:
    train_end_idx = train_start_idx + TRAIN_DAYS - 1
    val_start_idx = train_end_idx + 1
    val_end_idx   = val_start_idx + VAL_DAYS - 1
    test_start_idx = val_end_idx + 1
    test_end_idx   = test_start_idx + TEST_DAYS - 1

    if test_end_idx >= len(all_bdays):
        break

    rows.append({
        "Fold":        fold,
        "Train Start": fmt(train_start_idx),
        "Train End":   fmt(train_end_idx),
        "Val Start":   fmt(val_start_idx),
        "Val End":     fmt(val_end_idx),
        "Test Start":  fmt(test_start_idx),
        "Test End":    fmt(test_end_idx),
    })

    fold += 1
    train_start_idx += STRIDE   # rolling window — advance by one stride

df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------
csv_path = os.path.join(OUT_DIR, "table_t3_walkforward_calendar.csv")
df.to_csv(csv_path, index=False)
print(f"Saved CSV  → {csv_path}")

# ---------------------------------------------------------------------------
# LaTeX output (booktabs, no vertical lines)
# ---------------------------------------------------------------------------
tex_path = os.path.join(OUT_DIR, "table_t3_walkforward_calendar.tex")

col_spec = "r" + "c" * 6          # fold right-aligned; dates centred
bold = lambda s: r"\textbf{" + s + r"}"
header_cells = [bold(c) for c in df.columns]

lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"  \centering")
lines.append(r"  \small")
lines.append(r"  \setlength{\tabcolsep}{6pt}")
lines.append(
    r"  \caption{Walk-forward fold calendar. Each fold advances by one test window "
    r"(63 trading days). Dates are business days computed via \texttt{pd.bdate\_range}.}"
)
lines.append(r"  \label{tab:walkforward_calendar}")
lines.append(r"  \begin{tabular}{" + col_spec + r"}")
lines.append(r"    \toprule")
lines.append("    " + " & ".join(header_cells) + r" \\")
lines.append(r"    \midrule")

for _, row in df.iterrows():
    cells = [str(int(row["Fold"]))] + [row[c] for c in df.columns[1:]]
    lines.append("    " + " & ".join(cells) + r" \\")

lines.append(r"    \bottomrule")
lines.append(r"  \end{tabular}")
lines.append(r"\end{table}")

with open(tex_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"Saved LaTeX → {tex_path}")

# ---------------------------------------------------------------------------
# Console pretty-print
# ---------------------------------------------------------------------------
print()
print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
print()
print(f"Total folds : {len(df)}")
last = df.iloc[-1]
print(f"Last fold   : Test {last['Test Start']} → {last['Test End']}")
