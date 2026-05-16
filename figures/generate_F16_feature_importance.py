"""
Generate Figure F16 — Feature Importance Bar Charts (top-10, fold-averaged)
Single combined figure: 2 rows × 5 columns.
Row 0 = Large-Cap Universe, Row 1 = Small-Cap Universe.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------------------------------------------------------
# Load data from reports/
# ---------------------------------------------------------------------------

REPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "reports"
)

def load_importance_csv(universe_key: str) -> dict:
    path = os.path.join(REPORTS_DIR, f"{universe_key}_feature_importances_avg.csv")
    df = pd.read_csv(path)
    # NaN means the feature was not in that model's feature set; treat as 0 for ranking
    df = df.fillna(0.0)
    result = {"Feature": df["Feature"].tolist()}
    for col in ["LR_coef", "RF_importance", "XGB_gain", "LSTM_B_perm", "TCN_perm"]:
        result[col] = df[col].tolist()
    return result

large_cap_data = load_importance_csv("large_cap")
small_cap_data = load_importance_csv("small_cap")

display_names = {
    "Beta_60d":            "Beta (60d)",
    "RealVol_20d":         "Realized Vol (20d)",
    "Market_Vol_60d":      "Market Vol (60d)",
    "Market_Vol_20d":      "Market Vol (20d)",
    "RelToMarket_21d":     "Rel-to-Mkt (21d)",
    "RSI_14":              "RSI (14)",
    "BB_PctB":             "BB %B",
    "Return_21d":          "Return (21d)",
    "RelToMarket_5d":      "Rel-to-Mkt (5d)",
    "RelToMarket_1d":      "Rel-to-Mkt (1d)",
    "Return_5d":           "Return (5d)",
    "Volume_Ratio":        "Volume Ratio",
    "Market_Return_21d":   "Mkt Return (21d)",
    "Market_Return_5d":    "Mkt Return (5d)",
    "SectorRelReturn":     "Sector Rel Return",
    "Return_1d":           "Return (1d)",
    "Market_Return_1d":    "Mkt Return (1d)",
    "Sector_Return_1d":    "Sector Ret (1d)",
    "Sector_Return_5d":    "Sector Ret (5d)",
    "Sector_Return_21d":   "Sector Ret (21d)",
    "Sector_Vol_20d":      "Sector Vol (20d)",
    "Sector_Vol_60d":      "Sector Vol (60d)",
    "SectorRelZ_Return_1d":"Sector RelZ (1d)",
}

models = [
    ("LR",   "LR_coef",       "Logistic Regression", "| Coefficient |",        "#4878CF"),
    ("RF",   "RF_importance",  "Random Forest",        "Gini Importance",        "#6ACC65"),
    ("XGB",  "XGB_gain",       "XGBoost",              "Mean Gain",              "#D65F5F"),
    ("LSTM", "LSTM_B_perm",    "LSTM",                 "Permutation Importance", "#B47CC7"),
    ("TCN",  "TCN_perm",       "TCN",                  "Permutation Importance", "#C4AD66"),
]

universes = [
    ("large_cap", large_cap_data, "Large-Cap Universe"),
    ("small_cap", small_cap_data, "Small-Cap Universe"),
]

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

print("=" * 60)
print("SANITY CHECKS")
print("=" * 60)

for univ_key, data, univ_label in universes:
    print(f"\n[{univ_label}] {len(data['Feature'])} features loaded.")
    for model_key, col, model_title, _, _ in models:
        df = pd.DataFrame({"Feature": data["Feature"], "score": data[col]})
        top3 = df.nlargest(3, "score")
        print(f"  {model_title:22s} top-3: " + ", ".join(
            f"{display_names.get(r['Feature'], r['Feature'])} ({r['score']:.5f})"
            for _, r in top3.iterrows()
        ))

print()

# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------

mpl.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,   # controlled per-axis below
    "axes.axisbelow":    True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "grid.color":        "#888888",
})

# ---------------------------------------------------------------------------
# Figure generation — single 2×5 combined figure
# ---------------------------------------------------------------------------

output_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "figures"
)
os.makedirs(output_dir, exist_ok=True)

fig, axes = plt.subplots(
    2, 5,
    figsize=(15, 9.0),
)
# top margin for suptitle; left margin for row labels; generous hspace between rows
fig.subplots_adjust(left=0.10, right=0.99, top=0.93, bottom=0.07, wspace=0.55, hspace=0.52)

for row, (univ_key, data, univ_label) in enumerate(universes):
    for col_idx, (ax, (model_key, metric_col, title, xlabel, color)) in enumerate(
        zip(axes[row], models)
    ):
        df = pd.DataFrame({"Feature": data["Feature"], "score": data[metric_col]})
        top10 = df.nlargest(10, "score").sort_values("score", ascending=True)
        top10["Label"] = top10["Feature"].map(display_names)

        ax.barh(
            top10["Label"],
            top10["score"],
            color=color,
            edgecolor="none",
            alpha=1.0,
        )

        # Panel titles only on the top row to avoid repetition
        if row == 0:
            ax.set_title(title, fontsize=10, fontweight="bold", pad=6)

        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(axis="both", labelsize=8)
        ax.tick_params(axis="y", length=0)

        ax.xaxis.grid(True, alpha=0.3, linestyle="--", color="#888888")
        ax.yaxis.grid(False)
        ax.spines["left"].set_visible(False)
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))

    # Row label: rotated text to the left of the leftmost axis
    axes[row, 0].annotate(
        univ_label,
        xy=(0, 0.5),
        xycoords="axes fraction",
        xytext=(-0.72, 0.5),
        textcoords="axes fraction",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        rotation=90,
    )

fig.suptitle(
    "Figure F16 — Feature Importance Bar Charts (top-10, fold-averaged)",
    fontsize=11,
    fontweight="bold",
    y=0.97,
)

base_path = os.path.join(output_dir, "F16_feature_importance")
pdf_path = base_path + ".pdf"
png_path = base_path + ".png"

fig.savefig(pdf_path, format="pdf")
fig.savefig(png_path, format="png", dpi=300)
plt.close(fig)

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
print("\nDone.")
