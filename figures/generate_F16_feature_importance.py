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
# Data
# ---------------------------------------------------------------------------

large_cap_data = {
    "Feature": [
        "BB_PctB", "Beta_60d", "Market_Return_1d", "Market_Return_21d",
        "Market_Return_5d", "Market_Vol_20d", "Market_Vol_60d", "RSI_14",
        "RealVol_20d", "RelToMarket_1d", "RelToMarket_21d", "RelToMarket_5d",
        "Return_1d", "Return_21d", "Return_5d", "SectorRelReturn", "Volume_Ratio",
    ],
    "LR_coef":      [0.07974, 0.21162, 0.00899, 0.02553, 0.01207, 0.05695,
                     0.01510, 0.07530, 0.16727, 0.02659, 0.09323, 0.05572,
                     0.01466, 0.06505, 0.03809, 0.02793, 0.02618],
    "RF_importance":[0.03584, 0.22068, 0.01366, 0.04337, 0.02577, 0.07390,
                     0.10558, 0.05270, 0.11479, 0.01977, 0.08607, 0.04038,
                     0.01907, 0.06613, 0.03536, 0.01879, 0.02814],
    "XGB_gain":     [0.04425, 0.14007, 0.03067, 0.05883, 0.04835, 0.07097,
                     0.09391, 0.05649, 0.08600, 0.02951, 0.07931, 0.04936,
                     0.03161, 0.06652, 0.04625, 0.03062, 0.03726],
    "LSTM_B_perm":  [0.07495, 0.08132, 0.02697, 0.02126, 0.02098, 0.03470,
                     0.01284, 0.03790, 0.07238, 0.03961, 0.06561, 0.05610,
                     0.03224, 0.06164, 0.03035, 0.02930, 0.01881],
    "TCN_perm":     [0.07791, 0.07656, 0.02805, 0.03247, 0.02067, 0.02650,
                     0.01793, 0.09244, 0.06245, 0.04335, 0.05134, 0.04809,
                     0.02984, 0.07514, 0.03173, 0.01300, 0.04156],
}

small_cap_data = {
    "Feature": [
        "BB_PctB", "Beta_60d", "Market_Return_1d", "Market_Return_21d",
        "Market_Return_5d", "Market_Vol_20d", "Market_Vol_60d", "RSI_14",
        "RealVol_20d", "RelToMarket_1d", "RelToMarket_21d", "RelToMarket_5d",
        "Return_1d", "Return_21d", "Return_5d", "SectorRelReturn", "Volume_Ratio",
    ],
    "LR_coef":      [0.07661, 0.20066, 0.00819, 0.01584, 0.01581, 0.03980,
                     0.01377, 0.08004, 0.17870, 0.02414, 0.07057, 0.03540,
                     0.01812, 0.06120, 0.02936, 0.04975, 0.08205],
    "RF_importance":[0.03794, 0.18395, 0.01566, 0.05265, 0.02354, 0.07512,
                     0.09689, 0.05439, 0.14057, 0.02278, 0.07561, 0.03500,
                     0.02134, 0.07215, 0.03514, 0.02733, 0.02991],
    "XGB_gain":     [0.04491, 0.11211, 0.03439, 0.06585, 0.03805, 0.07888,
                     0.08630, 0.05649, 0.10242, 0.03909, 0.06733, 0.04458,
                     0.03651, 0.06518, 0.04740, 0.04201, 0.03849],
    "LSTM_B_perm":  [0.03027, 0.04777, 0.01623, 0.05395, 0.03314, 0.08434,
                     0.06354, 0.05754, 0.08216, 0.02482, 0.05933, 0.01851,
                     0.01225, 0.03715, 0.02391, 0.02528, 0.05174],
    "TCN_perm":     [0.04805, 0.08708, 0.03505, 0.04583, 0.08693, 0.01876,
                     0.02658, 0.01969, 0.11445, 0.01548, 0.04862, 0.03372,
                     0.02616, 0.05212, 0.04401, 0.02165, 0.03403],
}

display_names = {
    "Beta_60d":          "Beta (60d)",
    "RealVol_20d":       "Realized Vol (20d)",
    "Market_Vol_60d":    "Market Vol (60d)",
    "Market_Vol_20d":    "Market Vol (20d)",
    "RelToMarket_21d":   "Rel-to-Mkt (21d)",
    "RSI_14":            "RSI (14)",
    "BB_PctB":           "BB %B",
    "Return_21d":        "Return (21d)",
    "RelToMarket_5d":    "Rel-to-Mkt (5d)",
    "RelToMarket_1d":    "Rel-to-Mkt (1d)",
    "Return_5d":         "Return (5d)",
    "Volume_Ratio":      "Volume Ratio",
    "Market_Return_21d": "Mkt Return (21d)",
    "Market_Return_5d":  "Mkt Return (5d)",
    "SectorRelReturn":   "Sector Rel Return",
    "Return_1d":         "Return (1d)",
    "Market_Return_1d":  "Mkt Return (1d)",
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

EXPECTED_FEATURES = set(large_cap_data["Feature"])
print("=" * 60)
print("SANITY CHECKS")
print("=" * 60)

for univ_key, data, univ_label in universes:
    features_present = set(data["Feature"])
    assert features_present == EXPECTED_FEATURES, (
        f"[{univ_key}] Feature mismatch: {features_present.symmetric_difference(EXPECTED_FEATURES)}"
    )
    print(f"\n[{univ_label}] All {len(features_present)} features confirmed present.")
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
