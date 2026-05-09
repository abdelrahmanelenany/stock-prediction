"""
Generate Figure F15 — TCN Architecture Sweep.

2×2 grid: Row 1 = Large-Cap, Row 2 = Small-Cap
          Col 1 = Scatter (receptive field vs val Sharpe)
          Col 2 = Heatmap (kernel_size × num_levels, split by channel width)

Data is hardcoded — no CSV files are read.
"""

import os

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # used for style context
from matplotlib.colors import TwoSlopeNorm

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})

OUT_DIR = "outputs/figures"
os.makedirs(OUT_DIR, exist_ok=True)


# ── Hardcoded sweep data ──────────────────────────────────────────────────────

COLS = ["kernel_size", "num_levels", "num_channels", "receptive_field",
        "val_sharpe", "val_auc", "n_params"]

_LC = [
    (2, 4, 32,  31, -2.102, 0.500527,  16898),
    (2, 4, 64,  31, -1.333, 0.522513,  62466),
    (2, 5, 32,  63, -1.114, 0.498762,  21058),
    (2, 5, 64,  63, -0.487, 0.529887,  78978),
    (3, 4, 32,  61, -2.217, 0.540114,  24802),
    (3, 4, 64,  61, -0.256, 0.519940,  92610),
    (3, 5, 32, 125, -1.586, 0.523534,  31010),
    (3, 5, 64, 125, -1.473, 0.579356, 117314),
    (5, 3, 32,  57, -0.773, 0.522545,  30306),
    (5, 3, 64,  57, -2.050, 0.558688, 111810),
    (5, 4, 32, 121, -3.295, 0.543261,  40610),
    (5, 4, 64, 121,  0.185, 0.530830, 152898),   # SELECTED
    (5, 5, 32, 249, -2.709, 0.538039,  50914),
    (5, 5, 64, 249, -4.189, 0.535094, 193986),
]

_SC = [
    (2, 4, 32,  31, -4.254, 0.386096,  16898),
    (2, 4, 64,  31, -2.170, 0.469633,  62466),
    (2, 5, 32,  63, -6.563, 0.471428,  21058),
    (2, 5, 64,  63, -3.106, 0.477176,  78978),
    (3, 4, 32,  61, -4.473, 0.447754,  24802),
    (3, 4, 64,  61, -4.516, 0.473688,  92610),
    (3, 5, 32, 125, -2.427, 0.479024,  31010),
    (3, 5, 64, 125, -1.581, 0.489335, 117314),
    (5, 3, 32,  57, -2.621, 0.427924,  30306),
    (5, 3, 64,  57,  0.090, 0.546277, 111810),   # SELECTED
    (5, 4, 32, 121, -2.677, 0.458185,  40610),
    (5, 4, 64, 121, -0.588, 0.487320, 152898),
    (5, 5, 32, 249,  0.001, 0.455992,  50914),
    (5, 5, 64, 249, -4.087, 0.428816, 193986),
]

LC_DF = pd.DataFrame(_LC, columns=COLS)
SC_DF = pd.DataFrame(_SC, columns=COLS)

LC_SEL = LC_DF.loc[LC_DF["val_sharpe"].idxmax()]
SC_SEL = SC_DF.loc[SC_DF["val_sharpe"].idxmax()]

# ── Visual constants ──────────────────────────────────────────────────────────

KERNELS  = [2, 3, 5]
CHANNELS = [32, 64]
LEVELS   = [5, 4, 3]          # heatmap rows, top → bottom

KERNEL_COLORS = {2: "#4393c3", 3: "#e08048", 5: "#4dac26"}
CHAN_MARKER   = {32: "o", 64: "s"}
CHAN_SIZE     = {32: 70,  64: 100}
SEL_COLOR     = "#b2182b"

# Column index mapping for the 6-column heatmap grid
COL_IDX = {(k, c): ki * len(CHANNELS) + ci
           for ki, k in enumerate(KERNELS)
           for ci, c in enumerate(CHANNELS)}


# ── Panel A — Scatter ─────────────────────────────────────────────────────────

def plot_scatter(ax: plt.Axes, df: pd.DataFrame, sel: pd.Series,
                 universe_label: str) -> None:
    with sns.axes_style("whitegrid"):
        pass  # style applied globally; seaborn used for palette reference

    ax.axhline(0, color="grey", lw=0.9, ls="--", alpha=0.5, zorder=1)

    for kernel, kg in df.groupby("kernel_size"):
        for chan, cg in kg.groupby("num_channels"):
            ax.scatter(
                cg["receptive_field"], cg["val_sharpe"],
                color=KERNEL_COLORS[kernel],
                marker=CHAN_MARKER[chan],
                s=CHAN_SIZE[chan],
                edgecolors="white", linewidths=0.5,
                alpha=0.88, zorder=2,
            )

    # Selected star
    ax.scatter(
        sel["receptive_field"], sel["val_sharpe"],
        marker="*", s=320, color=SEL_COLOR,
        edgecolors="black", linewidths=0.6, zorder=5,
    )

    # Annotation offset: shift down when near the top, up otherwise
    y_range = df["val_sharpe"].max() - df["val_sharpe"].min()
    near_top = (sel["val_sharpe"] - df["val_sharpe"].min()) / y_range > 0.8
    yoff = -32 if near_top else 10
    ax.annotate(
        (f"k={int(sel['kernel_size'])}, L={int(sel['num_levels'])}, "
         f"c={int(sel['num_channels'])}\n"
         f"RF = {int(sel['receptive_field'])},  Sharpe = {sel['val_sharpe']:.3f}"),
        xy=(sel["receptive_field"], sel["val_sharpe"]),
        xytext=(10, yoff), textcoords="offset points",
        fontsize=7.5, color=SEL_COLOR, ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec=SEL_COLOR, alpha=0.92, lw=0.9),
        arrowprops=dict(arrowstyle="-", color=SEL_COLOR, lw=0.8),
    )

    # Legend: kernel colors + channel shapes + selected marker
    kern_handles = [
        mpatches.Patch(color=KERNEL_COLORS[k], label=f"Kernel = {k}")
        for k in KERNELS
    ]
    chan_handles = [
        plt.Line2D([0], [0], marker=CHAN_MARKER[c], color="grey",
                   markersize=7, ls="None", label=f"Channels = {c}")
        for c in CHANNELS
    ]
    sel_handle = plt.Line2D([0], [0], marker="*", color=SEL_COLOR,
                             markersize=11, ls="None", label="Selected")
    ax.legend(
        handles=kern_handles + chan_handles + [sel_handle],
        fontsize=7, loc="lower right", framealpha=0.90,
        edgecolor="#cccccc", ncol=2,
    )

    ax.set_xlabel("Receptive Field (time steps)")
    ax.set_ylabel("Validation Sharpe")
    ax.set_title(f"Receptive Field vs Validation Sharpe\n{universe_label}", pad=5)
    ax.grid(True, lw=0.4, alpha=0.35)
    ax.set_xlim(-5, 275)


# ── Panel B — Heatmap ────────────────────────────────────────────────────────

def plot_heatmap(ax: plt.Axes, df: pd.DataFrame, sel: pd.Series,
                 universe_label: str, fig: plt.Figure) -> None:
    """
    Rows = num_levels  [5, 4, 3]  top → bottom
    Cols = (k=2,c=32) (k=2,c=64) | (k=3,c=32) (k=3,c=64) | (k=5,c=32) (k=5,c=64)
    Missing cells (k=2/3 at L=3) shown as light gray.
    """
    n_rows = len(LEVELS)
    n_cols = len(KERNELS) * len(CHANNELS)   # 6

    grid = np.full((n_rows, n_cols), np.nan)
    for _, row in df.iterrows():
        lv = int(row["num_levels"])
        if lv in LEVELS:
            ri = LEVELS.index(lv)
            ci = COL_IDX[(int(row["kernel_size"]), int(row["num_channels"]))]
            grid[ri, ci] = row["val_sharpe"]

    # Diverging norm centred at 0
    vmin, vmax = float(np.nanmin(grid)), float(np.nanmax(grid))
    norm = TwoSlopeNorm(vmin=min(vmin, -0.01), vcenter=0, vmax=max(vmax, 0.01))

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#e4e4e4")

    masked = np.ma.array(grid, mask=np.isnan(grid))
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")

    # Colorbar
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Val Sharpe", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # Y-axis
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"L = {lv}" for lv in LEVELS])
    ax.set_ylabel("Num Levels")

    # X-axis — two-line labels "k=K\nc=C"
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [f"k={k}\nc={c}" for k in KERNELS for c in CHANNELS],
        fontsize=7.5, linespacing=1.4,
    )
    ax.tick_params(axis="x", length=0)   # hide tick marks; labels are descriptive

    # Vertical separators between kernel groups
    for sep_x in [1.5, 3.5]:
        ax.axvline(sep_x, color="white", lw=2.5, zorder=3)

    ax.set_title(f"Validation Sharpe: Kernel × Levels × Channels\n{universe_label}", pad=5)

    # Cell annotations
    for ri in range(n_rows):
        for ci in range(n_cols):
            val = grid[ri, ci]
            if np.isnan(val):
                ax.text(ci, ri, "n/a", ha="center", va="center",
                        fontsize=7.5, color="#bbbbbb", style="italic")
            else:
                normed_v = float(norm(val))
                tc = "white" if (normed_v < 0.28 or normed_v > 0.72) else "black"
                ax.text(ci, ri, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, fontweight="bold", color=tc)

    # Selected cell — double-rectangle border (white inner, black outer)
    sel_ri = LEVELS.index(int(sel["num_levels"]))
    sel_ci = COL_IDX[(int(sel["kernel_size"]), int(sel["num_channels"]))]
    for lw, ec, zo in [(2.8, "white", 5), (1.0, "black", 4)]:
        ax.add_patch(mpatches.Rectangle(
            (sel_ci - 0.48, sel_ri - 0.48), 0.96, 0.96,
            linewidth=lw, edgecolor=ec, facecolor="none", zorder=zo,
        ))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    datasets = [
        ("Large-Cap Universe", LC_DF, LC_SEL),
        ("Small-Cap Universe", SC_DF, SC_SEL),
    ]
    row_labels = ["Large-Cap", "Small-Cap"]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 9),
        gridspec_kw={"wspace": 0.52, "hspace": 0.72},
    )

    fig.suptitle(
        "Figure F15 — TCN Architecture Sweep",
        fontsize=13, fontweight="bold", y=0.99,
    )

    for row, (label, df, sel) in enumerate(datasets):
        # Rotated row label on the far left
        fig.text(
            0.01, 0.75 - row * 0.50,
            row_labels[row],
            va="center", ha="left",
            fontsize=11, fontweight="bold", rotation=90,
        )
        plot_scatter(axes[row, 0], df, sel, label)
        plot_heatmap(axes[row, 1], df, sel, label, fig)

    # Small unobtrusive figure label
    fig.text(0.015, 0.012, "F15", fontsize=8, color="#cccccc",
             va="bottom", ha="left")

    pdf_path = os.path.join(OUT_DIR, "F15_tcn_arch_sweep.pdf")
    png_path = os.path.join(OUT_DIR, "F15_tcn_arch_sweep.png")

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    print(f"Saved: {pdf_path}")

    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    print(f"Saved: {png_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
