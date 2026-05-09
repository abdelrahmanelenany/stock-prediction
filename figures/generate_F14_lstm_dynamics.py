"""
Generate Figure F14 — LSTM Training Dynamics and Hyperparameter Sweep.

2×2 grid:  Row 1 = Large-Cap, Row 2 = Small-Cap
           Col 1 = Training curves (chosen config)
           Col 2 = Val-Sharpe heatmap (LR × Hidden Size)
"""

import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
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

UNIVERSES = ["large_cap", "Small-Cap"]
UNIVERSE_LABELS = {"large_cap": "Large-Cap", "small_cap": "Small-Cap"}
TRAINING_LOG_DIR = "reports/training_logs"
REPORTS_DIR = "reports"
OUT_DIR = "outputs/figures"

os.makedirs(OUT_DIR, exist_ok=True)


# ─── helpers ────────────────────────────────────────────────────────────────

def load_training_log(universe: str) -> pd.DataFrame:
    """Load the sweep-best training log for a universe."""
    # Prefer the dedicated sweep-best file
    candidate = os.path.join(TRAINING_LOG_DIR, f"{universe}_lstm_sweep_best_training_log.csv")
    if os.path.isfile(candidate):
        df = pd.read_csv(candidate)
        if df.empty:
            raise FileNotFoundError(f"Training log exists but is empty: {candidate}")
        return df

    # Fall back: glob for any lstm CSV with the universe prefix
    pattern = os.path.join(TRAINING_LOG_DIR, f"{universe}*lstm*.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No LSTM training log found for universe '{universe}'. "
            f"Expected: {candidate}"
        )
    df = pd.read_csv(sorted(matches)[0])
    if df.empty:
        raise FileNotFoundError(f"Training log is empty: {matches[0]}")
    return df


def load_tuning_results(universe: str) -> pd.DataFrame:
    path = os.path.join(REPORTS_DIR, f"{universe}_lstm_tuning_results.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Tuning results not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise FileNotFoundError(f"Tuning results file is empty: {path}")
    return df


def best_config(tuning_df: pd.DataFrame) -> pd.Series:
    return tuning_df.loc[tuning_df["val_sharpe"].idxmax()]


# ─── panel A: training curves ───────────────────────────────────────────────

def plot_training_curves(ax: plt.Axes, log_df: pd.DataFrame, cfg: pd.Series,
                         universe_label: str) -> None:
    epochs = log_df["epoch"].values

    # Checkpoint epoch = epoch with minimum validation loss
    ckpt_epoch = int(log_df.loc[log_df["val_loss"].idxmin(), "epoch"])

    # Loss (left y-axis)
    color_train_loss = "#2166ac"
    color_val_loss = "#d6604d"
    color_train_auc = "#4dac26"
    color_val_auc = "#b8008a"

    lns = []
    lns += ax.plot(epochs, log_df["train_loss"], color=color_train_loss,
                   lw=1.6, label="Train Loss", zorder=3)
    lns += ax.plot(epochs, log_df["val_loss"], color=color_val_loss,
                   lw=1.6, ls="--", label="Val Loss", zorder=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss", color="black")
    ax.tick_params(axis="y")

    # AUC (right y-axis)
    ax2 = ax.twinx()
    if "train_auc" in log_df.columns:
        lns += ax2.plot(epochs, log_df["train_auc"], color=color_train_auc,
                        lw=1.4, ls=(0, (3, 1, 1, 1)), label="Train AUC", zorder=2)
    if "val_auc" in log_df.columns:
        lns += ax2.plot(epochs, log_df["val_auc"], color=color_val_auc,
                        lw=1.4, ls=":", label="Val AUC", zorder=2)
    ax2.set_ylabel("AUC", color="black")
    ax2.tick_params(axis="y")
    ax2.set_ylim(0.4, 0.85)

    # Vertical dashed line at checkpoint epoch
    ax.axvline(ckpt_epoch, color="black", lw=1.2, ls=(0, (5, 3)), zorder=4)
    # Use axes-fraction y so the label is always at the top regardless of y-scale
    ax.text(ckpt_epoch + 0.3, 1.0,
            f"ckpt ep {ckpt_epoch}",
            transform=ax.get_xaxis_transform(),   # x in data units, y in [0,1]
            fontsize=7, va="top", ha="left", color="black",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="grey",
                      alpha=0.85, lw=0.6))

    # Legend (combined, excluding the axvline from legend entries)
    labels = [l.get_label() for l in lns]
    ax.legend(lns, labels, loc="upper right", framealpha=0.85,
              edgecolor="grey", ncol=2)

    ax.set_title("LSTM Training Curves — Chosen Configuration", pad=6)

    # Annotation box with chosen config
    info = (f"lr = {cfg['lr']:.4f},  "
            f"hidden = {int(cfg['hidden_size'])},  "
            f"epochs = {int(cfg['n_epochs_trained'])}\n"
            f"val Sharpe = {cfg['val_sharpe']:.3f},  "
            f"val AUC = {cfg['val_auc']:.3f}")
    ax.text(0.02, 0.97, info, transform=ax.transAxes,
            fontsize=7.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.85))

    ax.grid(True, lw=0.4, alpha=0.5)
    # Add a half-epoch margin so a line at epoch[0] is never flush with the spine
    ax.set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)


# ─── panel B: heatmap ───────────────────────────────────────────────────────

def plot_heatmap(ax: plt.Axes, tuning_df: pd.DataFrame, cfg: pd.Series,
                 universe_label: str, fig: plt.Figure) -> None:
    # Pivot: rows = lr (sorted descending so highest LR is at bottom),
    #         cols = hidden_size (ascending)
    pivot = tuning_df.pivot(index="lr", columns="hidden_size", values="val_sharpe")
    pivot = pivot.sort_index(ascending=False)          # high LR at top
    hidden_sizes = sorted(pivot.columns.tolist())
    lrs = pivot.index.tolist()

    data = pivot[hidden_sizes].values

    # Diverging colormap centred at 0
    vmin, vmax = data.min(), data.max()
    # Ensure centre is 0; if all negative or all positive, just centre at 0
    norm = TwoSlopeNorm(vmin=min(vmin, -0.01), vcenter=0, vmax=max(vmax, 0.01))

    im = ax.imshow(data, cmap="RdYlGn", norm=norm, aspect="auto")

    # Colourbar
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Val Sharpe", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # Axis ticks
    ax.set_xticks(range(len(hidden_sizes)))
    ax.set_xticklabels([str(h) for h in hidden_sizes])
    ax.set_yticks(range(len(lrs)))
    ax.set_yticklabels([f"{lr:.4f}" for lr in lrs])
    ax.set_xlabel("Hidden Size")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Validation Sharpe — LR × Hidden Size Grid", pad=6)

    # Annotate cells
    for ri, lr in enumerate(lrs):
        for ci, hs in enumerate(hidden_sizes):
            val = data[ri, ci]
            text_color = "black" if abs(norm(val) - 0.5) < 0.35 else "white"
            ax.text(ci, ri, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=text_color, fontweight="bold")

    # Mark best config with a thick white border rectangle (avoids overlapping value text)
    best_lr = cfg["lr"]
    best_hs = int(cfg["hidden_size"])
    try:
        ri_best = lrs.index(best_lr)
        ci_best = hidden_sizes.index(best_hs)
        rect = mpatches.Rectangle(
            (ci_best - 0.48, ri_best - 0.48), 0.96, 0.96,
            linewidth=2.8, edgecolor="white", facecolor="none", zorder=5,
        )
        ax.add_patch(rect)
        # Thin black outer ring so the white border is visible on light cells
        rect_outer = mpatches.Rectangle(
            (ci_best - 0.50, ri_best - 0.50), 1.0, 1.0,
            linewidth=1.0, edgecolor="black", facecolor="none", zorder=4,
        )
        ax.add_patch(rect_outer)
    except ValueError:
        pass  # best config not in grid — shouldn't happen


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    universes = ["large_cap", "small_cap"]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 9),
        gridspec_kw={"wspace": 0.45, "hspace": 0.55},
    )

    fig.suptitle(
        "Figure F14 — LSTM Training Dynamics and Hyperparameter Sweep",
        fontsize=13, fontweight="bold", y=0.98,
    )

    row_labels = ["Large-Cap", "Small-Cap"]

    for row, universe in enumerate(universes):
        label = UNIVERSE_LABELS[universe]

        # Load data
        log_df = load_training_log(universe)
        tuning_df = load_tuning_results(universe)
        cfg = best_config(tuning_df)

        # Row label on the left
        fig.text(
            0.01,
            0.75 - row * 0.50,
            row_labels[row],
            va="center", ha="left",
            fontsize=11, fontweight="bold",
            rotation=90,
        )

        # Panel A — training curves
        ax_left = axes[row, 0]
        plot_training_curves(ax_left, log_df, cfg, label)

        # Panel B — heatmap
        ax_right = axes[row, 1]
        plot_heatmap(ax_right, tuning_df, cfg, label, fig)

    # Save
    pdf_path = os.path.join(OUT_DIR, "F14_lstm_training_dynamics.pdf")
    png_path = os.path.join(OUT_DIR, "F14_lstm_training_dynamics.png")

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    print(f"Saved: {pdf_path}")

    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    print(f"Saved: {png_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
