# Implementation Extensions — Bhandari et al. (2022)
## "Predicting Stock Market Index Using LSTM"

These extensions adapt selected methodology from Bhandari, Rimal, Pokhrel et al. (2022)
(*Machine Learning with Applications*, 9, 100320) into the existing pipeline defined in
`CLAUDE.md`. Each section maps directly to the paper's sections 3.3, 4.1/4.3, 4.4, and
4.5. **All adaptations respect the walk-forward, anti-leakage discipline already
established in `CLAUDE.md`.**

> **Important framing:** The Bhandari paper solves a *regression* problem (predicting the
> next-day S&P 500 closing price). This project solves a *binary classification* problem
> (predicting cross-sectional outperformance). Every adaptation below is explicitly
> re-framed for classification. Prediction quality metrics are already implemented in the
> existing project and are not duplicated here.

---

## Table of Contents
1. [LSTM Hyperparameter Tuning Loops (§3.3)](#1-lstm-hyperparameter-tuning-loops-section-33)
2. [Extended Technical Indicators (§4.3)](#2-extended-technical-indicators-section-43)
3. [Correlation Heatmap & Feature Selection (§4.4)](#3-correlation-heatmap--feature-selection-section-44)
4. [Wavelet Denoising (§4.5)](#4-wavelet-denoising-section-45)
5. [Normalization Choice (§4.5)](#5-normalization-choice-section-45)
6. [Remove Unused Features](#6-remove-unused-features)
7. [Per-Model Feature Sets: New LSTM-A and Unified Baseline Features](#7-per-model-feature-sets-new-lstm-a-and-unified-baseline-features)

---

## 1. LSTM Hyperparameter Tuning Loops (Section 3.3)

### What the paper does
Algorithm 1 in §3.3 runs nested loops over three hyperparameter axes — optimizer, learning
rate, and batch size — executing each combination `n_replicates` times and averaging the
validation-set RMSE. The best combination then feeds Algorithm 2, which loops over
architectural options (layer / neuron counts) for the full-scale training run.

### Adaptation for this project
Replace RMSE with **validation AUC** (area under the ROC curve) as the selection criterion,
since this is a classification task. The loop structure is otherwise identical. The tuning
must be performed **inside each walk-forward fold**, fitting only on `df_train` and
evaluating on `df_val` — never touching `df_test`.

### Files to modify
- `config.py` — add the hyperparameter search grid
- `models/lstm_model.py` — add `tune_lstm_hyperparams()` function
- `pipeline.py` — call the tuner before the main training loop per fold

### Config additions (`config.py`)
```python
# ── LSTM Hyperparameter Search Grid (Bhandari §3.3) ──────────────────────────
# Shared by both LSTM-A and LSTM-B for the training hyperparameter search.
LSTM_HYPERPARAM_GRID = {
    "optimizer":      ["adam", "adagrad", "nadam"],   # paper tests these three
    "learning_rate":  [0.1, 0.01, 0.001],             # paper tests these three
    "batch_size":     [32, 64, 128],                  # scaled up from paper's 4/8/16
                                                       # (paper used index data; we
                                                       # have more obs per fold)
}
LSTM_TUNE_REPLICATES = 3      # paper uses 10; 3 is feasible on M4 for a thesis
LSTM_TUNE_PATIENCE   = 5      # early stopping patience during tuning (paper §3.3)
LSTM_TUNE_MAX_EPOCHS = 50     # cap tuning runs; full training uses config.MAX_EPOCHS

# ── LSTM-A Architecture Search Grid (Bhandari §3.3 Algorithm 2 analogue) ─────
# LSTM-A's architecture (hidden_size, num_layers, dropout) is also data-driven,
# not fixed. This grid is swept jointly with LSTM_HYPERPARAM_GRID for LSTM-A only.
# LSTM-B keeps its fixed architecture (64 units, 2 layers, dropout 0.2) as
# established in the existing pipeline.
LSTM_A_ARCH_GRID = {
    "hidden_size": [16, 32, 64],   # small range appropriate for 4-feature input
    "num_layers":  [1, 2],         # Bhandari §5.5 found single-layer often wins
    "dropout":     [0.1, 0.2],
}
```

### New function in `models/lstm_model.py`
```python
import itertools
from sklearn.metrics import roc_auc_score
import torch, torch.nn as nn
from torch.utils.data import DataLoader

def tune_lstm_hyperparams(
    df_train, df_val, feature_cols, target_col, seq_len, device, cfg,
    arch_grid=None,
):
    """
    Bhandari §3.3 Algorithm 1 + Algorithm 2 — adapted for classification
    (AUC replaces RMSE).

    Phase 1 (always): sweeps (optimizer, lr, batch_size) from cfg.LSTM_HYPERPARAM_GRID.
    Phase 2 (when arch_grid is provided): sweeps (hidden_size, num_layers, dropout)
             from arch_grid, using the best Phase-1 hyperparameters as fixed context.
             This mirrors Bhandari Algorithm 2, which tunes architecture after fixing
             the training hyperparameters.

    LSTM-A calls this function with arch_grid=cfg.LSTM_A_ARCH_GRID.
    LSTM-B calls this function with arch_grid=None (architecture stays fixed).

    ANTI-LEAKAGE: Pass already-scaled DataFrames (scaler fit on train only).

    Returns
    -------
    dict with keys: optimizer, lr, batch_size, hidden_size, num_layers, dropout
    """
    from models.lstm_model import StockLSTM, StockSequenceDataset  # local import

    # ── Phase 1: tune training hyperparameters ────────────────────────────────
    grid   = cfg.LSTM_HYPERPARAM_GRID
    combos = list(itertools.product(
        grid["optimizer"], grid["learning_rate"], grid["batch_size"]
    ))

    # For Phase 1, use a fixed architecture seed:
    # if arch_grid provided use its first entry; else use the cfg defaults.
    if arch_grid is not None:
        seed_hidden = arch_grid["hidden_size"][0]
        seed_layers = arch_grid["num_layers"][0]
        seed_drop   = arch_grid["dropout"][0]
    else:
        seed_hidden = cfg.LSTM_B_HIDDEN_SIZE
        seed_layers = cfg.LSTM_B_NUM_LAYERS
        seed_drop   = cfg.LSTM_B_DROPOUT

    print(f"[LSTM Tuning — Phase 1] {len(combos)} training combos × "
          f"{cfg.LSTM_TUNE_REPLICATES} replicates")

    phase1_results = []
    for opt_name, lr, bs in combos:
        auc_scores = _run_replicates(
            df_train, df_val, feature_cols, target_col, seq_len, device, cfg,
            opt_name, lr, bs,
            hidden_size=seed_hidden, num_layers=seed_layers, dropout=seed_drop,
        )
        avg_auc = sum(auc_scores) / len(auc_scores)
        phase1_results.append({
            "optimizer": opt_name, "lr": lr, "batch_size": bs, "avg_val_auc": avg_auc
        })
        print(f"  opt={opt_name:7s}  lr={lr:.4f}  bs={bs:3d}  → avg AUC={avg_auc:.4f}")

    best_p1 = max(phase1_results, key=lambda x: x["avg_val_auc"])
    print(f"[Phase 1 best] {best_p1}")

    # If no arch grid, return Phase 1 results with fixed architecture
    if arch_grid is None:
        return {
            **best_p1,
            "hidden_size": seed_hidden,
            "num_layers":  seed_layers,
            "dropout":     seed_drop,
        }

    # ── Phase 2: tune architecture (LSTM-A only) ──────────────────────────────
    arch_combos = list(itertools.product(
        arch_grid["hidden_size"], arch_grid["num_layers"], arch_grid["dropout"]
    ))
    print(f"\n[LSTM Tuning — Phase 2] {len(arch_combos)} architecture combos × "
          f"{cfg.LSTM_TUNE_REPLICATES} replicates")

    phase2_results = []
    for hidden_size, num_layers, dropout in arch_combos:
        auc_scores = _run_replicates(
            df_train, df_val, feature_cols, target_col, seq_len, device, cfg,
            best_p1["optimizer"], best_p1["lr"], best_p1["batch_size"],
            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
        )
        avg_auc = sum(auc_scores) / len(auc_scores)
        phase2_results.append({
            "hidden_size": hidden_size, "num_layers": num_layers,
            "dropout": dropout, "avg_val_auc": avg_auc
        })
        print(f"  h={hidden_size:3d}  layers={num_layers}  drop={dropout:.1f}"
              f"  → avg AUC={avg_auc:.4f}")

    best_p2 = max(phase2_results, key=lambda x: x["avg_val_auc"])
    print(f"[Phase 2 best] {best_p2}")

    return {
        "optimizer":   best_p1["optimizer"],
        "lr":          best_p1["lr"],
        "batch_size":  best_p1["batch_size"],
        "hidden_size": best_p2["hidden_size"],
        "num_layers":  best_p2["num_layers"],
        "dropout":     best_p2["dropout"],
    }


def _run_replicates(
    df_train, df_val, feature_cols, target_col, seq_len, device, cfg,
    opt_name, lr, bs, hidden_size, num_layers, dropout,
):
    """
    Run cfg.LSTM_TUNE_REPLICATES independent training runs for one hyperparameter
    combination. Returns a list of validation AUC scores (one per replicate).
    """
    from models.lstm_model import StockLSTM, StockSequenceDataset

    train_ds = StockSequenceDataset(df_train, feature_cols, target_col, seq_len)
    val_ds   = StockSequenceDataset(df_val,   feature_cols, target_col, seq_len)

    auc_scores = []
    for _ in range(cfg.LSTM_TUNE_REPLICATES):
        model = StockLSTM(
            input_size=len(feature_cols),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)

        optimizer = _build_optimizer(model, opt_name, lr)
        criterion = nn.BCELoss()
        train_dl  = DataLoader(train_ds, batch_size=bs, shuffle=True)
        val_dl    = DataLoader(val_ds,   batch_size=bs * 2, shuffle=False)

        best_val_loss, patience_ctr = float("inf"), 0
        for epoch in range(cfg.LSTM_TUNE_MAX_EPOCHS):
            model.train()
            for Xb, yb in train_dl:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for Xb, yb in val_dl:
                    val_loss += criterion(model(Xb.to(device)), yb.to(device)).item()
            val_loss /= max(len(val_dl), 1)

            if val_loss < best_val_loss:
                best_val_loss, patience_ctr = val_loss, 0
            else:
                patience_ctr += 1
                if patience_ctr >= cfg.LSTM_TUNE_PATIENCE:
                    break

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for Xb, yb in val_dl:
                all_preds.extend(model(Xb.to(device)).cpu().numpy())
                all_labels.extend(yb.numpy())

        auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
        auc_scores.append(auc)

    return auc_scores


def _build_optimizer(model, name: str, lr: float):
    """Helper: instantiate optimizer by name string."""
    params = model.parameters()
    if   name == "adam":    return torch.optim.Adam(params, lr=lr)
    elif name == "adagrad": return torch.optim.Adagrad(params, lr=lr)
    elif name == "nadam":   return torch.optim.NAdam(params, lr=lr)
    else: raise ValueError(f"Unknown optimizer: {name}")
```

### Integration in `pipeline.py` (per-fold, inside the walk-forward loop)
```python
# LSTM-B: tune only training hyperparameters (architecture is fixed)
best_hp_b = tune_lstm_hyperparams(
    df_tr_scaled_b, df_val_scaled_b, cfg.LSTM_B_FEATURE_COLS,
    cfg.TARGET_COL, cfg.SEQ_LEN, device, cfg,
    arch_grid=None,   # architecture not swept for LSTM-B
)

# LSTM-A: tune training hyperparameters AND architecture jointly
best_hp_a = tune_lstm_hyperparams(
    df_tr_scaled_a, df_val_scaled_a, cfg.LSTM_A_FEATURE_COLS,
    cfg.TARGET_COL, cfg.SEQ_LEN, device, cfg,
    arch_grid=cfg.LSTM_A_ARCH_GRID,   # architecture is data-driven for LSTM-A
)

# best_hp_a now contains: optimizer, lr, batch_size, hidden_size, num_layers, dropout
# Use these to instantiate and train LSTM-A (see Section 7.5 for full pipeline code)
```

> **Thesis note:** For LSTM-A, report a two-part tuning table: (a) the Phase 1 table of
> all 27 optimizer × lr × batch_size combinations with their avg val AUC, and (b) the
> Phase 2 table of all 12 architecture combinations (3 hidden_sizes × 2 num_layers ×
> 2 dropouts) with their avg val AUC. This directly mirrors Bhandari Tables 5–7.

---

## 2. Extended Technical Indicators (Section 4.3)

### What the paper does
Bhandari §4.3 uses three specific technical indicators: **MACD** (12/26 EMA difference),
**ATR-14** (14-day average true range), and **RSI-14** (14-day relative strength index).
Explicit formulae and interpretations are given for each.

### Adaptation for this project
The project already includes RSI, MACD, ATR, and Bollinger Bands in `feature_engineering.py`.
Verify (and add if missing) that MACD, ATR-14, and RSI-14 match the exact parameters from
the paper. Additionally, add the **MACD signal line** (9-day EMA of MACD) and **MACD
histogram** (MACD − signal), which are standard extensions used in trading practice.

### Files to modify
- `feature_engineering.py` — verify/add indicator parameter alignment

### Verified indicator parameters (`feature_engineering.py`)
```python
import pandas_ta as ta   # or use manual pandas calculations

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators as specified in Bhandari §4.3.
    All indicators are computed per-ticker BEFORE any train/test split.
    Resulting NaN rows (warm-up period) are dropped at the end.

    ANTI-LEAKAGE: These are causal indicators — each value at time t
    uses only data from t and earlier. No future leakage.
    """
    df = df.sort_values(["Ticker", "Date"]).copy()

    for ticker, grp in df.groupby("Ticker"):
        idx = grp.index

        # ── RSI-14 (Bhandari §4.3 — Wilder 1978) ─────────────────────
        df.loc[idx, "RSI_14"] = ta.rsi(grp["Close"], length=14)

        # ── ATR-14 (Bhandari §4.3 — Wilder 1978) ─────────────────────
        df.loc[idx, "ATR_14"] = ta.atr(
            grp["High"], grp["Low"], grp["Close"], length=14
        )

        # ── MACD (Bhandari §4.3 — Appel; EMA12 − EMA26) ──────────────
        macd_df = ta.macd(grp["Close"], fast=12, slow=26, signal=9)
        df.loc[idx, "MACD"]        = macd_df["MACD_12_26_9"]
        df.loc[idx, "MACD_Signal"] = macd_df["MACDs_12_26_9"]   # signal line
        df.loc[idx, "MACD_Hist"]   = macd_df["MACDh_12_26_9"]   # histogram

        # ── Bollinger Bands (keep existing) ───────────────────────────
        bb = ta.bbands(grp["Close"], length=20, std=2)
        df.loc[idx, "BB_Upper"]  = bb["BBU_20_2.0"]
        df.loc[idx, "BB_Middle"] = bb["BBM_20_2.0"]
        df.loc[idx, "BB_Lower"]  = bb["BBL_20_2.0"]
        df.loc[idx, "BB_Width"]  = (
            (bb["BBU_20_2.0"] - bb["BBL_20_2.0"]) / bb["BBM_20_2.0"]
        )

    # Drop warm-up NaN rows (first 26 days per ticker for MACD)
    df.dropna(subset=["MACD", "RSI_14", "ATR_14"], inplace=True)
    return df
```

> **Feature list update:** With these additions the technical indicator block grows from
> ~4 features to ~8 (RSI_14, ATR_14, MACD, MACD_Signal, MACD_Hist, BB_Upper, BB_Lower,
> BB_Width). Update `config.py → TECHNICAL_FEATURE_COLS` accordingly. Update
> `StockLSTM(input_size=...)` to match.

---

## 3. Correlation Heatmap & Feature Selection (Section 4.4)

### What the paper does
Bhandari §4.4 builds a Pearson correlation matrix of all input features, visualises it as a
heatmap, and removes any feature pair with |r| > 0.80 (keeping one of the two). In the
paper, `Open` price is dropped because it is redundant with `Close` (r ≈ 1.0).

### Adaptation for this project
Apply the same strategy to the full feature set (~41 features after Section 2 above).
Because features are computed per-ticker and pooled across all tickers and all training-fold
dates, compute the correlation on the **training portion of Fold 1 only** for the
exploratory/thesis analysis. Then hard-code the resulting reduced feature list into
`config.py` so it is applied consistently across all folds.

> **Do NOT re-run correlation analysis inside each fold** — that would constitute a
> form of look-ahead if the selected features differ by fold. Run it once on Fold 1
> training data, document the result, and fix the feature list.

### Files to create / modify
- `analysis/feature_correlation.py` — new script (runs once, generates the plot)
- `config.py` — add `FEATURE_COLS_AFTER_SELECTION` list

### `analysis/feature_correlation.py`
```python
"""
Bhandari §4.4 — Correlation heatmap and redundant feature removal.

Run ONCE on Fold 1 training data. Output:
  outputs/figures/feature_correlation_heatmap.png
  outputs/feature_selection_log.txt

Usage:
    python analysis/feature_correlation.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

CORR_THRESHOLD = 0.80   # Bhandari §4.4 threshold


def compute_and_plot_heatmap(df_train: pd.DataFrame, feature_cols: list,
                              output_dir: str = "outputs/figures") -> list:
    """
    1. Compute Pearson correlation matrix on df_train[feature_cols].
    2. Plot and save a heatmap (Figure 4 equivalent from the paper).
    3. Identify and remove redundant features (|r| > CORR_THRESHOLD).
    4. Return the reduced feature list.
    """
    os.makedirs(output_dir, exist_ok=True)

    corr = df_train[feature_cols].corr(method="pearson")

    # ── Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(12, len(feature_cols) * 0.5),
                                    max(10, len(feature_cols) * 0.45)))
    mask = np.triu(np.ones_like(corr, dtype=bool))   # upper triangle mask
    sns.heatmap(
        corr,
        mask=mask,
        annot=True if len(feature_cols) <= 20 else False,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.3,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(
        "Pearson Correlation Heatmap — Training Features\n"
        "(Bhandari et al. §4.4 — Fold 1 Training Data)",
        fontsize=13
    )
    plt.tight_layout()
    fig.savefig(f"{output_dir}/feature_correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"[Feature Selection] Heatmap saved → {output_dir}/feature_correlation_heatmap.png")

    # ── Remove redundant features (|r| > threshold) ───────────────────
    # Greedy: traverse upper triangle; if |r_ij| > threshold, drop column j.
    to_drop = set()
    cols = list(corr.columns)
    for i in range(len(cols)):
        if cols[i] in to_drop:
            continue
        for j in range(i + 1, len(cols)):
            if cols[j] in to_drop:
                continue
            if abs(corr.iloc[i, j]) > CORR_THRESHOLD:
                to_drop.add(cols[j])
                print(f"  Drop '{cols[j]}' — |r| = {corr.iloc[i, j]:.3f} "
                      f"with '{cols[i]}'")

    reduced = [c for c in feature_cols if c not in to_drop]

    # ── Log ───────────────────────────────────────────────────────────
    log_path = "outputs/feature_selection_log.txt"
    os.makedirs("outputs", exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"Correlation threshold: {CORR_THRESHOLD}\n")
        f.write(f"Original features ({len(feature_cols)}): {feature_cols}\n\n")
        f.write(f"Dropped features ({len(to_drop)}): {sorted(to_drop)}\n\n")
        f.write(f"Retained features ({len(reduced)}): {reduced}\n")
    print(f"[Feature Selection] {len(to_drop)} features dropped, "
          f"{len(reduced)} retained. Log → {log_path}")

    return reduced


if __name__ == "__main__":
    # Load Fold 1 training data from the pipeline's output cache
    # Adjust path as needed
    from config import FEATURE_COLS, DATA_PATH
    df = pd.read_parquet(DATA_PATH)
    # Use first 500 trading days as a proxy for Fold 1 training data
    fold1_dates = sorted(df["Date"].unique())[:500]
    df_fold1_train = df[df["Date"].isin(fold1_dates)]

    reduced_features = compute_and_plot_heatmap(df_fold1_train, FEATURE_COLS)
    print("\nCopy this list into config.py → FEATURE_COLS_AFTER_SELECTION:")
    print(reduced_features)
```

### Config update (`config.py`)
```python
# After running analysis/feature_correlation.py, paste the output list here:
FEATURE_COLS_AFTER_SELECTION = [
    # ... paste output of feature_correlation.py here ...
    # Leave as None to use all FEATURE_COLS (before selection is run)
]

# Pipeline helper — returns whichever list is populated:
def get_active_feature_cols():
    if FEATURE_COLS_AFTER_SELECTION:
        return FEATURE_COLS_AFTER_SELECTION
    return FEATURE_COLS
```

> **Thesis figure:** The saved heatmap becomes a direct thesis figure ("Figure X —
> Pearson correlation heatmap of input features, training data"). Caption it analogously
> to Bhandari Figure 4.

---

## 4. Wavelet Denoising (Section 4.5)

### What the paper does
Bhandari §4.5 applies **soft-mode Haar wavelet denoising** (via `scikit-image`) to the
raw closing price series of the index before using it as a model feature. The motivation
is that stock price data is "noisy and in sequential discrete format", and Haar wavelets
are described as "most suitable and popular in stock price data" (citing Ortega &
Khashanah, 2014).

### Adaptation for this project
In this project, features are derived quantities (returns, technical indicators) rather
than raw prices. The correct adaptation is to apply wavelet denoising to the **raw Close
price series per ticker BEFORE computing any derived features** (returns, RSI, MACD, ATR).
This preserves the spirit of §4.5 while fitting the project's feature engineering pipeline.

> **Anti-leakage note:** Wavelet denoising is applied to the complete per-ticker price
> series as a preprocessing step before any train/test split, which is valid because the
> denoising operation at time t uses the entire price history to choose the denoising
> threshold — this is standard practice in financial signal processing. If you want to be
> maximally conservative, apply wavelet denoising only to the training portion and use
> the original (non-denoised) prices for validation/test. Implement the conservative
> variant as it is safer for a thesis.

### Files to modify
- `data_acquisition.py` or `feature_engineering.py` — add `denoise_close_price()` function
- `config.py` — add `USE_WAVELET_DENOISING = True/False` toggle

### Implementation
```python
# In feature_engineering.py, add BEFORE compute_technical_indicators()

import numpy as np

def denoise_close_price(close_series: pd.Series, wavelet: str = "haar",
                        level: int = 1, mode: str = "soft") -> pd.Series:
    """
    Bhandari §4.5 — Haar wavelet soft denoising of close price series.

    Uses PyWavelets (pywt) — a pure-Python library requiring no GPU.
    Install: pip install PyWavelets

    Parameters
    ----------
    close_series : pd.Series
        Raw daily close prices for a single ticker (chronological order).
    wavelet : str
        Wavelet family. 'haar' is the paper's choice.
    level : int
        Decomposition level. Level 1 is sufficient for daily data.
    mode : str
        Thresholding mode: 'soft' (paper) or 'hard'.

    Returns
    -------
    pd.Series
        Denoised close prices with the same index.
    """
    import pywt

    prices = close_series.values.astype(float)

    # Wavelet decomposition
    coeffs = pywt.wavedec(prices, wavelet=wavelet, level=level)

    # Estimate noise sigma from detail coefficients (median absolute deviation)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745   # universal threshold formula

    # Universal threshold (Donoho & Johnstone 1994)
    threshold = sigma * np.sqrt(2 * np.log(len(prices)))

    # Apply soft thresholding to all detail coefficient levels
    coeffs_thresholded = [coeffs[0]]  # keep approximation coefficients unchanged
    for detail in coeffs[1:]:
        coeffs_thresholded.append(pywt.threshold(detail, threshold, mode=mode))

    # Reconstruct denoised signal
    denoised = pywt.waverec(coeffs_thresholded, wavelet=wavelet)

    # waverec may produce one extra sample due to odd-length inputs — trim
    denoised = denoised[:len(prices)]

    return pd.Series(denoised, index=close_series.index, name=close_series.name)


def apply_wavelet_denoising(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply denoise_close_price() to each ticker independently.
    Replaces the 'Close' column with the denoised version.
    Call this BEFORE compute_technical_indicators() and compute_returns().
    """
    df = df.copy()
    denoised_rows = []
    for ticker, grp in df.groupby("Ticker"):
        grp = grp.sort_values("Date").copy()
        grp["Close"] = denoise_close_price(grp["Close"])
        denoised_rows.append(grp)
    return pd.concat(denoised_rows).sort_values(["Date", "Ticker"])
```

### Call order in `feature_engineering.py` main function
```python
def build_features(df_raw: pd.DataFrame, cfg) -> pd.DataFrame:
    """Full feature engineering pipeline."""

    # Step 1 — Optional wavelet denoising (Bhandari §4.5)
    if cfg.USE_WAVELET_DENOISING:
        df_raw = apply_wavelet_denoising(df_raw)

    # Step 2 — Compute returns (uses denoised Close if Step 1 ran)
    df_raw = compute_returns(df_raw)

    # Step 3 — Compute technical indicators (uses denoised Close)
    df_raw = compute_technical_indicators(df_raw)

    # Step 4 — Lagged features, target construction, etc.
    # ... (existing pipeline) ...

    return df_raw
```

### Config toggle (`config.py`)
```python
# ── Wavelet Denoising (Bhandari §4.5) ────────────────────────────────────────
USE_WAVELET_DENOISING = True    # Set False to use raw prices (ablation study)
WAVELET_TYPE          = "haar"  # Paper uses Haar wavelets
WAVELET_LEVEL         = 1       # Decomposition level; 1 is appropriate for daily data
WAVELET_MODE          = "soft"  # Thresholding mode: 'soft' (paper) or 'hard'
```

### Dependency
```bash
pip install PyWavelets   # pure Python, no GPU needed, M4-compatible
```

> **Thesis ablation:** Run the full pipeline twice — `USE_WAVELET_DENOISING = True` and
> `False` — and report the difference in Sharpe, AUC, and accuracy. This is a clean
> ablation study directly inspired by the paper's methodology.

---

## 5. Normalization Choice (Section 4.5)

### What the paper does
Bhandari §4.5 uses **min-max normalization** (scaling all features to [0, 1]):
`z = (x − x_min) / (x_max − x_min)`.
The existing pipeline uses `StandardScaler` (z-score: mean 0, std 1).

### Adaptation for this project
Both approaches are valid for neural networks. For this project:
- Keep `StandardScaler` as the **default** (it is standard practice in cross-sectional ML
  and handles outliers in return distributions better than min-max).
- Add a `MinMaxScaler` option as an **ablation variant**, configurable via `config.py`.

> **Anti-leakage:** Regardless of which scaler is used, it must be fit on the **training
> fold only** and applied to val/test. This rule is unchanged.

### Config update (`config.py`)
```python
# ── Normalization (Bhandari §4.5 uses MinMax; our default is Standard) ───────
SCALER_TYPE = "standard"   # Options: "standard" (default) | "minmax"
```

### Scaler factory in `pipeline.py`
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def get_scaler(scaler_type: str):
    if scaler_type == "minmax":
        return MinMaxScaler()
    return StandardScaler()   # default

# Inside fold loop (existing standardize_fold logic):
scaler = get_scaler(cfg.SCALER_TYPE)
X_train_scaled = scaler.fit_transform(X_train)   # fit on train only
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)
```

> **Thesis note:** Run ablation: `SCALER_TYPE = "standard"` vs `"minmax"`. Report both
> in a table. For most cross-sectional ML tasks, StandardScaler is expected to perform
> equal or better due to return distributions having heavier tails than MinMax assumes.

---

## 6. Remove Unused Features

### Motivation

With the per-model feature sets now fixed (Section 7.1), all features not belonging to
either LSTM-A's or LSTM-B's feature set serve no purpose in the pipeline. Continuing to
compute them wastes memory, increases DataFrame size, and creates noise during the
correlation heatmap analysis (Section 3). They should be removed from `feature_engineering.py`
entirely rather than computed and then discarded.

### Active feature union

Only features present in at least one model's input set need to be computed:

| Feature | Used by |
|---------|---------|
| `Return_1d` | LSTM-A, LSTM-B, LR, RF, XGBoost |
| `RSI_14` | LSTM-A, LSTM-B, LR, RF, XGBoost |
| `MACD` | LSTM-A |
| `ATR_14` | LSTM-A |
| `BB_PctB` | LSTM-B, LR, RF, XGBoost |
| `RealVol_20d` | LSTM-B, LR, RF, XGBoost |
| `Volume_Ratio` | LSTM-B, LR, RF, XGBoost |
| `SectorRelReturn` | LSTM-B, LR, RF, XGBoost |

**Total: 8 features.** Everything else must be removed.

### Features to remove

The following computed features are no longer needed and must be deleted from
`feature_engineering.py` and from any feature column list in `config.py`:

**Lagged return series (from Fischer & Krauss replication):**
`Return_2d`, `Return_3d`, ..., `Return_20d`, `Return_40d`, `Return_60d`, `Return_80d`,
`Return_100d`, `Return_120d`, `Return_140d`, `Return_160d`, `Return_180d`, `Return_200d`,
`Return_220d`, `Return_240d` — **30 features removed.**

**Technical indicators no longer in any feature set:**
`MACD_Signal`, `BB_Width`, `OBV`, `HL_Pct`, `HL_Pct_5d`, `Mom_10d` — **6 features removed.**

> **Note on `MACD_Signal`:** Section 2 of this document added `MACD_Signal` as a Bhandari
> §4.3 extension. Since it is not in either active feature set, do not implement it.
> `MACD` alone is sufficient for LSTM-A.

### Files to modify

- `feature_engineering.py` — remove the computation of all 36 listed features above;
  keep only the 8 active features plus the raw OHLCV columns and `Target`
- `config.py` — replace any master `FEATURE_COLS` list with the exact 8-feature union,
  and remove `feature_cols_returns` (the 31-lagged-return list from `CLAUDE.md` §4.1)

### Updated `compute_technical_features()` in `feature_engineering.py`

Replace the existing function body entirely with this leaner version:

```python
def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute only the 8 features actively used across all models.
    All other previously computed features (lagged returns, MACD_Signal,
    BB_Width, OBV, HL_Pct, Mom_10d) are intentionally omitted.

    ANTI-LEAKAGE: All indicators are causal — each value at time t
    uses only data from t and earlier.
    """
    df = df.copy().sort_values("Date")

    # ── Return_1d ─────────────────────────────────────────────────────
    df["Return_1d"] = df["Close"].pct_change(1)

    # ── RSI_14 (Bhandari §4.3) ────────────────────────────────────────
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI_14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # ── MACD (Bhandari §4.3 — 12/26 EMA difference) ───────────────────
    ema12       = df["Close"].ewm(span=12, adjust=False).mean()
    ema26       = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]  = ema12 - ema26
    # Note: MACD_Signal is NOT computed — not in any active feature set

    # ── ATR_14 (Bhandari §4.3) ────────────────────────────────────────
    hl  = df["High"] - df["Low"]
    hpc = (df["High"] - df["Close"].shift()).abs()
    lpc = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()

    # ── BB_PctB (%B position within Bollinger Bands) ───────────────────
    sma20        = df["Close"].rolling(20).mean()
    std20        = df["Close"].rolling(20).std()
    upper        = sma20 + 2 * std20
    lower        = sma20 - 2 * std20
    df["BB_PctB"] = (df["Close"] - lower) / (upper - lower + 1e-10)
    # Note: BB_Width is NOT computed — not in any active feature set

    # ── RealVol_20d (annualised 20-day realised volatility) ───────────
    log_ret          = np.log(df["Close"] / df["Close"].shift(1))
    df["RealVol_20d"] = log_ret.rolling(20).std() * np.sqrt(252)

    # ── Volume_Ratio ──────────────────────────────────────────────────
    df["Volume_Ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1)

    # Note: OBV, HL_Pct, HL_Pct_5d, Mom_10d are NOT computed

    return df
```

### Updated `config.py` — master feature list

```python
# ── Master feature union (all features computed by feature_engineering.py) ───
# This is the complete set of engineered features. Per-model subsets are defined
# below. Do NOT add features here unless they are used by at least one model.
ALL_FEATURE_COLS = [
    "Return_1d",
    "RSI_14",
    "MACD",
    "ATR_14",
    "BB_PctB",
    "RealVol_20d",
    "Volume_Ratio",
    "SectorRelReturn",   # computed separately in compute_sector_rel_return()
]

# Remove the old feature_cols_returns list (31 lagged returns) — no longer used.
# The Fischer & Krauss (2017) replication feature set has been superseded by the
# per-model feature sets defined in Section 7.
```

> **Impact on Section 3 (Correlation Heatmap):** The heatmap now operates on 8 features
> rather than 41. This is actually preferable — the reduced heatmap is readable without
> annotation suppression, and the off-diagonal correlations among these 8 features are the
> ones that actually matter for the models. Update `analysis/feature_correlation.py` to
> pass `cfg.ALL_FEATURE_COLS` as the feature list.

---

## Summary of New/Modified Files

| File | Action | Section |
|------|--------|---------|
| `config.py` | Add `LSTM_HYPERPARAM_GRID`, `LSTM_A_ARCH_GRID`, `LSTM_TUNE_REPLICATES`, `LSTM_TUNE_PATIENCE`, `LSTM_TUNE_MAX_EPOCHS`, `USE_WAVELET_DENOISING`, `WAVELET_*`, `SCALER_TYPE`, `FEATURE_COLS_AFTER_SELECTION`, `ALL_FEATURE_COLS`, `LSTM_A_FEATURE_COLS`, `LSTM_B_FEATURE_COLS`, `BASELINE_FEATURE_COLS`, `SECTOR_MAP`; remove `feature_cols_returns` and all lagged-return references | §3.3 / §4.4 / §4.5 / §6 / §7 |
| `feature_engineering.py` | Add `denoise_close_price()`, `apply_wavelet_denoising()`, `compute_sector_rel_return()`; rewrite `compute_technical_features()` to compute only the 8 active features; remove lagged returns computation entirely | §4.3 / §4.5 / §6 / §7 |
| `models/lstm_model.py` | Add `tune_lstm_hyperparams()` (with optional `arch_grid` for LSTM-A), `_build_optimizer()`; parameterise `StockLSTM` constructor fully | §3.3 / §7 |
| `analysis/feature_correlation.py` | **New file.** Runs once on Fold 1 training data; heatmap now operates on 8 features | §4.4 |
| `pipeline.py` | Two separate scalers per fold (Scaler A / Scaler B); LSTM-A architecture resolved from tuning results; updated `evaluate_fold()` signature; use `get_scaler(cfg.SCALER_TYPE)` | §3.3 / §4.5 / §7 |

## New Dependency

```bash
pip install PyWavelets   # for wavelet denoising (§4.5)
```
`pandas_ta` should already be installed for technical indicators; if not:
```bash
pip install pandas_ta
```

---

## Implementation Order (Recommended)

Follow this sequence to minimise integration conflicts:

1. **`config.py`** — add all new config keys and remove lagged-return feature lists (safe, no logic)
2. **`feature_engineering.py`** — rewrite `compute_technical_features()` to 8-feature version; add wavelet, RealVol, and SectorRelReturn functions; re-run data prep
3. **`analysis/feature_correlation.py`** — run once on the new 8-feature output; inspect heatmap; update `FEATURE_COLS_AFTER_SELECTION` if any pair exceeds the 0.80 threshold
4. **`models/lstm_model.py`** — add tuner with arch grid support
5. **`pipeline.py`** — integrate dual scalers, LSTM-A tuning, and updated `evaluate_fold()`
6. **Run ablation study** — wavelet on/off, scaler type, with/without feature selection

---

## 7. Per-Model Feature Sets: New LSTM-A and Unified Baseline Features

### Context and motivation

The current pipeline runs a single shared feature set for all models. The results show a
clear divergence between LSTM-A (Sharpe = −0.517) and LSTM-B (Sharpe = +0.278), suggesting
that feature set composition has a large effect on model performance — larger, in these
results, than model architecture. This section formalises two separate feature regimes:

- **LSTM-A (new definition):** A compact, technically-oriented 4-feature set — MACD, RSI,
  ATR, Return_1D. Motivated by Bhandari §4.3, which demonstrates strong predictive
  performance with exactly these three technical indicators plus raw return information.
- **LSTM-B (unchanged):** The existing 6-feature set that produced Sharpe = +0.278.
- **Baseline models (LR, RF, XGBoost):** Standardised onto LSTM-B's 6-feature set.
  This creates a fair, apples-to-apples comparison: baselines and LSTM-B compete on
  identical inputs, while LSTM-A tests whether a purely technical-indicator-driven LSTM
  can match or exceed LSTM-B despite using less information.

---

### 7.1 Feature Set Definitions

#### LSTM-B feature set (unchanged — also adopted by all baselines)

| # | Feature | Already in pipeline? | Notes |
|---|---------|----------------------|-------|
| 1 | `Return_1d` | ✅ Yes | 1-day simple return |
| 2 | `RSI_14` | ✅ Yes | 14-day RSI per §4.3 |
| 3 | `BB_PctB` | ✅ Yes (`BB_PctB`) | %B position within Bollinger Bands |
| 4 | `RealVol_20d` | ❌ **New** | 20-day realised volatility — see §7.2 |
| 5 | `Volume_Ratio` | ✅ Yes | Volume / 20-day average volume |
| 6 | `SectorRelReturn` | ❌ **New** | Stock return minus sector mean return — see §7.3 |

#### LSTM-A feature set (new definition)

| # | Feature | Already in pipeline? | Notes |
|---|---------|----------------------|-------|
| 1 | `MACD` | ✅ Yes | 12/26 EMA difference per §4.3 |
| 2 | `RSI_14` | ✅ Yes | 14-day RSI per §4.3 |
| 3 | `ATR_14` | ✅ Yes | 14-day ATR per §4.3 |
| 4 | `Return_1d` | ✅ Yes | 1-day simple return |

Two new features must be engineered before this section can be used: `RealVol_20d` and
`SectorRelReturn`. Both are defined below.

---

### 7.2 New Feature: RealVol_20d (20-Day Realised Volatility)

**Definition:** The annualised standard deviation of daily log-returns over a trailing
20-day window. This is the most common realised volatility estimator in quantitative
finance and captures current volatility regime per stock.

```
RealVol_20d(t) = std(log(Close_t / Close_{t-1}), window=20) × sqrt(252)
```

This is distinct from ATR (which measures average true range) — `RealVol_20d` is a
pure return-based volatility measure and is the standard input to volatility-targeting
strategies.

#### Implementation in `feature_engineering.py`

```python
def compute_realvol(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute 20-day annualised realised volatility per ticker.

    RealVol_20d = rolling std of log-returns × sqrt(252).

    ANTI-LEAKAGE: Uses only past data (rolling window looks backward).
    Each value at time t uses days [t-19, ..., t] — no future information.
    """
    df = df.copy()
    log_ret = np.log(df['Close'] / df['Close'].shift(1))
    df['RealVol_20d'] = log_ret.rolling(window).std() * np.sqrt(252)
    return df
```

Add the call inside the per-ticker loop in `compute_technical_features()`:

```python
# Inside the per-ticker loop in feature_engineering.py:
for ticker, group in data.groupby('Ticker'):
    enriched = compute_technical_features(group)   # existing call
    enriched = compute_realvol(enriched, window=20) # new call
    all_stocks.append(enriched)
```

---

### 7.3 New Feature: SectorRelReturn

**Definition:** The stock's 1-day return minus the equal-weighted mean 1-day return of
all other stocks in the same sector on that day. This is a cross-sectional, sector-neutral
return signal that captures stock-specific momentum or reversal relative to sector peers —
a finer-grained signal than raw return.

```
SectorRelReturn(s, t) = Return_1d(s, t) − mean(Return_1d(s', t) for s' in sector(s))
```

This is computationally similar to the cross-sectional median target construction already
in the pipeline, but computed at the sector level and used as a feature, not a target.

#### Sector assignments (from the project's 10-stock universe)

```python
# config.py — add this mapping
SECTOR_MAP = {
    "AAPL":  "Technology",
    "MSFT":  "Technology",
    "NVDA":  "Technology",
    "GOOGL": "Communication",
    "META":  "Communication",
    "AMZN":  "Consumer",
    "TSLA":  "Consumer",       # Auto/Tech — closest peer group
    "BRK-B": "Financials",
    "JPM":   "Financials",
    "V":     "Financials",
}
```

> **Note on small sectors:** With only 10 stocks, sector means are computed over 2–3
> stocks per sector (e.g. Technology = AAPL, MSFT, NVDA). This is a very thin mean.
> The resulting `SectorRelReturn` will largely reflect idiosyncratic stock behaviour
> rather than true sector effects. This is acceptable for a thesis — document it as a
> limitation and note that the feature becomes more meaningful in a larger universe.

#### Implementation in `feature_engineering.py`

```python
def compute_sector_rel_return(df: pd.DataFrame, sector_map: dict) -> pd.DataFrame:
    """
    Compute SectorRelReturn: each stock's Return_1d minus the equal-weighted
    mean Return_1d of all stocks in the same sector on that date.

    ANTI-LEAKAGE: Uses only same-day cross-sectional returns — no future data.
    The sector mean at time t is the realised (not forecast) mean of t, which is
    legitimate: it is a feature known after market close on day t, used to predict
    day t+1.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with columns [Date, Ticker, Return_1d].
        Must already have Return_1d computed.
    sector_map : dict
        {ticker: sector_name} mapping from config.py.

    Returns
    -------
    pd.DataFrame with added column 'SectorRelReturn'.
    """
    df = df.copy()
    df['Sector'] = df['Ticker'].map(sector_map)

    # Equal-weighted sector mean per date (exclude the stock itself
    # to avoid self-inclusion bias in small sectors)
    def sector_mean_excl_self(row, daily_sector_means):
        sector = row['Sector']
        ticker = row['Ticker']
        sector_stocks = df[
            (df['Date'] == row['Date']) &
            (df['Sector'] == sector) &
            (df['Ticker'] != ticker)
        ]['Return_1d']
        if len(sector_stocks) == 0:
            return 0.0   # only stock in sector — relative return is 0
        return sector_stocks.mean()

    # Vectorised implementation (faster than apply for large DataFrames)
    # Step 1: compute sector mean per (Date, Sector) over all tickers
    sector_means = (
        df.groupby(['Date', 'Sector'])['Return_1d']
          .transform('mean')
    )
    # Step 2: subtract self-contribution to get leave-one-out mean
    sector_counts = (
        df.groupby(['Date', 'Sector'])['Return_1d']
          .transform('count')
    )
    # Leave-one-out sector mean:
    # mean_excl_self = (sum - self_value) / (count - 1)
    sector_sums = sector_means * sector_counts
    df['SectorRelReturn'] = (
        df['Return_1d']
        - (sector_sums - df['Return_1d']) / (sector_counts - 1).clip(lower=1)
    )

    df.drop(columns=['Sector'], inplace=True)
    return df
```

Add the call in the main `build_features()` function, after returns are computed:

```python
def build_features(df_raw: pd.DataFrame, cfg) -> pd.DataFrame:
    # ... existing steps ...

    # After compute_returns() and compute_technical_features():
    df = compute_sector_rel_return(df, sector_map=cfg.SECTOR_MAP)

    # ... rest of pipeline ...
    return df
```

---

### 7.4 New LSTM-A Architecture — Data-Driven via Tuning

LSTM-A's architecture is **not fixed in advance**. Instead, it is determined by the
two-phase hyperparameter tuning procedure defined in Section 1, extended with the
`LSTM_A_ARCH_GRID` from `config.py`. This is the correct methodological approach:
prescribing architecture before training is an arbitrary choice that undermines the
empirical rigour of the comparison with LSTM-B.

The rationale for the search space is that a 4-feature input warrants a narrower,
potentially shallower model than LSTM-B, so the grid is bounded accordingly:
- `hidden_size ∈ {16, 32, 64}` — smaller ceiling than LSTM-B's fixed 64 units
- `num_layers ∈ {1, 2}` — includes the single-layer case favoured by Bhandari §5.5
- `dropout ∈ {0.1, 0.2}` — lower floor than LSTM-B's 0.2, appropriate for a compact model

The tuner (Section 1) runs Phase 1 (fix seed architecture, sweep optimizer/lr/batch),
then Phase 2 (fix best Phase-1 hyperparams, sweep all 12 architecture combinations).
The best architecture is selected by highest average validation AUC across
`cfg.LSTM_TUNE_REPLICATES` independent runs.

#### Implementation in `models/lstm_model.py`

`StockLSTM` must accept all architecture parameters via its constructor so both LSTM-A
(tuner-resolved values) and LSTM-B (fixed values) can be instantiated from the same class:

```python
class StockLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,        # 4 for LSTM-A, 6 for LSTM-B
        hidden_size: int = 64,  # tuner-resolved for LSTM-A; fixed 64 for LSTM-B
        num_layers: int = 2,    # tuner-resolved for LSTM-A; fixed 2 for LSTM-B
        dropout: float = 0.2,   # tuner-resolved for LSTM-A; fixed 0.2 for LSTM-B
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,  # PyTorch ignores dropout
                                                           # for single-layer LSTM
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(hidden_size, 16)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.relu(self.fc1(out))
        return self.sigmoid(self.fc2(out)).squeeze(1)
```

#### Config additions for LSTM-A (`config.py`)

```python
# ── Feature sets per model ────────────────────────────────────────────────────

LSTM_A_FEATURE_COLS = [
    "MACD",        # 12/26 EMA difference (Bhandari §4.3)
    "RSI_14",      # 14-day RSI (Bhandari §4.3)
    "ATR_14",      # 14-day ATR (Bhandari §4.3)
    "Return_1d",   # 1-day simple return
]

LSTM_B_FEATURE_COLS = [
    "Return_1d",
    "RSI_14",
    "BB_PctB",
    "RealVol_20d",
    "Volume_Ratio",
    "SectorRelReturn",
]

# Baselines use LSTM-B features for fair comparison
BASELINE_FEATURE_COLS = LSTM_B_FEATURE_COLS   # LR, RF, XGBoost all use this

# ── LSTM-B architecture (fixed — established in existing pipeline) ─────────────
LSTM_B_HIDDEN_SIZE = 64
LSTM_B_NUM_LAYERS  = 2
LSTM_B_DROPOUT     = 0.2

# Note: LSTM-A has NO static architecture constants.
# Its hidden_size, num_layers, and dropout are resolved at runtime by the tuner
# (tune_lstm_hyperparams with arch_grid=cfg.LSTM_A_ARCH_GRID) and stored in
# best_hp_a returned by the tuner within each walk-forward fold.
```

---

### 7.5 Pipeline Changes — Per-Model Feature Routing

The walk-forward pipeline must be updated to route each model to its correct feature set.
The key change is that `standardize_fold()` is called **separately** for each feature set,
since each scaler must be fit only on the columns it will actually see.

#### Modified fold execution in `pipeline.py`

```python
def run_walk_forward_pipeline(data_features, folds, cfg, device):
    """
    Updated pipeline: each model uses its own feature set and its own scaler.
    LSTM-A  → cfg.LSTM_A_FEATURE_COLS (4 features)
    LSTM-B  → cfg.LSTM_B_FEATURE_COLS (6 features)
    LR/RF/XGB → cfg.BASELINE_FEATURE_COLS (same 6 as LSTM-B)
    """
    all_results = []
    dates = sorted(data_features['Date'].unique())

    for fold in folds:
        print(f"\n{'='*50}\nFOLD {fold['fold']}\n{'='*50}")

        # ── Date slices ───────────────────────────────────────────────
        t_s, t_e   = fold['train']
        v_s, v_e   = fold['val']
        ts_s, ts_e = fold['test']
        train_dates = dates[t_s:t_e]
        val_dates   = dates[v_s:v_e]
        test_dates  = dates[ts_s:ts_e]

        df_train = data_features[data_features['Date'].isin(train_dates)]
        df_val   = data_features[data_features['Date'].isin(val_dates)]
        df_test  = data_features[data_features['Date'].isin(test_dates)]

        # ── Scaler A: fit on LSTM-A features (4 cols) ─────────────────
        feat_a = cfg.LSTM_A_FEATURE_COLS
        Xtr_a, Xv_a, Xts_a, scaler_a = standardize_fold(
            df_train[feat_a].values,
            df_val[feat_a].values,
            df_test[feat_a].values,
        )

        # ── Scaler B: fit on LSTM-B / baseline features (6 cols) ──────
        feat_b = cfg.LSTM_B_FEATURE_COLS
        Xtr_b, Xv_b, Xts_b, scaler_b = standardize_fold(
            df_train[feat_b].values,
            df_val[feat_b].values,
            df_test[feat_b].values,
        )

        y_tr = df_train[cfg.TARGET_COL].values
        y_v  = df_val[cfg.TARGET_COL].values

        # ── Baseline models (all use Scaler B / feat_b) ───────────────
        lr_model  = train_logistic(Xtr_b, y_tr)
        rf_model  = train_random_forest(Xtr_b, y_tr)
        xgb_model = train_xgboost(Xtr_b, y_tr, Xv_b, y_v)

        # ── LSTM-B (unchanged) ────────────────────────────────────────
        df_tr_b = df_train.copy(); df_tr_b[feat_b] = Xtr_b
        df_v_b  = df_val.copy();   df_v_b[feat_b]  = Xv_b
        df_ts_b = df_test.copy();  df_ts_b[feat_b] = Xts_b

        lstm_b = StockLSTM(
            input_size  = len(feat_b),
            hidden_size = cfg.LSTM_B_HIDDEN_SIZE,
            num_layers  = cfg.LSTM_B_NUM_LAYERS,
            dropout     = cfg.LSTM_B_DROPOUT,
        )
        lstm_b_ds_tr  = StockSequenceDataset(df_tr_b, feat_b, cfg.TARGET_COL, cfg.SEQ_LEN)
        lstm_b_ds_v   = StockSequenceDataset(df_v_b,  feat_b, cfg.TARGET_COL, cfg.SEQ_LEN)
        lstm_b_ds_ts  = StockSequenceDataset(df_ts_b, feat_b, cfg.TARGET_COL, cfg.SEQ_LEN)
        lstm_b = train_lstm(lstm_b,
                            DataLoader(lstm_b_ds_tr, batch_size=cfg.BATCH_SIZE, shuffle=True),
                            DataLoader(lstm_b_ds_v,  batch_size=cfg.BATCH_SIZE * 2),
                            device)

        # ── LSTM-A (tuned architecture + hyperparameters) ─────────────
        # Run tuner FIRST — it resolves architecture via Phase 2 arch sweep
        best_hp_a = tune_lstm_hyperparams(
            df_tr_a, df_v_a, feat_a, cfg.TARGET_COL,
            cfg.SEQ_LEN, device, cfg,
            arch_grid=cfg.LSTM_A_ARCH_GRID,   # architecture is data-driven
        )

        lstm_a_ds_tr = StockSequenceDataset(df_tr_a, feat_a, cfg.TARGET_COL, cfg.SEQ_LEN)
        lstm_a_ds_v  = StockSequenceDataset(df_v_a,  feat_a, cfg.TARGET_COL, cfg.SEQ_LEN)
        lstm_a_ds_ts = StockSequenceDataset(df_ts_a, feat_a, cfg.TARGET_COL, cfg.SEQ_LEN)

        lstm_a = StockLSTM(
            input_size  = len(feat_a),              # always 4
            hidden_size = best_hp_a["hidden_size"], # resolved by Phase-2 tuner
            num_layers  = best_hp_a["num_layers"],  # resolved by Phase-2 tuner
            dropout     = best_hp_a["dropout"],     # resolved by Phase-2 tuner
        )
        lstm_a = train_lstm(lstm_a,
                            DataLoader(lstm_a_ds_tr, batch_size=best_hp_a["batch_size"],
                                       shuffle=True),
                            DataLoader(lstm_a_ds_v,  batch_size=best_hp_a["batch_size"] * 2),
                            device,
                            optimizer_name=best_hp_a["optimizer"],
                            lr=best_hp_a["lr"])

        # ── Collect predictions on test set ───────────────────────────
        fold_result = evaluate_fold(
            fold       = fold,
            df_test    = df_test,
            y_test     = df_test[cfg.TARGET_COL].values,
            # Baseline models + their scaled test arrays
            lr_model   = lr_model,  Xts_lr  = Xts_b,
            rf_model   = rf_model,  Xts_rf  = Xts_b,
            xgb_model  = xgb_model, Xts_xgb = Xts_b,
            # LSTM models + their sequence datasets
            lstm_a     = lstm_a, lstm_a_ds_ts = lstm_a_ds_ts,
            lstm_b     = lstm_b, lstm_b_ds_ts = lstm_b_ds_ts,
            device     = device,
        )
        all_results.append(fold_result)

    return pd.concat(all_results)
```

---

### 7.6 Updated `evaluate_fold()` Signature

The `evaluate_fold()` function must accept separate test arrays for each model.
Update its signature to match the per-model feature routing above:

```python
def evaluate_fold(
    fold, df_test, y_test,
    lr_model,  Xts_lr,
    rf_model,  Xts_rf,
    xgb_model, Xts_xgb,
    lstm_a, lstm_a_ds_ts,
    lstm_b, lstm_b_ds_ts,
    device,
):
    """
    Collect out-of-sample predictions for all five models.
    Each model receives its own correctly-scaled test array.
    Returns a DataFrame of [Date, Ticker, Prob_LR, Prob_RF,
    Prob_XGB, Prob_LSTM_A, Prob_LSTM_B, Target].
    """
    import xgboost as xgb
    from torch.utils.data import DataLoader

    # Baseline predictions
    lr_probs  = lr_model.predict_proba(Xts_lr)[:, 1]
    rf_probs  = rf_model.predict_proba(Xts_rf)[:, 1]
    xgb_probs = xgb_model.predict(xgb.DMatrix(Xts_xgb))

    # LSTM-A predictions
    lstm_a.eval()
    a_preds = []
    with torch.no_grad():
        for Xb, _ in DataLoader(lstm_a_ds_ts, batch_size=256):
            a_preds.extend(lstm_a(Xb.to(device)).cpu().numpy())
    lstm_a_probs = np.array(a_preds)

    # LSTM-B predictions
    lstm_b.eval()
    b_preds = []
    with torch.no_grad():
        for Xb, _ in DataLoader(lstm_b_ds_ts, batch_size=256):
            b_preds.extend(lstm_b(Xb.to(device)).cpu().numpy())
    lstm_b_probs = np.array(b_preds)

    # Assemble result DataFrame
    # Note: LSTM sequence datasets may have fewer rows than df_test
    # due to the seq_len warm-up. Align on the shorter of the two.
    n = min(len(lr_probs), len(lstm_a_probs), len(lstm_b_probs))
    result = df_test.tail(n).copy().reset_index(drop=True)
    result['Prob_LR']     = lr_probs[-n:]
    result['Prob_RF']     = rf_probs[-n:]
    result['Prob_XGB']    = xgb_probs[-n:]
    result['Prob_LSTM_A'] = lstm_a_probs
    result['Prob_LSTM_B'] = lstm_b_probs

    return result
```

---

### 7.7 Summary of Changes to Existing Functions

| Function / File | Change required |
|----------------|----------------|
| `config.py` | Add `LSTM_A_FEATURE_COLS`, `LSTM_B_FEATURE_COLS`, `BASELINE_FEATURE_COLS`, `SECTOR_MAP`, `LSTM_A_ARCH_GRID`, `LSTM_B_HIDDEN_SIZE`, `LSTM_B_NUM_LAYERS`, `LSTM_B_DROPOUT`; **remove** `LSTM_A_HIDDEN_SIZE`, `LSTM_A_NUM_LAYERS`, `LSTM_A_DROPOUT` (these are now resolved at runtime by the tuner); **remove** `feature_cols_returns` and all other unused feature lists |
| `feature_engineering.py` | Add `compute_realvol()` and `compute_sector_rel_return()`; rewrite `compute_technical_features()` to the 8-feature-only version from Section 6 |
| `models/lstm_model.py` | Ensure `StockLSTM` constructor accepts `input_size`, `hidden_size`, `num_layers`, `dropout`; update `tune_lstm_hyperparams()` to accept `arch_grid=None` parameter and implement Phase-2 architecture sweep when it is provided |
| `pipeline.py` | (1) Two separate `standardize_fold` calls per fold — Scaler A for `LSTM_A_FEATURE_COLS`, Scaler B for `LSTM_B_FEATURE_COLS`; (2) call `tune_lstm_hyperparams(..., arch_grid=cfg.LSTM_A_ARCH_GRID)` before LSTM-A training and use the returned `best_hp_a` dict for instantiation; (3) call `tune_lstm_hyperparams(..., arch_grid=None)` for LSTM-B and use `best_hp_b` for its training call; (4) update `evaluate_fold()` signature to accept separate test arrays per model |
| `evaluation/metrics.py` | Rename `Prob_LSTM` → `Prob_LSTM_A` and `Prob_LSTM_B` in any column references |
| `signal_generation.py` | Add `Prob_LSTM_A` as a separate signal column; do not ensemble with `Prob_LSTM_B` |

---

### 7.8 Thesis Interpretation Notes

**Why not ensemble LSTM-A and LSTM-B?**
The two LSTM variants test different hypotheses about what information drives short-term
cross-sectional returns. LSTM-A asks: *can purely technical indicators predict relative
outperformance?* LSTM-B asks: *does adding volatility, volume, and sector context help?*
Combining them would obscure this comparison. Report them as separate rows in the results
table (as currently done) rather than blending.

**Expected behaviour:**
LSTM-A's 4-feature set contains MACD, RSI, and ATR — the same indicators validated in
Bhandari §4.3 for index-level prediction. For cross-sectional ranking, these indicators
are most useful when their *relative* level across stocks matters. Since all 10 stocks
share a common market factor, a technical indicator at the same level across all stocks
provides no cross-sectional signal. LSTM-A is therefore expected to have lower signal
content than LSTM-B, which includes `SectorRelReturn` — an explicitly cross-sectional
feature. This makes for a clean and defensible thesis comparison.

**What to report:**
Add a "Feature Set Description" table to the thesis methodology chapter listing the two
LSTM feature sets and the baseline feature set side by side. For each model, report the
number of input features, sequence length, and total parameter count alongside the usual
Sharpe/AUC/Accuracy table. This is the direct parallel to Bhandari Tables 8 and 9, which
report parameter counts alongside performance metrics.
