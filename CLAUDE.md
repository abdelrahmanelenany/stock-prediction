# CLAUDE.md — Neural Networks for Stock Behavior Prediction
> Bachelor's Project · Complete Implementation Reference for IDE Copilot
> Based on Fischer & Krauss (2017), Krauss, Do & Huck (2017), and Bhandari et al. (2022)
> **Last Updated:** 2026-04-18 (Reflects current implementation)

---

## Project Summary

Build a **walk-forward validated, backtested long-short trading strategy** using ML models
(Logistic Regression, Random Forest, XGBoost, LSTM, TCN, Ensemble) on a configurable stock universe
over **2019–2024**. The pipeline predicts each stock's probability of outperforming
the cross-sectional median N-day forward return, ranks stocks by that probability, and constructs
an equal-weighted long-short portfolio. Features include technical indicators plus optional
**market-context and sector-context features**, with optional **causal wavelet denoising** (disabled
by default). All performance is reported net of transaction costs.

**Two universe modes:**
- `large_cap`: 50 S&P 500 large caps balanced across 5 sectors (Tech, Finance, Healthcare, Consumer, Industrial)
- `small_cap`: 30 true small-cap stocks (Russell 2000 / S&P SmallCap 600 constituents)

**Target hardware:** MacBook Air M4 (CPU / MPS backend for PyTorch).

---

## Repository Layout

```
stock_prediction/
├── config.py                         # All hyperparameters and constants (single source of truth)
├── main.py                           # Orchestrates the full pipeline end-to-end
├── combine_and_backtest.py           # Stitch together separate baseline/LSTM runs
├── generate_feature_importance_plot.py  # Feature importance visualisation
├── generate_latex_tables.py          # LaTeX table generation from reports CSVs
├── generate_thesis_outputs.py        # Batch thesis output generator
├── requirements.txt                  # Python dependencies
├── data/
│   ├── raw/
│   │   ├── ohlcv_raw.csv             # Multi-index download from yfinance
│   │   └── ohlcv_long.csv            # Restructured: Date, Ticker, OHLCV
│   └── processed/
│       ├── features_large_cap.csv    # All features + target for large-cap universe (cached)
│       └── features_small_cap.csv    # All features + target for small-cap universe (cached)
├── pipeline/
│   ├── data_loader.py                # Download + clean + save raw data
│   ├── features.py                   # Technical + context features + causal wavelet denoising
│   ├── targets.py                    # Configurable N-day cross-sectional median target
│   ├── walk_forward.py               # Walk-forward fold generator (rolling / expanding)
│   ├── standardizer.py               # Standard/MinMax scaler + winsorizer (fit on train only)
│   ├── fold_reporting.py             # Per-fold JSON/CSV diagnostic artifacts
│   └── diagnostics.py               # Dataset diagnostics utilities
├── models/
│   ├── baselines.py                  # LR, RF, XGBoost (with grid search + feature importances)
│   ├── lstm_model.py                 # LSTM architecture + Bhandari-style tuning
│   ├── tcn_model.py                  # TCN architecture (Bai et al. 2018) + tuning
│   └── calibration.py               # Probability calibration (isotonic/Platt)
├── backtest/
│   ├── signals.py                    # Rank → Long/Short/Hold signals (smoothing, z-scoring, holding)
│   ├── portfolio.py                  # Daily P&L with transaction costs + slippage + invert_signals
│   └── metrics.py                    # Sharpe, Sortino, MDD, Calmar, AUC, Daily AUC, sub-period
├── evaluation/
│   └── metrics_utils.py              # binary_auc_safe, classification_sanity_checks, log_split_balance
├── experiments/
│   ├── lstm_lr_sweep.py              # LSTM learning rate grid sweep
│   ├── tcn_arch_sweep.py             # TCN kernel × levels × channels diagnostic sweep
│   └── train_window_sweep.py         # Train-window length comparison (~1y / ~3y / ~5y)
├── analysis/
│   └── feature_correlation.py        # Feature correlation analysis and selection
├── outputs/
│   └── figures/                      # Generated plots (cumulative returns, feature importance, etc.)
├── reports/                          # Output tables and CSVs (universe-prefixed)
│   ├── large_cap_table_T5_*.csv      # Gross/net returns (large-cap)
│   ├── large_cap_table_T6_*.csv      # Sub-period performance (large-cap)
│   ├── large_cap_table_T8_*.csv      # Classification metrics (large-cap)
│   ├── large_cap_feature_importances_*.csv  # Per-fold + averaged feature importances
│   ├── large_cap_fold_sharpe_per_model.csv  # Fold-level Sharpe diagnostics
│   ├── large_cap_lstm_tuning_results.csv    # LSTM hyperparameter tuning results
│   ├── large_cap_tcn_tuning_results.csv     # TCN hyperparameter tuning results
│   ├── large_cap_full_predictions.csv       # Raw per-fold model probabilities
│   ├── large_cap_daily_returns_*.csv        # Daily return series
│   ├── large_cap_signals_all_models.csv     # All model signals
│   ├── large_cap_backtest_summary.txt       # Human-readable summary
│   ├── small_cap_*/                  # Same outputs for small-cap universe
│   ├── fold_reports/                 # Per-fold JSON diagnostic artifacts
│   └── training_logs/               # LSTM + TCN epoch-level training logs (CSV)
├── verify_bug_fixes.py               # Verification script for known bug fixes
└── notebooks/                        # Visualisation & reporting (placeholder)
```

---

## Environment Setup

```bash
# Core data & ML
pip install yfinance pandas numpy scikit-learn xgboost tqdm

# PyTorch (Apple Silicon MPS support)
pip install torch torchvision

# Wavelet denoising (Bhandari extension)
pip install PyWavelets

# Visualization & stats
pip install matplotlib seaborn plotly scipy statsmodels
```

**M4 device selection (always include at top of training scripts):**
Device priority: CUDA → MPS → CPU. The pipeline auto-detects in `main.py`.

---

## config.py — Single Source of Truth

### Universe Configuration

The active universe is controlled by `UNIVERSE_MODE` ("large_cap" | "small_cap").

A `UniverseConfig` dataclass captures all universe-specific settings:
- `tickers`, `baseline_feature_cols`, `lstm_b_feature_cols`
- `invert_signals` — large-cap inverts portfolio direction (mean-reversion)
- `invert_features` — flips momentum features for reversal-oriented universes
- `sector_min_size`, `sector_winsorize`, `sector_winsorize_pct`
- `k_stocks`, `include_lstm_b_in_ensemble`, `include_tcn_in_ensemble`

**Active config objects:** `LARGE_CAP_CONFIG` and `SMALL_CAP_CONFIG`.

### Key Parameters (current values)

| Parameter | Value | Notes |
|---|---|---|
| `UNIVERSE_MODE` | `"large_cap"` | Toggle to `"small_cap"` for comparison |
| `START_DATE` | `2019-01-01` | 6-year sample |
| `END_DATE` | `2024-12-31` | |
| `TRAIN_DAYS` | 252 | ~1 trading year |
| `VAL_DAYS` | 63 | ~1 quarter |
| `TEST_DAYS` | 63 | ~1 quarter |
| `MAX_FOLDS` | None | Set to limit folds for development runs |
| `TRAIN_WINDOW_MODE` | `"rolling"` | "rolling" or "expanding" |
| `WALK_FORWARD_STRIDE` | None | Defaults to TEST_DAYS |
| `SEQ_LEN` | 30 | LSTM lookback window (trading days) |
| `K_STOCKS` | 5 | Long top-5, short bottom-5 per day |
| `TC_BPS` | 5 | Transaction cost per half-turn |
| `SLIPPAGE_BPS` | 0.0 | Extra execution cost (0 = off) |
| `TARGET_HORIZON_DAYS` | 21 | Forward return horizon for Target (1/5/21) |
| `RANDOM_SEED` | 42 | |

### Signal Parameters

| Parameter | Value | Notes |
|---|---|---|
| `SIGNAL_SMOOTH_ALPHA` | 0.0 | EMA alpha (0 = no smoothing) |
| `SIGNAL_EMA_METHOD` | `"alpha"` | "alpha" or "span" |
| `SIGNAL_CONFIDENCE_THRESHOLD` | 0.55 | z-score threshold; sit out low-conviction signals |
| `SIGNAL_USE_ZSCORE` | True | Cross-sectional z-scoring before ranking |
| `MIN_HOLDING_DAYS` | 5 | Minimum holding period constraint |
| `RUN_SIGNAL_ABLATION` | False | Compare raw-rank vs full pipeline |

### Feature Sets

**Master feature union** (`ALL_FEATURE_COLS`) — dynamically built, includes:
- Core: `Return_1d`, `NegReturn_1d`, `Return_5d`, `NegReturn_5d`, `Return_21d`, `RSI_14`, `MACD`, `ATR_14`, `BB_PctB`, `RealVol_20d`, `Volume_Ratio`, `SectorRelReturn`
- Reversal aliases: `RSI_Reversal`, `NegMACD`, `BB_Reversal`
- Market context (if `MARKET_FEATURES_ENABLED=True`): `Market_Return_1d/5d/21d`, `Market_Vol_20d/60d`, `RelToMarket_1d/5d/21d`, `Beta_60d`
- Sector context (if `SECTOR_FEATURES_ENABLED=True`): `Sector_Return_1d/5d/21d`, `Sector_Vol_20d/60d`, `SectorRelZ_Return_1d`

**LSTM features** (`LSTM_B_FEATURE_COLS`): `Return_1d`, `Return_5d`, `Return_21d`, `RSI_14`, `BB_PctB`, `RealVol_20d`, `Volume_Ratio`, `SectorRelReturn` + market/sector context features (if enabled by `LSTM_MARKET_FEATURES_ENABLED` / `LSTM_SECTOR_FEATURES_ENABLED`).

**TCN features**: Same as LSTM by default (`TCN_FEATURE_COLS_FULL = LSTM_B_FEATURE_COLS`). A `core`-only subset (`TCN_FEATURE_COLS_CORE`) is also available and may be selected during tuning.

**Baseline features** (`BASELINE_FEATURE_COLS`): Core features + market features (`BASELINE_MARKET_FEATURES_ENABLED=True`) but **no sector features** (`BASELINE_SECTOR_FEATURES_ENABLED=False`).

**Per-model feature flags** (control which computed features each model receives):
| Flag | LR/RF/XGBoost | LSTM | TCN |
|---|---|---|---|
| `*_MARKET_FEATURES_ENABLED` | True | True | True (via full set) |
| `*_SECTOR_FEATURES_ENABLED` | False | True | True (via full set) |

### LSTM Architecture (Fixed)

| Parameter | Value |
|---|---|
| Hidden size | 32 |
| Num layers | 1 |
| Dropout | 0.0 |
| Learning rate | 0.001 |
| Batch size | 256 |
| Optimizer | Adam |
| Sequence length | 30 |
| Max epochs | 200 |
| Early stopping patience | 15 |
| LR scheduler patience | 7 |
| LR scheduler factor | 0.5 |
| Weight decay | 1e-4 |

**Tuning:** `LSTM_B_ENABLE_TUNING` (currently True), `LSTM_B_TUNE_ON_FIRST_FOLD_ONLY` (True — tune on fold 1 and reuse).
Selection criterion: validation net Sharpe (then annualised return as tiebreaker) — NOT AUC alone.
Return-aware guardrail: tuned candidate is only accepted if it outperforms the default on val trading metrics.

### TCN Architecture (Fixed defaults, optional tuning)

| Parameter | Value | Grid (if tuned) |
|---|---|---|
| Num channels | [16, 16, 16] (3 levels × 16 filters) | [[16,16,16], [32,32,32], [32,32,32,32]] |
| Kernel size | 3 | [3, 5] |
| Dropout | 0.2 | [0.1, 0.2, 0.3] |
| Learning rate | 0.001 | [3e-4, 1e-3, 3e-3] |
| Batch size | 256 | [64, 128] |
| Optimizer | Adam | [adam, nadam] |
| Label smoothing | 0.1 | — |
| Weight norm | False | — |
| Sequence length | 30 | — |
| Max epochs | 200 (35 during tuning) | — |
| Early stopping patience | 15 (4 during tuning) | — |
| LR scheduler patience | 7 | — |
| LR scheduler factor | 0.5 | — |
| Weight decay | 1e-4 | — |
| Feature set | full | [core, full] |

**TCN tuning:** `TCN_ENABLE_TUNING=True`, `TCN_TUNE_ON_FIRST_FOLD_ONLY=True` (same first-fold pattern as LSTM).
Two-phase tuning: Phase 1 = optimizer/lr/batch; Phase 2 = architecture + feature set.
`TCN_USE_WEIGHT_NORM=False` — disabled because weight_norm causes MPS non-determinism and breaks kaiming init.

### LSTM Diagnostics

| Parameter | Value |
|---|---|
| `LSTM_LOG_EVERY_EPOCH` | True |
| `LSTM_SAVE_TRAINING_CSV` | True |
| `LSTM_AUDIT_GRAD_NORM` | True |
| `LSTM_MAX_GRAD_NORM` | 1.0 (gradient clipping) |
| `LSTM_FLAT_AUC_WARN_epochs` | 8 |
| `LSTM_OVERFIT_LOSS_RATIO` | 3.0 |

### Normalization & Regularization

- `SCALER_TYPE`: "standard" (Z-score normalisation)
- `WINSORIZE_ENABLED`: **False** (disabled — enable for small-cap where extreme outliers destabilise training)
- `WINSORIZE_LOWER_Q`: 0.005, `WINSORIZE_UPPER_Q`: 0.995

### Wavelet Denoising

`USE_WAVELET_DENOISING`: **False** (disabled by default — domain-shift risk in OOS evaluation).
When enabled: Haar wavelet, level 1, soft thresholding, causal rolling window of 128 days.

---

## Step 1 — Data Download & Cleaning (`pipeline/data_loader.py`)

**Status:** ✅ Implemented

- Downloads OHLCV via `yfinance` with `auto_adjust=True` (handles splits/dividends).
- Restructures from MultiIndex to long format: `[Date, Open, High, Low, Close, Volume, Ticker]`.
- Cleaning: removes Volume=0 days, forward-fills gaps ≤2 days, flags `|Return_1d| > 20%`.

---

## Step 2 — Feature Engineering (`pipeline/features.py`)

**Status:** ✅ Implemented with technical + context features

### 2a. Technical Features (computed per ticker from raw Close)

| Feature | Formula | Notes |
|---|---|---|
| `Return_1d` | 1-day simple return | Daily momentum |
| `Return_5d` | 5-day simple return | Weekly momentum |
| `Return_21d` | 21-day simple return | Monthly momentum |
| `RSI_14` | Relative Strength Index, 14-day | Overbought / oversold |
| `MACD` | 12d EMA – 26d EMA | Trend momentum |
| `ATR_14` | Average True Range, 14-day | Daily volatility |
| `BB_PctB` | (Close–Lower)/(Upper–Lower), 20d | Bollinger Band position |
| `RealVol_20d` | Annualised 20-day realised volatility | Volatility regime |
| `Volume_Ratio` | Volume / 20d avg volume | Unusual activity |

Reversal aliases computed alongside (same underlying data, sign-flipped or complement):
`NegReturn_1d`, `NegReturn_5d`, `RSI_Reversal`, `NegMACD`, `BB_Reversal`.

### 2b. Cross-Sectional Feature (`SectorRelReturn`)

Leave-one-out sector mean return: each stock's `Return_1d` minus the equal-weighted mean
return of all other stocks in the same sector on that date. Optional winsorization
at `sector_winsorize_pct` percentile cross-sectionally per date.

### 2c. Market Context Features (optional, `MARKET_FEATURES_ENABLED`)

Cross-sectional mean returns at 1d/5d/21d horizons, rolling market volatility at 20d/60d,
excess returns vs market at 1d/5d/21d, and rolling 60-day CAPM-style beta.
All computed causally from same-day or historical data only.

### 2d. Sector Context Features (optional, `SECTOR_FEATURES_ENABLED`)

Leave-one-out sector mean returns at 1d/5d/21d, rolling sector volatility at 20d/60d,
and within-(Date, Sector) z-score of 1d return.

### 2e. Causal Wavelet Denoising (optional, `USE_WAVELET_DENOISING=False`)

When enabled: Haar wavelet soft denoising applied per fold using training-only thresholds.
Rolling-window causal approach: each value at time t uses only data from [t-128, t].
Targets are preserved based on raw returns after denoising.

**Critical anti-leakage measures:**
1. Thresholds computed from **training data only** per fold
2. Causal denoising: each value uses only **historical data** (rolling window)
3. Applied to each split **independently** (no concatenation)
4. **Targets remain based on raw returns** (not denoised) for honest evaluation

---

## Step 3 — Target Variable (`pipeline/targets.py`)

**Status:** ✅ Implemented with configurable horizon

**Two output columns:**
- `Return_NextDay` — always the 1-day forward return; used by `portfolio.py` for P&L only.
- `Target` — binary label: 1 if stock's `TARGET_HORIZON_DAYS`-day forward return ≥ cross-sectional median, else 0.

**Current setting:** `TARGET_HORIZON_DAYS = 21` (monthly momentum target for large-cap).
Rationale: 1-day predictability is near-zero for large-cap liquid stocks; multi-day horizons
yield more persistent, learnable signals (Jegadeesh & Titman 1993).

Cache horizon is embedded in the cached CSV (`_target_horizon` column) so that a
config change automatically triggers target recomputation on next load.

---

## Step 4 — Walk-Forward Fold Generator (`pipeline/walk_forward.py`)

**Status:** ✅ Implemented with rolling and expanding modes

```
Windows: Train=252d / Val=63d / Test=63d (rolling by default)
Date range: 2019-01-01 → 2024-12-31

Rolling mode:
  Fold 1: |---Train 252---|---Val 63---|---Test 63---|
  Fold 2:                 |---Train 252---|---Val 63---|---Test 63---|
  ...

Expanding mode:
  Fold 1: |---Train 252---|---Val 63---|---Test 63---|
  Fold 2: |------Train 315------|---Val 63---|---Test 63---|
  ...
```

Minimum 8 folds enforced (assertion). `MAX_FOLDS` config cap for development.
Configurable `stride_days` (default = `TEST_DAYS`). Full boundary dates stored in each fold dict.

---

## Step 5 — Preprocessing (`pipeline/standardizer.py`)

**Status:** ✅ Implemented with winsorizer + dual-scaler approach

Pipeline per fold:
1. **Winsorize** (if `WINSORIZE_ENABLED`): clip features at 0.5th/99.5th percentile — fit on train rows, applied to all splits.
2. **Standardize**: StandardScaler fit on training data, transform val/test.

Separate scalers are fit for baseline features vs LSTM features (different feature sets).

---

## Step 6 — Models

**Status:** ✅ 5 base models + Ensemble implemented (LSTM-A removed)

### Summary Table

| Model | Features | Architecture | Notes |
|---|---|---|---|
| LR | Baseline features (core + market) | Logistic Regression | L2 regularization, TimeSeriesSplit CV |
| RF | Baseline features (core + market) | Random Forest | Val-based hyperparameter selection |
| XGBoost | Baseline features (core + market) | Gradient Boosting | Early stopping on val AUC |
| LSTM | LSTM features (core + market + sector) | 32h, 1L, 0.0d (fixed) | Optional Bhandari-style tuning |
| TCN | TCN features (core + market + sector) | 3×[16,16,16] dilated residual blocks | Optional two-phase tuning, label smoothing |
| Ensemble | LR + LSTM + TCN | Mean probability | RF and XGBoost excluded (negative Sharpe) |

### 6a. Logistic Regression

L2-regularised with TimeSeriesSplit(n_splits=5) grid-search over C.
Also outputs `LR_coef` (absolute normalized coefficient) for feature importance.

### 6b. Random Forest

Validation-based hyperparameter selection (not CV) to preserve time-series order.
Grid: `n_estimators` [300], `max_depth` [5, 10], `min_samples_leaf` [30, 50].
Outputs `RF_importance` (mean decrease impurity, normalised) for feature importance.

### 6c. XGBoost

Grid search with early stopping on validation AUC.
Grid: `max_depth` [3,4,5], `eta` [0.01], `subsample` [0.6,0.7].
L1 regularization (alpha=0.1), L2 (lambda=1.0), colsample_bytree=0.5.
Outputs `XGB_gain` (average gain, normalised) for feature importance.

### 6d. LSTM

Fixed architecture: 32 hidden, 1 layer, no dropout, batch=256, seq_len=30.
Temporal split (by date, not ticker) to prevent temporal leakage.
Optional Bhandari §3.3-style two-phase tuning (Phase 1: optimizer/lr/batch; Phase 2: arch):
- Tuning is only run on fold 1 (`LSTM_B_TUNE_ON_FIRST_FOLD_ONLY=True`), then reused.
- Selection criterion: best validation **net Sharpe** (trading metric), not AUC.
- Return-aware guardrail: default is kept if tuned candidate underperforms on val returns.
Per-fold permutation importance computed for LSTM (feature drop in AUC after shuffling).
Gradient clipping at `LSTM_MAX_GRAD_NORM=1.0`.

### 6e. TCN (Temporal Convolutional Network)

Bai et al. (2018) architecture: stack of dilated causal `Conv1d` residual blocks (`TemporalBlock`).
Input: `(N, seq_len, n_features)` — transposed internally to `(N, channels, seq_len)` for `Conv1d`.
Final-timestep output feeds a 2-class linear decoder with `CrossEntropyLoss` + label smoothing (0.1).
`Chomp1d` removes right-side padding to enforce causality.
Reuses LSTM infrastructure helpers (`_build_optimizer`, `_build_sequences_multi`, `align_predictions_to_df`, etc.).
Optional two-phase tuning: Phase 1 = optimizer/lr/batch; Phase 2 = architecture + feature set.
Permutation importance computed per fold on test set.

### 6f. Ensemble

Mean probability of **LR + LSTM + TCN** (RF and XGBoost excluded due to negative Sharpe
in large-cap universe). Controlled by `include_lstm_b_in_ensemble` and `include_tcn_in_ensemble` in `UniverseConfig`.

### 6g. Feature Importance (all models)

`extract_feature_importances()` in `models/baselines.py` extracts per-fold importance from
all three baseline models. LSTM uses permutation importance (AUC drop after feature shuffle).
Saved to `reports/{universe}_feature_importances_per_fold.csv` and `..._avg.csv`.

---

## Step 7 — Trading Signals (`backtest/signals.py`)

**Status:** ✅ Implemented with smoothing, holding constraints, and confidence threshold

Pipeline:
1. **EMA smoothing** (if `SIGNAL_SMOOTH_ALPHA > 0`): per-ticker exponential smoothing.
2. **Signal generation**: cross-sectional z-scoring of probabilities, then rank top-K as Long,
   bottom-K as Short. Slots below confidence threshold are filtered to Hold.
3. **Holding period constraint** (`MIN_HOLDING_DAYS=5`): once a position is entered,
   held for at least 5 days before exit or flip.
4. **Diagnostics**: `compute_turnover_and_holding_stats()` reports mean daily turnover and
   average holding period per model.
5. **Signal ablation** (`RUN_SIGNAL_ABLATION=False`): compares raw ranking vs full pipeline.

**`invert_signals` flag** (`INVERT_SIGNALS=True` for large-cap): applied at portfolio level —
Long positions earn Short leg returns and vice versa. Classification metrics evaluated on
`(1 - prob)` when `invert_signals=True`.

---

## Step 8 — Portfolio & Transaction Costs (`backtest/portfolio.py`)

**Status:** ✅ Implemented with slippage support

Equal-weighted long-short portfolio with:
- TC model: `TC_BPS` per half-turn (buy or sell). Flip = 2 half-turns.
- Slippage: `SLIPPAGE_BPS` per half-turn (additive to TC).
- `invert_signals` parameter: passes through from universe config.
- Position fraction: TC per-position scaled by `1 / (2 * K_STOCKS)`.

---

## Step 9 — Performance Metrics (`backtest/metrics.py`)

**Status:** ✅ Implemented with Daily AUC and sub-period analysis

**Risk-return metrics** (Table T5): N Days, Mean Daily Return, Annualised Return, Annualised Std Dev,
Sharpe Ratio, Sortino Ratio, Max Drawdown, Calmar Ratio, Win Rate, VaR 1%.

**Classification metrics** (Table T8): Accuracy, AUC-ROC, F1 Score, Daily AUC (mean/std).
Daily AUC measures within-day ranking quality — more appropriate for cross-sectional strategies.
When `invert_signals=True`, all metrics are computed on `(1 - prob)`.

**Sub-period analysis** (Table T6): Metrics broken down by market regime.

---

## Step 10 — Evaluation Utilities (`evaluation/metrics_utils.py`)

**Status:** ✅ Implemented

- `binary_auc_safe()`: AUC with graceful handling of degenerate cases.
- `classification_sanity_checks()`: validates probabilities are in [0,1], non-constant, etc.
- `log_split_balance()`: logs class balance per fold split to detect target drift.

---

## Step 11 — Fold Reporting (`pipeline/fold_reporting.py`)

**Status:** ✅ Implemented

When `SAVE_FOLD_REPORTS=True`, saves a per-fold JSON artifact under
`reports/fold_reports/fold_{N}.json` with split sizes, dates, class balance,
and validation AUC for each fold. Stale fold reports are cleared at the start of each run.

---

## Step 12 — main.py Pipeline Overview

**Status:** ✅ Implemented with full orchestration

Entry points:
- `main(load_cached=True)` — CLI entry point.
- `run_walk_forward_pipeline(load_cached, train_days, reports_dir)` — full pipeline.

**Pipeline flags:**
- `RUN_BASELINES` (True): enable/disable LR, RF, XGBoost.
- `RUN_LSTMS` (True): enable/disable LSTM.
- `RUN_TCNS` (True): enable/disable TCN.
- All results prefixed by `config.UNIVERSE_MODE` in reports dir.

**Key implementation highlights:**
1. **Universe-aware config**: `_UNIVERSE_CFG` drives `invert_signals`, `k_stocks`, feature sets.
2. **Deterministic fold seeds**: `fold_seed_base = RANDOM_SEED + fold * 1000`.
3. **Stale artifact cleanup**: fold_reports/ and training_logs/ cleared at run start.
4. **Cache horizon check**: detects `TARGET_HORIZON_DAYS` mismatch and auto-recomputes.
5. **Causal wavelet denoising** (when enabled): thresholds from train only, applied per-split.
6. **Winsorization + standardisation**: per fold, fit on train only.
7. **LSTM tuning**: optional, first-fold only, return-aware selection.
8. **TCN tuning**: optional, first-fold only, two-phase (training hyperparams + architecture).
9. **LSTM permutation importance**: computed per fold on test set.
10. **TCN permutation importance**: computed per fold on test set.
11. **Signal pipeline**: smoothing → z-score → threshold → holding constraint.
12. **Ensemble**: LR + LSTM + TCN (RF/XGBoost excluded empirically).
13. **Per-fold fold reports** saved when `SAVE_FOLD_REPORTS=True`.
14. **Full predictions CSV** saved for downstream stitching (`combine_and_backtest.py`).

---

## Anti-Leakage Rules

| Rule | How to enforce |
|---|---|
| Standardize on training data only | `scaler.fit_transform(X_train)`, then `.transform()` on val/test |
| Winsorize on training quantiles only | `winsorize_fold()` fits on train, applies to all splits |
| Wavelet thresholds from train only | `compute_wavelet_thresholds(df_train)`, apply to all splits |
| **CAUSAL wavelet denoising** | **Rolling window: each value uses only historical data** |
| Never shuffle time-series | `shuffle=False` for all val/test DataLoaders |
| Tune hyperparameters on val, not test | All early stopping uses val loss / val Sharpe only |
| Features use only data ≤ t | All rolling windows are causal (no `center=True`) |
| Target uses only realised N-day returns | `shift(-N)` on actual realised close prices |
| **Temporal LSTM split** | **Split by date, not by ticker** (prevents leakage across time) |
| **Target unchanged after denoising** | **Targets remain based on raw returns** (honest evaluation) |
| Cache horizon tag | `_target_horizon` column detects and repairs stale caches |
| Fold seed determinism | `fold_seed_base = RANDOM_SEED + fold * 1000` |

---

## Required Thesis Outputs

### Tables

| # | Title | Output File | Status |
|---|---|---|---|
| T1 | Descriptive stats per stock | Manual analysis | 📝 TODO |
| T2 | Walk-forward fold dates | `print_fold_summary()` | ✅ |
| T3 | Hyperparameter configurations | config.py | ✅ |
| T4 | Daily return characteristics | `daily_returns_*.csv` | ✅ |
| T5 | Annualized risk-return metrics | `{universe}_table_T5_*.csv` | ✅ |
| T6 | Sub-period performance | `{universe}_table_T6_*.csv` | ✅ |
| T7 | TC sensitivity | `compute_tc_sensitivity()` | ✅ |
| T8 | Classification metrics | `{universe}_table_T8_*.csv` | ✅ (with Daily AUC) |
| T9 | Feature importance | `{universe}_feature_importances_*.csv` | ✅ |

### Figures

| # | Title | How to create | Status |
|---|---|---|---|
| F1 | Walk-forward timeline | matplotlib gantt chart | 📝 TODO |
| F2 | LSTM architecture diagram | matplotlib / PowerPoint | 📝 TODO |
| F3 | Cumulative equity curves | `(1 + r).cumprod()` | 📝 TODO |
| F4 | Drawdown underwater chart | Rolling max drawdown | 📝 TODO |
| F5 | Feature importance bar chart | `generate_feature_importance_plot.py` | ✅ |
| F6 | Return distribution | seaborn.violinplot | 📝 TODO |
| F7 | Sharpe vs TC | Loop over TC values | 📝 TODO |
| F8 | Confusion matrices | sklearn + seaborn | 📝 TODO |
| F9 | Sub-period performance | seaborn.barplot | 📝 TODO |
| F10 | Correlation heatmap | seaborn.heatmap | 📝 TODO |
| F11 | LSTM train/val loss curves | Saved epoch CSV in training_logs/ | 📝 TODO |
| F12 | Feature correlation heatmap | `analysis/feature_correlation.py` | 📝 TODO |

### Generated Reports

| File | Description | Status |
|---|---|---|
| `{universe}_backtest_summary.txt` | Human-readable performance summary | ✅ |
| `{universe}_signals_all_models.csv` | All model signals for analysis | ✅ |
| `{universe}_lstm_tuning_results.csv` | LSTM hyperparameter tuning results | ✅ |
| `{universe}_feature_importances_*.csv` | Per-fold and averaged feature importances | ✅ |
| `{universe}_full_predictions.csv` | Raw per-fold model probabilities | ✅ |
| `{universe}_fold_sharpe_per_model.csv` | Fold-level Sharpe diagnostics | ✅ |
| `fold_reports/fold_N.json` | Per-fold diagnostic artifacts | ✅ |
| `training_logs/` | LSTM epoch-level training CSVs | ✅ |

---

## Sub-Period Analysis

| Period | Dates | Regime |
|---|---|---|
| Pre-COVID | 2019-01-01 to 2020-02-19 | Normal bull market |
| COVID crash | 2020-02-20 to 2020-04-30 | Extreme volatility / dislocation |
| Recovery / bull | 2020-05-01 to 2021-12-31 | Low rates, momentum-driven |
| 2022 bear market | 2022-01-01 to 2022-12-31 | Rate hikes, drawdowns |
| 2023-2024 AI rally | 2023-01-01 to 2024-12-31 | AI / large-cap concentration |

---

## Experiments

### `experiments/train_window_sweep.py`

Compares walk-forward performance across train window lengths:
`TRAIN_DAYS_CANDIDATES = [504, 756, 1260]` (~2y / ~3y / ~5y).
Results saved to `reports/train_window_comparison.csv`.

### `experiments/lstm_lr_sweep.py`

LSTM learning rate grid sweep: `LSTM_LR_GRID = [0.0005, 0.001, 0.003, 0.005]`.
Capped at `LSTM_LR_SWEEP_MAX_EPOCHS = 40` for budget control.

### `experiments/tcn_arch_sweep.py`

Quick diagnostic sweep of TCN kernel × levels × channels on fold 0 only.
Grid: `TCN_SWEEP_KERNEL_GRID=[2,3,5]`, `TCN_SWEEP_LEVELS_GRID=[3,4,5]`, `TCN_SWEEP_CHANNEL_GRID=[32,64]`.
Capped at `TCN_SWEEP_MAX_EPOCHS=40`. Reports receptive field, param count, AUC.

### `combine_and_backtest.py`

Stitches together separate baseline and LSTM run predictions when `RUN_BASELINES` and
`RUN_LSTMS` are run independently (e.g., for parallelism or debugging).
Loads `{universe}_full_predictions.csv` from both runs and runs combined backtest.

---

## Hyperparameter Reference

### LSTM (Fixed architecture, optional tuning)

| Parameter | Value | Grid (if tuned) |
|---|---|---|
| Hidden size | 32 | [32, 64] |
| Num layers | 1 | [1, 2] |
| Dropout | 0.0 | [0.0, 0.2] |
| Learning rate | 0.001 | [0.0003, 0.001, 0.003] |
| Batch size | 256 (128 during tuning) | [64, 128] |
| Optimizer | Adam | [adam, nadam] |
| Sequence length | 30 | — |
| Max epochs | 200 (35 during tuning) | — |
| Early stopping patience | 15 (4 during tuning) | — |
| LR scheduler patience | 7 | — |
| LR scheduler factor | 0.5 | — |
| Weight decay | 1e-4 | — |
| Gradient clip norm | 1.0 | — |

### TCN (Fixed defaults, optional two-phase tuning)

| Parameter | Value | Grid (if tuned) |
|---|---|---|
| Num channels | [16, 16, 16] | [[16,16,16], [32,32,32], [32,32,32,32]] |
| Kernel size | 3 | [3, 5] |
| Dropout | 0.2 | [0.1, 0.2, 0.3] |
| Learning rate | 0.001 | [3e-4, 1e-3, 3e-3] |
| Batch size | 256 | [64, 128] |
| Optimizer | Adam | [adam, nadam] |
| Label smoothing | 0.1 | — |
| Weight norm | False | — |
| Sequence length | 30 | — |
| Max epochs | 200 (35 during tuning) | — |
| Early stopping patience | 15 (4 during tuning) | — |
| LR scheduler patience | 7 | — |
| LR scheduler factor | 0.5 | — |
| Weight decay | 1e-4 | — |
| Feature set | full | [core, full] |

### XGBoost

| Parameter | Grid | Fixed |
|---|---|---|
| max_depth | 3, 4, 5 | |
| eta | 0.01 | |
| subsample | 0.6, 0.7 | |
| colsample_bytree | | 0.5 |
| alpha (L1) | | 0.1 |
| lambda (L2) | | 1.0 |
| num_boost_round | | 1000 (early stop @ 50) |

### Random Forest

| Parameter | Grid |
|---|---|
| n_estimators | 300 |
| max_depth | 5, 10 |
| min_samples_leaf | 30, 50 |
| max_features | sqrt |

### Logistic Regression

| Parameter | Value |
|---|---|
| C | 1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0 |
| penalty | L2 |
| CV | TimeSeriesSplit(5) |

### Trading

| Parameter | Value | Notes |
|---|---|---|
| k (long/short stocks) | 5 | 5 per side out of 50 large-cap stocks |
| TC (bps) | 5 | Per half-turn |
| Slippage (bps) | 0.0 | Off by default |
| Signal smoothing alpha | 0.0 | No EMA smoothing (disabled) |
| Confidence threshold | 0.55 | Sit out low-conviction z-scores |
| Z-score normalization | True | Cross-sectional |
| Minimum holding days | 5 | Enforce holding period |

---

## Implementation Status Summary

### ✅ Completed

1. **Data Pipeline**
   - Download & cleaning (yfinance)
   - Technical features: Return_1d/5d/21d, RSI_14, MACD, ATR_14, BB_PctB, RealVol_20d, Volume_Ratio
   - Reversal aliases: NegReturn_1d/5d, RSI_Reversal, NegMACD, BB_Reversal
   - Cross-sectional SectorRelReturn (LOO, configurable winsorization)
   - Market context features (Market returns, vol, RelToMarket, Beta)
   - Sector context features (Sector returns, vol, SectorRelZ)
   - Causal wavelet denoising (rolling window, disabled by default)
   - Configurable N-day cross-sectional median target (currently 21d)
   - Walk-forward fold generator (rolling + expanding modes)
   - Winsorization + standardisation pipeline

2. **Models**
   - Logistic Regression (CV grid search + feature importance)
   - Random Forest (validation-based selection + feature importance)
   - XGBoost (early stopping + feature importance)
   - LSTM (fixed architecture, optional first-fold tuning, return-aware selection)
   - TCN (dilated causal Conv1d residual blocks, optional two-phase tuning, label smoothing)
   - Ensemble (LR + LSTM + TCN mean probability)
   - LSTM permutation importance
   - TCN permutation importance

3. **Backtest**
   - Signal generation (z-scoring, confidence threshold)
   - Holding period constraint
   - Transaction cost + slippage model
   - invert_signals support (large-cap mean reversion)
   - Comprehensive metrics (Sharpe, Sortino, MDD, Calmar, Win Rate, VaR)
   - Daily AUC (within-day ranking quality)
   - Sub-period analysis
   - Signal ablation study support

4. **Reports & Diagnostics**
   - Table T5 (gross/net returns, universe-prefixed)
   - Table T6 (sub-period performance)
   - Table T8 (classification + Daily AUC)
   - Feature importances (per-fold + averaged, all 4 models)
   - Daily returns CSV
   - Signals CSV
   - LSTM tuning results CSV
   - Full predictions CSV
   - Per-fold fold reports (JSON)
   - LSTM epoch-level training logs

5. **Experiments**
   - Train window sweep (504/756/1260 days)
   - LSTM learning rate sweep
   - TCN architecture sweep (kernel × levels × channels)

### 📝 TODO

1. **Visualization**
   - Figures F1-F4, F6-F12
   - Cumulative return curves with drawdown overlay
   - LSTM loss curves from training_logs/

2. **Analysis**
   - Descriptive statistics (Table T1)
   - TC sensitivity analysis plots (T7)

3. **Documentation**
   - Thesis write-up
   - Results interpretation
   - Ablation study findings

---

## Realistic Expectations

- **Directional accuracy:** 51-54% is a solid result (random = 50%). With a 21-day target, slightly higher predictability is achievable for large-cap.
- **Large-cap inversion:** `invert_signals=True` means the model identifies stocks that will underperform and the portfolio shorts the model's "high probability" stocks — mean reversion in liquid large-caps.
- **Sharpe ratio:** With 50 stocks and k=5, diversification is limited. Expect lower Sharpe than the papers' 5.83 (which used 500 stocks with k=50).
- **LSTM:** Small architecture (32h, 1L) empirically best for the small-data regime (50 stocks, 252-day train window).
- **Ensemble:** LR + LSTM + TCN; RF and XGBoost found to have negative Sharpe in large-cap universe.
- **Wavelet denoising:** Disabled by default — domain shift risk when applied across OOS folds.
- **COVID crash (Feb-Apr 2020):** Expect elevated returns — extreme cross-sectional dispersion creates exploitable patterns.
- **2022 bear market:** Expect elevated drawdown. Analyse separately in sub-period section.

---

## Key Technical Changes from Original Design

### 1. Universe Mode System
**Change:** Single 70-stock fixed universe → configurable `UNIVERSE_MODE` with `UniverseConfig` dataclass.
**Why:** Enables comparison of large-cap momentum vs small-cap strategies in one codebase.

### 2. Removal of LSTM-A
**Change:** LSTM-A (6 features, two-phase tuning per fold) removed.
**Why:** LSTM with return-aware first-fold tuning is sufficient; LSTM-A added run-time without benefit.

### 3. Configurable Target Horizon
**Change:** Fixed 1-day target → configurable `TARGET_HORIZON_DAYS` (currently 21).
**Why:** 1-day predictability is near-zero for liquid large-caps; 21d target follows Jegadeesh & Titman (1993).

### 4. Walk-Forward Structure
**Change:** 500/125/125 days → 252/63/63 days (1 year / 1 quarter / 1 quarter).
**Why:** Matches standard practitioner fold sizing; 6-year sample requires shorter windows.

### 5. Signal Inversion for Large-Cap
**Change:** `invert_signals=True` in `LARGE_CAP_CONFIG`.
**Why:** Large liquid stocks exhibit mean reversion; shorting model "winners" earns positive returns.

### 6. Return-Aware LSTM Tuning
**Change:** Tuning selection by val Sharpe (not AUC); first fold only.
**Why:** AUC does not correlate reliably with trading performance. Return-aware selection is more practical.

### 7. Market & Sector Context Features
**Change:** Added market-wide and sector-level features as optional feature groups.
**Why:** Relative-to-market and relative-to-sector features provide richer cross-sectional context.

### 8. Winsorization
**Change:** Added quantile clipping before standardisation (currently disabled, `WINSORIZE_ENABLED=False`).
**Why:** Extreme feature outliers (especially in small-cap) destabilise training. Disabled in large-cap where outliers are less extreme.

### 9. TCN Model Added
**Change:** `models/tcn_model.py` implementing Bai et al. (2018) Temporal Convolutional Network alongside LSTM.
**Why:** Provides a second neural-network model for comparison; TCN's parallelisable dilated convolutions complement LSTM's recurrent inductive bias. TCN is included in the Ensemble.

### 10. Per-Model Feature Differentiation
**Change:** Baselines (LR/RF/XGBoost) use market features but not sector features; LSTM and TCN use both.
**Why:** Sector features add noise for tree/linear models at this universe size but provide useful cross-sectional context for sequence models.

---

## Key References

- Fischer, T. & Krauss, C. (2017). *Deep learning with long short-term memory networks for financial market predictions.* FAU Discussion Papers in Economics, No. 11/2017.
- Krauss, C., Do, X.A. & Huck, N. (2017). *Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500.* European Journal of Operational Research, 259, 689-702.
- Bhandari, H.N., et al. (2022). *Predicting stock market index using LSTM.* Machine Learning with Applications.
- Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.
- Jegadeesh, N. & Titman, S. (1993). Returns to buying winners and selling losers. *Journal of Finance*, 48(1), 65-91.
- Fama, E. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work. *Journal of Finance*, 25(2), 383-417.
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.
- Donoho, D.L. & Johnstone, I.M. (1994). Ideal spatial adaptation by wavelet shrinkage. *Biometrika*, 81(3), 425-455.
- Bai, S., Kolter, J.Z. & Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.* arXiv:1803.01271.

---

**END OF CLAUDE.MD**
