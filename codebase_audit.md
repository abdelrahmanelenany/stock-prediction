## SECTION 1 — CONFIG SNAPSHOT

### config.py (verbatim)

```python
# config.py — Single source of truth for all hyperparameters and constants
# Implements Bhandari et al. (2022) extensions from IMPLEMENTATION_EXTENSIONS.md
# Universe-mode setup supports large-cap vs relative small-cap S&P 500 experiments.
# =============================================================================
# 0. UNIVERSE MODE — toggle between large-cap and small-cap experiments
# =============================================================================
UNIVERSE_MODE = "small_cap"   # Options: "large_cap" | "small_cap"

# Large-cap: 30 S&P 500 large caps balanced across 5 sectors
LARGE_CAP_TICKERS = [
    # Technology (6)
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META',
    # Finance (6)
    'JPM', 'V', 'MA', 'BRK-B', 'GS', 'BAC',
    # Healthcare (6)
    'JNJ', 'LLY', 'UNH', 'ABBV', 'MRK', 'AMGN',
    # Consumer (6)
    'HD', 'MCD', 'KO', 'WMT', 'COST', 'NKE',
    # Industrial (6)
    'CAT', 'HON', 'UPS', 'UNP', 'GE', 'LMT',
]  # Total: 30

# Small-cap: 30 TRUE small-cap stocks (Russell 2000 / S&P SmallCap 600 constituents)
# Market cap range: ~300M – 5B USD (actual small-cap territory)
# Better reflects size-factor effects vs S&P 500 "pseudo small caps"

SMALL_CAP_TICKERS = [
    # Technology / Growth
    'SMCI', 'FSLY', 'AI', 'PLUG', 'RUN', 'ARRY',
    
    # Healthcare / Biotech
    'NVAX', 'ICPT', 'SRPT', 'BLUE', 'EXEL', 'IONS',
    
    # Consumer / Retail
    'GME', 'BOOT', 'CROX', 'SHOO', 'CAL', 'MOV',
    
    # Industrials / Manufacturing
    'AA', 'CLF', 'X', 'ATI', 'WCC', 'LPX',
    
    # Financials / REITs
    'FHN', 'ZION', 'CMA', 'PACW', 'NYCB', 'STWD'
]  # Total: 30

# Active ticker list — set by UNIVERSE_MODE
TICKERS = LARGE_CAP_TICKERS if UNIVERSE_MODE == "large_cap" else SMALL_CAP_TICKERS
N_STOCKS = len(TICKERS)

LARGE_CAP_SECTOR_MAP = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech', 'GOOGL': 'Tech',
    'AMZN': 'Tech', 'META': 'Tech',
    'JPM': 'Finance', 'V': 'Finance', 'MA': 'Finance', 'BRK-B': 'Finance', 'GS': 'Finance', 'BAC': 'Finance',
    'JNJ': 'Healthcare', 'LLY': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare', 'MRK': 'Healthcare', 'AMGN': 'Healthcare',
    'HD': 'Consumer', 'MCD': 'Consumer', 'KO': 'Consumer', 'WMT': 'Consumer', 'COST': 'Consumer', 'NKE': 'Consumer',
    'CAT': 'Industrial', 'HON': 'Industrial', 'UPS': 'Industrial', 'UNP': 'Industrial', 'GE': 'Industrial', 'LMT': 'Industrial',
}

SMALL_CAP_SECTOR_MAP = {
    'HAS': 'Consumer', 'MHK': 'Consumer', 'PVH': 'Consumer', 'RL': 'Consumer', 'TPR': 'Consumer',
    'DVA': 'Healthcare', 'BEN': 'Finance', 'IVZ': 'Finance', 'NCLH': 'Consumer', 'CCL': 'Consumer',
    'FMC': 'Materials', 'AES': 'Utilities', 'SEE': 'Materials', 'AIZ': 'Finance', 'PARA': 'Comm',
    'WHR': 'Consumer', 'MOS': 'Materials', 'NWL': 'Consumer', 'HSIC': 'Healthcare', 'FRT': 'REIT',
}

SECTOR_MAP = LARGE_CAP_SECTOR_MAP if UNIVERSE_MODE == "large_cap" else SMALL_CAP_SECTOR_MAP

# ── Development Mode: faster iteration with shorter sequences and larger batches ───
DEV_MODE = True  # Set False for final thesis run

START_DATE = '2019-01-01'
END_DATE   = '2024-12-31'

# Walk-forward fold structure
TRAIN_DAYS = 252   # 1 trading year
VAL_DAYS   = 63    # 1 quarter
TEST_DAYS  = 63    # 1 quarter
MAX_FOLDS  = None     # development cap; set None for full walk-forward run

# Walk-forward: stride between folds (None = roll by one test window)
WALK_FORWARD_STRIDE = None  # resolved to TEST_DAYS when None
# "rolling" = fixed-length train window slides; "expanding" = train always from index 0 to val start
TRAIN_WINDOW_MODE = "rolling"

# Train-window experiment grid (~2y / 3y / 5y trading days at 252 d/y)
TRAIN_DAYS_CANDIDATES = [504, 756, 1260]

# Optional train-only quantile clipping applied before scaler (fit quantiles on train rows)
WINSORIZE_ENABLED = False
WINSORIZE_LOWER_Q = 0.005
WINSORIZE_UPPER_Q = 0.995

# Fold-level JSON/CSV artifacts under reports/fold_reports/
SAVE_FOLD_REPORTS = True

# Extra execution cost (same half-turn structure as TC_BPS); 0 = off
SLIPPAGE_BPS = 0.0

# Signal EMA: use alpha (SIGNAL_SMOOTH_ALPHA) or pandas span
SIGNAL_EMA_METHOD = "alpha"  # "alpha" | "span"
SIGNAL_EMA_SPAN = None  # if set and METHOD=span, e.g. 10

# Run raw ranking vs full post-process pipeline; writes reports/signal_ablation_summary.csv
RUN_SIGNAL_ABLATION = False

# LSTM training audit / diagnostics
LSTM_LOG_EVERY_EPOCH = True
LSTM_SAVE_TRAINING_CSV = True
LSTM_AUDIT_GRAD_NORM = False
LSTM_MAX_GRAD_NORM = None  # if float, clip gradients to this norm
LSTM_FLAT_AUC_WARN_epochs = 8
LSTM_FLAT_AUC_EPS = 0.02
LSTM_OVERFIT_LOSS_RATIO = 3.0
LSTM_OVERFIT_WARN_epochs = 6

# LSTM LR grid for experiments/lstm_lr_sweep.py (single value = no sweep)
LSTM_LR_GRID = [0.0005, 0.001, 0.003, 0.005]
LSTM_LR_SWEEP_MAX_EPOCHS = 40  # capped budget for experiments/lstm_lr_sweep.py

# Market/sector feature horizons (used in pipeline/features.py)
MARKET_RETURN_HORIZONS = (1, 5, 21)
MARKET_VOL_WINDOWS = (20, 60)
BETA_WINDOW = 60
SECTOR_RETURN_EXTRA_HORIZONS = (21,)
SECTOR_VOL_EXTRA_WINDOWS = (60,)
SECTOR_REL_ZSCORE_RETURN_COLS = ("Return_1d",)

# ── Feature config (10 active features including momentum + Context features) ────────────────
SEQ_LEN               = 30

# Context features flags
MARKET_FEATURES_ENABLED = False
SECTOR_FEATURES_ENABLED = False

# Master feature union: all features used by at least one model
ALL_FEATURE_COLS = [
    "Return_1d",        # LSTM-A, LSTM, Baselines
    "Return_5d",        # LSTM, Baselines (weekly momentum)
    "Return_21d",       # LSTM, Baselines (monthly momentum)
    "RSI_14",           # LSTM-A, LSTM, Baselines
    "MACD",             # LSTM-A only
    "ATR_14",           # LSTM-A only
    "BB_PctB",          # LSTM, Baselines
    "RealVol_20d",      # LSTM, Baselines
    "Volume_Ratio",     # LSTM, Baselines
    "SectorRelReturn",  # LSTM, Baselines
]

if MARKET_FEATURES_ENABLED:
    ALL_FEATURE_COLS.extend([
        "Market_Return_1d",
        "Market_Return_5d",
        "Market_Return_21d",
        "Market_Vol_20d",
        "Market_Vol_60d",
        "RelToMarket_1d",
        "RelToMarket_5d",
        "RelToMarket_21d",
        f"Beta_{BETA_WINDOW}d",
    ])

if SECTOR_FEATURES_ENABLED:
    ALL_FEATURE_COLS.extend([
        "Sector_Return_1d",
        "Sector_Return_5d",
        "Sector_Return_21d",
        "Sector_Vol_20d",
        "Sector_Vol_60d",
        "SectorRelZ_Return_1d",
    ])

N_TOTAL_FEATURES = len(ALL_FEATURE_COLS)  # Dynamically computed

# ── Per-model feature sets (Section 7.1) ────────────────────────────────────
LSTM_A_FEATURE_COLS = [
    "MACD",        # 12/26 EMA difference (Bhandari §4.3)
    "RSI_14",      # 14-day RSI (Bhandari §4.3)
    "ATR_14",      # 14-day ATR (Bhandari §4.3)
    "Return_1d",   # 1-day simple return
    "Return_5d",   # 5-day simple return (weekly momentum)
    "Return_21d",  # 21-day simple return (monthly momentum)
]

LSTM_B_FEATURE_COLS = [
    "Return_1d",
    "Return_5d",        # Weekly momentum
    "Return_21d",       # Monthly momentum
    "RSI_14",
    "BB_PctB",
    "RealVol_20d",
    "Volume_Ratio",
    "SectorRelReturn",
]

if MARKET_FEATURES_ENABLED:
    LSTM_B_FEATURE_COLS.extend([
        "Market_Return_1d",
        "Market_Return_5d",
        "Market_Return_21d",
        "Market_Vol_20d",
        "Market_Vol_60d",
        "RelToMarket_1d",
        "RelToMarket_5d",
        "RelToMarket_21d",
        f"Beta_{BETA_WINDOW}d",
    ])

if SECTOR_FEATURES_ENABLED:
    LSTM_B_FEATURE_COLS.extend([
        "Sector_Return_1d",
        "Sector_Return_5d",
        "Sector_Return_21d",
        "Sector_Vol_20d",
        "Sector_Vol_60d",
        "SectorRelZ_Return_1d",
    ])

# Baselines use LSTM features for fair comparison
BASELINE_FEATURE_COLS = LSTM_B_FEATURE_COLS

# Trading
K_STOCKS = 5   # Long top-5, short bottom-5 per day
TC_BPS   = 5   # Transaction cost per half-turn in basis points (0.0005)
SIGNAL_SMOOTH_ALPHA = 0.0
SIGNAL_CONFIDENCE_THRESHOLD = 0.55  # Requires prob to be >= 0.5 + threshold or <= 0.5 - threshold
SIGNAL_USE_ZSCORE = True  # Use cross-sectional z-score for more robust signal generation
MIN_HOLDING_DAYS = 5

# Execution semantics (see backtest/portfolio.py): features at date t use data through t;
# signals rank at t; portfolio earns Return_NextDay (close t to close t+1).

# ── LSTM-A: Bhandari-inspired technical indicator LSTM (4 features) ─────────
# Architecture is determined by hyperparameter tuning (Section 1 / 7.4)
LSTM_A_DEV_MODE     = True                  # Set False for final thesis run only
LSTM_A_FEATURES      = LSTM_A_FEATURE_COLS  # 4 features: MACD, RSI, ATR, Return_1d
LSTM_A_SEQ_LEN       = SEQ_LEN
LSTM_A_OPTIMIZER     = 'adam'                # will be tuned
LSTM_A_LR            = 0.001                 # will be tuned
LSTM_A_BATCH         = 256 if DEV_MODE else 128   # DEV: faster batches
LSTM_A_MAX_EPOCHS    = 200
LSTM_A_PATIENCE      = 15
LSTM_A_VAL_SPLIT     = 0.2

# LSTM-A architecture search grid (Bhandari §3.3 Algorithm 2)
# Architecture is data-driven, not fixed
LSTM_A_ARCH_GRID = {
    "hidden_size": [16, 32, 64],   # small range appropriate for 4-feature input
    "num_layers":  [1, 2],         # Bhandari §5.5 found single-layer often wins
    "dropout":     [0.1, 0.2],
}

# ── LSTM: Extended ablation — curated 6-feature set (fixed architecture) ────
LSTM_B_FEATURES      = LSTM_B_FEATURE_COLS
LSTM_B_SEQ_LEN       = SEQ_LEN
LSTM_B_HIDDEN_SIZE   = 32                     # fixed architecture for LSTM
LSTM_B_NUM_LAYERS    = 1
LSTM_B_DROPOUT       = 0.0
LSTM_B_HIDDEN        = LSTM_B_HIDDEN_SIZE     # alias for backward compatibility
LSTM_B_LAYERS        = LSTM_B_NUM_LAYERS
LSTM_B_OPTIMIZER     = 'adam'
LSTM_B_LR            = 0.001
LSTM_B_BATCH         = 256 if DEV_MODE else 128   # DEV: faster batches
LSTM_B_MAX_EPOCHS    = 200
LSTM_B_PATIENCE      = 15
LSTM_B_LR_PATIENCE   = 7
LSTM_B_LR_FACTOR     = 0.5
LSTM_B_VAL_SPLIT     = 0.2

# ── Shared LSTM settings ──────────────────────────────────────────────────
LSTM_WD              = 1e-5                   # weight decay (both models)

# ── LSTM Hyperparameter Search Grid (Bhandari §3.3) ──────────────────────────
# Shared by both LSTM-A and LSTM for the training hyperparameter search
LSTM_HYPERPARAM_GRID = {
    "optimizer":      ["adam", "adagrad", "nadam"],   # paper tests these three
    "learning_rate":  [0.1, 0.01, 0.001],             # paper tests these three
    "batch_size":     [32, 64, 128],                  # scaled up from paper's 4/8/16
}
LSTM_TUNE_REPLICATES = 3      # paper uses 10; 3 is feasible on M4 for a thesis
LSTM_TUNE_PATIENCE   = 5      # early stopping patience during tuning (paper §3.3)
LSTM_TUNE_MAX_EPOCHS = 50     # cap tuning runs; full training uses MAX_EPOCHS

# LSTM focused tuning controls (bounded search to keep wall-time manageable)
LSTM_B_ENABLE_TUNING = True
LSTM_B_TUNE_ON_FIRST_FOLD_ONLY = True
LSTM_B_HYPERPARAM_GRID = {
    "optimizer": ["adam", "nadam"],
    "learning_rate": [0.0003, 0.001, 0.003],
    "batch_size": [64, 128],
}
LSTM_B_ARCH_GRID = {
    "hidden_size": [32, 64],
    "num_layers": [1, 2],
    "dropout": [0.0, 0.2],
}
LSTM_B_TUNE_REPLICATES = 1
LSTM_B_TUNE_PATIENCE = 4
LSTM_B_TUNE_MAX_EPOCHS = 35

# ── Wavelet Denoising (Bhandari §4.5) ────────────────────────────────────────
USE_WAVELET_DENOISING = False    # Set False to use raw prices (Fixes OOS domain shift)
WAVELET_TYPE          = "haar"  # Paper uses Haar wavelets
WAVELET_LEVEL         = 1       # Decomposition level; 1 is appropriate for daily data
WAVELET_MODE          = "soft"  # Thresholding mode: 'soft' (paper) or 'hard'
WAVELET_WINDOW_SIZE   = 128     # Lookback window for causal denoising (prevents leakage)

# ── Normalization (Bhandari §4.5 uses MinMax; our default is Standard) ───────
SCALER_TYPE = "standard"   # Options: "standard" (default) | "minmax"

# ── Feature Selection (Bhandari §4.4) ────────────────────────────────────────
FEATURE_CORR_THRESHOLD = 0.80   # Drop features with |r| > threshold

# After running analysis/feature_correlation.py, paste the output list here:
# Leave as None to use all ALL_FEATURE_COLS (before selection is run)
FEATURE_COLS_AFTER_SELECTION = ['Return_1d', 'Return_5d', 'Return_21d', 'RSI_14', 'MACD', 'ATR_14', 'RealVol_20d', 'Volume_Ratio', 'SectorRelReturn']

# Random Forest — reduced development grid
RF_PARAM_GRID = {
    'n_estimators':     [300],
    'max_depth':        [5, 10],
    'min_samples_leaf': [30, 50],
}

# XGBoost — grid search over key hyperparameters
XGB_PARAM_GRID = {
    'max_depth':  [3, 4, 5],
    'eta':        [0.01],
    'subsample':  [0.6, 0.7],
}
XGB_COLSAMPLE    = 0.5
XGB_ROUNDS       = 1000
XGB_EARLY_STOP   = 50
XGB_REG_ALPHA    = 0.1    # L1 regularization
XGB_REG_LAMBDA   = 1.0    # L2 regularization

RANDOM_SEED = 42

# =============================================================================
# DEV MODE — Set False for final thesis run only
# =============================================================================
DEV_MODE = True  # When True, skips LSTM-A to reduce runtime
MODELS_DEV  = ['LR', 'RF', 'XGBoost', 'LSTM']
MODELS_FULL = ['LR', 'RF', 'XGBoost', 'LSTM-A', 'LSTM']

# ── Model registry (after refactor) ──────────────────────────────────────────
MODELS = ['LR', 'RF', 'XGBoost', 'LSTM-A', 'LSTM']

```

### Computed: Active TICKERS sector counts

```text
UNIVERSE_MODE small_cap
N_TICKERS 30
SECTOR_COUNTS_ACTIVE
Unknown: 30
MISSING_IN_SECTOR_MAP 30
SMCI,FSLY,AI,PLUG,RUN,ARRY,NVAX,ICPT,SRPT,BLUE,EXEL,IONS,GME,BOOT,CROX,SHOO,CAL,MOV,AA,CLF,X,ATI,WCC,LPX,FHN,ZION,CMA,PACW,NYCB,STWD
FLAGS_PREFIX_ENABLE_OR_USE
USE_WAVELET_DENOISING False
```

## SECTION 2 — FEATURE ENGINEERING AUDIT

### pipeline/features.py (verbatim)

```python
"""
pipeline/features.py
Compute only the 8 active features used across all models (Section 6 of IMPLEMENTATION_EXTENSIONS.md).
Includes wavelet denoising (Bhandari §4.5) and sector-relative return computation.

Features computed:
    Return_1d, RSI_14, MACD, ATR_14, BB_PctB, RealVol_20d, Volume_Ratio, SectorRelReturn
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config

# ── Feature column names (8 total per Section 6) ──────────────────────────────
FEATURE_COLS = config.ALL_FEATURE_COLS
TARGET_COL = 'Target'


def compute_wavelet_threshold(close_series: pd.Series, wavelet: str = "haar",
                               level: int = 1) -> float:
    """
    Compute the wavelet denoising threshold from a training price series.
    
    ANTI-LEAKAGE: This must only be called on training data. The returned
    threshold is then applied to denoise val/test data without recomputation.

    Parameters
    ----------
    close_series : pd.Series
        Raw daily close prices for training window only.
    wavelet : str
        Wavelet family. 'haar' is the paper's choice.
    level : int
        Decomposition level.

    Returns
    -------
    float
        The universal threshold value computed from training data.
    """
    try:
        import pywt
    except ImportError:
        return 0.0

    prices = close_series.values.astype(float)
    if len(prices) < 4:
        return 0.0

    coeffs = pywt.wavedec(prices, wavelet=wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(prices)))
    return threshold


def denoise_close_price(close_series: pd.Series, wavelet: str = "haar",
                        level: int = 1, mode: str = "soft",
                        threshold: float = None) -> pd.Series:
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
    threshold : float, optional
        Pre-computed threshold from training data. If None, threshold is
        computed from close_series itself (use only for training data).

    Returns
    -------
    pd.Series
        Denoised close prices with the same index.
    """
    try:
        import pywt
    except ImportError:
        print("Warning: PyWavelets not installed. Skipping wavelet denoising.")
        print("Install with: pip install PyWavelets")
        return close_series

    prices = close_series.values.astype(float)

    # Handle edge case of too few data points
    if len(prices) < 4:
        return close_series

    # Wavelet decomposition
    coeffs = pywt.wavedec(prices, wavelet=wavelet, level=level)

    # Use provided threshold or compute from this series (training only)
    if threshold is None:
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
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


def denoise_close_price_causal(close_series: pd.Series, threshold: float,
                                window_size: int = 128,
                                wavelet: str = "haar",
                                level: int = 1,
                                mode: str = "soft") -> pd.Series:
    """
    CAUSAL wavelet denoising: each value at time t uses only a fixed lookback window.

    This prevents look-ahead bias by ensuring the denoised value at time t
    depends only on prices from times [t - window_size, t], not future data.

    Parameters
    ----------
    close_series : pd.Series
        Raw daily close prices (chronological order).
    threshold : float
        Pre-computed threshold from training data (required).
    window_size : int
        Number of historical points to use for each denoising operation.
        Must be >= 4 for wavelet decomposition.
    wavelet : str
        Wavelet family ('haar' is the paper's choice).
    level : int
        Decomposition level.
    mode : str
        Thresholding mode: 'soft' (paper) or 'hard'.

    Returns
    -------
    pd.Series
        Causally denoised close prices (same index as input).
    """
    try:
        import pywt
    except ImportError:
        print("Warning: PyWavelets not installed. Returning raw prices.")
        return close_series

    prices = close_series.values.astype(float)
    denoised = np.full(len(prices), np.nan)

    # For each time point >= window_size, apply wavelet denoising to lookback window
    for t in range(window_size, len(prices)):
        # Fixed lookback window ending at t (inclusive)
        window = prices[t - window_size:t + 1]

        # Wavelet decomposition
        coeffs = pywt.wavedec(window, wavelet=wavelet, level=level)

        # Apply thresholding to detail coefficients
        coeffs_thresh = [coeffs[0]]  # Keep approximation unchanged
        for detail in coeffs[1:]:
            coeffs_thresh.append(pywt.threshold(detail, threshold, mode=mode))

        # Reconstruct and take only the LAST value (current time t)
        reconstructed = pywt.waverec(coeffs_thresh, wavelet=wavelet)
        denoised[t] = reconstructed[-1]

    # For early points (warm-up period), use raw prices
    denoised[:window_size] = prices[:window_size]

    return pd.Series(denoised, index=close_series.index, name=close_series.name)


def compute_wavelet_thresholds(df_train: pd.DataFrame) -> dict:
    """
    Compute wavelet denoising thresholds from training data only.
    
    ANTI-LEAKAGE: These thresholds are computed once per fold on df_train,
    then reused to denoise df_val and df_test without recomputation.
    
    Parameters
    ----------
    df_train : pd.DataFrame
        Training fold data with columns [Date, Ticker, Close, ...].
        
    Returns
    -------
    dict
        {ticker: threshold} mapping for each ticker in the training set.
    """
    thresholds = {}
    for ticker, grp in df_train.groupby("Ticker"):
        grp = grp.sort_values("Date")
        thresholds[ticker] = compute_wavelet_threshold(
            grp["Close"],
            wavelet=config.WAVELET_TYPE,
            level=config.WAVELET_LEVEL,
        )
    return thresholds


def apply_wavelet_denoising(df: pd.DataFrame, 
                            thresholds: dict = None) -> pd.DataFrame:
    """
    Apply denoise_close_price() to each ticker independently.
    Replaces the 'Close' column with the denoised version.
    Call this BEFORE compute_technical_features() and compute_returns().
    
    ANTI-LEAKAGE: If thresholds dict is provided, uses pre-computed thresholds
    (for val/test data). If None, computes thresholds from the data itself
    (only appropriate for training data).
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with columns [Date, Ticker, Close, ...].
    thresholds : dict, optional
        {ticker: threshold} pre-computed from training data.
        If None, thresholds are computed per-ticker from df itself.
        
    Returns
    -------
    pd.DataFrame
        Copy of df with 'Close' column replaced by denoised values.
    """
    df = df.copy()
    denoised_rows = []
    for ticker, grp in df.groupby("Ticker"):
        grp = grp.sort_values("Date").copy()
        # Use pre-computed threshold if available, else compute from this series
        thresh = thresholds.get(ticker) if thresholds else None
        grp["Close"] = denoise_close_price(
            grp["Close"],
            wavelet=config.WAVELET_TYPE,
            level=config.WAVELET_LEVEL,
            mode=config.WAVELET_MODE,
            threshold=thresh,
        )
        denoised_rows.append(grp)
    return pd.concat(denoised_rows).sort_values(["Date", "Ticker"])


def apply_wavelet_denoising_causal(df: pd.DataFrame,
                                    thresholds: dict,
                                    window_size: int = None) -> pd.DataFrame:
    """
    Apply CAUSAL wavelet denoising to each ticker independently.

    CRITICAL FIX: This function uses rolling-window denoising to prevent
    look-ahead bias. Each denoised value at time t only uses historical data
    from [t - window_size, t].

    Unlike apply_wavelet_denoising(), this function:
    1. REQUIRES pre-computed thresholds (no auto-computation)
    2. Uses denoise_close_price_causal() instead of denoise_close_price()
    3. Applies denoising per-split independently (no concatenation)

    Parameters
    ----------
    df : pd.DataFrame
        Data with columns [Date, Ticker, Close, ...].
    thresholds : dict
        {ticker: threshold} pre-computed from training data (REQUIRED).
    window_size : int, optional
        Lookback window size. Defaults to config.WAVELET_WINDOW_SIZE.

    Returns
    -------
    pd.DataFrame
        Copy of df with 'Close' column replaced by causally denoised values.
    """
    if window_size is None:
        window_size = getattr(config, 'WAVELET_WINDOW_SIZE', 128)

    df = df.copy()
    denoised_rows = []

    for ticker, grp in df.groupby("Ticker"):
        grp = grp.sort_values("Date").copy()

        # Get pre-computed threshold (required for causal denoising)
        thresh = thresholds.get(ticker)
        if thresh is None:
            # Ticker not in training set - use raw prices
            denoised_rows.append(grp)
            continue

        grp["Close"] = denoise_close_price_causal(
            grp["Close"],
            threshold=thresh,
            window_size=window_size,
            wavelet=config.WAVELET_TYPE,
            level=config.WAVELET_LEVEL,
            mode=config.WAVELET_MODE,
        )
        denoised_rows.append(grp)

    return pd.concat(denoised_rows).sort_values(["Date", "Ticker"])


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 10 features actively used across all models.
    Includes multi-day momentum features (Return_5d, Return_21d) for
    improved predictive power.

    ANTI-LEAKAGE: All indicators are causal — each value at time t
    uses only data from t and earlier.
    """
    df = df.copy().sort_values("Date")

    # ── Return_1d ─────────────────────────────────────────────────────
    df["Return_1d"] = df["Close"].pct_change(1)

    # ── Multi-day Momentum (Fischer & Krauss 2017 style) ───────────────
    df["Return_5d"] = df["Close"].pct_change(5)    # Weekly momentum
    df["Return_21d"] = df["Close"].pct_change(21)  # Monthly momentum

    # ── RSI_14 (Bhandari §4.3) ────────────────────────────────────────
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI_14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # ── MACD (Bhandari §4.3 — 12/26 EMA difference) ───────────────────
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    # Note: MACD_Signal is NOT computed — not in any active feature set

    # ── ATR_14 (Bhandari §4.3) ────────────────────────────────────────
    hl = df["High"] - df["Low"]
    hpc = (df["High"] - df["Close"].shift()).abs()
    lpc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()

    # ── BB_PctB (%B position within Bollinger Bands) ───────────────────
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    df["BB_PctB"] = (df["Close"] - lower) / (upper - lower + 1e-10)
    # Note: BB_Width is NOT computed — not in any active feature set

    # ── RealVol_20d (annualised 20-day realised volatility) ───────────
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    df["RealVol_20d"] = log_ret.rolling(20).std() * np.sqrt(252)

    # ── Volume_Ratio ──────────────────────────────────────────────────
    df["Volume_Ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1)

    # Note: OBV, HL_Pct, HL_Pct_5d, Mom_10d are NOT computed

    return df


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

    # Handle tickers not in sector_map (assign to 'Unknown')
    df['Sector'] = df['Sector'].fillna('Unknown')

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


def compute_market_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Market-wide features: cross-sectional mean returns (same day, PIT at t),
    rolling market vol on the market return series, excess returns vs market,
    rolling CAPM-style beta vs market (past window only per ticker).
    """
    df = df.copy()
    vol_windows = getattr(config, 'MARKET_VOL_WINDOWS', (20, 60))

    agg_cols = ['Return_1d', 'Return_5d']
    if 'Return_21d' in df.columns:
        agg_cols.append('Return_21d')

    market_daily = df.groupby('Date')[agg_cols].mean().reset_index()
    rename = {'Return_1d': 'Market_Return_1d', 'Return_5d': 'Market_Return_5d'}
    if 'Return_21d' in agg_cols:
        rename['Return_21d'] = 'Market_Return_21d'
    market_daily.rename(columns=rename, inplace=True)
    market_daily = market_daily.sort_values('Date')

    for w in vol_windows:
        col = f'Market_Vol_{w}d'
        market_daily[col] = (
            market_daily['Market_Return_1d'].rolling(int(w)).std() * np.sqrt(252)
        )

    df = df.merge(market_daily, on='Date', how='left')

    df['RelToMarket_1d'] = df['Return_1d'] - df['Market_Return_1d']
    df['RelToMarket_5d'] = df['Return_5d'] - df['Market_Return_5d']
    if 'Market_Return_21d' in df.columns and 'Return_21d' in df.columns:
        df['RelToMarket_21d'] = df['Return_21d'] - df['Market_Return_21d']

    beta_w = int(getattr(config, 'BETA_WINDOW', 60))
    beta_col = f'Beta_{beta_w}d'
    beta_parts = []
    for _, g in df.groupby('Ticker'):
        g = g.sort_values('Date').copy()
        rm = g['Market_Return_1d']
        rs = g['Return_1d']
        c = rs.rolling(beta_w).cov(rm)
        v = rm.rolling(beta_w).var() + 1e-12
        g[beta_col] = c / v
        beta_parts.append(g)
    df = pd.concat(beta_parts).sort_values(['Date', 'Ticker'])
    return df


def compute_sector_context_features(df: pd.DataFrame, sector_map: dict) -> pd.DataFrame:
    """
    Leave-one-out sector mean returns, rolling sector vol on sector mean return,
    within (Date, Sector) z-score of returns (same-day peers only).
    """
    df = df.copy()
    if 'Sector' not in df.columns:
        df['Sector'] = df['Ticker'].map(sector_map).fillna('Unknown')

    def leave_one_out_mean(col_name: str, new_col_name: str) -> None:
        sector_means = df.groupby(['Date', 'Sector'])[col_name].transform('mean')
        sector_counts = df.groupby(['Date', 'Sector'])[col_name].transform('count')
        sector_sums = sector_means * sector_counts
        df[new_col_name] = np.where(
            sector_counts > 1,
            (sector_sums - df[col_name]) / (sector_counts - 1),
            0.0,
        )

    leave_one_out_mean('Return_1d', 'Sector_Return_1d')
    leave_one_out_mean('Return_5d', 'Sector_Return_5d')
    if 'Return_21d' in df.columns:
        leave_one_out_mean('Return_21d', 'Sector_Return_21d')

    extra_vol_w = getattr(config, 'SECTOR_VOL_EXTRA_WINDOWS', (60,))
    sector_daily = df.groupby(['Date', 'Sector'])['Return_1d'].mean().reset_index()
    sector_daily = sector_daily.sort_values('Date')
    sector_daily['Sector_Vol_20d'] = sector_daily.groupby('Sector')['Return_1d'].transform(
        lambda x: x.rolling(20).std() * np.sqrt(252)
    )
    for w in extra_vol_w:
        if int(w) == 20:
            continue
        sector_daily[f'Sector_Vol_{int(w)}d'] = sector_daily.groupby('Sector')['Return_1d'].transform(
            lambda x, ww=int(w): x.rolling(ww).std() * np.sqrt(252)
        )

    merge_cols = ['Date', 'Sector'] + [c for c in sector_daily.columns if c.startswith('Sector_Vol')]
    df = df.merge(sector_daily[merge_cols], on=['Date', 'Sector'], how='left')

    df['SectorRelZ_Return_1d'] = df.groupby(['Date', 'Sector'])['Return_1d'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-12)
    )

    df.drop(columns=['Sector'], inplace=True)
    return df


def recompute_features_from_denoised(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute all Close-dependent features from denoised Close prices.
    
    This function is called per-fold AFTER wavelet denoising has been applied
    with training-derived thresholds. It recomputes:
      - Return_1d, RSI_14, MACD, ATR_14, BB_PctB, RealVol_20d
      
    Volume_Ratio uses only Volume (no Close dependency) so is unchanged.
    SectorRelReturn depends on Return_1d, so must be recomputed after this.
    
    ANTI-LEAKAGE: Called on train/val/test splits separately after denoising
    with training-only thresholds.
    
    Parameters
    ----------
    df : pd.DataFrame
        Fold data with denoised 'Close' column and OHLCV.
        
    Returns
    -------
    pd.DataFrame
        Copy with recomputed feature columns.
    """
    result_parts = []
    for ticker, grp in df.groupby('Ticker'):
        grp = grp.copy().sort_values('Date')
        
        # ── Return_1d ─────────────────────────────────────────────────────
        grp["Return_1d"] = grp["Close"].pct_change(1)

        # ── RSI_14 (Bhandari §4.3) ────────────────────────────────────────
        delta = grp["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        grp["RSI_14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

        # ── MACD (Bhandari §4.3 — 12/26 EMA difference) ───────────────────
        ema12 = grp["Close"].ewm(span=12, adjust=False).mean()
        ema26 = grp["Close"].ewm(span=26, adjust=False).mean()
        grp["MACD"] = ema12 - ema26

        # ── ATR_14 (Bhandari §4.3) ────────────────────────────────────────
        hl = grp["High"] - grp["Low"]
        hpc = (grp["High"] - grp["Close"].shift()).abs()
        lpc = (grp["Low"] - grp["Close"].shift()).abs()
        tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
        grp["ATR_14"] = tr.rolling(14).mean()

        # ── BB_PctB (%B position within Bollinger Bands) ───────────────────
        sma20 = grp["Close"].rolling(20).mean()
        std20 = grp["Close"].rolling(20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        grp["BB_PctB"] = (grp["Close"] - lower) / (upper - lower + 1e-10)

        # ── RealVol_20d (annualised 20-day realised volatility) ───────────
        log_ret = np.log(grp["Close"] / grp["Close"].shift(1))
        grp["RealVol_20d"] = log_ret.rolling(20).std() * np.sqrt(252)

        # Volume_Ratio is unchanged (no Close dependency)
        
        result_parts.append(grp)
    
    result = pd.concat(result_parts).sort_values(['Date', 'Ticker'])
    
    # Recompute SectorRelReturn (depends on Return_1d)
    result = compute_sector_rel_return(result, sector_map=config.SECTOR_MAP)
    
    if getattr(config, 'MARKET_FEATURES_ENABLED', False):
        result = compute_market_context_features(result)
        
    if getattr(config, 'SECTOR_FEATURES_ENABLED', False):
        result = compute_sector_context_features(result, sector_map=config.SECTOR_MAP)
        
    return result


def build_feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the full feature pipeline on the long-format OHLCV DataFrame:
      1. Compute 8 technical features per ticker (from RAW Close prices)
      2. Compute SectorRelReturn cross-sectional feature
      3. Drops rows with any NaN in feature columns
    4. Saves to data/processed/features_{UNIVERSE_MODE}.csv
      
    NOTE: Wavelet denoising is intentionally NOT applied here to prevent
    look-ahead bias. Denoising must be done per fold inside the walk-forward
    loop in main.py, with thresholds computed only on training data.
    The cached CSV preserves raw OHLCV columns for per-fold recomputation.

    Returns the enriched DataFrame (Date, Ticker, OHLCV, features...).
    """
    # Step 1 — Compute technical indicators per ticker (from raw Close)
    print("Computing technical indicators per ticker (raw Close)...")
    enriched = []
    for ticker, group in data.groupby('Ticker'):
        enriched.append(compute_technical_features(group))
    result = pd.concat(enriched).sort_values(['Date', 'Ticker']).reset_index(drop=True)

    # Step 2 — Compute SectorRelReturn and context features
    print("Computing SectorRelReturn and Context Features...")
    result = compute_sector_rel_return(result, sector_map=config.SECTOR_MAP)
    
    if getattr(config, 'MARKET_FEATURES_ENABLED', False):
        result = compute_market_context_features(result)
        
    if getattr(config, 'SECTOR_FEATURES_ENABLED', False):
        result = compute_sector_context_features(result, sector_map=config.SECTOR_MAP)

    # Step 3 — Drop NaN rows
    before = len(result)
    result.dropna(subset=FEATURE_COLS, inplace=True)
    result.reset_index(drop=True, inplace=True)
    print(f"Dropped {before - len(result)} rows with NaN features "
          f"(expected: ~26 per ticker for MACD warmup)")

    # Step 4 — Save
    features_cache = f'data/processed/features_{config.UNIVERSE_MODE}.csv'
    result.to_csv(features_cache, index=False)
    print(f"\nSaved {len(result)} rows to {features_cache}")
    print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"Date range: {result['Date'].min()} -> {result['Date'].max()}")
    print(f"Tickers: {len(result['Ticker'].unique())}")
    print(f"NOTE: Wavelet denoising is applied per-fold in main.py (not cached)")
    return result


if __name__ == '__main__':
    data = pd.read_csv('data/raw/ohlcv_long.csv', parse_dates=['Date'])
    build_feature_matrix(data)

```

### Call order evidence from main.py (feature timing and wavelet per-fold)

```python
42:    compute_wavelet_thresholds, apply_wavelet_denoising, apply_wavelet_denoising_causal,
43:    recompute_features_from_denoised,
276:    if load_cached:
288:                data = create_targets(data)
298:        raw = download_and_save()
299:        data = build_feature_matrix(raw)
300:        data = create_targets(data)
381:    for fold in tqdm(folds, desc="Walk-Forward Folds", unit="fold"):
404:        if config.USE_WAVELET_DENOISING:
406:            wavelet_thresholds = compute_wavelet_thresholds(df_tr)
411:            df_tr = apply_wavelet_denoising_causal(df_tr, thresholds=wavelet_thresholds)
412:            df_v = apply_wavelet_denoising_causal(df_v, thresholds=wavelet_thresholds)
413:            df_ts = apply_wavelet_denoising_causal(df_ts, thresholds=wavelet_thresholds)
429:            df_tr = recompute_features_from_denoised(df_tr)
430:            df_v = recompute_features_from_denoised(df_v)
431:            df_ts = recompute_features_from_denoised(df_ts)

```

### Groupby+transform leakage-risk candidates in features.py

```text
204:    for ticker, grp in df_train.groupby("Ticker"):
240:    for ticker, grp in df.groupby("Ticker"):
290:    for ticker, grp in df.groupby("Ticker"):
401:        df.groupby(['Date', 'Sector'])['Return_1d']
402:          .transform('mean')
406:        df.groupby(['Date', 'Sector'])['Return_1d']
407:          .transform('count')
434:    market_daily = df.groupby('Date')[agg_cols].mean().reset_index()
457:    for _, g in df.groupby('Ticker'):
479:        sector_means = df.groupby(['Date', 'Sector'])[col_name].transform('mean')
480:        sector_counts = df.groupby(['Date', 'Sector'])[col_name].transform('count')
494:    sector_daily = df.groupby(['Date', 'Sector'])['Return_1d'].mean().reset_index()
496:    sector_daily['Sector_Vol_20d'] = sector_daily.groupby('Sector')['Return_1d'].transform(
502:        sector_daily[f'Sector_Vol_{int(w)}d'] = sector_daily.groupby('Sector')['Return_1d'].transform(
509:    df['SectorRelZ_Return_1d'] = df.groupby(['Date', 'Sector'])['Return_1d'].transform(
542:    for ticker, grp in df.groupby('Ticker'):
613:    for ticker, group in data.groupby('Ticker'):

```

## SECTION 3 — TARGET CONSTRUCTION

### pipeline/targets.py (verbatim)

```python
"""
pipeline/targets.py
Step 3: Create binary cross-sectional median-based target variable.

On each day, compute the cross-sectional median of next-day returns.
Target = 1 if stock's next-day return >= median, else 0.
This mirrors Fischer & Krauss (2017) and Krauss et al. (2017) exactly.
"""
import pandas as pd
import numpy as np
import sys
import os
import config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

FEATURES_CACHE = f'data/processed/features_{config.UNIVERSE_MODE}.csv'


def create_targets(data: pd.DataFrame, return_col: str = 'Return_1d') -> pd.DataFrame:
    """
    Adds columns to the feature DataFrame:
      - Return_NextDay : realized t+1 return for each stock (used in backtesting)
      - Target         : 1 if Return_NextDay >= cross-sectional median, else 0

    Parameters
    ----------
    data : pd.DataFrame
        Output of build_feature_matrix() — must contain 'Date', 'Ticker', return_col.
    return_col : str
        Column to shift forward as the prediction target (default 'Return_1d').

    Returns
    -------
    pd.DataFrame
        Original DataFrame plus 'Return_NextDay' and 'Target' columns.
    """
    data = data.copy().sort_values(['Date', 'Ticker']).reset_index(drop=True)

    # Next-day return: shift Return_1d backward by 1 within each ticker
    # (i.e. today's row gets tomorrow's realized return)
    data['Return_NextDay'] = data.groupby('Ticker')[return_col].shift(-1)

    # Drop the last day per ticker (Return_NextDay is NaN — no future return exists)
    before = len(data)
    data.dropna(subset=['Return_NextDay'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    print(f"Dropped {before - len(data)} rows (last trading day per ticker, no future return)")

    # Cross-sectional median per date
    daily_median = data.groupby('Date')['Return_NextDay'].transform('median')

    # Binary label: 1 if stock outperforms or matches median, 0 otherwise
    data['Target'] = (data['Return_NextDay'] >= daily_median).astype(int)

    n_pos = (data['Target'] == 1).sum()
    n_neg = (data['Target'] == 0).sum()
    print(f"\nClass distribution:")
    print(f"  Target=1 (>= median): {n_pos} ({n_pos/len(data)*100:.1f}%)")
    print(f"  Target=0 (<  median): {n_neg} ({n_neg/len(data)*100:.1f}%)")
    print(f"\nTotal samples: {len(data)}")
    print(f"Date range:    {data['Date'].min()} → {data['Date'].max()}")
    return data


if __name__ == '__main__':
    data = pd.read_csv(FEATURES_CACHE, parse_dates=['Date'])
    result = create_targets(data)

    # Quick sanity: per-ticker class balance
    print("\nPer-ticker class balance:")
    print(result.groupby('Ticker')['Target'].mean().round(3))

    # Save augmented feature+target file
    result.to_csv(FEATURES_CACHE, index=False)
    print(f"\nUpdated {FEATURES_CACHE} with Target and Return_NextDay columns.")

```

### Computed class balance on active cached dataset

```text
ROWS 34653
TARGET_COUNTS
Target
1    17397
0    17256
TARGET_NORMALIZED
Target
1    0.502034
0    0.497966
```

## SECTION 4 — WALK-FORWARD FOLD GENERATOR

### pipeline/walk_forward.py (verbatim)

```python
"""
pipeline/walk_forward.py
Step 4: Walk-forward fold generator for time-series cross-validation.

Fold structure (rolls forward by one test period each time):
    |--- Train 252 ---|--- Val 63 ---|--- Test 63 ---|
                                        |--- Train 252 ---|--- Val 63 ---|--- Test 63 ---|
                                     ...

Modes:
  rolling    — fixed train length; window slides by stride_days
  expanding  — train always dates[0:val_start); val/test blocks slide by stride_days
"""
from __future__ import annotations

import sys
import os
from typing import Literal
import pandas as pd
import config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import TRAIN_DAYS, VAL_DAYS, TEST_DAYS, WALK_FORWARD_STRIDE, TRAIN_WINDOW_MODE


def generate_walk_forward_folds(
    dates_sorted: list,
    train_days: int = TRAIN_DAYS,
    val_days: int = VAL_DAYS,
    test_days: int = TEST_DAYS,
    stride_days: int | None = None,
    train_window_mode: Literal["rolling", "expanding"] | str | None = None,
) -> list[dict]:
    """
    Generates walk-forward folds with strict chronological ordering.

    Parameters
    ----------
    dates_sorted : list
        Sorted list of unique trading dates (strings or Timestamps).
    train_days, val_days, test_days : int
        Sizes of each segment (rolling mode) or minimum train span (expanding).
    stride_days : int, optional
        Index advance between folds. Default: config WALK_FORWARD_STRIDE or test_days.
    train_window_mode : str
        'rolling' (default) or 'expanding'.

    Returns
    -------
    list of dict with index slices into dates_sorted and boundary dates.
    """
    if stride_days is None:
        stride_days = WALK_FORWARD_STRIDE if WALK_FORWARD_STRIDE is not None else test_days
    mode = (train_window_mode or TRAIN_WINDOW_MODE or "rolling").lower()
    if mode not in ("rolling", "expanding"):
        raise ValueError(f"train_window_mode must be 'rolling' or 'expanding', got {mode!r}")

    total = len(dates_sorted)
    folds: list[dict] = []

    if mode == "rolling":
        window = train_days + val_days + test_days
        start = 0
        while start + window <= total:
            train_end = start + train_days
            val_end = train_end + val_days
            test_end = val_end + test_days

            folds.append({
                'fold': len(folds) + 1,
                'train': (start, train_end),
                'val': (train_end, val_end),
                'test': (val_end, test_end),
                'train_start_date': dates_sorted[start],
                'train_end_date': dates_sorted[train_end - 1],
                'val_start_date': dates_sorted[train_end],
                'val_end_date': dates_sorted[val_end - 1],
                'test_start_date': dates_sorted[val_end],
                'test_end_date': dates_sorted[test_end - 1],
                'train_window_mode': mode,
                'stride_days': stride_days,
            })
            start += stride_days
    else:
        # Expanding train: train indices [0, val_start), val [val_start, val_end), test [...]
        val_start = train_days
        while True:
            val_end = val_start + val_days
            test_end = val_end + test_days
            if test_end > total:
                break
            folds.append({
                'fold': len(folds) + 1,
                'train': (0, val_start),
                'val': (val_start, val_end),
                'test': (val_end, test_end),
                'train_start_date': dates_sorted[0],
                'train_end_date': dates_sorted[val_start - 1],
                'val_start_date': dates_sorted[val_start],
                'val_end_date': dates_sorted[val_end - 1],
                'test_start_date': dates_sorted[val_end],
                'test_end_date': dates_sorted[test_end - 1],
                'train_window_mode': mode,
                'stride_days': stride_days,
            })
            val_start += stride_days

    print(
        f"Generated {len(folds)} folds mode={mode} stride={stride_days} "
        f"(train={'expanding from 0' if mode == 'expanding' else train_days}, "
        f"val={val_days}d, test={test_days}d, total dates={total})"
    )
    assert len(folds) >= 8, (
        f"Only {len(folds)} folds generated. Check TRAIN/VAL/TEST_DAYS vs date range. "
        f"Need at least 8 folds for statistically meaningful walk-forward evaluation."
    )
    return folds


def print_fold_summary(folds: list[dict]) -> None:
    """Pretty-print a summary table of all folds (for thesis Table T2)."""
    header = (f"{'Fold':>4} | {'Train start':>12} | {'Train end':>12} | "
              f"{'Val start':>12} | {'Val end':>12} | "
              f"{'Test start':>12} | {'Test end':>12}")
    print(header)
    print('-' * len(header))
    for f in folds:
        print(
            f"{f['fold']:>4} | "
            f"{str(f['train_start_date'])[:10]:>12} | "
            f"{str(f['train_end_date'])[:10]:>12} | "
            f"{str(f['val_start_date'])[:10]:>12} | "
            f"{str(f['val_end_date'])[:10]:>12} | "
            f"{str(f['test_start_date'])[:10]:>12} | "
            f"{str(f['test_end_date'])[:10]:>12}"
        )


if __name__ == '__main__':
    features_cache = f'data/processed/features_{config.UNIVERSE_MODE}.csv'
    data = pd.read_csv(features_cache, parse_dates=['Date'])
    dates = sorted(data['Date'].unique())
    folds = generate_walk_forward_folds(dates)
    print()
    print_fold_summary(folds)

```

### Computed fold counts/ranges/overlap checks

```text
Generated 18 folds mode=rolling stride=63 (train=252, val=63d, test=63d, total dates=1487)
N_DATES 1487
N_FOLDS 18
FOLD1 {'fold': 1, 'train': (0, 252), 'val': (252, 315), 'test': (315, 378), 'train_start_date': Timestamp('2019-02-01 00:00:00'), 'train_end_date': Timestamp('2020-01-31 00:00:00'), 'val_start_date': Timestamp('2020-02-03 00:00:00'), 'val_end_date': Timestamp('2020-05-01 00:00:00'), 'test_start_date': Timestamp('2020-05-04 00:00:00'), 'test_end_date': Timestamp('2020-07-31 00:00:00'), 'train_window_mode': 'rolling', 'stride_days': 63}
FOLDN {'fold': 18, 'train': (1071, 1323), 'val': (1323, 1386), 'test': (1386, 1449), 'train_start_date': Timestamp('2023-05-04 00:00:00'), 'train_end_date': Timestamp('2024-05-03 00:00:00'), 'val_start_date': Timestamp('2024-05-06 00:00:00'), 'val_end_date': Timestamp('2024-08-05 00:00:00'), 'test_start_date': Timestamp('2024-08-06 00:00:00'), 'test_end_date': Timestamp('2024-11-01 00:00:00'), 'train_window_mode': 'rolling', 'stride_days': 63}
SAME_FOLD_VAL_TEST_OVERLAP False
ADJ_VAL_i_TEST_i1_OVERLAP False
ADJ_TEST_i_VAL_i1_OVERLAP True
```

## SECTION 5 — SCALER AND PREPROCESSING

### pipeline/standardizer.py (verbatim)

```python
"""
pipeline/standardizer.py
Step 5: Feature standardization — fit on train only, transform all splits.

Supports both StandardScaler (default) and MinMaxScaler (Bhandari §4.5).
The scaler type is configured via config.SCALER_TYPE.

ANTI-LEAKAGE: Scaler is ALWAYS fit on training data only, then applied to
validation and test sets.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Union

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


def get_scaler(scaler_type: str = None) -> Union[StandardScaler, MinMaxScaler]:
    """
    Factory function for scaler instantiation.

    Parameters
    ----------
    scaler_type : str, optional
        'standard' (default) or 'minmax'. If None, uses config.SCALER_TYPE.

    Returns
    -------
    sklearn scaler instance (unfitted)
    """
    if scaler_type is None:
        scaler_type = getattr(config, 'SCALER_TYPE', 'standard')

    if scaler_type.lower() == 'minmax':
        return MinMaxScaler()
    return StandardScaler()


def standardize_fold(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    scaler_type: str = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Union[StandardScaler, MinMaxScaler]]:
    """
    Fit a scaler on training data, then apply to all three splits.

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray, shape (n_samples, n_features)
        Raw (unscaled) feature arrays for each split.
    scaler_type : str, optional
        'standard' or 'minmax'. Defaults to config.SCALER_TYPE.

    Returns
    -------
    X_train_s, X_val_s, X_test_s : np.ndarray
        Scaled arrays.
    scaler : StandardScaler or MinMaxScaler
        Fitted scaler (saved per fold so LSTM sequences can be scaled
        consistently via scaled_df helper in main.py).

    Notes
    -----
    StandardScaler: zero mean, unit variance (robust to outliers in returns)
    MinMaxScaler: scales to [0, 1] (Bhandari §4.5 approach)
    """
    scaler = get_scaler(scaler_type)
    X_train_s = scaler.fit_transform(X_train)  # fit + transform on train only
    X_val_s   = scaler.transform(X_val)        # transform only
    X_test_s  = scaler.transform(X_test)       # transform only
    return X_train_s, X_val_s, X_test_s, scaler


def winsorize_fold(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    lower_q: float = 0.005,
    upper_q: float = 0.995,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Clip features to train-derived quantiles (column-wise), then apply same
    bounds to val and test. PIT-safe: quantiles use training rows only.
    """
    X_train = np.asarray(X_train, dtype=np.float64).copy()
    X_val = np.asarray(X_val, dtype=np.float64).copy()
    X_test = np.asarray(X_test, dtype=np.float64).copy()
    n_feat = X_train.shape[1]
    for j in range(n_feat):
        col = X_train[:, j]
        lo = float(np.quantile(col, lower_q))
        hi = float(np.quantile(col, upper_q))
        if lo > hi:
            lo, hi = hi, lo
        X_train[:, j] = np.clip(X_train[:, j], lo, hi)
        X_val[:, j] = np.clip(X_val[:, j], lo, hi)
        X_test[:, j] = np.clip(X_test[:, j], lo, hi)
    return X_train, X_val, X_test


def standardize_train_val(
    X_train: np.ndarray,
    X_val: np.ndarray,
    scaler_type: str = None,
) -> tuple[np.ndarray, np.ndarray, Union[StandardScaler, MinMaxScaler]]:
    """
    Fit a scaler on training data, then apply to train and validation splits.
    Useful for hyperparameter tuning where test set is not yet needed.

    Parameters
    ----------
    X_train, X_val : np.ndarray, shape (n_samples, n_features)
        Raw (unscaled) feature arrays.
    scaler_type : str, optional
        'standard' or 'minmax'. Defaults to config.SCALER_TYPE.

    Returns
    -------
    X_train_s, X_val_s : np.ndarray
        Scaled arrays.
    scaler : StandardScaler or MinMaxScaler
        Fitted scaler.
    """
    scaler = get_scaler(scaler_type)
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    return X_train_s, X_val_s, scaler

```

### Scaler-fit lines in main.py and models/lstm_model.py

```text
main.py:268:    print(f"Scaler Type: {config.SCALER_TYPE}")
main.py:461:            Xb_tr, Xb_v, Xb_ts = winsorize_fold(
main.py:466:        X_tr_b_s, X_v_b_s, X_ts_b_s, _ = standardize_fold(Xb_tr, Xb_v, Xb_ts)
pipeline/standardizer.py:6:The scaler type is configured via config.SCALER_TYPE.
pipeline/standardizer.py:28:        'standard' (default) or 'minmax'. If None, uses config.SCALER_TYPE.
pipeline/standardizer.py:35:        scaler_type = getattr(config, 'SCALER_TYPE', 'standard')
pipeline/standardizer.py:39:    return StandardScaler()
pipeline/standardizer.py:42:def standardize_fold(
pipeline/standardizer.py:56:        'standard' or 'minmax'. Defaults to config.SCALER_TYPE.
pipeline/standardizer.py:72:    X_train_s = scaler.fit_transform(X_train)  # fit + transform on train only
pipeline/standardizer.py:78:def winsorize_fold(
pipeline/standardizer.py:119:        'standard' or 'minmax'. Defaults to config.SCALER_TYPE.
pipeline/standardizer.py:129:    X_train_s = scaler.fit_transform(X_train)
models/lstm_model.py:463:    scaler = StandardScaler()
models/lstm_model.py:464:    scaler.fit(df_train[feature_cols].values)
models/lstm_model.py:508:    scaler = StandardScaler()
models/lstm_model.py:509:    scaler.fit(df_train[feature_cols].values)
models/lstm_model.py:563:    scaler = StandardScaler()
models/lstm_model.py:564:    scaler.fit(df_true_train[feature_cols].values)
models/lstm_model.py:639:    scaler = StandardScaler()
models/lstm_model.py:640:    scaler.fit(df_true_train[feature_cols].values)

```

## SECTION 6 — LSTM SEQUENCE CONSTRUCTION

### models/lstm_model.py (verbatim)

```python
"""
models/lstm_model.py
LSTM-A: Bhandari-inspired technical indicator LSTM (4 features, tuned architecture)
LSTM: Extended ablation — 6 curated features, fixed architecture (64 units, 2 layers)

Both models output raw logits (2 classes) — use CrossEntropyLoss in training.
Inference applies softmax to get class probabilities.

Implements Bhandari §3.3 hyperparameter tuning (Section 1 of IMPLEMENTATION_EXTENSIONS.md):
- Phase 1: tune optimizer, learning rate, batch size
- Phase 2 (LSTM-A only): tune architecture (hidden_size, num_layers, dropout)
"""
import itertools
import logging
import os
import random
import sys
import csv
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from evaluation.metrics_utils import binary_auc_safe

logger = logging.getLogger(__name__)


def _seed_everything(seed: int):
    """Seed all RNGs used by LSTM training for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_torch_generator(seed: int) -> torch.Generator:
    """Create a seeded CPU generator for deterministic DataLoader shuffling."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


# ────────────────────────────────────────────────────────────────────────────
# Hyperparameter Tuning (Bhandari §3.3)
# ────────────────────────────────────────────────────────────────────────────

def _build_optimizer(model, name: str, lr: float):
    """Helper: instantiate optimizer by name string."""
    params = model.parameters()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=config.LSTM_WD)
    elif name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr, weight_decay=config.LSTM_WD)
    elif name == "nadam":
        return torch.optim.NAdam(params, lr=lr, weight_decay=config.LSTM_WD)
    elif name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=config.LSTM_WD)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def _run_tuning_replicates(
    X_train, y_train, X_val, y_val, device,
    opt_name, lr, bs, input_size, hidden_size, num_layers, dropout,
    max_epochs, patience, seed,
    n_replicates=None,
):
    """
    Run cfg.LSTM_TUNE_REPLICATES independent training runs for one hyperparameter
    combination. Returns a list of validation AUC scores (one per replicate).
    """
    if n_replicates is None:
        n_replicates = config.LSTM_TUNE_REPLICATES
    auc_scores = []

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

    for rep in range(n_replicates):
        rep_seed = seed + rep
        _seed_everything(rep_seed)
        train_gen = _make_torch_generator(rep_seed)

        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, generator=train_gen)
        val_dl = DataLoader(val_ds, batch_size=bs * 2, shuffle=False)

        model = StockLSTMTunable(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)

        optimizer = _build_optimizer(model, opt_name, lr)
        criterion = nn.CrossEntropyLoss()

        best_val_loss, patience_ctr = float("inf"), 0
        for epoch in range(max_epochs):
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
                if patience_ctr >= patience:
                    break

        # Compute AUC on validation set
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for Xb, yb in val_dl:
                logits = model(Xb.to(device))
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(yb.numpy())

        if len(set(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_preds)
        else:
            auc = 0.5
        auc_scores.append(auc)

    return auc_scores


def tune_lstm_hyperparams(
    X_train, y_train, X_val, y_val,
    input_size, device,
    arch_grid=None,
    train_grid=None,
    tune_replicates=None,
    tune_patience=None,
    tune_max_epochs=None,
    seed_hidden=64, seed_layers=2, seed_dropout=0.2,
    seed=None,
):
    """
    Bhandari §3.3 Algorithm 1 + Algorithm 2 — adapted for classification
    (AUC replaces RMSE).

    Phase 1 (always): sweeps (optimizer, lr, batch_size) from config.LSTM_HYPERPARAM_GRID.
    Phase 2 (when arch_grid is provided): sweeps (hidden_size, num_layers, dropout)
             from arch_grid, using the best Phase-1 hyperparameters as fixed context.
             This mirrors Bhandari Algorithm 2, which tunes architecture after fixing
             the training hyperparameters.

    LSTM-A calls this function with arch_grid=config.LSTM_A_ARCH_GRID.
    LSTM calls this function with arch_grid=None (architecture stays fixed).

    Parameters
    ----------
    X_train : np.ndarray
        Training sequences, shape (N, seq_len, n_features)
    y_train : np.ndarray
        Training labels
    X_val : np.ndarray
        Validation sequences
    y_val : np.ndarray
        Validation labels
    input_size : int
        Number of input features
    device : torch.device
        Device to train on
    arch_grid : dict or None
        If provided, contains hidden_size, num_layers, dropout options to search
    seed_hidden, seed_layers, seed_dropout : int/float
        Default architecture to use if arch_grid is None (Phase 1 only)

    Returns
    -------
    dict with keys: optimizer, lr, batch_size, hidden_size, num_layers, dropout
    """
    if seed is None:
        seed = config.RANDOM_SEED

    # ── Phase 1: tune training hyperparameters ────────────────────────────────
    grid = train_grid if train_grid is not None else config.LSTM_HYPERPARAM_GRID
    replicates = tune_replicates if tune_replicates is not None else config.LSTM_TUNE_REPLICATES
    max_epochs = tune_max_epochs if tune_max_epochs is not None else config.LSTM_TUNE_MAX_EPOCHS
    patience = tune_patience if tune_patience is not None else config.LSTM_TUNE_PATIENCE
    combos = list(itertools.product(
        grid["optimizer"], grid["learning_rate"], grid["batch_size"]
    ))

    # For Phase 1, use a fixed architecture seed
    if arch_grid is not None:
        p1_hidden = arch_grid["hidden_size"][0]
        p1_layers = arch_grid["num_layers"][0]
        p1_drop = arch_grid["dropout"][0]
    else:
        p1_hidden = seed_hidden
        p1_layers = seed_layers
        p1_drop = seed_dropout

        print(f"[LSTM Tuning - Phase 1] {len(combos)} training combos x "
            f"{replicates} replicates")

    phase1_results = []
    for opt_name, lr, bs in combos:
        auc_scores = _run_tuning_replicates(
            X_train, y_train, X_val, y_val, device,
            opt_name, lr, bs, input_size,
            hidden_size=p1_hidden, num_layers=p1_layers, dropout=p1_drop,
            max_epochs=max_epochs,
            patience=patience,
            seed=seed,
            n_replicates=replicates,
        )
        avg_auc = sum(auc_scores) / len(auc_scores)
        phase1_results.append({
            "optimizer": opt_name, "lr": lr, "batch_size": bs, "avg_val_auc": avg_auc
        })
        print(f"  opt={opt_name:7s}  lr={lr:.4f}  bs={bs:3d}  -> avg AUC={avg_auc:.4f}")

    best_p1 = max(phase1_results, key=lambda x: x["avg_val_auc"])
    print(f"[Phase 1 best] {best_p1}")

    # If no arch grid, return Phase 1 results with fixed architecture
    if arch_grid is None:
        return {
            "optimizer": best_p1["optimizer"],
            "lr": best_p1["lr"],
            "batch_size": best_p1["batch_size"],
            "hidden_size": seed_hidden,
            "num_layers": seed_layers,
            "dropout": seed_dropout,
        }

    # ── Phase 2: tune architecture (LSTM-A only) ──────────────────────────────
    arch_combos = list(itertools.product(
        arch_grid["hidden_size"], arch_grid["num_layers"], arch_grid["dropout"]
    ))
    print(f"\n[LSTM Tuning - Phase 2] {len(arch_combos)} architecture combos x "
          f"{replicates} replicates")

    phase2_results = []
    for hidden_size, num_layers, dropout in arch_combos:
        auc_scores = _run_tuning_replicates(
            X_train, y_train, X_val, y_val, device,
            best_p1["optimizer"], best_p1["lr"], best_p1["batch_size"], input_size,
            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
            max_epochs=max_epochs,
            patience=patience,
            seed=seed + 10_000,
            n_replicates=replicates,
        )
        avg_auc = sum(auc_scores) / len(auc_scores)
        phase2_results.append({
            "hidden_size": hidden_size, "num_layers": num_layers,
            "dropout": dropout, "avg_val_auc": avg_auc
        })
        print(f"  h={hidden_size:3d}  layers={num_layers}  drop={dropout:.1f}"
              f"  -> avg AUC={avg_auc:.4f}")

    best_p2 = max(phase2_results, key=lambda x: x["avg_val_auc"])
    print(f"[Phase 2 best] {best_p2}")

    return {
        "optimizer": best_p1["optimizer"],
        "lr": best_p1["lr"],
        "batch_size": best_p1["batch_size"],
        "hidden_size": best_p2["hidden_size"],
        "num_layers": best_p2["num_layers"],
        "dropout": best_p2["dropout"],
    }


# ────────────────────────────────────────────────────────────────────────────
# Sequence Building Functions
# ────────────────────────────────────────────────────────────────────────────

def _build_sequences(df: pd.DataFrame, seq_len: int, feature_col: str):
    """
    Constructs overlapping sequences per ticker for single-feature input.
    df must have columns: [Date, Ticker, {feature_col}, Target]
    sorted by [Ticker, Date] ascending.

    Returns:
        X: np.array shape (N, seq_len, 1)
        y: np.array shape (N,)
        keys: list of (date, ticker) tuples for alignment
    """
    X_list, y_list, keys = [], [], []
    for ticker, grp in df.sort_values(['Ticker', 'Date']).groupby('Ticker'):
        grp = grp.reset_index(drop=True)
        vals = grp[feature_col].values
        labels = grp['Target'].values
        dates = grp['Date'].values
        for i in range(seq_len, len(vals)):
            X_list.append(vals[i - seq_len:i])
            y_list.append(labels[i])
            keys.append((dates[i], ticker))
    if len(X_list) == 0:
        # No sequences could be built (not enough data points per ticker)
        X = np.zeros((0, seq_len, 1), dtype=np.float32)
        y = np.zeros((0,), dtype=np.int64)
        return X, y, keys
    X = np.array(X_list)[:, :, np.newaxis].astype(np.float32)  # (N, seq_len, 1)
    y = np.array(y_list).astype(np.int64)
    return X, y, keys


def _build_sequences_multi(df: pd.DataFrame, seq_len: int, feature_cols: list):
    """
    Constructs overlapping multi-feature sequences per ticker.
    df must have columns: [Date, Ticker, *feature_cols, Target]
    sorted by [Ticker, Date] ascending.

    Returns:
        X: np.array shape (N, seq_len, n_feat)
        y: np.array shape (N,)
        keys: list of (date, ticker) tuples for alignment
    """
    X_list, y_list, keys = [], [], []
    n_feat = len(feature_cols)
    for ticker, grp in df.sort_values(['Ticker', 'Date']).groupby('Ticker'):
        grp = grp.reset_index(drop=True)
        vals = grp[feature_cols].values
        labels = grp['Target'].values
        dates = grp['Date'].values
        for i in range(seq_len, len(vals)):
            X_list.append(vals[i - seq_len:i])  # shape (seq_len, n_feat)
            y_list.append(labels[i])
            keys.append((dates[i], ticker))
    if len(X_list) == 0:
        # No sequences could be built (not enough data points per ticker)
        X = np.zeros((0, seq_len, n_feat), dtype=np.float32)
        y = np.zeros((0,), dtype=np.int64)
        return X, y, keys
    X = np.array(X_list).astype(np.float32)  # (N, seq_len, n_feat)
    y = np.array(y_list).astype(np.int64)
    return X, y, keys


def _build_sequences_multi_with_lookback(df_combined: pd.DataFrame, seq_len: int,
                                          feature_cols: list, test_dates: set):
    """
    Build multi-feature sequences from combined train+test data, but only output
    sequences where the target day is in test_dates.

    Args:
        df_combined: DataFrame with train+test data
        seq_len: lookback window length
        feature_cols: list of column names for features
        test_dates: set of dates that define the test period

    Returns:
        X, y, keys for test period only
    """
    X_list, y_list, keys = [], [], []
    n_feat = len(feature_cols)
    for ticker, grp in df_combined.sort_values(['Ticker', 'Date']).groupby('Ticker'):
        grp = grp.reset_index(drop=True)
        vals = grp[feature_cols].values
        labels = grp['Target'].values
        dates = grp['Date'].values
        for i in range(seq_len, len(vals)):
            date_str = pd.Timestamp(dates[i]).strftime('%Y-%m-%d')
            if date_str in test_dates:
                X_list.append(vals[i - seq_len:i])
                y_list.append(labels[i])
                keys.append((dates[i], ticker))

    if len(X_list) == 0:
        X = np.zeros((0, seq_len, n_feat), dtype=np.float32)
        y = np.zeros((0,), dtype=np.int64)
        return X, y, keys
    X = np.array(X_list).astype(np.float32)
    y = np.array(y_list).astype(np.int64)
    return X, y, keys


def _build_sequences_with_lookback(df_combined: pd.DataFrame, seq_len: int,
                                    feature_col: str, test_dates: set):
    """
    Build sequences from combined train+test data, but only output sequences
    where the target day is in test_dates. This allows using training data
    as lookback history for test predictions.

    Args:
        df_combined: DataFrame with train+test data, sorted by [Ticker, Date]
        seq_len: lookback window length
        feature_col: column name for the feature
        test_dates: set of dates (as strings or Timestamps) that define the test period

    Returns:
        X, y, keys for test period only
    """
    X_list, y_list, keys = [], [], []
    for ticker, grp in df_combined.sort_values(['Ticker', 'Date']).groupby('Ticker'):
        grp = grp.reset_index(drop=True)
        vals = grp[feature_col].values
        labels = grp['Target'].values
        dates = grp['Date'].values
        for i in range(seq_len, len(vals)):
            date_str = pd.Timestamp(dates[i]).strftime('%Y-%m-%d')
            # Only include if target day is in test period
            if date_str in test_dates:
                X_list.append(vals[i - seq_len:i])
                y_list.append(labels[i])
                keys.append((dates[i], ticker))

    if len(X_list) == 0:
        X = np.zeros((0, seq_len, 1), dtype=np.float32)
        y = np.zeros((0,), dtype=np.int64)
        return X, y, keys
    X = np.array(X_list)[:, :, np.newaxis].astype(np.float32)
    y = np.array(y_list).astype(np.int64)
    return X, y, keys


def prepare_lstm_a_sequences(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    Builds overlapping multi-feature sequences for LSTM-A.
    Uses 4 features: MACD, RSI_14, ATR_14, Return_1d (Bhandari §4.3 inspired).

    Standardization is fit on training data ONLY, then applied to both splits.

    For test sequences, training data is used as lookback history so that
    predictions can be made even when test period < seq_len.

    Args:
        df_train: DataFrame with columns [Date, Ticker, *LSTM_A_FEATURES, Target]
        df_test:  DataFrame with same columns

    Returns:
        X_train: np.array shape (N_train, seq_len, n_feat)
        y_train: np.array shape (N_train,)
        X_test:  np.array shape (N_test, seq_len, n_feat)
        y_test:  np.array shape (N_test,)
        keys_train: list of (date, ticker)
        keys_test:  list of (date, ticker)
    """
    seq_len = config.LSTM_A_SEQ_LEN
    feature_cols = config.LSTM_A_FEATURES  # 4 features

    # Fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols].values)

    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train[feature_cols] = scaler.transform(df_train[feature_cols].values)
    df_test[feature_cols] = scaler.transform(df_test[feature_cols].values)

    # Build training sequences from train data only
    X_train, y_train, keys_train = _build_sequences_multi(df_train, seq_len, feature_cols)

    # Build test sequences using train data as lookback history
    test_dates = set(df_test['Date'].apply(lambda d: pd.Timestamp(d).strftime('%Y-%m-%d')))
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    X_test, y_test, keys_test = _build_sequences_multi_with_lookback(
        df_combined, seq_len, feature_cols, test_dates
    )

    return X_train, y_train, X_test, y_test, keys_train, keys_test


def prepare_lstm_b_sequences(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    Builds overlapping multi-feature sequences for LSTM.
    Scaler is fit on training fold only and applied to both splits.

    For test sequences, training data is used as lookback history so that
    predictions can be made even when test period < seq_len.

    Args:
        df_train: DataFrame with columns [Date, Ticker, *LSTM_B_FEATURES, Target]
        df_test:  DataFrame with same columns

    Returns:
        X_train: np.array shape (N_train, seq_len, n_feat)
        y_train: np.array shape (N_train,)
        X_test:  np.array shape (N_test, seq_len, n_feat)
        y_test:  np.array shape (N_test,)
        keys_train: list of (date, ticker)
        keys_test:  list of (date, ticker)
    """
    seq_len = config.LSTM_B_SEQ_LEN
    feature_cols = config.LSTM_B_FEATURES

    # Fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols].values)

    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train[feature_cols] = scaler.transform(df_train[feature_cols].values)
    df_test[feature_cols] = scaler.transform(df_test[feature_cols].values)

    # Build training sequences from train data only
    X_train, y_train, keys_train = _build_sequences_multi(df_train, seq_len, feature_cols)

    # Build test sequences using train data as lookback history
    test_dates = set(df_test['Date'].apply(lambda d: pd.Timestamp(d).strftime('%Y-%m-%d')))
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    X_test, y_test, keys_test = _build_sequences_multi_with_lookback(
        df_combined, seq_len, feature_cols, test_dates
    )

    return X_train, y_train, X_test, y_test, keys_train, keys_test


def prepare_lstm_a_sequences_temporal_split(df_train: pd.DataFrame, df_test: pd.DataFrame,
                                             val_ratio: float = 0.2):
    """
    Build LSTM-A sequences with TEMPORAL train/val split.

    FIX: Instead of splitting by index (which splits by ticker), this function
    splits by DATE - the last val_ratio of training DATES become validation.

    Args:
        df_train: DataFrame with columns [Date, Ticker, *LSTM_A_FEATURES, Target]
        df_test:  DataFrame with same columns
        val_ratio: Fraction of training dates to use for validation (default 0.2)

    Returns:
        X_train, y_train: Training sequences
        X_val, y_val: Validation sequences (last 20% of dates)
        X_test, y_test: Test sequences
        keys_train, keys_val, keys_test: (date, ticker) tuples for alignment
    """
    seq_len = config.LSTM_A_SEQ_LEN
    feature_cols = config.LSTM_A_FEATURES

    # Split training dates temporally
    train_dates_sorted = sorted(df_train['Date'].unique())
    n_dates = len(train_dates_sorted)
    val_start_idx = int(n_dates * (1 - val_ratio))

    train_date_set = set(pd.Timestamp(d).strftime('%Y-%m-%d')
                         for d in train_dates_sorted[:val_start_idx])
    val_date_set = set(pd.Timestamp(d).strftime('%Y-%m-%d')
                       for d in train_dates_sorted[val_start_idx:])

    # Fit scaler on TRUE training data only (excluding validation dates)
    df_true_train = df_train[df_train['Date'].isin(train_dates_sorted[:val_start_idx])]
    scaler = StandardScaler()
    scaler.fit(df_true_train[feature_cols].values)

    # Apply scaler to all data
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train[feature_cols] = scaler.transform(df_train[feature_cols].values)
    df_test[feature_cols] = scaler.transform(df_test[feature_cols].values)

    # Build ALL sequences from training data
    X_all, y_all, keys_all = _build_sequences_multi(df_train, seq_len, feature_cols)

    # Split sequences by DATE (not by index!)
    X_train_list, y_train_list, keys_train = [], [], []
    X_val_list, y_val_list, keys_val = [], [], []

    for i, (date, ticker) in enumerate(keys_all):
        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        if date_str in train_date_set:
            X_train_list.append(X_all[i])
            y_train_list.append(y_all[i])
            keys_train.append((date, ticker))
        elif date_str in val_date_set:
            X_val_list.append(X_all[i])
            y_val_list.append(y_all[i])
            keys_val.append((date, ticker))

    X_train = np.array(X_train_list, dtype=np.float32) if X_train_list else np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int64) if y_train_list else np.zeros(0, dtype=np.int64)
    X_val = np.array(X_val_list, dtype=np.float32) if X_val_list else np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.int64) if y_val_list else np.zeros(0, dtype=np.int64)

    # Build test sequences using full train data as lookback history
    test_dates = set(df_test['Date'].apply(lambda d: pd.Timestamp(d).strftime('%Y-%m-%d')))
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    X_test, y_test, keys_test = _build_sequences_multi_with_lookback(
        df_combined, seq_len, feature_cols, test_dates
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, keys_train, keys_val, keys_test


def prepare_lstm_b_sequences_temporal_split(df_train: pd.DataFrame, df_test: pd.DataFrame,
                                             val_ratio: float = 0.2):
    """
    Build LSTM sequences with TEMPORAL train/val split.

    FIX: Instead of splitting by index (which splits by ticker), this function
    splits by DATE - the last val_ratio of training DATES become validation.

    Args:
        df_train: DataFrame with columns [Date, Ticker, *LSTM_B_FEATURES, Target]
        df_test:  DataFrame with same columns
        val_ratio: Fraction of training dates to use for validation (default 0.2)

    Returns:
        X_train, y_train: Training sequences
        X_val, y_val: Validation sequences (last 20% of dates)
        X_test, y_test: Test sequences
        keys_train, keys_val, keys_test: (date, ticker) tuples for alignment
    """
    seq_len = config.LSTM_B_SEQ_LEN
    feature_cols = config.LSTM_B_FEATURES

    # Split training dates temporally
    train_dates_sorted = sorted(df_train['Date'].unique())
    n_dates = len(train_dates_sorted)
    val_start_idx = int(n_dates * (1 - val_ratio))

    train_date_set = set(pd.Timestamp(d).strftime('%Y-%m-%d')
                         for d in train_dates_sorted[:val_start_idx])
    val_date_set = set(pd.Timestamp(d).strftime('%Y-%m-%d')
                       for d in train_dates_sorted[val_start_idx:])

    # Fit scaler on TRUE training data only (excluding validation dates)
    df_true_train = df_train[df_train['Date'].isin(train_dates_sorted[:val_start_idx])]
    scaler = StandardScaler()
    scaler.fit(df_true_train[feature_cols].values)

    # Apply scaler to all data
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train[feature_cols] = scaler.transform(df_train[feature_cols].values)
    df_test[feature_cols] = scaler.transform(df_test[feature_cols].values)

    # Build ALL sequences from training data
    X_all, y_all, keys_all = _build_sequences_multi(df_train, seq_len, feature_cols)

    # Split sequences by DATE (not by index!)
    X_train_list, y_train_list, keys_train = [], [], []
    X_val_list, y_val_list, keys_val = [], [], []

    for i, (date, ticker) in enumerate(keys_all):
        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        if date_str in train_date_set:
            X_train_list.append(X_all[i])
            y_train_list.append(y_all[i])
            keys_train.append((date, ticker))
        elif date_str in val_date_set:
            X_val_list.append(X_all[i])
            y_val_list.append(y_all[i])
            keys_val.append((date, ticker))

    X_train = np.array(X_train_list, dtype=np.float32) if X_train_list else np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int64) if y_train_list else np.zeros(0, dtype=np.int64)
    X_val = np.array(X_val_list, dtype=np.float32) if X_val_list else np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.int64) if y_val_list else np.zeros(0, dtype=np.int64)

    # Build test sequences using full train data as lookback history
    test_dates = set(df_test['Date'].apply(lambda d: pd.Timestamp(d).strftime('%Y-%m-%d')))
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    X_test, y_test, keys_test = _build_sequences_multi_with_lookback(
        df_combined, seq_len, feature_cols, test_dates
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, keys_train, keys_val, keys_test


# ────────────────────────────────────────────────────────────────────────────
# Model Architectures
# ────────────────────────────────────────────────────────────────────────────

class StockLSTMTunable(nn.Module):
    """
    Flexible LSTM architecture for hyperparameter tuning.
    All architecture parameters are configurable via constructor.
    Used for both LSTM-A (tuned) and LSTM (fixed architecture).
    Outputs logits for 2 classes.
    """
    def __init__(
        self,
        input_size: int,        # 4 for LSTM-A, 6 for LSTM
        hidden_size: int = 64,  # tuner-resolved for LSTM-A; fixed 64 for LSTM
        num_layers: int = 2,    # tuner-resolved for LSTM-A; fixed 2 for LSTM
        dropout: float = 0.2,   # tuner-resolved for LSTM-A; fixed 0.2 for LSTM
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
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # take last timestep
        out = self.relu(self.fc1(out))
        return self.fc2(out)  # logits for 2 classes


class LSTMModelA(nn.Module):
    """
    LSTM-A: Bhandari-inspired technical indicator LSTM.
    4 features (MACD, RSI, ATR, Return_1d), architecture from hyperparameter tuning.
    Outputs logits for 2 classes.
    """
    def __init__(self, hidden_size=None, num_layers=None, dropout=None):
        super().__init__()
        n_feat = len(config.LSTM_A_FEATURES)
        # Use tuned values if provided, otherwise use first value from grid
        hidden = hidden_size if hidden_size else config.LSTM_A_ARCH_GRID["hidden_size"][0]
        layers = num_layers if num_layers else config.LSTM_A_ARCH_GRID["num_layers"][0]
        drop = dropout if dropout else config.LSTM_A_ARCH_GRID["dropout"][0]

        self.lstm = nn.LSTM(
            input_size=n_feat,
            hidden_size=hidden,
            num_layers=layers,
            dropout=drop if layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(drop)
        self.fc1 = nn.Linear(hidden, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc2(self.relu(self.fc1(out)))


class LSTMModelB(nn.Module):
    """
    Extended LSTM with 6 input features, 2 layers.
    Outputs logits for 2 classes.
    """
    def __init__(self, hidden_size=None, num_layers=None, dropout=None):
        super().__init__()
        n_feat = len(config.LSTM_B_FEATURES)
        hidden = config.LSTM_B_HIDDEN if hidden_size is None else int(hidden_size)
        layers = config.LSTM_B_LAYERS if num_layers is None else int(num_layers)
        drop = config.LSTM_B_DROPOUT if dropout is None else float(dropout)
        self.lstm = nn.LSTM(
            input_size=n_feat,
            hidden_size=hidden,
            num_layers=layers,
            dropout=drop if layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(drop)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


# ────────────────────────────────────────────────────────────────────────────
# Training Functions
# ────────────────────────────────────────────────────────────────────────────

def _clear_mps_cache():
    """Clear MPS memory cache if available."""
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _eval_loader_loss_auc(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, Optional[float]]:
    """Mean loss and ROC-AUC (positive class prob vs labels) on a loader."""
    model.eval()
    total_loss = 0.0
    n = 0
    probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * len(yb)
            n += len(yb)
            pr = torch.softmax(logits, dim=1)[:, 1]
            probs_list.append(pr.cpu().numpy())
            labels_list.append(yb.cpu().numpy())
    mean_loss = total_loss / max(n, 1)
    if not probs_list:
        return mean_loss, None
    y_score = np.concatenate(probs_list)
    y_true = np.concatenate(labels_list)
    auc = binary_auc_safe(y_true, y_score, log_on_fail=False)
    return mean_loss, auc


def _train_lstm_impl(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    max_epochs: int,
    patience: int,
    desc: str,
    batch_size: int | None = None,
    seed: int | None = None,
    lr_scheduler: Any | None = None,
    fold_idx: int | None = None,
) -> nn.Module:
    """
    Training loop with per-epoch train/val loss, AUC, LR logging, optional CSV,
    gradient norm audit, and heuristic warnings (flat AUC, train/val loss gap).
    """
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val),
    )

    if batch_size is None:
        batch_size = config.LSTM_A_BATCH if desc == "LSTM-A" else config.LSTM_B_BATCH

    if seed is None:
        seed = config.RANDOM_SEED
    train_gen = _make_torch_generator(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_gen,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

    max_grad_norm = getattr(config, "LSTM_MAX_GRAD_NORM", None)
    audit_grad = getattr(config, "LSTM_AUDIT_GRAD_NORM", False)
    log_every = getattr(config, "LSTM_LOG_EVERY_EPOCH", True)
    save_csv = getattr(config, "LSTM_SAVE_TRAINING_CSV", False)
    flat_n = getattr(config, "LSTM_FLAT_AUC_WARN_epochs", 8)
    flat_eps = getattr(config, "LSTM_FLAT_AUC_EPS", 0.02)
    of_ratio = getattr(config, "LSTM_OVERFIT_LOSS_RATIO", 3.0)
    of_n = getattr(config, "LSTM_OVERFIT_WARN_epochs", 6)

    epoch_rows: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_state: dict | None = None
    best_epoch = -1
    patience_ctr = 0
    flat_streak = 0
    flat_warned = False
    of_streak = 0
    of_warned = False
    stop_reason = "max_epochs"
    epoch = -1

    with tqdm(range(max_epochs), desc=desc, unit="epoch") as pbar:
        for epoch in pbar:
            model.train()
            train_loss_sum = 0.0
            train_count = 0
            last_gnorm: float | None = None

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()

                if max_grad_norm is not None:
                    gnorm_t = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(max_grad_norm)
                    )
                else:
                    gnorm_t = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float("inf")
                    )
                last_gnorm = float(gnorm_t)

                if audit_grad and last_gnorm is not None:
                    if last_gnorm < 1e-6 or last_gnorm > 1e3:
                        logger.warning(
                            "[%s] epoch %s gradient norm=%.4e (vanish/explode check)",
                            desc,
                            epoch + 1,
                            last_gnorm,
                        )

                optimizer.step()
                train_loss_sum += loss.item() * len(yb)
                train_count += len(yb)

            train_loss_batch = train_loss_sum / max(train_count, 1)

            tr_eval_loss, tr_auc = _eval_loader_loss_auc(
                model, train_loader, device, criterion
            )
            val_loss, val_auc = _eval_loader_loss_auc(
                model, val_loader, device, criterion
            )

            if lr_scheduler is not None:
                lr_scheduler.step(val_loss)
            lr = float(optimizer.param_groups[0]["lr"])

            row = {
                "epoch": epoch + 1,
                "train_loss_batch": round(train_loss_batch, 6),
                "train_eval_loss": round(tr_eval_loss, 6),
                "val_loss": round(val_loss, 6),
                "train_auc": tr_auc,
                "val_auc": val_auc,
                "lr": lr,
                "grad_norm_last": last_gnorm,
            }
            epoch_rows.append(row)

            if log_every:
                au = f" tr_auc={tr_auc:.4f} va_auc={val_auc:.4f}" if tr_auc is not None and val_auc is not None else ""
                logger.info(
                    "[%s] epoch %d/%d train_loss=%.5f tr_eval=%.5f val_loss=%.5f lr=%.2e%s",
                    desc,
                    epoch + 1,
                    max_epochs,
                    train_loss_batch,
                    tr_eval_loss,
                    val_loss,
                    lr,
                    au,
                )

            if tr_auc is not None and val_auc is not None:
                if (
                    abs(tr_auc - 0.5) < flat_eps
                    and abs(val_auc - 0.5) < flat_eps
                ):
                    flat_streak += 1
                else:
                    flat_streak = 0
                if flat_streak >= flat_n and not flat_warned:
                    logger.warning(
                        "[%s] Train and val AUC near 0.5 for %d consecutive epochs (no discrimination).",
                        desc,
                        flat_streak,
                    )
                    flat_warned = True

            if tr_eval_loss > 1e-12 and val_loss > of_ratio * tr_eval_loss:
                of_streak += 1
            else:
                of_streak = 0
            if of_streak >= of_n and not of_warned:
                logger.warning(
                    "[%s] Val loss >> train eval loss for %d epochs (possible overfitting).",
                    desc,
                    of_streak,
                )
                of_warned = True

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    stop_reason = "early_stop_patience"
                    pbar.set_postfix(
                        {"val_loss": f"{best_val_loss:.4f}", "status": "early stop"}
                    )
                    break

            pbar.set_postfix(
                {"val_loss": f"{val_loss:.4f}", "best": f"{best_val_loss:.4f}"}
            )

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epoch + 1

    model.load_state_dict(best_state)
    model.to(device)

    print(
        f"  [{desc}] stopped: {stop_reason} | best_epoch={best_epoch} "
        f"| best_val_loss={best_val_loss:.4f}"
    )

    if save_csv and epoch_rows:
        tag = desc.replace(" ", "_").lower()
        fd = fold_idx if fold_idx is not None else "na"
        out_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "training_logs")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"fold{fd}_{tag}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(epoch_rows[0].keys()))
            w.writeheader()
            w.writerows(epoch_rows)
        print(f"  [{desc}] training log: {path}")

    return model


def train_lstm_a(
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    optimizer_name=None,
    lr=None,
    hidden_size=None,
    num_layers=None,
    dropout=None,
    batch_size=None,
    seed=None,
    fold_idx: int | None = None,
):
    """
    Trains LSTM-A using specified or default hyperparameters.
    Returns trained model with best validation loss weights restored.
    Falls back to CPU if MPS runs out of memory.

    Parameters
    ----------
    optimizer_name : str, optional
        Optimizer name ('adam', 'adagrad', 'nadam'). Defaults to config value.
    lr : float, optional
        Learning rate. Defaults to config value.
    hidden_size, num_layers, dropout : int/float, optional
        Architecture params. Defaults to first value in arch grid.
    batch_size : int, optional
        Batch size. Defaults to config value.
    """
    _clear_mps_cache()

    # Use tuned or default values
    opt_name = optimizer_name or config.LSTM_A_OPTIMIZER
    learning_rate = lr or config.LSTM_A_LR
    bs = batch_size or config.LSTM_A_BATCH
    h_size = hidden_size or config.LSTM_A_ARCH_GRID["hidden_size"][0]
    n_layers = num_layers or config.LSTM_A_ARCH_GRID["num_layers"][0]
    drop = dropout or config.LSTM_A_ARCH_GRID["dropout"][0]

    n_feat = len(config.LSTM_A_FEATURES)
    train_seed = config.RANDOM_SEED if seed is None else seed
    _seed_everything(train_seed)

    # Try with the given device first
    try:
        model = StockLSTMTunable(
            input_size=n_feat,
            hidden_size=h_size,
            num_layers=n_layers,
            dropout=drop,
        ).to(device)

        optimizer = _build_optimizer(model, opt_name, learning_rate)
        criterion = nn.CrossEntropyLoss()

        return _train_lstm_impl(
            model, X_train, y_train, X_val, y_val, device,
            optimizer, criterion, config.LSTM_A_MAX_EPOCHS,
            config.LSTM_A_PATIENCE, "LSTM-A", bs, seed=train_seed,
            lr_scheduler=None, fold_idx=fold_idx,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "MPS" in str(e):
            print(f"  [LSTM-A] MPS out of memory, falling back to CPU...")
            _clear_mps_cache()
            cpu_device = torch.device('cpu')
            model = StockLSTMTunable(
                input_size=n_feat,
                hidden_size=h_size,
                num_layers=n_layers,
                dropout=drop,
            ).to(cpu_device)

            optimizer = _build_optimizer(model, opt_name, learning_rate)
            criterion = nn.CrossEntropyLoss()

            return _train_lstm_impl(
                model, X_train, y_train, X_val, y_val, cpu_device,
                optimizer, criterion, config.LSTM_A_MAX_EPOCHS,
                config.LSTM_A_PATIENCE, "LSTM-A", bs, seed=train_seed,
                lr_scheduler=None, fold_idx=fold_idx,
            )
        raise


def _train_lstm_b_impl(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    optimizer,
    scheduler,
    criterion,
    max_epochs,
    patience,
    batch_size,
    seed=None,
    fold_idx: int | None = None,
):
    """Delegates to _train_lstm_impl with ReduceLROnPlateau."""
    return _train_lstm_impl(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        device,
        optimizer,
        criterion,
        max_epochs,
        patience,
        "LSTM",
        batch_size=batch_size,
        seed=seed,
        lr_scheduler=scheduler,
        fold_idx=fold_idx,
    )


def train_lstm_b(
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    seed=None,
    fold_idx: int | None = None,
    learning_rate: float | None = None,
    optimizer_name: str | None = None,
    batch_size: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
    dropout: float | None = None,
):
    """
    Trains LSTM using Adam with ReduceLROnPlateau scheduler.
    Returns trained model with best validation loss weights restored.
    Falls back to CPU if MPS runs out of memory.
    """
    _clear_mps_cache()
    train_seed = config.RANDOM_SEED if seed is None else seed
    _seed_everything(train_seed)
    lr_use = float(learning_rate) if learning_rate is not None else config.LSTM_B_LR
    opt_name = optimizer_name if optimizer_name is not None else config.LSTM_B_OPTIMIZER
    bs = int(batch_size) if batch_size is not None else int(config.LSTM_B_BATCH)
    hs = int(hidden_size) if hidden_size is not None else int(config.LSTM_B_HIDDEN)
    nl = int(num_layers) if num_layers is not None else int(config.LSTM_B_LAYERS)
    dr = float(dropout) if dropout is not None else float(config.LSTM_B_DROPOUT)

    def _create_model_and_optim(dev):
        model = LSTMModelB(hidden_size=hs, num_layers=nl, dropout=dr).to(dev)
        optimizer = _build_optimizer(model, opt_name, lr_use)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=config.LSTM_B_LR_PATIENCE,
            factor=config.LSTM_B_LR_FACTOR,
        )
        criterion = nn.CrossEntropyLoss()
        return model, optimizer, scheduler, criterion

    # Try with the given device first
    try:
        model, optimizer, scheduler, criterion = _create_model_and_optim(device)
        return _train_lstm_b_impl(
            model, X_train, y_train, X_val, y_val, device,
            optimizer, scheduler, criterion, config.LSTM_B_MAX_EPOCHS,
            config.LSTM_B_PATIENCE,
            bs,
            seed=train_seed,
            fold_idx=fold_idx,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "MPS" in str(e):
            print(f"  [LSTM] MPS out of memory, falling back to CPU...")
            _clear_mps_cache()
            cpu_device = torch.device('cpu')
            model, optimizer, scheduler, criterion = _create_model_and_optim(cpu_device)
            return _train_lstm_b_impl(
                model, X_train, y_train, X_val, y_val, cpu_device,
                optimizer, scheduler, criterion, config.LSTM_B_MAX_EPOCHS,
                config.LSTM_B_PATIENCE,
                bs,
                seed=train_seed,
                fold_idx=fold_idx,
            )
        raise


# ────────────────────────────────────────────────────────────────────────────
# Prediction Functions
# ────────────────────────────────────────────────────────────────────────────

def predict_lstm(model, X: np.ndarray, device=None) -> np.ndarray:
    """
    Run inference and return predicted probability of class 1.
    Model outputs logits — softmax is applied here.
    Uses batched inference for memory efficiency.

    Args:
        model: trained LSTMModelA or LSTMModelB
        X: np.array of sequences
        device: torch device (if None, uses model's current device)

    Returns:
        np.array of probabilities for class 1
    """
    # Detect model's actual device
    model_device = next(model.parameters()).device
    if device is None:
        device = model_device
    elif device != model_device:
        # Model might have been moved to CPU due to OOM, use model's device
        device = model_device

    model.eval()
    batch_size = 512  # Use reasonable batch size for inference
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i + batch_size]).to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs)


def align_predictions_to_df(probs: np.ndarray, keys: list, df: pd.DataFrame) -> np.ndarray:
    """
    Map LSTM output (aligned to dataset keys) back to all rows of df.

    Args:
        probs: predicted probabilities from predict_lstm
        keys: list of (date, ticker) from sequence building
        df: original DataFrame to align to

    Returns:
        np.array aligned to df rows (NaN where no prediction)
    """
    # If a model was intentionally skipped (e.g., DEV mode), keep alignment safe.
    if probs is None or keys is None:
        return np.full(len(df), np.nan)

    prob_map = {
        (pd.Timestamp(date).strftime('%Y-%m-%d'), ticker): float(prob)
        for (date, ticker), prob in zip(keys, probs)
    }

    result = np.full(len(df), np.nan)
    for i, (_, row) in enumerate(df.iterrows()):
        key = (pd.Timestamp(row['Date']).strftime('%Y-%m-%d'), row['Ticker'])
        if key in prob_map:
            result[i] = prob_map[key]
    return result

```

### Computed sequence totals across current folds

```text
Generated 18 folds mode=rolling stride=63 (train=252, val=63d, test=63d, total dates=1487)
N_FOLDS 18
A_TRAIN 93026
A_VAL 26719
A_TEST 26845
B_TRAIN 93026
B_VAL 26719
B_TEST 26845
A_TOTAL 146590
B_TOTAL 146590
```

## SECTION 7 — MODEL TRAINING LOOPS

### models/baselines.py (verbatim)

```python
"""
models/baselines.py
Step 6a/b/c: Logistic Regression, Random Forest, XGBoost classifiers.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    RF_PARAM_GRID,
    XGB_PARAM_GRID, XGB_COLSAMPLE,
    XGB_ROUNDS, XGB_EARLY_STOP, XGB_REG_ALPHA, XGB_REG_LAMBDA,
    RANDOM_SEED,
)


def train_logistic(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """
    L2-regularised Logistic Regression with time-series-aware CV over C.
    Uses TimeSeriesSplit to respect temporal ordering within the training window.
    """
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    lr = LogisticRegression(
        solver='lbfgs', max_iter=1000, random_state=RANDOM_SEED
    )
    tscv = TimeSeriesSplit(n_splits=5)
    cv = GridSearchCV(lr, param_grid, cv=tscv, scoring='roc_auc', n_jobs=-1)
    cv.fit(X_train, y_train)
    print(f"  LR  best C={cv.best_params_['C']:.4f}  CV AUC={cv.best_score_:.4f}")
    return cv.best_estimator_


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> RandomForestClassifier:
    """
    Random Forest with validation-based hyperparameter selection.
    Evaluates all combinations in RF_PARAM_GRID on the held-out
    validation set (AUC-ROC) to select the best configuration.
    """
    best_auc = -1.0
    best_model = None
    best_params = {}

    from itertools import product
    keys = list(RF_PARAM_GRID.keys())
    values = list(RF_PARAM_GRID.values())

    for combo in product(*values):
        params = dict(zip(keys, combo))
        rf = RandomForestClassifier(
            max_features='sqrt',
            n_jobs=-1,
            random_state=RANDOM_SEED,
            **params,
        )
        rf.fit(X_train, y_train)
        val_probs = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_probs)

        if auc > best_auc:
            best_auc = auc
            best_model = rf
            best_params = params

    print(f"  RF  best params={best_params}  val AUC={best_auc:.4f}")
    return best_model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> xgb.Booster:
    """
    XGBoost with grid search over XGB_PARAM_GRID.
    Each combo uses early stopping on validation AUC.
    Returns the booster with the best validation AUC.
    """
    from itertools import product

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)

    best_auc   = -1.0
    best_model = None
    best_p     = {}

    keys   = list(XGB_PARAM_GRID.keys())
    values = list(XGB_PARAM_GRID.values())

    for combo in product(*values):
        p = dict(zip(keys, combo))
        params = {
            'max_depth':        p['max_depth'],
            'eta':              p['eta'],
            'subsample':        p['subsample'],
            'colsample_bytree': XGB_COLSAMPLE,
            'reg_alpha':        XGB_REG_ALPHA,
            'reg_lambda':       XGB_REG_LAMBDA,
            'objective':        'binary:logistic',
            'eval_metric':      'auc',
            'seed':             RANDOM_SEED,
        }
        model = xgb.train(
            params, dtrain,
            num_boost_round=XGB_ROUNDS,
            evals=[(dval, 'val')],
            early_stopping_rounds=XGB_EARLY_STOP,
            verbose_eval=0,
        )
        if model.best_score > best_auc:
            best_auc   = model.best_score
            best_model = model
            best_p     = p

    print(f"  XGB best params={best_p}  val AUC={best_auc:.4f}")
    return best_model

```

### Training-loop key lines (LSTM + baselines)

```text
models/baselines.py:25:    Uses TimeSeriesSplit to respect temporal ordering within the training window.
models/baselines.py:27:    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
models/baselines.py:32:    tscv = TimeSeriesSplit(n_splits=5)
models/baselines.py:50:    best_auc = -1.0
models/baselines.py:70:        if auc > best_auc:
models/baselines.py:71:            best_auc = auc
models/baselines.py:75:    print(f"  RF  best params={best_params}  val AUC={best_auc:.4f}")
models/baselines.py:95:    best_auc   = -1.0
models/baselines.py:112:            'eval_metric':      'auc',
models/baselines.py:118:            evals=[(dval, 'val')],
models/baselines.py:119:            early_stopping_rounds=XGB_EARLY_STOP,
models/baselines.py:122:        if model.best_score > best_auc:
models/baselines.py:123:            best_auc   = model.best_score
models/baselines.py:127:    print(f"  XGB best params={best_p}  val AUC={best_auc:.4f}")
models/lstm_model.py:6:Both models output raw logits (2 classes) — use CrossEntropyLoss in training.
models/lstm_model.py:57:def _build_optimizer(model, name: str, lr: float):
models/lstm_model.py:75:    max_epochs, patience, seed,
models/lstm_model.py:104:        optimizer = _build_optimizer(model, opt_name, lr)
models/lstm_model.py:105:        criterion = nn.CrossEntropyLoss()
models/lstm_model.py:107:        best_val_loss, patience_ctr = float("inf"), 0
models/lstm_model.py:125:            if val_loss < best_val_loss:
models/lstm_model.py:126:                best_val_loss, patience_ctr = val_loss, 0
models/lstm_model.py:128:                patience_ctr += 1
models/lstm_model.py:129:                if patience_ctr >= patience:
models/lstm_model.py:157:    tune_patience=None,
models/lstm_model.py:205:    patience = tune_patience if tune_patience is not None else config.LSTM_TUNE_PATIENCE
models/lstm_model.py:230:            patience=patience,
models/lstm_model.py:268:            patience=patience,
models/lstm_model.py:823:def _train_lstm_impl(
models/lstm_model.py:833:    patience: int,
models/lstm_model.py:878:    best_val_loss = float("inf")
models/lstm_model.py:881:    patience_ctr = 0
models/lstm_model.py:992:            if val_loss < best_val_loss:
models/lstm_model.py:993:                best_val_loss = val_loss
models/lstm_model.py:996:                patience_ctr = 0
models/lstm_model.py:998:                patience_ctr += 1
models/lstm_model.py:999:                if patience_ctr >= patience:
models/lstm_model.py:1000:                    stop_reason = "early_stop_patience"
models/lstm_model.py:1002:                        {"val_loss": f"{best_val_loss:.4f}", "status": "early stop"}
models/lstm_model.py:1007:                {"val_loss": f"{val_loss:.4f}", "best": f"{best_val_loss:.4f}"}
models/lstm_model.py:1019:        f"| best_val_loss={best_val_loss:.4f}"
models/lstm_model.py:1091:        optimizer = _build_optimizer(model, opt_name, learning_rate)
models/lstm_model.py:1092:        criterion = nn.CrossEntropyLoss()
models/lstm_model.py:1094:        return _train_lstm_impl(
models/lstm_model.py:1112:            optimizer = _build_optimizer(model, opt_name, learning_rate)
models/lstm_model.py:1113:            criterion = nn.CrossEntropyLoss()
models/lstm_model.py:1115:            return _train_lstm_impl(
models/lstm_model.py:1135:    patience,
models/lstm_model.py:1140:    """Delegates to _train_lstm_impl with ReduceLROnPlateau."""
models/lstm_model.py:1141:    return _train_lstm_impl(
models/lstm_model.py:1151:        patience,
models/lstm_model.py:1176:    Trains LSTM using Adam with ReduceLROnPlateau scheduler.
models/lstm_model.py:1192:        optimizer = _build_optimizer(model, opt_name, lr_use)
models/lstm_model.py:1193:        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
models/lstm_model.py:1195:            patience=config.LSTM_B_LR_PATIENCE,
models/lstm_model.py:1198:        criterion = nn.CrossEntropyLoss()
main.py:90:ENABLE_LSTM_A_TUNING = False  # Keep False in dev unless explicitly needed.
main.py:251:    lstm_a_tuning_enabled = ENABLE_LSTM_A_TUNING and not lstm_a_dev_mode
main.py:500:            if not DEV_MODE:
main.py:611:                    tune_patience=config.LSTM_B_TUNE_PATIENCE,
main.py:793:    if not DEV_MODE:

```

## SECTION 8 — SIGNAL GENERATION

### backtest/signals.py (verbatim)

```python
"""
backtest/signals.py
Step 7: Convert per-stock daily probabilities to Long / Short / Hold signals.

Ranking logic (Fischer & Krauss 2017 / Krauss et al. 2017):
    - Use an explicit model probability column (e.g., Prob_LR_Smooth)
    - Sort all stocks by score descending each day
  - Top-k → Long   (predicted to outperform the median)
  - Bottom-k → Short (predicted to underperform the median)
  - Remaining → Hold

With SIGNAL_USE_ZSCORE=True, probabilities are z-scored within each day's
cross-section before ranking. This makes signal generation more robust to
probability calibration issues.
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import K_STOCKS, SIGNAL_CONFIDENCE_THRESHOLD, SIGNAL_USE_ZSCORE, MIN_HOLDING_DAYS


def smooth_probabilities(
    preds_df: pd.DataFrame,
    prob_col: str,
    alpha: float = 0.3,
    ema_method: str = 'alpha',
    ema_span: int | None = None,
) -> pd.DataFrame:
    """
    Apply per-ticker exponential smoothing to predicted probabilities.

    Reduces daily position turnover by making signals "stickier".
    Lower alpha = more smoothing (more weight on history).
    Supports pandas EWM smoothing configured either by alpha or span.

    Returns a copy of preds_df with a new column '{prob_col}_Smooth'.
    """
    preds_df = preds_df.sort_values(['Ticker', 'Date']).copy()
    smoothed_col = f'{prob_col}_Smooth'

    if ema_method not in ('alpha', 'span'):
        raise ValueError(f"Unsupported ema_method='{ema_method}'. Expected 'alpha' or 'span'.")
    if ema_method == 'span':
        if ema_span is None or ema_span <= 0:
            raise ValueError('ema_span must be a positive integer when ema_method="span".')
    else:
        # Allow alpha <= 0 as an explicit "no smoothing" mode.
        if alpha is None or alpha <= 0:
            alpha = None
        elif alpha > 1:
            raise ValueError('alpha must satisfy 0 < alpha <= 1 when ema_method="alpha".')

    parts = []
    for ticker, group in preds_df.groupby('Ticker'):
        g = group.sort_values('Date').copy()
        if ema_method == 'span':
            g[smoothed_col] = g[prob_col].ewm(span=ema_span, adjust=False).mean()
        elif alpha is None:
            g[smoothed_col] = g[prob_col]
        else:
            g[smoothed_col] = g[prob_col].ewm(alpha=alpha, adjust=False).mean()
        parts.append(g)

    return pd.concat(parts).sort_values(['Date', 'Ticker']).reset_index(drop=True)


def apply_holding_period_constraint(
    signals_df: pd.DataFrame,
    min_hold_days: int = MIN_HOLDING_DAYS,
) -> pd.DataFrame:
    """
    Apply minimum holding period constraint to reduce turnover.

    Once a stock enters a Long or Short position, it must stay in that
    position for at least min_hold_days before it can exit or flip.

    Parameters
    ----------
    signals_df : pd.DataFrame
        Must contain Date, Ticker, Signal columns. Should be sorted by Date.
    min_hold_days : int
        Minimum number of days to hold before allowing position change.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with Signal column modified to enforce holding period.
    """
    if min_hold_days <= 1:
        return signals_df

    signals_df = signals_df.sort_values(['Ticker', 'Date']).copy()
    dates = sorted(signals_df['Date'].unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}

    # Track entry dates per ticker
    entry_dates = {}  # {ticker: (signal, date_idx)}

    results = []
    for date in dates:
        date_idx = date_to_idx[date]
        day_signals = signals_df[signals_df['Date'] == date].copy()

        for idx, row in day_signals.iterrows():
            ticker = row['Ticker']
            new_signal = row['Signal']

            if ticker in entry_dates:
                old_signal, entry_idx = entry_dates[ticker]
                days_held = date_idx - entry_idx

                # Check if we're trying to change position before min hold
                if new_signal != old_signal and days_held < min_hold_days:
                    # Force keep old signal
                    day_signals.loc[idx, 'Signal'] = old_signal
                else:
                    # Allow change, update entry date
                    if new_signal != old_signal:
                        if new_signal in ('Long', 'Short'):
                            entry_dates[ticker] = (new_signal, date_idx)
                        else:
                            # Exiting to Hold, remove tracking
                            del entry_dates[ticker]
            else:
                # New position entry
                if new_signal in ('Long', 'Short'):
                    entry_dates[ticker] = (new_signal, date_idx)

        results.append(day_signals)

    return pd.concat(results).sort_values(['Date', 'Ticker']).reset_index(drop=True)


def generate_signals(
    predictions_df: pd.DataFrame,
    k: int | None = None,
    prob_col: str = None,
    confidence_threshold: float = SIGNAL_CONFIDENCE_THRESHOLD,
    use_cross_sectional_z: bool = SIGNAL_USE_ZSCORE,
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, int | float | bool | str | None]]:
    """
    Generate Long / Short / Hold signals for each (Date, Ticker) row.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Must contain: Date, Ticker, Return_NextDay, Target, and a single
        probability column specified via `prob_col`.
    k : int
        Number of long and short positions per day (default K_STOCKS).
    prob_col : str or None
        Single column used as ranking signal (e.g. prob_col='Prob_RF').
        Required to prevent accidental ensemble usage.
    confidence_threshold : float
        Minimum distance from 0.5 (or z-score if use_cross_sectional_z=True)
        required to generate a Long or Short signal.
        Set to 0.0 for pure-ranking behavior.
    use_cross_sectional_z : bool
        If True, z-score probabilities within each day's cross-section
        before applying confidence threshold. This makes the threshold
        work on standard deviations rather than raw probability units,
        which is more stable across different probability calibrations.
    return_diagnostics : bool
        If True, return a `(signals_df, diagnostics)` tuple. Otherwise
        return only the signals DataFrame for backward compatibility.

    Returns
    -------
    pd.DataFrame or tuple[pd.DataFrame, dict]
        Input DataFrame with added columns:
                    Prob_ENS : copy of prob_col for ranking diagnostics
          Prob_Z   : cross-sectional z-score (if use_cross_sectional_z=True)
          Signal   : 'Long', 'Short', or 'Hold'
        If `return_diagnostics=True`, also returns summary diagnostics about
        requested versus assigned long/short slots.
    """
    if k is None:
        from config import K_STOCKS as CFG_K_STOCKS
        k = CFG_K_STOCKS
    if prob_col is None:
        raise ValueError('prob_col is required. Implicit ensemble scoring has been removed.')

    results = []
    long_slots_requested = 0
    short_slots_requested = 0
    long_slots_assigned = 0
    short_slots_assigned = 0

    for date, group in predictions_df.sort_values('Date').groupby('Date'):
        g = group.copy()
        n = len(g)

        g['Prob_ENS'] = g[prob_col]

        # ── Cross-sectional z-score ─────────────────────────────────────────
        if use_cross_sectional_z:
            prob_mean = g['Prob_ENS'].mean()
            prob_std = g['Prob_ENS'].std()
            if prob_std > 1e-8:
                g['Prob_Z'] = (g['Prob_ENS'] - prob_mean) / prob_std
            else:
                g['Prob_Z'] = 0.0
            sort_col = 'Prob_Z'
        else:
            g['Prob_Z'] = np.nan
            sort_col = 'Prob_ENS'

        # ── Rank by score (descending); stable sort for ties ────────────────
        g = g.sort_values(sort_col, ascending=False, kind='stable').reset_index(drop=True)

        # ── Assign signals: top-k Long, bottom-k Short, rest Hold ──────────
        g['Signal'] = 'Hold'

        long_k  = min(k, n // 2)
        short_k = min(k, n // 2)
        long_slots_requested += int(long_k)
        short_slots_requested += int(short_k)

        if confidence_threshold > 0.0:
            long_candidates  = g.head(long_k)
            short_candidates = g.tail(short_k)

            if use_cross_sectional_z:
                # Threshold on z-score (e.g., threshold=0.3 means top/bottom must
                # be 0.3 std away from cross-sectional mean)
                long_mask  = long_candidates['Prob_Z'] > confidence_threshold
                short_mask = short_candidates['Prob_Z'] < -confidence_threshold
            else:
                # Original behavior: threshold on raw probability
                long_mask  = long_candidates['Prob_ENS'] > (0.5 + confidence_threshold)
                short_mask = short_candidates['Prob_ENS'] < (0.5 - confidence_threshold)

            g.loc[long_candidates[long_mask].index,   'Signal'] = 'Long'
            g.loc[short_candidates[short_mask].index, 'Signal'] = 'Short'
        else:
            # Pure-ranking behavior (confidence_threshold=0.0)
            g.loc[:long_k - 1, 'Signal'] = 'Long'
            g.loc[n - short_k:, 'Signal'] = 'Short'

        long_slots_assigned += int((g['Signal'] == 'Long').sum())
        short_slots_assigned += int((g['Signal'] == 'Short').sum())
        results.append(g)

    out = pd.concat(results).reset_index(drop=True)
    diagnostics = {
        'prob_col_used': prob_col,
        'confidence_threshold': float(confidence_threshold),
        'use_cross_sectional_z': bool(use_cross_sectional_z),
        'n_dates': int(out['Date'].nunique()) if not out.empty else 0,
        'n_rows': int(len(out)),
        'long_slots_requested': int(long_slots_requested),
        'short_slots_requested': int(short_slots_requested),
        'long_slots_assigned': int(long_slots_assigned),
        'short_slots_assigned': int(short_slots_assigned),
        'long_slots_filtered_by_threshold': int(long_slots_requested - long_slots_assigned),
        'short_slots_filtered_by_threshold': int(short_slots_requested - short_slots_assigned),
    }

    # Sanity summary
    counts = out['Signal'].value_counts()
    print(f"Signal counts — Long: {counts.get('Long', 0)}  "
          f"Short: {counts.get('Short', 0)}  "
          f"Hold: {counts.get('Hold', 0)}")
    if return_diagnostics:
        return out, diagnostics
    return out


def portfolio_half_turns_per_day(signals_df: pd.DataFrame) -> pd.Series:
    """Half-turn counts per date (same counting as portfolio.compute_portfolio_returns)."""
    daily_turns = []
    prev_signals: dict[str, str] = {}
    for date, group in signals_df.sort_values('Date').groupby('Date'):
        curr = dict(zip(group['Ticker'], group['Signal']))
        half_turns = 0
        for t, sig in curr.items():
            prev = prev_signals.get(t, 'Hold')
            if sig == prev:
                continue
            if prev in ('Long', 'Short') and sig in ('Long', 'Short'):
                half_turns += 2
            else:
                half_turns += 1
        prev_signals = curr
        daily_turns.append((date, half_turns))
    idx, vals = zip(*daily_turns) if daily_turns else ([], [])
    return pd.Series(vals, index=idx, name='Turnover')


def compute_turnover_and_holding_stats(
    signals_df: pd.DataFrame,
    k: int = K_STOCKS,
):
    """
    Mean daily half-turn turnover (portfolio definition) and average length
    of uninterrupted Long or Short legs per ticker (trading days).
    """
    turn = portfolio_half_turns_per_day(signals_df)
    mean_turnover = float(turn.mean()) if len(turn) else 0.0

    hold_lengths: list[int] = []
    for _, g in signals_df.sort_values(['Ticker', 'Date']).groupby('Ticker'):
        sigs = g['Signal'].tolist()
        i = 0
        while i < len(sigs):
            s = sigs[i]
            j = i + 1
            while j < len(sigs) and sigs[j] == s:
                j += 1
            if s in ('Long', 'Short'):
                hold_lengths.append(j - i)
            i = j

    avg_hold = float(np.mean(hold_lengths)) if hold_lengths else float('nan')
    return {
        'mean_daily_turnover_half_turns': mean_turnover,
        'avg_holding_period_trading_days': avg_hold,
        'n_long_short_runs': len(hold_lengths),
    }


if __name__ == '__main__':
    # Quick demo with synthetic data to verify signal assignment
    import numpy as np
    import config
    np.random.seed(42)
    dates = pd.date_range('2022-01-03', periods=5, freq='B')
    tickers = config.TICKERS[:min(10, len(config.TICKERS))]
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                'Date': d, 'Ticker': t,
                'Prob_LR':   np.random.rand(),
                'Prob_RF':   np.random.rand(),
                'Prob_XGB':  np.random.rand(),
                'Prob_LSTM_A': np.random.rand(),
                'Prob_LSTM_B': np.random.rand(),
                'Return_NextDay': np.random.randn() * 0.01,
                'Target': np.random.randint(0, 2),
            })
    df = pd.DataFrame(rows)
    signals = generate_signals(df, k=2, prob_col='Prob_LR')
    print('\nSample output (first date):')
    print(signals[signals['Date'] == signals['Date'].iloc[0]]
          [['Date', 'Ticker', 'Prob_ENS', 'Signal']].to_string(index=False))
    assert (signals.groupby('Date')['Signal'].apply(lambda x: (x == 'Long').sum()) == 2).all()
    assert (signals.groupby('Date')['Signal'].apply(lambda x: (x == 'Short').sum()) == 2).all()
    print('\nAll signal counts verified (2 long, 2 short per day).')

```

## SECTION 9 — PORTFOLIO AND TRANSACTION COST

### backtest/portfolio.py (verbatim)

```python
"""
backtest/portfolio.py
Step 8: Daily equal-weighted long-short P&L with transaction costs.

Execution timing (aligned with pipeline/targets):
  - Row at date t uses features known through close t.
  - Signals are formed at t from those features.
  - Return_NextDay is the close-to-close return from t to t+1; the portfolio
    earns long minus short averages of that return.

Transaction cost model (Fischer & Krauss 2017):
  - tc_bps basis points charged per half-turn
  - Optional slippage_bps added per half-turn (same structural formula)
  - Each position change affects 1/(2*k) of the total portfolio
    (2 legs × k positions per leg = 2k active positions)
  - Day 1: all positions are new → full turnover cost on all active positions
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import TC_BPS, K_STOCKS


def compute_portfolio_returns(
    signals_df: pd.DataFrame,
    tc_bps: float = TC_BPS,
    k: int = K_STOCKS,
    slippage_bps: float = 0.0,
) -> pd.DataFrame:
    """
    Compute daily gross and net portfolio returns for a long-short strategy.

    Portfolio construction per day:
      long_ret  = equal-weighted mean return of Long positions
      short_ret = equal-weighted mean return of Short positions
      gross_ret = long_ret - short_ret
      net_ret   = gross_ret - turnover * tc / (2 * k)

    Each position change only affects 1/(2*k) of the portfolio, since
    we have 2 legs (long + short) with k equal-weight positions each.

    Parameters
    ----------
    signals_df : pd.DataFrame
        Output of backtest.signals.generate_signals() — must contain:
        Date, Ticker, Signal ('Long'/'Short'/'Hold'), Return_NextDay.
    tc_bps : float
        Transaction cost per half-turn in basis points (default 5 bps = 0.0005).
    k : int
        Number of long (and short) positions per day.
    slippage_bps : float
        Additional basis points per half-turn (e.g. bid-ask; 0 = disabled).

    Returns
    -------
    pd.DataFrame indexed by Date with columns:
        Gross_Return, Net_Return, Long_Return, Short_Return,
        TC, Slippage_Cost, Turnover (number of half-turn position changes that day)
    """
    tc = tc_bps / 10_000
    slip = slippage_bps / 10_000
    daily = []
    prev_signals: dict[str, str] = {}   # ticker → last signal

    for date, group in signals_df.sort_values('Date').groupby('Date'):
        longs  = group[group['Signal'] == 'Long']
        shorts = group[group['Signal'] == 'Short']

        long_ret  = longs['Return_NextDay'].mean()  if len(longs)  > 0 else 0.0
        short_ret = shorts['Return_NextDay'].mean() if len(shorts) > 0 else 0.0
        gross_ret = long_ret - short_ret

        # Count half-turns vs previous day
        # Long↔Short flip = 2 half-turns (close + open), all others = 1
        curr = dict(zip(group['Ticker'], group['Signal']))
        half_turns = 0
        for t, sig in curr.items():
            prev = prev_signals.get(t, 'Hold')
            if sig == prev:
                continue
            if prev in ('Long', 'Short') and sig in ('Long', 'Short'):
                half_turns += 2  # close one side + open the other
            else:
                half_turns += 1
        turnover = half_turns

        denom = (2 * k)
        tc_cost = turnover * tc / denom
        slip_cost = turnover * slip / denom
        frict = tc_cost + slip_cost
        net_ret = gross_ret - frict
        prev_signals = curr

        daily.append({
            'Date':          date,
            'Gross_Return':  gross_ret,
            'Net_Return':    net_ret,
            'Long_Return':   long_ret,
            'Short_Return':  short_ret,
            'TC':            tc_cost,
            'Slippage_Cost': slip_cost,
            'Turnover':      turnover,
        })

    return pd.DataFrame(daily).set_index('Date')


if __name__ == '__main__':
    # Smoke test with synthetic signals
    import numpy as np
    import config
    from backtest.signals import generate_signals

    np.random.seed(42)
    dates   = pd.date_range('2022-01-03', periods=10, freq='B')
    tickers = config.TICKERS[:min(10, len(config.TICKERS))]

    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                'Date': d, 'Ticker': t,
                'Prob_LR':   np.random.rand(),
                'Prob_RF':   np.random.rand(),
                'Prob_XGB':  np.random.rand(),
                'Prob_LSTM_A': np.random.rand(),
                'Prob_LSTM_B': np.random.rand(),
                'Return_NextDay': np.random.randn() * 0.01,
                'Target': np.random.randint(0, 2),
            })
    pred_df  = pd.DataFrame(rows)
    sig_df   = generate_signals(pred_df, k=2, prob_col='Prob_LR')
    port     = compute_portfolio_returns(sig_df, tc_bps=5, k=2)

    print('\nPortfolio daily returns (10 days):')
    print(port.round(6).to_string())
    print(f'\nMean gross return : {port["Gross_Return"].mean()*100:.4f}%')
    print(f'Mean net return   : {port["Net_Return"].mean()*100:.4f}%')
    print(f'Mean TC per day   : {port["TC"].mean()*100:.4f}%')
    print(f'Mean turnover/day : {port["Turnover"].mean():.1f} position changes')

    assert len(port) == len(dates), 'Wrong number of rows'
    assert (port['Net_Return'] <= port['Gross_Return']).all(), 'Net > Gross — TC bug'
    print('\nAll portfolio checks passed.')

```

## SECTION 10 — METRICS AND EVALUATION

### backtest/metrics.py (verbatim)

```python
"""
backtest/metrics.py
Step 9: Portfolio performance and classification metrics.

Covers all thesis table requirements:
  T4/T5 — compute_metrics()           (Sharpe, Sortino, MDD, Calmar, etc.)
  T6     — compute_subperiod_metrics() (per market-regime breakdown)
  T7     — compute_tc_sensitivity()    (Sharpe vs transaction cost grid)
  T8     — evaluate_classification()   (AUC, Accuracy, F1)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ~3.8% annual risk-free rate / 252 trading days
RF_DAILY = 0.00015

# Sub-period definitions for T6 / F9 (CLAUDE.md Section "Sub-Period Analysis")
SUBPERIODS = {
    'Pre-COVID':        ('2019-01-01', '2020-02-19'),
    'COVID crash':      ('2020-02-20', '2020-04-30'),
    'Recovery/bull':    ('2020-05-01', '2021-12-31'),
    '2022 bear':        ('2022-01-01', '2022-12-31'),
    '2023-24 AI rally': ('2023-01-01', '2024-12-31'),
}


def compute_metrics(
    returns_series: pd.Series,
    rf_daily: float = RF_DAILY,
) -> dict:
    """
    Annualised risk-return metrics for a daily return series.

    Parameters
    ----------
    returns_series : pd.Series
        Daily net (or gross) portfolio returns, indexed by date.
    rf_daily : float
        Daily risk-free rate (default ≈ 3.8% p.a. / 252).

    Returns
    -------
    dict with keys matching thesis Table T5 columns.
    """
    r = returns_series.dropna()
    if len(r) == 0:
        return {}

    mean_d = r.mean()
    std_d  = r.std()
    excess = r - rf_daily

    sharpe = (
        (excess.mean() / excess.std()) * np.sqrt(252)
        if excess.std() > 0 else 0.0
    )

    downside = r[r < rf_daily]
    if len(downside) > 1:
        downside_std = downside.std()
        if downside_std > 0:
            sortino = ((mean_d - rf_daily) / downside_std) * np.sqrt(252)
        else:
            # All downside returns are identical (zero variance)
            sortino = np.inf if (mean_d - rf_daily) > 0 else 0.0
    else:
        # No downside periods (all returns >= rf_daily) = infinite risk-adjusted return
        sortino = np.inf if (mean_d - rf_daily) > 0 else 0.0

    cum     = (1 + r).cumprod()
    max_dd  = ((cum - cum.cummax()) / cum.cummax()).min()
    ann_ret = (1 + mean_d) ** 252 - 1

    return {
        'N Days':                 int(len(r)),
        'Mean Daily Return (%)':  round(mean_d * 100, 4),
        'Annualized Return (%)':  round(ann_ret * 100, 2),
        'Annualized Std Dev (%)': round(std_d * np.sqrt(252) * 100, 2),
        'Sharpe Ratio':           round(sharpe, 3),
        'Sortino Ratio':          round(sortino, 3),
        'Max Drawdown (%)':       round(max_dd * 100, 2),
        'Calmar Ratio':           round(ann_ret / abs(max_dd), 3) if max_dd != 0 else 0.0,
        'Win Rate (%)':           round((r > 0).mean() * 100, 2),
        'VaR 1% (%)':             round(np.percentile(r, 1) * 100, 4),
    }


def evaluate_classification(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Binary classification metrics for thesis Table T8.

    Parameters
    ----------
    y_true    : array-like of 0/1 ground-truth labels
    y_prob    : array-like of predicted probabilities for class 1
    threshold : decision boundary (default 0.5)
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        'Accuracy (%)': round(accuracy_score(y_true, y_pred) * 100, 2),
        'AUC-ROC':      round(roc_auc_score(y_true, y_prob), 4),
        'F1 Score':     round(f1_score(y_true, y_pred), 4),
    }


def compute_daily_auc(
    predictions_df: pd.DataFrame,
    prob_col: str,
    target_col: str = 'Target',
) -> dict:
    """
    Compute average AUC-ROC per day (cross-sectional ranking quality).

    This metric is more appropriate for ranking strategies than pooled AUC
    because it measures within-day ranking ability, which is what the
    signal generation actually uses.

    NOTE: Pooled AUC (from evaluate_classification) may differ from daily AUC
    because pooled AUC measures global ranking while daily AUC measures
    within-day ranking. A model can have good within-day ranking (high Sharpe)
    but poor global ranking (low pooled AUC) if its probability calibration
    varies across days.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Must contain Date, prob_col, and target_col columns.
    prob_col : str
        Column name for predicted probabilities.
    target_col : str
        Column name for ground truth labels.

    Returns
    -------
    dict with:
        - 'Daily AUC (mean)': average daily AUC
        - 'Daily AUC (std)': standard deviation of daily AUC
        - 'Days with valid AUC': number of days with computable AUC
    """
    daily_aucs = []

    for date, group in predictions_df.groupby('Date'):
        y_true = group[target_col].values
        y_prob = group[prob_col].values

        # AUC requires both classes present
        if len(np.unique(y_true)) < 2:
            continue

        try:
            auc = roc_auc_score(y_true, y_prob)
            daily_aucs.append(auc)
        except ValueError:
            continue

    if len(daily_aucs) == 0:
        return {
            'Daily AUC (mean)': np.nan,
            'Daily AUC (std)': np.nan,
            'Days with valid AUC': 0,
        }

    return {
        'Daily AUC (mean)': round(np.mean(daily_aucs), 4),
        'Daily AUC (std)': round(np.std(daily_aucs), 4),
        'Days with valid AUC': len(daily_aucs),
    }


def compute_subperiod_metrics(
    returns_series: pd.Series,
    rf_daily: float = RF_DAILY,
    subperiods: dict = None,
) -> pd.DataFrame:
    """
    Compute compute_metrics() for each market-regime sub-period (thesis T6 / F9).

    Parameters
    ----------
    returns_series : pd.Series indexed by date (datetime or string).
    subperiods     : dict mapping label → (start_str, end_str).
                     Defaults to the five regimes defined in CLAUDE.md.

    Returns
    -------
    pd.DataFrame with one row per sub-period and metric columns.
    """
    if subperiods is None:
        subperiods = SUBPERIODS

    idx = pd.to_datetime(returns_series.index)
    rows = {}
    for label, (start, end) in subperiods.items():
        mask = (idx >= start) & (idx <= end)
        sub  = returns_series[mask]
        rows[label] = compute_metrics(sub, rf_daily) if len(sub) > 0 else {}

    return pd.DataFrame(rows).T


def compute_tc_sensitivity(
    signals_df: pd.DataFrame,
    k: int = 2,
    tc_grid: list = None,
) -> pd.DataFrame:
    """
    Compute Sharpe ratio and annualised return for a range of TC values (thesis T7 / F7).

    Parameters
    ----------
    signals_df : output of generate_signals() — must contain Date, Ticker,
                 Signal, Return_NextDay.
    k          : number of long (and short) positions per day.
    tc_grid    : list of TC values in basis points to evaluate
                 (default: 0, 2, 5, 10, 15, 20, 25, 30).

    Returns
    -------
    pd.DataFrame indexed by TC (bps) with Sharpe and Annualized Return columns.
    """
    from backtest.portfolio import compute_portfolio_returns

    if tc_grid is None:
        tc_grid = [0, 2, 5, 10, 15, 20, 25, 30]

    rows = {}
    for tc in tc_grid:
        port = compute_portfolio_returns(signals_df, tc_bps=tc, k=k)
        m    = compute_metrics(port['Net_Return'])
        rows[tc] = {
            'Sharpe Ratio':          m.get('Sharpe Ratio', np.nan),
            'Annualized Return (%)':  m.get('Annualized Return (%)', np.nan),
            'Max Drawdown (%)':       m.get('Max Drawdown (%)', np.nan),
        }

    df = pd.DataFrame(rows).T
    df.index.name = 'TC (bps)'
    return df

```

## SECTION 11 — MAIN PIPELINE ORCHESTRATION

### main.py (verbatim)

```python
"""
main.py — Full pipeline orchestrator
Walk-forward validated long-short strategy using LR, RF, XGBoost, LSTM-A, LSTM.

Models after refactor:
  - LR:      Logistic Regression (baseline) — 6 features
  - RF:      Random Forest (baseline) — 6 features
  - XGBoost: XGBoost (baseline) — 6 features
  - LSTM-A:  Bhandari-inspired (4 technical features, tuned architecture)
  - LSTM:  Extended ablation (6 features, fixed architecture)

Implements Bhandari et al. (2022) extensions:
  - Dual scalers: separate feature normalization for LSTM-A (4 features) and
    LSTM/baselines (6 features)
  - Optional LSTM hyperparameter tuning (Phase 1 + Phase 2 for LSTM-A)
  - Configurable scaler type (standard/minmax)

Run:
    .venv/bin/python3 main.py
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'  # Prevent XGBoost/PyTorch dual-OpenMP segfault

import time
import random
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger('pipeline')

from pipeline.data_loader import download_and_save
from pipeline.features import (
    build_feature_matrix, FEATURE_COLS,
    compute_wavelet_thresholds, apply_wavelet_denoising, apply_wavelet_denoising_causal,
    recompute_features_from_denoised,
)
from pipeline.targets import create_targets
from pipeline.walk_forward import generate_walk_forward_folds, print_fold_summary
from pipeline.standardizer import standardize_fold, winsorize_fold
from pipeline.fold_reporting import save_fold_report
from evaluation.metrics_utils import binary_auc_safe, classification_sanity_checks, log_split_balance
from models.baselines import train_logistic, train_random_forest, train_xgboost
from models.lstm_model import (
    prepare_lstm_a_sequences, prepare_lstm_b_sequences,
    prepare_lstm_a_sequences_temporal_split, prepare_lstm_b_sequences_temporal_split,
    train_lstm_a, train_lstm_b,
    predict_lstm, align_predictions_to_df,
    tune_lstm_hyperparams,
)
from backtest.signals import (
    generate_signals,
    smooth_probabilities,
    apply_holding_period_constraint,
    compute_turnover_and_holding_stats,
)
from backtest.portfolio import compute_portfolio_returns
from backtest.metrics import (
    compute_metrics, evaluate_classification,
    compute_subperiod_metrics, compute_daily_auc,
)
import config
from config import (
    TRAIN_DAYS, VAL_DAYS, TEST_DAYS,
    K_STOCKS, TC_BPS, MODELS,
    LSTM_A_VAL_SPLIT, LSTM_B_VAL_SPLIT,
    LSTM_A_FEATURES, LSTM_B_FEATURES,
    BASELINE_FEATURE_COLS,
    SIGNAL_SMOOTH_ALPHA, MIN_HOLDING_DAYS,
    SLIPPAGE_BPS,
    WINSORIZE_ENABLED, WINSORIZE_LOWER_Q, WINSORIZE_UPPER_Q,
    TRAIN_WINDOW_MODE,
    RUN_SIGNAL_ABLATION,
    SAVE_FOLD_REPORTS,
    SIGNAL_EMA_METHOD, SIGNAL_EMA_SPAN,
    DEV_MODE,
)

TARGET_COL = 'Target'
CACHE_FEATURES_PATH = f'data/processed/features_{config.UNIVERSE_MODE}.csv'

# ── Pipeline options ─────────────────────────────────────────────────────────
ENABLE_LSTM_A_TUNING = False  # Keep False in dev unless explicitly needed.

RUN_BASELINES = True        # Set False to skip LR, RF, XGB
RUN_LSTMS = True            # Set False to skip LSTM-A, LSTM

device = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('mps') if torch.backends.mps.is_available()
    else torch.device('cpu')
)
print(f'Using device: {device}')


def set_global_seed(seed: int):
    """Set deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Keep deterministic mode best-effort to avoid runtime failures on unsupported ops.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# ────────────────────────────────────────────────────────────────────────────
# Results saving functions
# ────────────────────────────────────────────────────────────────────────────

def save_all_results(
    results_dict: dict,
    daily_returns_dict: dict,
    signals_dict: pd.DataFrame,
    tuning_results: list = None,
    reports_dir: str = 'reports'
):
    """
    Saves all backtest results to the /reports folder.

    Args:
        results_dict: {
            'gross':   list of metric dicts (one per model),
            'net_5':   list of metric dicts,
            'classification': list of classification metric dicts,
            'subperiod': DataFrame of sub-period metrics,
        }
        daily_returns_dict: {
            'gross':  pd.DataFrame columns=[Date, LR, RF, XGBoost, LSTM-A, LSTM],
            'net_5':  pd.DataFrame same columns,
        }
        signals_dict: pd.DataFrame with all signals
        tuning_results: list of dicts with tuning results per fold (optional)
        reports_dir: output directory path
    """
    os.makedirs(reports_dir, exist_ok=True)
    prefix = config.UNIVERSE_MODE

    # Table T5: Risk-Return Metrics
    pd.DataFrame(results_dict['gross']).to_csv(
        f'{reports_dir}/{prefix}_table_T5_gross_returns.csv', index=False
    )
    pd.DataFrame(results_dict['net_5']).to_csv(
        f'{reports_dir}/{prefix}_table_T5_net_returns_5bps.csv', index=False
    )

    # Table T8: Classification Metrics
    pd.DataFrame(results_dict['classification']).to_csv(
        f'{reports_dir}/{prefix}_table_T8_classification_metrics.csv', index=False
    )

    # Table T6: Sub-Period Performance
    if results_dict['subperiod'] is not None:
        results_dict['subperiod'].to_csv(
            f'{reports_dir}/{prefix}_table_T6_subperiod_performance.csv', index=False
        )

    # LSTM Tuning Results (Bhandari §3.3 Tables)
    if tuning_results and len(tuning_results) > 0:
        pd.DataFrame(tuning_results).to_csv(
            f'{reports_dir}/{prefix}_lstm_tuning_results.csv', index=False
        )

    # Raw daily returns
    daily_returns_dict['gross'].to_csv(
        f'{reports_dir}/{prefix}_daily_returns_gross.csv', index=False
    )
    daily_returns_dict['net_5'].to_csv(
        f'{reports_dir}/{prefix}_daily_returns_net_5bps.csv', index=False
    )

    # Signals
    signals_dict.to_csv(f'{reports_dir}/{prefix}_signals_all_models.csv', index=False)

    # Human-readable summary
    with open(f'{reports_dir}/{prefix}_backtest_summary.txt', 'w') as f:
        f.write(_format_summary(results_dict))

    print(f"\nAll results saved to /{reports_dir}/")


def _format_summary(results_dict: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("BACKTEST RESULTS SUMMARY")
    lines.append("=" * 60)

    for label, key in [
        ("GROSS RETURNS (0 bps TC)", 'gross'),
        ("NET RETURNS  (5 bps TC)", 'net_5'),
    ]:
        lines.append(f"\n{'─' * 60}")
        lines.append(label)
        lines.append(f"{'─' * 60}")
        for row in results_dict[key]:
            lines.append(
                f"  {row['Model']:<12}  "
                f"Sharpe={row['Sharpe Ratio']:>6.3f}  "
                f"Sortino={row['Sortino Ratio']:>6.3f}  "
                f"Ann.Ret={row['Annualized Return (%)']:>6.2f}%  "
                f"MDD={row['Max Drawdown (%)']:>6.2f}%"
            )

    lines.append("\n" + "=" * 60)
    lines.append("CLASSIFICATION METRICS")
    lines.append("=" * 60)
    for row in results_dict['classification']:
        lines.append(
            f"  {row['Model']:<12}  "
            f"Acc={row['Accuracy (%)']:>5.2f}%  "
            f"AUC={row['AUC-ROC']:.4f}  "
            f"F1={row['F1 Score']:.4f}"
        )

    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ────────────────────────────────────────────────────────────────────────────

def run_walk_forward_pipeline(
    load_cached: bool = True,
    train_days: int | None = None,
    reports_dir: str = 'reports',
):
    """
    Parameters
    ----------
    load_cached : bool
        If True (default), load data from the universe-specific cache
        data/processed/features_{UNIVERSE_MODE}.csv instead
        of re-downloading and recomputing. Set False to run from scratch.
    train_days : int, optional
        Override config TRAIN_DAYS for walk-forward train window length.
    reports_dir : str
        Directory for tables and fold reports.
    """
    lstm_a_dev_mode = getattr(config, 'LSTM_A_DEV_MODE', False)
    lstm_a_tuning_enabled = ENABLE_LSTM_A_TUNING and not lstm_a_dev_mode
    lstm_b_tuning_enabled = bool(getattr(config, 'LSTM_B_ENABLE_TUNING', False))
    lstm_b_tune_once = bool(getattr(config, 'LSTM_B_TUNE_ON_FIRST_FOLD_ONLY', True))

    print("=" * 60)
    print("EXPERIMENT CONFIGURATION")
    print(f"  Universe mode : {config.UNIVERSE_MODE}")
    print(f"  Tickers       : {config.N_STOCKS} stocks")
    print(f"  Date range    : {config.START_DATE} -> {config.END_DATE}")
    print(f"  Windows       : {config.TRAIN_DAYS}/{config.VAL_DAYS}/{config.TEST_DAYS} days")
    print(f"  Sequence len  : {config.SEQ_LEN} days")
    print(f"  K (long/short): {config.K_STOCKS} stocks per side")
    print("=" * 60)
    print(f"BACKTEST — {len(MODELS)} MODELS × N FOLDS")
    print(f"LSTM-A Dev Mode: {'ENABLED' if lstm_a_dev_mode else 'DISABLED'}")
    print(f"LSTM-A Tuning: {'ENABLED' if lstm_a_tuning_enabled else 'DISABLED'}")
    print(f"LSTM Tuning: {'ENABLED' if lstm_b_tuning_enabled else 'DISABLED'}")
    print(f"Scaler Type: {config.SCALER_TYPE}")
    print(f"Configured tickers: {len(config.TICKERS)}")
    print("=" * 60)

    # Ensure run-to-run reproducibility for all stochastic components.
    set_global_seed(config.RANDOM_SEED)

    # ── Steps 1-3: Data ─────────────────────────────────────────────────────
    if load_cached:
        print(f'\nLoading cached {os.path.basename(CACHE_FEATURES_PATH)} ...')
        data = pd.read_csv(CACHE_FEATURES_PATH, parse_dates=['Date'])
        print(f'Loaded {len(data)} rows, {len(data.columns)} columns.')

        # Backward compatibility: older caches may contain features but not targets.
        # If enough columns exist, rebuild targets from cached Return_1d and persist.
        missing_target_cols = [c for c in ['Return_NextDay', TARGET_COL] if c not in data.columns]
        if missing_target_cols:
            print(f"Cached file missing target columns: {missing_target_cols}")
            if {'Date', 'Ticker', 'Return_1d'}.issubset(data.columns):
                print('Recomputing Return_NextDay/Target from cached features...')
                data = create_targets(data)
                data.to_csv(CACHE_FEATURES_PATH, index=False)
                print(f'Repaired and saved cache to {CACHE_FEATURES_PATH}')
            else:
                raise ValueError(
                    f'Cached {os.path.basename(CACHE_FEATURES_PATH)} is missing target columns and lacks the columns '
                    "required to rebuild them ('Date', 'Ticker', 'Return_1d'). "
                    'Run once with load_cached=False to regenerate cache.'
                )
    else:
        raw = download_and_save()
        data = build_feature_matrix(raw)
        data = create_targets(data)
        # Persist full dataset (features + Return_NextDay + Target) for future cached runs.
        data.to_csv(CACHE_FEATURES_PATH, index=False)
        print(f'Saved full cache (with targets) to {CACHE_FEATURES_PATH}')

    # Verify required columns
    required = FEATURE_COLS + [TARGET_COL, 'Return_NextDay', 'Date', 'Ticker']
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f'Missing columns in {os.path.basename(CACHE_FEATURES_PATH)}: {missing}')

    # Verify LSTM-A features are available (4 features)
    lstm_a_missing = [c for c in LSTM_A_FEATURES if c not in data.columns]
    if lstm_a_missing:
        raise ValueError(f'Missing LSTM-A features: {lstm_a_missing}')

    # Verify LSTM features are available (6 features)
    lstm_b_missing = [c for c in LSTM_B_FEATURES if c not in data.columns]
    if lstm_b_missing:
        raise ValueError(f'Missing LSTM features: {lstm_b_missing}')

    print(f'\nFeature sets:')
    print(f'  LSTM-A: {LSTM_A_FEATURES} ({len(LSTM_A_FEATURES)} features)')
    print(f'  LSTM: {LSTM_B_FEATURES} ({len(LSTM_B_FEATURES)} features)')
    print(f'  Baselines (LR/RF/XGB): {BASELINE_FEATURE_COLS} ({len(BASELINE_FEATURE_COLS)} features)')

    # Step 4: Walk-forward folds
    dates = sorted(data['Date'].unique())
    td = train_days if train_days is not None else TRAIN_DAYS
    wf_stride = getattr(config, 'WALK_FORWARD_STRIDE', None)
    folds = generate_walk_forward_folds(
        dates, td, VAL_DAYS, getattr(config, 'TEST_DAYS', TEST_DAYS),
        stride_days=wf_stride,
        train_window_mode=TRAIN_WINDOW_MODE,
    )
    max_folds = getattr(config, 'MAX_FOLDS', None)
    if max_folds is not None:
        folds = folds[:max_folds]
        print(f'Limiting walk-forward run to first {len(folds)} fold(s) (MAX_FOLDS={max_folds}).')
    print()
    print_fold_summary(folds)

    ablation_rows: list[dict] = []

    def _score_lstm_b_candidate_on_val(preds_val: pd.DataFrame) -> tuple[float, float]:
        """
        Score a candidate on validation TRADING performance (not only AUC).
        Returns (val_sharpe_net, val_annual_return_net_pct).
        """
        valid = preds_val.dropna(subset=['Prob_LSTM_B']).copy()
        if len(valid) == 0:
            return float('-inf'), float('-inf')

        smoothed = smooth_probabilities(
            valid,
            'Prob_LSTM_B',
            alpha=SIGNAL_SMOOTH_ALPHA,
            ema_method=SIGNAL_EMA_METHOD,
            ema_span=SIGNAL_EMA_SPAN,
        )
        sig_df, _ = generate_signals(
            smoothed,
            k=K_STOCKS,
            prob_col='Prob_LSTM_B_Smooth',
            return_diagnostics=True,
        )
        sig_df = apply_holding_period_constraint(sig_df, min_hold_days=MIN_HOLDING_DAYS)
        port_val = compute_portfolio_returns(
            sig_df,
            tc_bps=TC_BPS,
            k=K_STOCKS,
            slippage_bps=SLIPPAGE_BPS,
        )
        met_val = compute_metrics(port_val['Net_Return'])
        return float(met_val['Sharpe Ratio']), float(met_val['Annualized Return (%)'])

    # ── Steps 5-6: Train models per fold ────────────────────────────────────
    all_preds = []
    tuning_results = []  # Store tuning results for thesis reporting
    best_hp_b_global = None  # dict for tuned params or string 'DEFAULT'

    for fold in tqdm(folds, desc="Walk-Forward Folds", unit="fold"):
        print(f'\n{"=" * 60}')
        print(f'=== Fold {fold["fold"]}/{len(folds)} ===')
        print('=' * 60)

        # Use deterministic fold-specific seeds so reruns are stable while
        # keeping each fold/model independent.
        fold_seed_base = config.RANDOM_SEED + (fold['fold'] * 1000)
        set_global_seed(fold_seed_base)

        df_tr = data[data['Date'].isin(dates[fold['train'][0]:fold['train'][1]])]
        df_v = data[data['Date'].isin(dates[fold['val'][0]:fold['val'][1]])]
        df_ts = data[data['Date'].isin(dates[fold['test'][0]:fold['test'][1]])]

        print(f'  Train: {len(df_tr):>6} rows | '
              f'Val: {len(df_v):>5} rows | '
              f'Test: {len(df_ts):>5} rows')

        # ── Per-fold wavelet denoising (CAUSAL - NO LEAKAGE) ─────────────────────
        # 1. Threshold computed from training data only
        # 2. CAUSAL denoising: each value uses only historical data (rolling window)
        # 3. Apply to each split INDEPENDENTLY (no concatenation)
        # 4. CRITICAL: Do NOT recompute Target - it must remain based on raw returns
        if config.USE_WAVELET_DENOISING:
            print('  [Wavelet] Computing thresholds from training data...')
            wavelet_thresholds = compute_wavelet_thresholds(df_tr)

            # Apply CAUSAL denoising to each split INDEPENDENTLY
            # This prevents future data from leaking into training
            print('  [Wavelet] Applying causal denoising per split...')
            df_tr = apply_wavelet_denoising_causal(df_tr, thresholds=wavelet_thresholds)
            df_v = apply_wavelet_denoising_causal(df_v, thresholds=wavelet_thresholds)
            df_ts = apply_wavelet_denoising_causal(df_ts, thresholds=wavelet_thresholds)

            # Recompute Close-dependent FEATURES only (RSI, MACD, BB, etc.)
            # CRITICAL: Do NOT recompute Return_NextDay or Target - they must
            # remain based on RAW realized returns for honest evaluation
            print('  [Wavelet] Recomputing features from denoised Close...')

            # Save original targets before feature recomputation
            tr_target = df_tr['Target'].copy()
            tr_return_next = df_tr['Return_NextDay'].copy()
            v_target = df_v['Target'].copy()
            v_return_next = df_v['Return_NextDay'].copy()
            ts_target = df_ts['Target'].copy()
            ts_return_next = df_ts['Return_NextDay'].copy()

            # Recompute features from denoised Close
            df_tr = recompute_features_from_denoised(df_tr)
            df_v = recompute_features_from_denoised(df_v)
            df_ts = recompute_features_from_denoised(df_ts)

            # Restore original targets (based on raw returns, not denoised)
            df_tr['Target'] = tr_target.values
            df_tr['Return_NextDay'] = tr_return_next.values
            df_v['Target'] = v_target.values
            df_v['Return_NextDay'] = v_return_next.values
            df_ts['Target'] = ts_target.values
            df_ts['Return_NextDay'] = ts_return_next.values

            # Drop rows with NaN from warm-up period after recomputation
            df_tr = df_tr.dropna(subset=FEATURE_COLS).reset_index(drop=True)
            df_v = df_v.dropna(subset=FEATURE_COLS).reset_index(drop=True)
            df_ts = df_ts.dropna(subset=FEATURE_COLS).reset_index(drop=True)

            print(f'  [Wavelet] After denoising: Train={len(df_tr)}, '
                  f'Val={len(df_v)}, Test={len(df_ts)}')

        y_tr = df_tr[TARGET_COL].values.astype(int)
        y_v = df_v[TARGET_COL].values.astype(int)
        y_ts = df_ts[TARGET_COL].values.astype(int)

        log_split_balance(y_tr, f'fold{fold["fold"]} train', logger)
        log_split_balance(y_v, f'fold{fold["fold"]} val', logger)
        log_split_balance(y_ts, f'fold{fold["fold"]} test', logger)

        Xb_tr = df_tr[BASELINE_FEATURE_COLS].values
        Xb_v = df_v[BASELINE_FEATURE_COLS].values
        Xb_ts = df_ts[BASELINE_FEATURE_COLS].values
        if WINSORIZE_ENABLED:
            Xb_tr, Xb_v, Xb_ts = winsorize_fold(
                Xb_tr, Xb_v, Xb_ts,
                lower_q=WINSORIZE_LOWER_Q, upper_q=WINSORIZE_UPPER_Q,
            )

        X_tr_b_s, X_v_b_s, X_ts_b_s, _ = standardize_fold(Xb_tr, Xb_v, Xb_ts)

        # Baseline Models
        if RUN_BASELINES:
            t0 = time.time()
            print('\n  [LR]      fitting...')
            lr_m = train_logistic(X_tr_b_s, y_tr)
            print(f'  [LR]      fit done in {time.time()-t0:.1f}s')

            t0 = time.time()
            print('  [RF]      fitting...')
            rf_m = train_random_forest(X_tr_b_s, y_tr, X_v_b_s, y_v)
            print(f'  [RF]      fit done in {time.time()-t0:.1f}s')

            t0 = time.time()
            print('  [XGBoost] fitting...')
            xgb_m = train_xgboost(X_tr_b_s, y_tr, X_v_b_s, y_v)
            print(f'  [XGBoost] fit done in {time.time()-t0:.1f}s')
        else:
            print('\n  [Baselines] Skipping LR, RF, XGBoost (RUN_BASELINES=False)')
            lr_m = rf_m = xgb_m = None

        # ── LSTM-A: Bhandari-inspired (4 technical features) ─────────────────
        probs_a = None
        keys_te_a = None
        probs_b = None
        keys_te_b = None
        
        if RUN_LSTMS:
            # Prepare shared data for LSTM models
            df_train_fold = pd.concat([df_tr, df_v]).sort_values(['Ticker', 'Date'])
            df_test_fold = df_ts.copy()

            # ── LSTM-A: Bhandari-inspired (4 technical features) ─────────────────
            if not DEV_MODE:
                t0 = time.time()
                print('  [LSTM-A]  building sequences & training...')

                # Use TEMPORAL split (splits by date, not index) - FIX for ticker-based split bug
                X_tr_a, y_tr_a, X_val_a, y_val_a, X_te_a, y_te_a, keys_tr_a, keys_val_a, keys_te_a = \
                    prepare_lstm_a_sequences_temporal_split(df_train_fold, df_test_fold, val_ratio=LSTM_A_VAL_SPLIT)

                print(f'    LSTM-A sequences: train={len(X_tr_a)}, val={len(X_val_a)}, test={len(X_te_a)}')

                # Optional: hyperparameter tuning for LSTM-A (using temporal val split)
                best_hp_a = None
                if lstm_a_dev_mode:
                    print(
                        '    [LSTM-A Dev Mode] Skipping Phase 1 + Phase 2 tuning; '
                        'using fixed config defaults.'
                    )
                    print(
                        f'    [LSTM-A Dev Mode] optimizer={config.LSTM_A_OPTIMIZER} '
                        f'lr={config.LSTM_A_LR} batch={config.LSTM_A_BATCH} '
                        f'hidden={config.LSTM_B_HIDDEN_SIZE} layers={config.LSTM_B_NUM_LAYERS} '
                        f'dropout={config.LSTM_B_DROPOUT}'
                    )
                    model_a = train_lstm_a(
                        X_tr_a, y_tr_a,
                        X_val_a, y_val_a,
                        device,
                        optimizer_name=config.LSTM_A_OPTIMIZER,
                        lr=config.LSTM_A_LR,
                        hidden_size=config.LSTM_B_HIDDEN_SIZE,
                        num_layers=config.LSTM_B_NUM_LAYERS,
                        dropout=config.LSTM_B_DROPOUT,
                        batch_size=config.LSTM_A_BATCH,
                        seed=fold_seed_base + 20,
                        fold_idx=fold['fold'],
                    )
                elif lstm_a_tuning_enabled:
                    print('    [LSTM-A Tuning] Running Phase 1 + Phase 2...')
                    best_hp_a = tune_lstm_hyperparams(
                        X_tr_a, y_tr_a,
                        X_val_a, y_val_a,
                        input_size=len(LSTM_A_FEATURES),
                        device=device,
                        arch_grid=config.LSTM_A_ARCH_GRID,  # Phase 2 architecture search
                        seed=fold_seed_base + 10,
                    )
                    tuning_results.append({
                        'fold': fold['fold'],
                        'model': 'LSTM-A',
                        **best_hp_a
                    })
                    print(f'    [LSTM-A Tuning] Best: {best_hp_a}')
                    model_a = train_lstm_a(
                        X_tr_a, y_tr_a,
                        X_val_a, y_val_a,
                        device,
                        optimizer_name=best_hp_a['optimizer'],
                        lr=best_hp_a['lr'],
                        hidden_size=best_hp_a['hidden_size'],
                        num_layers=best_hp_a['num_layers'],
                        dropout=best_hp_a['dropout'],
                        batch_size=best_hp_a['batch_size'],
                        seed=fold_seed_base + 20,
                        fold_idx=fold['fold'],
                    )
                else:
                    # Train with default hyperparameters when tuning is disabled.
                    model_a = train_lstm_a(
                        X_tr_a, y_tr_a,
                        X_val_a, y_val_a,
                        device,
                        seed=fold_seed_base + 20,
                        fold_idx=fold['fold'],
                    )
                print(f'  [LSTM-A]  fit done in {time.time()-t0:.1f}s')

                # LSTM-A inference
                probs_a = predict_lstm(model_a, X_te_a, device)

                # Free LSTM-A memory before training LSTM
                del model_a, X_tr_a, y_tr_a, X_val_a, y_val_a, X_te_a, y_te_a
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            else:
                print('  [LSTM-A]  skipped (DEV_MODE=True)')
                probs_a = None

            # ── LSTM: Extended feature LSTM with optional tuning ───────────────
            t0 = time.time()
            print('  [LSTM]  building sequences & training...')

            # Use TEMPORAL split (splits by date, not index) - FIX for ticker-based split bug
            X_tr_b, y_tr_b, X_val_b, y_val_b, X_te_b, y_te_b, keys_tr_b, keys_val_b, keys_te_b = \
                prepare_lstm_b_sequences_temporal_split(df_train_fold, df_test_fold, val_ratio=LSTM_B_VAL_SPLIT)

            print(f'    LSTM sequences: train={len(X_tr_b)}, val={len(X_val_b)}, test={len(X_te_b)}')

            best_hp_b = None
            should_tune_b = lstm_b_tuning_enabled and (
                (best_hp_b_global is None) or (not lstm_b_tune_once)
            )
            if should_tune_b:
                print('    [LSTM Tuning] Running Phase 1 + Phase 2...')
                tuned_hp_b = tune_lstm_hyperparams(
                    X_tr_b, y_tr_b,
                    X_val_b, y_val_b,
                    input_size=len(LSTM_B_FEATURES),
                    device=device,
                    arch_grid=config.LSTM_B_ARCH_GRID,
                    train_grid=config.LSTM_B_HYPERPARAM_GRID,
                    tune_replicates=config.LSTM_B_TUNE_REPLICATES,
                    tune_patience=config.LSTM_B_TUNE_PATIENCE,
                    tune_max_epochs=config.LSTM_B_TUNE_MAX_EPOCHS,
                    seed_hidden=config.LSTM_B_HIDDEN_SIZE,
                    seed_layers=config.LSTM_B_NUM_LAYERS,
                    seed_dropout=config.LSTM_B_DROPOUT,
                    seed=fold_seed_base + 25,
                )

                # Return-aware guardrail: compare tuned-vs-default on val trading metrics.
                candidate_hps = [
                    ('default', None),
                    ('tuned', tuned_hp_b),
                ]
                candidate_scores = []

                for cand_name, cand_hp in candidate_hps:
                    if cand_hp is None:
                        model_b_cand = train_lstm_b(
                            X_tr_b, y_tr_b,
                            X_val_b, y_val_b,
                            device,
                            seed=fold_seed_base + 28,
                            fold_idx=fold['fold'],
                        )
                    else:
                        model_b_cand = train_lstm_b(
                            X_tr_b, y_tr_b,
                            X_val_b, y_val_b,
                            device,
                            seed=fold_seed_base + 28,
                            fold_idx=fold['fold'],
                            optimizer_name=cand_hp['optimizer'],
                            learning_rate=cand_hp['lr'],
                            batch_size=cand_hp['batch_size'],
                            hidden_size=cand_hp['hidden_size'],
                            num_layers=cand_hp['num_layers'],
                            dropout=cand_hp['dropout'],
                        )

                    probs_val_b = predict_lstm(model_b_cand, X_val_b, device)
                    pred_val_b = df_v.copy().reset_index(drop=True)
                    pred_val_b['Prob_LSTM_B'] = align_predictions_to_df(
                        probs_val_b, keys_val_b, df_v
                    )
                    val_sharpe_b, val_ann_ret_b = _score_lstm_b_candidate_on_val(pred_val_b)
                    candidate_scores.append({
                        'name': cand_name,
                        'hp': cand_hp,
                        'val_sharpe_net': val_sharpe_b,
                        'val_ann_ret_net_pct': val_ann_ret_b,
                    })

                    del model_b_cand
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()

                candidate_scores = sorted(
                    candidate_scores,
                    key=lambda x: (x['val_sharpe_net'], x['val_ann_ret_net_pct']),
                    reverse=True,
                )
                selected = candidate_scores[0]
                best_hp_b = selected['hp']

                print(
                    '    [LSTM Tuning] Candidate scores: '
                    + ', '.join(
                        f"{c['name']}(Sharpe={c['val_sharpe_net']:.3f},AnnRet={c['val_ann_ret_net_pct']:.2f}%)"
                        for c in candidate_scores
                    )
                )
                if selected['name'] == 'default':
                    print('    [LSTM Tuning] Tuned candidate underperformed on val returns; using default.')

                tuning_results.append({
                    'fold': fold['fold'],
                    'model': 'LSTM',
                    'selection_basis': 'val_net_sharpe_then_annret',
                    'selected_candidate': selected['name'],
                    'val_sharpe_selected': selected['val_sharpe_net'],
                    'val_ann_ret_selected': selected['val_ann_ret_net_pct'],
                    **(tuned_hp_b if tuned_hp_b is not None else {}),
                })
                print(f'    [LSTM Tuning] Selected params: {best_hp_b}')
                if lstm_b_tune_once:
                    best_hp_b_global = best_hp_b if best_hp_b is not None else 'DEFAULT'
            elif best_hp_b_global is not None:
                if best_hp_b_global == 'DEFAULT':
                    best_hp_b = None
                else:
                    best_hp_b = best_hp_b_global
                print(f'    [LSTM Tuning] Reusing tuned params: {best_hp_b}')

            if best_hp_b is not None:
                model_b = train_lstm_b(
                    X_tr_b, y_tr_b,
                    X_val_b, y_val_b,
                    device,
                    seed=fold_seed_base + 30,
                    fold_idx=fold['fold'],
                    optimizer_name=best_hp_b['optimizer'],
                    learning_rate=best_hp_b['lr'],
                    batch_size=best_hp_b['batch_size'],
                    hidden_size=best_hp_b['hidden_size'],
                    num_layers=best_hp_b['num_layers'],
                    dropout=best_hp_b['dropout'],
                )
            else:
                model_b = train_lstm_b(
                    X_tr_b, y_tr_b,
                    X_val_b, y_val_b,
                    device,
                    seed=fold_seed_base + 30,
                    fold_idx=fold['fold'],
                )
            print(f'  [LSTM]  fit done in {time.time()-t0:.1f}s')

            # LSTM inference
            probs_b = predict_lstm(model_b, X_te_b, device)

            # Free LSTM memory for next fold
            del model_b, X_tr_b, y_tr_b, X_val_b, y_val_b, X_te_b, y_te_b
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        else:
            print('\n  [LSTMs]     Skipping LSTM-A and LSTM (RUN_LSTMS=False)')

        # ── Collect predictions for this fold ────────────────────────────────
        pred = df_ts.copy().reset_index(drop=True)
        pred['Prob_LR'] = lr_m.predict_proba(X_ts_b_s)[:, 1] if RUN_BASELINES else np.nan
        pred['Prob_RF'] = rf_m.predict_proba(X_ts_b_s)[:, 1] if RUN_BASELINES else np.nan
        pred['Prob_XGB'] = xgb_m.predict(xgb.DMatrix(X_ts_b_s)) if RUN_BASELINES else np.nan
        pred['Prob_LSTM_A'] = align_predictions_to_df(probs_a, keys_te_a, df_ts) if RUN_LSTMS else np.nan
        pred['Prob_LSTM_B'] = align_predictions_to_df(probs_b, keys_te_b, df_ts) if RUN_LSTMS else np.nan
        pred['Fold'] = fold['fold']

        if RUN_BASELINES:
            classification_sanity_checks(
                y_ts, pred['Prob_LR'].values, name=f"fold{fold['fold']} test LR",
            )
            val_auc_lr = binary_auc_safe(y_v, lr_m.predict_proba(X_v_b_s)[:, 1], log_on_fail=False)
            test_auc_lr = binary_auc_safe(y_ts, pred['Prob_LR'].values, log_on_fail=False)
        else:
            val_auc_lr = test_auc_lr = float('nan')

        if SAVE_FOLD_REPORTS:
            fr_extra = {
                'val_auc_lr': val_auc_lr,
                'test_auc_lr': test_auc_lr,
            }
            path_fr = save_fold_report(
                fold, df_tr, df_v, df_ts, TARGET_COL,
                extra=fr_extra,
                reports_dir=os.path.join(reports_dir, 'fold_reports'),
            )
            print(f'  Fold report: {path_fr}')

        # Report coverage
        n_lstm_a = pred['Prob_LSTM_A'].notna().sum()
        n_lstm_b = pred['Prob_LSTM_B'].notna().sum()
        print(f'  Predictions: LR/RF/XGB={len(pred)}, '
              f'LSTM-A={n_lstm_a}, LSTM={n_lstm_b}')

        all_preds.append(pred)

    # ── Combine folds ─────────────────────────────────────────────────────────
    full_preds = pd.concat(all_preds).reset_index(drop=True)
    print(f'\nTotal predictions: {len(full_preds)}')
    print(f'  LSTM-A valid: {full_preds["Prob_LSTM_A"].notna().sum()}')
    print(f'  LSTM valid: {full_preds["Prob_LSTM_B"].notna().sum()}')

    # ── Backtest each model independently ─────────────────────────────────────
    print('\n' + '=' * 60)
    print('RESULTS — GROSS (0 bps)')
    print('=' * 60)

    model_cols = {
        'LR': 'Prob_LR',
        'RF': 'Prob_RF',
        'XGBoost': 'Prob_XGB',
        'LSTM': 'Prob_LSTM_B',
    }
    if not DEV_MODE:
        model_cols['LSTM-A'] = 'Prob_LSTM_A'

    port_returns_gross = {}
    port_returns_net_5 = {}
    class_metrics = []
    all_signals = []
    daily_returns_gross = {'Date': None}
    daily_returns_net_5 = {'Date': None}

    for model_name, prob_col in model_cols.items():
        # Filter to rows that have predictions for this model
        valid_preds = full_preds.dropna(subset=[prob_col]).copy()

        if len(valid_preds) == 0:
            print(f'  {model_name:<12}  [SKIPPED - no valid predictions]')
            continue

        smoothed_preds = smooth_probabilities(
            valid_preds, prob_col,
            alpha=SIGNAL_SMOOTH_ALPHA,
            ema_method=SIGNAL_EMA_METHOD,
            ema_span=SIGNAL_EMA_SPAN,
        )
        smoothed_col = f'{prob_col}_Smooth'

        sig_df, sig_diag = generate_signals(
            smoothed_preds, k=K_STOCKS, prob_col=smoothed_col,
            return_diagnostics=True,
        )
        sig_df = apply_holding_period_constraint(sig_df, min_hold_days=MIN_HOLDING_DAYS)
        hold_st = compute_turnover_and_holding_stats(sig_df, k=K_STOCKS)
        print(f'  [{model_name}] turnover~{hold_st["mean_daily_turnover_half_turns"]:.2f}  '
              f'avg_hold~{hold_st["avg_holding_period_trading_days"]:.1f}  '
              f'threshold_filtered(L/S)={sig_diag["long_slots_filtered_by_threshold"]}/'
              f'{sig_diag["short_slots_filtered_by_threshold"]}')

        sig_df['Model'] = model_name
        all_signals.append(sig_df)

        port_gross = compute_portfolio_returns(
            sig_df, tc_bps=0, k=K_STOCKS, slippage_bps=SLIPPAGE_BPS,
        )
        port_net_5 = compute_portfolio_returns(
            sig_df, tc_bps=TC_BPS, k=K_STOCKS, slippage_bps=SLIPPAGE_BPS,
        )

        if RUN_SIGNAL_ABLATION:
            sig_raw, _ = generate_signals(
                valid_preds, k=K_STOCKS, prob_col=prob_col,
                confidence_threshold=0.0, return_diagnostics=True,
            )
            sig_raw = apply_holding_period_constraint(sig_raw, min_hold_days=1)
            port_raw = compute_portfolio_returns(
                sig_raw, tc_bps=TC_BPS, k=K_STOCKS, slippage_bps=SLIPPAGE_BPS,
            )
            st_raw = compute_turnover_and_holding_stats(sig_raw, k=K_STOCKS)
            ablation_rows.append({
                'Model': model_name,
                'variant': 'raw_rank_min_hold_1',
                'Sharpe Net': compute_metrics(port_raw['Net_Return'])['Sharpe Ratio'],
                'mean_turnover': st_raw['mean_daily_turnover_half_turns'],
                'avg_hold': st_raw['avg_holding_period_trading_days'],
            })
            ablation_rows.append({
                'Model': model_name,
                'variant': 'ema_threshold_min_hold',
                'Sharpe Net': compute_metrics(port_net_5['Net_Return'])['Sharpe Ratio'],
                'mean_turnover': hold_st['mean_daily_turnover_half_turns'],
                'avg_hold': hold_st['avg_holding_period_trading_days'],
            })

        port_returns_gross[model_name] = port_gross
        port_returns_net_5[model_name] = port_net_5

        # Store daily returns for export
        if daily_returns_gross['Date'] is None:
            daily_returns_gross['Date'] = port_gross.index
            daily_returns_net_5['Date'] = port_net_5.index
        daily_returns_gross[model_name] = port_gross['Gross_Return'].values
        daily_returns_net_5[model_name] = port_net_5['Net_Return'].values

        # Classification metrics (pooled + daily AUC for diagnostic)
        y_true = valid_preds[TARGET_COL].values
        y_prob = valid_preds[prob_col].values
        cm = evaluate_classification(y_true, y_prob)

        # Add daily AUC to diagnose pooled vs within-day ranking
        daily_auc = compute_daily_auc(valid_preds, prob_col, TARGET_COL)
        cm['Daily AUC (mean)'] = daily_auc['Daily AUC (mean)']
        cm['Daily AUC (std)'] = daily_auc['Daily AUC (std)']

        cm['Model'] = model_name
        class_metrics.append(cm)

        # Print gross metrics
        m = compute_metrics(port_gross['Gross_Return'])
        print(f'  {model_name:<12}  '
              f'Sharpe={m["Sharpe Ratio"]:>6.3f}  '
              f'Sortino={m["Sortino Ratio"]:>6.3f}  '
              f'Ann.Ret={m["Annualized Return (%)"]:.2f}%  '
              f'MDD={m["Max Drawdown (%)"]:.2f}%')

    # ── Print net results ─────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print(f'RESULTS — NET ({TC_BPS:g} bps TC + {SLIPPAGE_BPS:g} bps slippage per half-turn)')
    print('=' * 60)

    results_gross = []
    results_net_5 = []

    all_model_names = list(model_cols.keys())

    for model_name in all_model_names:
        if model_name not in port_returns_gross:
            continue

        # Gross
        m = compute_metrics(port_returns_gross[model_name]['Gross_Return'])
        m['Model'] = model_name
        results_gross.append(m)

        # Net 5 bps
        m = compute_metrics(port_returns_net_5[model_name]['Net_Return'])
        m['Model'] = model_name
        results_net_5.append(m)
        print(f'  {model_name:<12}  '
              f'Sharpe={m["Sharpe Ratio"]:>6.3f}  '
              f'Sortino={m["Sortino Ratio"]:>6.3f}  '
              f'Ann.Ret={m["Annualized Return (%)"]:.2f}%  '
              f'MDD={m["Max Drawdown (%)"]:.2f}%')

    # ── Sub-period analysis (using the best performing model) ─────────────────
    subperiod_metrics = None
    if 'LSTM-A' in port_returns_net_5:
        try:
            subperiod_metrics = compute_subperiod_metrics(
                port_returns_net_5['LSTM-A']['Net_Return']
            )
        except Exception as e:
            print(f'Warning: Could not compute sub-period metrics: {e}')

    # ── Save all results ─────────────────────────────────────────────────────
    results_dict = {
        'gross': results_gross,
        'net_5': results_net_5,
        'classification': class_metrics,
        'subperiod': subperiod_metrics,
    }

    daily_returns_gross_df = pd.DataFrame(daily_returns_gross)
    daily_returns_net_5_df = pd.DataFrame(daily_returns_net_5)

    signals_df = pd.concat(all_signals).reset_index(drop=True) if all_signals else pd.DataFrame()

    # Save full predictions so Baseline/LSTM runs can be stitched together later
    full_preds_path = os.path.join(reports_dir, f'{config.UNIVERSE_MODE}_full_predictions.csv')
    full_preds.to_csv(full_preds_path, index=False)
    print(f'  Saved raw probabilities to {full_preds_path}')

    save_all_results(
        results_dict=results_dict,
        daily_returns_dict={
            'gross': daily_returns_gross_df,
            'net_5': daily_returns_net_5_df,
        },
        signals_dict=signals_df,
        tuning_results=tuning_results,
        reports_dir=reports_dir,
    )

    if ablation_rows:
        ab_path = os.path.join(reports_dir, f'{config.UNIVERSE_MODE}_signal_ablation_summary.csv')
        pd.DataFrame(ablation_rows).to_csv(ab_path, index=False)
        print(f'Signal ablation summary: {ab_path}')

    print('\n\nPipeline complete.')
    return {
        'full_preds': full_preds,
        'port_returns': port_returns_net_5,
        'class_metrics': class_metrics,
        'folds': folds,
    }


def main(load_cached: bool = True):
    """CLI entry: delegates to run_walk_forward_pipeline with default reports dir."""
    return run_walk_forward_pipeline(load_cached=load_cached, reports_dir='reports')


if __name__ == '__main__':
    main(load_cached=False)

```

### Orchestration line index extracts

```text
103:def set_global_seed(seed: int):
112:    try:
114:    except Exception:
233:def run_walk_forward_pipeline(
273:    set_global_seed(config.RANDOM_SEED)
276:    if load_cached:
388:        fold_seed_base = config.RANDOM_SEED + (fold['fold'] * 1000)
389:        set_global_seed(fold_seed_base)
500:            if not DEV_MODE:
533:                        seed=fold_seed_base + 20,
544:                        seed=fold_seed_base + 10,
562:                        seed=fold_seed_base + 20,
571:                        seed=fold_seed_base + 20,
616:                    seed=fold_seed_base + 25,
632:                            seed=fold_seed_base + 28,
640:                            seed=fold_seed_base + 28,
709:                    seed=fold_seed_base + 30,
723:                    seed=fold_seed_base + 30,
777:    full_preds = pd.concat(all_preds).reset_index(drop=True)
793:    if not DEV_MODE:
928:        try:
932:        except Exception as e:
978:def main(load_cached: bool = True):
979:    """CLI entry: delegates to run_walk_forward_pipeline with default reports dir."""
980:    return run_walk_forward_pipeline(load_cached=load_cached, reports_dir='reports')

```

## SECTION 12 — OBSERVED RUNTIME OUTPUTS

### reports/backtest_summary.txt (verbatim)

```text
============================================================
BACKTEST RESULTS SUMMARY
============================================================

────────────────────────────────────────────────────────────
GROSS RETURNS (0 bps TC)
────────────────────────────────────────────────────────────
  LR            Sharpe=-0.530  Sortino=-0.722  Ann.Ret= -2.92%  MDD=-27.92%
  RF            Sharpe=-0.246  Sortino=-0.354  Ann.Ret=  1.01%  MDD=-24.19%
  XGBoost       Sharpe=-0.199  Sortino=-0.288  Ann.Ret=  1.65%  MDD=-21.38%
  LSTM        Sharpe=-0.095  Sortino=-0.133  Ann.Ret=  2.48%  MDD=-34.03%
  Ensemble      Sharpe=-0.781  Sortino=-1.142  Ann.Ret= -5.62%  MDD=-38.03%

────────────────────────────────────────────────────────────
NET RETURNS  (5 bps TC)
────────────────────────────────────────────────────────────
  LR            Sharpe=-0.968  Sortino=-1.321  Ann.Ret= -8.19%  MDD=-40.02%
  RF            Sharpe=-0.814  Sortino=-1.170  Ann.Ret= -5.24%  MDD=-36.42%
  XGBoost       Sharpe=-0.801  Sortino=-1.153  Ann.Ret= -4.72%  MDD=-33.11%
  LSTM        Sharpe=-0.351  Sortino=-0.493  Ann.Ret= -1.14%  MDD=-37.60%
  Ensemble      Sharpe=-1.245  Sortino=-1.813  Ann.Ret=-10.85%  MDD=-48.81%

============================================================
CLASSIFICATION METRICS
============================================================
  LR            Acc=49.91%  AUC=0.4972  F1=0.5041
  RF            Acc=50.10%  AUC=0.5019  F1=0.4970
  XGBoost       Acc=50.13%  AUC=0.5012  F1=0.4972
  LSTM        Acc=50.17%  AUC=0.5004  F1=0.5240
  Ensemble      Acc=50.22%  AUC=0.5008  F1=0.5117
```

### reports/small_cap_backtest_summary.txt (verbatim)

```text
============================================================
BACKTEST RESULTS SUMMARY
============================================================

────────────────────────────────────────────────────────────
GROSS RETURNS (0 bps TC)
────────────────────────────────────────────────────────────
  LR            Sharpe= 0.317  Sortino= 0.405  Ann.Ret= 21.99%  MDD=-70.45%
  RF            Sharpe=-0.700  Sortino=-0.917  Ann.Ret=-19.25%  MDD=-78.77%
  XGBoost       Sharpe= 0.158  Sortino= 0.200  Ann.Ret=  9.91%  MDD=-60.77%
  LSTM        Sharpe=-0.373  Sortino=-0.451  Ann.Ret=-12.63%  MDD=-87.58%

────────────────────────────────────────────────────────────
NET RETURNS  (5 bps TC)
────────────────────────────────────────────────────────────
  LR            Sharpe= 0.205  Sortino= 0.263  Ann.Ret= 15.27%  MDD=-75.18%
  RF            Sharpe=-0.902  Sortino=-1.182  Ann.Ret=-24.91%  MDD=-84.36%
  XGBoost       Sharpe=-0.050  Sortino=-0.063  Ann.Ret=  2.02%  MDD=-67.72%
  LSTM        Sharpe=-0.472  Sortino=-0.571  Ann.Ret=-16.53%  MDD=-89.13%

============================================================
CLASSIFICATION METRICS
============================================================
  LR            Acc=50.48%  AUC=0.5028  F1=0.5729
  RF            Acc=50.94%  AUC=0.5112  F1=0.5367
  XGBoost       Acc=51.10%  AUC=0.5097  F1=0.5498
  LSTM        Acc=51.00%  AUC=0.5149  F1=0.5373
```

### reports/large_cap_backtest_summary.txt (verbatim)

```text
============================================================
BACKTEST RESULTS SUMMARY
============================================================

────────────────────────────────────────────────────────────
GROSS RETURNS (0 bps TC)
────────────────────────────────────────────────────────────
  LR            Sharpe=-0.828  Sortino=-1.124  Ann.Ret=-13.58%  MDD=-72.38%
  RF            Sharpe=-1.124  Sortino=-1.523  Ann.Ret=-16.76%  MDD=-65.93%
  XGBoost       Sharpe=-0.773  Sortino=-1.091  Ann.Ret= -9.25%  MDD=-53.73%
  LSTM        Sharpe= 0.035  Sortino= 0.054  Ann.Ret=  4.72%  MDD=-54.38%

────────────────────────────────────────────────────────────
NET RETURNS  (5 bps TC)
────────────────────────────────────────────────────────────
  LR            Sharpe=-1.132  Sortino=-1.537  Ann.Ret=-19.21%  MDD=-77.90%
  RF            Sharpe=-1.507  Sortino=-2.043  Ann.Ret=-22.81%  MDD=-72.83%
  XGBoost       Sharpe=-1.227  Sortino=-1.728  Ann.Ret=-16.16%  MDD=-63.84%
  LSTM        Sharpe=-0.157  Sortino=-0.238  Ann.Ret=  0.09%  MDD=-56.51%

============================================================
CLASSIFICATION METRICS
============================================================
  LR            Acc=49.56%  AUC=0.4938  F1=0.5060
  RF            Acc=48.94%  AUC=0.4878  F1=0.4816
  XGBoost       Acc=49.19%  AUC=0.4912  F1=0.4848
  LSTM        Acc=49.67%  AUC=0.4982  F1=0.5312
```

### Fold-by-fold AUC found in reports/fold_reports/*.json

```text
reports/fold_reports/fold_001.json: fold=1 val_auc_lr=0.49090575930402774 test_auc_lr=0.490671486398694
reports/fold_reports/fold_002.json: fold=2 val_auc_lr=0.5139053155869467 test_auc_lr=0.5129089284933441
reports/fold_reports/fold_003.json: fold=3 val_auc_lr=0.5182977996830811 test_auc_lr=0.4730975750140107
reports/fold_reports/fold_004.json: fold=4 val_auc_lr=0.4719464261371382 test_auc_lr=0.514238823101257
reports/fold_reports/fold_005.json: fold=5 val_auc_lr=0.5076303155006858 test_auc_lr=0.49481572464376694
reports/fold_reports/fold_006.json: fold=6 val_auc_lr=0.48785728283082785 test_auc_lr=0.48981866409115093
reports/fold_reports/fold_007.json: fold=7 val_auc_lr=0.5096704319587917 test_auc_lr=0.5373939699336525
reports/fold_reports/fold_008.json: fold=8 val_auc_lr=0.5272091346826796 test_auc_lr=0.47573906105652136
reports/fold_reports/fold_009.json: fold=9 val_auc_lr=0.4885029114526469 test_auc_lr=0.48850116178158504
reports/fold_reports/fold_010.json: fold=10 val_auc_lr=0.49049228744995943 test_auc_lr=0.5007856023067664
reports/fold_reports/fold_011.json: fold=11 val_auc_lr=0.5014592256655749 test_auc_lr=0.4922419585117998
reports/fold_reports/fold_012.json: fold=12 val_auc_lr=0.4992598891408415 test_auc_lr=0.5101638391982308
reports/fold_reports/fold_013.json: fold=13 val_auc_lr=0.5051282858822541 test_auc_lr=0.5183082798976838
reports/fold_reports/fold_014.json: fold=14 val_auc_lr=0.5119450576404219 test_auc_lr=0.5028939559362839
reports/fold_reports/fold_015.json: fold=15 val_auc_lr=0.5125626382240138 test_auc_lr=0.5044179194311469
reports/fold_reports/fold_016.json: fold=16 val_auc_lr=0.5128775790151452 test_auc_lr=0.5169648106156043
reports/fold_reports/fold_017.json: fold=17 val_auc_lr=0.50874660563814 test_auc_lr=0.5060809924151627
reports/fold_reports/fold_018.json: fold=18 val_auc_lr=0.5247220205236774 test_auc_lr=0.5190801629293693
reports/fold_reports/fold_019.json: fold=19 val_auc_lr=0.4909328125 test_auc_lr=0.51629375
reports/fold_reports/fold_020.json: fold=20 val_auc_lr=0.5093890624999999 test_auc_lr=0.5115875
reports/fold_reports/fold_021.json: fold=21 val_auc_lr=0.5120359375 test_auc_lr=0.49211874999999994
reports/fold_reports/fold_022.json: fold=22 val_auc_lr=0.49057500000000004 test_auc_lr=0.49317500000000003
reports/fold_reports/fold_023.json: fold=23 val_auc_lr=0.49833906250000004 test_auc_lr=0.494625
reports/fold_reports/fold_024.json: fold=24 val_auc_lr=0.491378125 test_auc_lr=0.47608125
reports/fold_reports/fold_025.json: fold=25 val_auc_lr=0.4828984375 test_auc_lr=0.5147437499999999
reports/fold_reports/fold_026.json: fold=26 val_auc_lr=0.4976015625 test_auc_lr=0.51595
reports/fold_reports/fold_027.json: fold=27 val_auc_lr=0.5086578125 test_auc_lr=0.49475625
reports/fold_reports/fold_028.json: fold=28 val_auc_lr=0.5142578125 test_auc_lr=0.53735
reports/fold_reports/fold_029.json: fold=29 val_auc_lr=0.5157 test_auc_lr=0.51325
reports/fold_reports/fold_030.json: fold=30 val_auc_lr=0.5238765625000001 test_auc_lr=0.5104562500000001
reports/fold_reports/fold_031.json: fold=31 val_auc_lr=0.5097359374999999 test_auc_lr=0.52288125
reports/fold_reports/fold_032.json: fold=32 val_auc_lr=0.5294546875 test_auc_lr=0.48289375
reports/fold_reports/fold_033.json: fold=33 val_auc_lr=0.496846875 test_auc_lr=0.49874999999999997
reports/fold_reports/fold_034.json: fold=34 val_auc_lr=0.4892328125 test_auc_lr=0.49901875000000007
reports/fold_reports/fold_035.json: fold=35 val_auc_lr=0.50549375 test_auc_lr=0.487425
reports/fold_reports/fold_036.json: fold=36 val_auc_lr=0.49151718749999995 test_auc_lr=0.43366874999999994
reports/fold_reports/fold_037.json: fold=37 val_auc_lr=0.46388437499999996 test_auc_lr=0.52076875
reports/fold_reports/fold_038.json: fold=38 val_auc_lr=0.47170312499999995 test_auc_lr=0.46670625
reports/fold_reports/fold_039.json: fold=39 val_auc_lr=0.49223281249999995 test_auc_lr=0.461925
reports/fold_reports/fold_040.json: fold=40 val_auc_lr=0.47706875000000004 test_auc_lr=0.48691875
reports/fold_reports/fold_041.json: fold=41 val_auc_lr=0.470134375 test_auc_lr=0.49989999999999996
reports/fold_reports/fold_042.json: fold=42 val_auc_lr=0.48486093750000003 test_auc_lr=0.48282500000000006
reports/fold_reports/fold_043.json: fold=43 val_auc_lr=0.4742015625 test_auc_lr=0.44276875
reports/fold_reports/fold_044.json: fold=44 val_auc_lr=0.4738484375 test_auc_lr=0.48485624999999993
reports/fold_reports/fold_045.json: fold=45 val_auc_lr=0.48053749999999995 test_auc_lr=0.50090625
reports/fold_reports/fold_046.json: fold=46 val_auc_lr=0.5023328125 test_auc_lr=0.47048419052619084
reports/fold_reports/fold_047.json: fold=47 val_auc_lr=0.4850960704626101 test_auc_lr=0.47429999999999994
reports/fold_reports/fold_048.json: fold=48 val_auc_lr=0.478503872662301 test_auc_lr=0.5131
reports/fold_reports/fold_049.json: fold=49 val_auc_lr=0.49709062499999995 test_auc_lr=0.4975625
reports/fold_reports/fold_050.json: fold=50 val_auc_lr=0.492078125 test_auc_lr=0.46701875
reports/fold_reports/fold_051.json: fold=51 val_auc_lr=0.48566718750000004 test_auc_lr=0.5133875
reports/fold_reports/fold_052.json: fold=52 val_auc_lr=0.4904171875 test_auc_lr=0.5289312500000001
reports/fold_reports/fold_053.json: fold=53 val_auc_lr=0.5243421875000001 test_auc_lr=0.52780625
reports/fold_reports/fold_054.json: fold=54 val_auc_lr=0.5273359375 test_auc_lr=0.51904375
reports/fold_reports/fold_055.json: fold=55 val_auc_lr=0.5202046875 test_auc_lr=0.4718375
reports/fold_reports/fold_056.json: fold=56 val_auc_lr=0.49746406249999997 test_auc_lr=0.5264
reports/fold_reports/fold_057.json: fold=57 val_auc_lr=0.5052875 test_auc_lr=0.5069625
reports/fold_reports/fold_058.json: fold=58 val_auc_lr=0.512303125 test_auc_lr=0.47270625
reports/fold_reports/fold_059.json: fold=59 val_auc_lr=0.49287968749999994 test_auc_lr=0.5216
reports/fold_reports/fold_060.json: fold=60 val_auc_lr=0.49045781250000003 test_auc_lr=0.50944375
```

### LSTM train/val loss summaries from reports/training_logs

```text
LSTM_A_LOG_SUMMARY
file,last_epoch,last_train_loss,last_val_loss,last_train_auc,last_val_auc,best_epoch,best_val_loss,best_val_auc
fold10_lstm-a.csv,23,0.688846,0.696001,0.559909718709,0.503032194060,8,0.692608,0.509600614439
fold11_lstm-a.csv,26,0.686608,0.696379,0.565405254541,0.532082128329,11,0.691896,0.531342125369
fold12_lstm-a.csv,16,0.687975,0.695171,0.561905801727,0.500816000000,1,0.694152,0.472516000000
fold13_lstm-a.csv,20,0.687277,0.693182,0.575439192092,0.531668000000,5,0.691361,0.548680000000
fold14_lstm-a.csv,17,0.690252,0.6968,0.554880873556,0.472466000000,2,0.692524,0.532780000000
fold15_lstm-a.csv,41,0.686622,0.693649,0.573931753616,0.508932000000,26,0.690684,0.539988000000
fold16_lstm-a.csv,16,0.690587,0.694608,0.548948056560,0.505752000000,1,0.693051,0.507996000000
fold17_lstm-a.csv,20,0.687684,0.69477,0.557647856405,0.525900000000,5,0.691632,0.533330000000
fold18_lstm-a.csv,18,0.688384,0.694265,0.560091038223,0.520872000000,3,0.693149,0.503836000000
fold19_lstm-a.csv,43,0.678637,0.693003,0.601429009556,0.538392000000,28,0.689306,0.555534000000
fold1_lstm-a.csv,36,0.685725,0.693149,0.566277194050,0.515660445558,21,0.69275,0.519827272521
fold20_lstm-a.csv,19,0.688438,0.697018,0.562752776343,0.484602000000,4,0.69318,0.498428000000
fold21_lstm-a.csv,18,0.689197,0.699012,0.559668130165,0.488652000000,3,0.692687,0.518700000000
fold22_lstm-a.csv,21,0.689362,0.694171,0.557744866994,0.495996000000,6,0.692998,0.495076000000
fold23_lstm-a.csv,17,0.690961,0.694805,0.535628712552,0.493626000000,2,0.692824,0.520868000000
fold24_lstm-a.csv,59,0.679762,0.689095,0.598427169421,0.556192000000,44,0.687983,0.546704000000
fold25_lstm-a.csv,16,0.69046,0.695165,0.543107082903,0.490336000000,1,0.693132,0.505206000000
fold26_lstm-a.csv,17,0.690923,0.694108,0.543926588326,0.499800000000,2,0.693745,0.461872000000
fold27_lstm-a.csv,19,0.69044,0.694256,0.547871577996,0.498320000000,4,0.692764,0.509600000000
fold28_lstm-a.csv,23,0.689725,0.695783,0.551347010589,0.510496000000,8,0.69328,0.513584000000
fold29_lstm-a.csv,34,0.685285,0.693891,0.587329545455,0.528196000000,19,0.690803,0.534956000000
fold2_lstm-a.csv,17,0.690935,0.693721,0.547659127289,0.503247004444,2,0.693121,0.505652622222
fold30_lstm-a.csv,16,0.68981,0.694557,0.556539740444,0.501948000000,1,0.692956,0.515158000000
fold31_lstm-a.csv,24,0.684957,0.698393,0.583750645661,0.498832000000,9,0.692677,0.516414000000
fold32_lstm-a.csv,17,0.690223,0.693501,0.540074735279,0.523488000000,2,0.692616,0.522904000000
fold33_lstm-a.csv,21,0.688705,0.69548,0.548477692407,0.533468000000,6,0.690644,0.546956000000
fold34_lstm-a.csv,18,0.68937,0.707119,0.552622837035,0.498368000000,3,0.692903,0.504490000000
fold35_lstm-a.csv,17,0.689139,0.694679,0.553470267304,0.491100000000,2,0.693024,0.514748000000
fold36_lstm-a.csv,33,0.688298,0.693473,0.561399470558,0.514114000000,18,0.692281,0.529474000000
fold37_lstm-a.csv,20,0.689052,0.695494,0.567118898502,0.491752000000,5,0.693507,0.493200000000
fold38_lstm-a.csv,18,0.687078,0.696687,0.551513591167,0.493264000000,3,0.693139,0.501164000000
fold39_lstm-a.csv,16,0.692344,0.694249,0.525718298037,0.475480000000,1,0.692971,0.514400000000
fold3_lstm-a.csv,30,0.690984,0.693263,0.542786710113,0.503038044444,15,0.693157,0.504814257778
fold40_lstm-a.csv,17,0.691132,0.696502,0.545206611570,0.486708000000,2,0.694937,0.477396000000
fold41_lstm-a.csv,40,0.686983,0.695571,0.581307786674,0.512692000000,25,0.692456,0.525096000000
fold42_lstm-a.csv,44,0.684806,0.694289,0.590235666322,0.536718146873,29,0.691118,0.542712170849
fold43_lstm-a.csv,33,0.688098,0.694053,0.571413375327,0.512560000000,18,0.692926,0.515296000000
fold44_lstm-a.csv,18,0.689814,0.696159,0.558213474372,0.497132000000,3,0.692828,0.513464000000
fold45_lstm-a.csv,38,0.685634,0.693405,0.581348651004,0.527826000000,23,0.692431,0.523632000000
fold46_lstm-a.csv,17,0.690552,0.696688,0.548699315599,0.443649774599,2,0.693284,0.502270009080
fold4_lstm-a.csv,30,0.686773,0.697547,0.566137768784,0.489248924892,15,0.691733,0.534225422542
fold5_lstm-a.csv,49,0.682498,0.691618,0.589406806474,0.528559253198,34,0.686492,0.560101472877
fold6_lstm-a.csv,21,0.687404,0.693679,0.565514395612,0.524414390630,6,0.692146,0.529298468776
fold7_lstm-a.csv,25,0.689045,0.692317,0.557175762025,0.527863461238,10,0.691118,0.530772031318
fold8_lstm-a.csv,17,0.689722,0.699943,0.551207735864,0.489579833277,2,0.692381,0.520276324421
fold9_lstm-a.csv,43,0.683345,0.69151,0.585782965348,0.541267525472,28,0.688965,0.544498064082
LSTM_B_LOG_SUMMARY
file,last_epoch,last_train_loss,last_val_loss,last_train_auc,last_val_auc,best_epoch,best_val_loss,best_val_auc
fold0_lstm-b.csv,9,0.683872,0.70129,0.587000538709,0.504451706997,1,0.692818,0.516823710876
fold10_lstm-b.csv,16,0.68948,0.701663,0.547698529047,0.488969198791,1,0.694412,0.476465174547
fold11_lstm-b.csv,19,0.687761,0.694103,0.560331939485,0.518833459310,4,0.691747,0.531535196383
fold12_lstm-b.csv,22,0.686697,0.694942,0.565106773440,0.509915385907,7,0.693627,0.511420977856
fold13_lstm-b.csv,34,0.680129,0.694883,0.593560691886,0.530062848185,19,0.691483,0.539857506789
fold14_lstm-b.csv,25,0.685638,0.692983,0.568626763024,0.527600826939,10,0.69156,0.526150180455
fold15_lstm-b.csv,26,0.683438,0.694722,0.577580561222,0.519573570169,11,0.691017,0.543638545953
fold16_lstm-b.csv,19,0.686744,0.696541,0.561863426513,0.522623246830,4,0.69204,0.525304617732
fold17_lstm-b.csv,23,0.683429,0.698755,0.578545429583,0.507567327342,8,0.691362,0.534905062848
fold18_lstm-b.csv,19,0.687568,0.693137,0.560224559896,0.533579745772,4,0.691156,0.540761283211
fold19_lstm-b.csv,17,0.646337,0.705266,0.688412677916,0.488245442708,2,0.694786,0.482215169271
fold1_lstm-b.csv,16,0.690734,0.698152,0.538268781470,0.505968778696,1,0.696831,0.481560607102
fold20_lstm-b.csv,24,0.60747,0.708795,0.750844524793,0.503388671875,9,0.693828,0.501431206597
fold21_lstm-b.csv,25,0.600518,0.708205,0.758266758494,0.513738064236,10,0.692508,0.520093315972
fold22_lstm-b.csv,19,0.640777,0.695689,0.689135746671,0.503444010417,4,0.692929,0.509597981771
fold23_lstm-b.csv,17,0.648273,0.695857,0.686993873393,0.496343858507,2,0.692929,0.508933919271
fold24_lstm-b.csv,16,0.651373,0.700796,0.676530503903,0.486242404514,1,0.693598,0.507519531250
fold25_lstm-b.csv,16,0.651779,0.699349,0.682977645776,0.472481553819,1,0.693397,0.503263346354
fold26_lstm-b.csv,19,0.634815,0.704629,0.705938934803,0.499935438368,4,0.69478,0.493924153646
fold27_lstm-b.csv,24,0.601235,0.703599,0.753429321625,0.511357421875,9,0.691781,0.530565321181
fold28_lstm-b.csv,16,0.645671,0.702361,0.692934386478,0.504929470486,1,0.693603,0.479733072917
fold29_lstm-b.csv,20,0.634322,0.703138,0.710700111915,0.492995876736,5,0.693848,0.496226128472
fold2_lstm-b.csv,16,0.689477,0.699774,0.548355424394,0.474341381953,1,0.693684,0.489873981776
fold30_lstm-b.csv,23,0.622922,0.694605,0.724314451331,0.531100802951,8,0.692379,0.524180230035
fold31_lstm-b.csv,29,0.594968,0.706131,0.764930842516,0.536674262153,14,0.691707,0.534958224826
fold32_lstm-b.csv,37,0.552048,0.703953,0.805121742998,0.535032552083,22,0.690304,0.548653428819
fold33_lstm-b.csv,20,0.635504,0.698497,0.709483040634,0.500032552083,5,0.692952,0.511717122396
fold34_lstm-b.csv,23,0.62034,0.700164,0.732388802801,0.514447699653,8,0.692249,0.527464192708
fold35_lstm-b.csv,27,0.590181,0.708044,0.773755452250,0.529352756076,12,0.691658,0.532134331597
fold36_lstm-b.csv,16,0.655222,0.696542,0.666464359504,0.507962239583,1,0.693399,0.495383572049
fold37_lstm-b.csv,16,0.64885,0.69833,0.679980701905,0.487226019965,1,0.693644,0.481577690972
fold38_lstm-b.csv,18,0.651447,0.69627,0.687516069789,0.495230577257,3,0.693182,0.507275933160
fold39_lstm-b.csv,21,0.634834,0.700695,0.708093147383,0.490660807292,6,0.692715,0.518953993056
fold3_lstm-b.csv,16,0.69003,0.693982,0.541089084006,0.493933355405,1,0.693134,0.501565854380
fold40_lstm-b.csv,18,0.642103,0.697974,0.699090622130,0.496085069444,3,0.6936,0.521599392361
fold41_lstm-b.csv,16,0.654363,0.697046,0.668319344008,0.482022026910,1,0.693453,0.493079969618
fold42_lstm-b.csv,19,0.635162,0.698674,0.706225536616,0.502887369792,4,0.693425,0.494120551215
fold43_lstm-b.csv,16,0.648787,0.695455,0.688540447084,0.489685872396,1,0.693365,0.507949761285
fold44_lstm-b.csv,16,0.653121,0.69662,0.671575413223,0.483397352431,1,0.694375,0.473849826389
fold45_lstm-b.csv,16,0.650682,0.698245,0.672687887397,0.473786349826,1,0.694786,0.459319118924
fold46_lstm-b.csv,16,0.657408,0.698121,0.668550347222,0.530661892361,1,0.69457,0.486923285590
fold47_lstm-b.csv,16,0.657094,0.695496,0.670176337236,0.503077802819,1,0.692895,0.518808614159
fold48_lstm-b.csv,16,0.647438,0.696015,0.692798080234,0.488939332616,1,0.693029,0.506357428773
fold49_lstm-b.csv,17,0.643463,0.69606,0.700330463776,0.506094292535,2,0.694162,0.504830186632
fold4_lstm-b.csv,28,0.686602,0.6944,0.563394178755,0.490648778776,13,0.692712,0.503218892773
fold50_lstm-b.csv,19,0.625359,0.69735,0.727845030970,0.513694118924,4,0.692629,0.514526909722
fold51_lstm-b.csv,21,0.622047,0.697689,0.739030179498,0.506657443576,6,0.692711,0.518984375000
fold52_lstm-b.csv,17,0.64261,0.69446,0.702521264136,0.517048611111,2,0.693132,0.506332465278
fold53_lstm-b.csv,22,0.599109,0.705354,0.765373732406,0.519228515625,7,0.690637,0.534202473958
fold54_lstm-b.csv,17,0.641506,0.702643,0.697327134603,0.505718315972,2,0.694555,0.499458550347
fold55_lstm-b.csv,22,0.624994,0.699414,0.730097240889,0.524471571181,7,0.692367,0.524638129340
fold56_lstm-b.csv,19,0.634392,0.702488,0.704820506198,0.500717230903,4,0.693596,0.527030164931
fold57_lstm-b.csv,17,0.641392,0.695722,0.691155374770,0.513183051215,2,0.693544,0.504272460938
fold58_lstm-b.csv,24,0.602078,0.700351,0.756533158287,0.510378146701,9,0.692082,0.525107964410
fold59_lstm-b.csv,24,0.605681,0.695978,0.748883077938,0.505251736111,9,0.692605,0.511676432292
fold5_lstm-b.csv,16,0.68949,0.694289,0.547886857602,0.514556469463,1,0.692569,0.533128239729
fold60_lstm-b.csv,24,0.606808,0.695678,0.744581037649,0.514538845486,9,0.692721,0.513515625000
fold6_lstm-b.csv,18,0.686883,0.695413,0.557862770782,0.498892458218,3,0.693542,0.499142661180
fold7_lstm-b.csv,16,0.687796,0.695387,0.551230336276,0.494862965762,1,0.693426,0.505680307102
fold8_lstm-b.csv,24,0.686739,0.692514,0.557250661059,0.526084096190,9,0.691336,0.539026413034
fold9_lstm-b.csv,16,0.689528,0.693845,0.547228887108,0.518756473783,1,0.692035,0.524532137958
```

### outputs/feature_selection_log.txt (verbatim)

```text
Correlation threshold: 0.8
Original features (8): ['Return_1d', 'RSI_14', 'MACD', 'ATR_14', 'BB_PctB', 'RealVol_20d', 'Volume_Ratio', 'SectorRelReturn']

Dropped features (1): ['BB_PctB']

Retained features (7): ['Return_1d', 'RSI_14', 'MACD', 'ATR_14', 'RealVol_20d', 'Volume_Ratio', 'SectorRelReturn']

```

## SECTION 13 — KNOWN DEVIATIONS FROM CLAUDE.md

```text
File: config.py
Location: Universe setup and date range
CLAUDE.md says: 70-stock S&P 500 development universe over 2015-2024.
Code does: Uses UNIVERSE_MODE with 30-stock large-cap or 30-stock small-cap lists; active mode is small_cap; START_DATE='2019-01-01'.
Severity: [HIGH]
```

```text
File: config.py
Location: TRAIN_DAYS, VAL_DAYS, TEST_DAYS
CLAUDE.md says: TRAIN_DAYS=500, VAL_DAYS=125, TEST_DAYS=125.
Code does: TRAIN_DAYS=252, VAL_DAYS=63, TEST_DAYS=63.
Severity: [MEDIUM]
```

```text
File: config.py
Location: Trading controls
CLAUDE.md says: K_STOCKS=10 and SIGNAL_CONFIDENCE_THRESHOLD=0.0.
Code does: K_STOCKS=5 and SIGNAL_CONFIDENCE_THRESHOLD=0.55.
Severity: [MEDIUM]
```

```text
File: config.py
Location: Wavelet toggle
CLAUDE.md says: USE_WAVELET_DENOISING=True (causal per-fold denoising active).
Code does: USE_WAVELET_DENOISING=False by default.
Severity: [MEDIUM]
```

```text
File: config.py
Location: SMALL_CAP_SECTOR_MAP vs active SMALL_CAP_TICKERS
CLAUDE.md says: SectorRelReturn should be computed with valid sector mapping.
Code does: Active small-cap tickers are absent from SMALL_CAP_SECTOR_MAP; active set maps to Unknown in current mode.
Severity: [HIGH]
```

```text
File: main.py
Location: model_cols and post-fold backtest block
CLAUDE.md says: Backtests LR, RF, XGBoost, LSTM-A, LSTM, and Ensemble.
Code does: No Ensemble backtest block present; model_cols includes LR, RF, XGBoost, LSTM and LSTM-A only when DEV_MODE=False.
Severity: [HIGH]
```

```text
File: main.py
Location: LSTM-A execution branch in fold loop
CLAUDE.md says: LSTM-A is part of the model set (with optional tuning behavior).
Code does: Entire LSTM-A path skipped when DEV_MODE=True.
Severity: [HIGH]
```

```text
File: main.py
Location: __main__ entry point
CLAUDE.md says: Main example flow emphasizes cached run path (main(load_cached=True)).
Code does: __main__ executes main(load_cached=False), forcing rebuild when run directly.
Severity: [MEDIUM]
```

```text
File: models/lstm_model.py
Location: prepare_lstm_a_sequences_temporal_split / prepare_lstm_b_sequences_temporal_split
CLAUDE.md says: Scaling policy is config-driven (SCALER_TYPE) via pipeline standardizer.
Code does: Hardcodes StandardScaler() inside sequence prep for LSTM-A/LSTM.
Severity: [MEDIUM]
```

```text
File: main.py
Location: LSTM-A dev-mode train_lstm_a call
CLAUDE.md says: LSTM-A architecture should follow LSTM-A tuning/defaults.
Code does: In dev-mode path, LSTM-A call uses hidden_size/num_layers/dropout from LSTM_B_* values.
Severity: [MEDIUM]
```

```text
File: main.py
Location: Sub-period analysis block
CLAUDE.md says: Sub-period section example computes metrics for Ensemble net returns.
Code does: Computes sub-period metrics only if LSTM-A exists, using LSTM-A net returns.
Severity: [MEDIUM]
```
