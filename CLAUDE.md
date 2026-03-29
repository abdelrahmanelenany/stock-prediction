# CLAUDE.md — Neural Networks for Stock Behavior Prediction
> Bachelor's Project · Complete Implementation Reference for IDE Copilot
> Based on Fischer & Krauss (2017), Krauss, Do & Huck (2017), and Bhandari et al. (2022)
> **Last Updated:** 2025-01-XX (Reflects current implementation)

---

## Project Summary

Build a **walk-forward validated, backtested long-short trading strategy** using ML models
(Logistic Regression, Random Forest, XGBoost, LSTM-A, LSTM-B, Ensemble) on **105 S&P 500 stocks** across
10 sectors over **2015–2024**. The pipeline predicts each stock's probability of outperforming
the cross-sectional median return the next day, ranks stocks by that probability, and constructs
an equal-weighted long-short portfolio. Features include **10 technical indicators** (including multi-day momentum)
with **causal wavelet denoising** (Bhandari extension). All performance is reported net of transaction costs.

**Target hardware:** MacBook Air M4 (CPU / MPS backend for PyTorch).

---

## Repository Layout

```
stock_prediction/
├── config.py                  # All hyperparameters and constants (single source of truth)
├── main.py                    # Orchestrates the full pipeline end-to-end
├── requirements.txt           # Python dependencies
├── data/
│   ├── raw/
│   │   ├── ohlcv_raw.csv      # Multi-index download from yfinance
│   │   └── ohlcv_long.csv     # Restructured: Date, Ticker, OHLCV
│   └── processed/
│       └── features.csv       # All features + target (cached)
├── pipeline/
│   ├── data_loader.py         # Download + clean + save raw data
│   ├── features.py            # Compute 10 technical features + causal wavelet denoising
│   ├── targets.py             # Binary cross-sectional median target
│   ├── walk_forward.py        # Walk-forward fold generator (train/val/test)
│   └── standardizer.py        # Standard/MinMax scaler (fit on train only)
├── models/
│   ├── baselines.py           # LogisticRegression, RandomForest, XGBoost (with grid search)
│   ├── lstm_model.py          # LSTM-A and LSTM-B architectures + hyperparameter tuning
│   └── calibration.py         # Probability calibration (isotonic/Platt)
├── backtest/
│   ├── signals.py             # Rank -> Long/Short/Hold signals (with smoothing & z-scoring)
│   ├── portfolio.py           # Daily P&L with transaction costs
│   └── metrics.py             # Sharpe, Sortino, MDD, Calmar, AUC, Daily AUC, sub-period analysis
├── analysis/
│   └── feature_correlation.py # Feature correlation analysis and selection
├── outputs/
│   ├── figures/               # Generated plots
│   └── feature_selection_log.txt
├── reports/                   # Output tables and CSVs
│   ├── table_T5_*.csv         # Gross/net returns
│   ├── table_T6_*.csv         # Sub-period performance
│   ├── table_T8_*.csv         # Classification metrics
│   ├── lstm_tuning_results.csv # LSTM hyperparameter tuning results
│   ├── daily_returns_*.csv    # Daily return series
│   ├── signals_all_models.csv # All model signals
│   └── backtest_summary.txt   # Human-readable summary
└── notebooks/                 # Visualization & reporting (placeholder)
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
```python
import torch
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")  # Should print: mps
```

---

## config.py — Single Source of Truth

```python
# config.py — Current Implementation

# =============================================================================
# 1. UNIVERSE: 105 S&P 500 stocks across 10 sectors
# =============================================================================
TICKERS = [
    # Technology (20)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'ADBE', 'CRM', 'INTC', 'CSCO',
    'NFLX', 'ORCL', 'AVGO', 'QCOM', 'TXN', 'AMD', 'IBM', 'NOW', 'INTU', 'AMAT',
    # Finance (15)
    'JPM', 'V', 'MA', 'BRK-B', 'GS', 'BAC', 'WFC', 'MS', 'AXP', 'C',
    'BLK', 'SCHW', 'CME', 'ICE', 'USB',
    # Healthcare (15)
    'JNJ', 'UNH', 'ABT', 'MRK', 'PFE', 'LLY', 'ABBV', 'TMO', 'DHR', 'BMY',
    'AMGN', 'GILD', 'MDT', 'ISRG', 'CVS',
    # Consumer Discretionary (12)
    'TSLA', 'HD', 'NKE', 'MCD', 'SBUX', 'TGT', 'LOW', 'TJX', 'BKNG', 'MAR',
    'ORLY', 'CMG',
    # Consumer Staples (8)
    'PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'CL',
    # Communication Services (6)
    'DIS', 'CMCSA', 'TMUS', 'VZ', 'T', 'CHTR',
    # Energy (8)
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO',
    # Industrials (10)
    'HON', 'CAT', 'UPS', 'GE', 'RTX', 'BA', 'DE', 'LMT', 'UNP', 'FDX',
    # Utilities (4)
    'NEE', 'DUK', 'SO', 'D',
    # Real Estate (4)
    'AMT', 'PLD', 'CCI', 'EQIX',
    # Materials (3)
    'LIN', 'APD', 'SHW',
]

START_DATE = '2015-01-01'
END_DATE   = '2024-12-31'

# =============================================================================
# 2. WALK-FORWARD FOLD STRUCTURE
# =============================================================================
TRAIN_DAYS = 500   # ~2 years
VAL_DAYS   = 125   # ~6 months (hyperparameter tuning)
TEST_DAYS  = 125   # ~6 months (out-of-sample evaluation)

# =============================================================================
# 3. SEQUENCE CONFIG
# =============================================================================
SEQ_LEN = 60  # LSTM lookback window (trading days)

# =============================================================================
# 4. TRADING
# =============================================================================
K_STOCKS = 10  # Number of long / short positions per day
TC_BPS   = 5   # Transaction cost per half-turn in basis points (0.0005)

# Signal generation parameters
SIGNAL_SMOOTH_ALPHA = 0.3              # EMA smoothing for turnover reduction
SIGNAL_CONFIDENCE_THRESHOLD = 0.0      # Pure ranking (no threshold)
SIGNAL_USE_ZSCORE = True               # Cross-sectional z-scoring
MIN_HOLDING_DAYS = 5                   # Minimum days to hold position

# =============================================================================
# 5. RANDOM SEED
# =============================================================================
RANDOM_SEED = 42

# =============================================================================
# 6. FEATURE SETS (10 FEATURES TOTAL)
# =============================================================================
ALL_FEATURE_COLS = [
    "Return_1d",        # Daily return
    "Return_5d",        # Weekly momentum (NEW)
    "Return_21d",       # Monthly momentum (NEW)
    "RSI_14",           # Relative Strength Index
    "MACD",             # Moving Average Convergence Divergence
    "ATR_14",           # Average True Range
    "BB_PctB",          # Bollinger Band %B
    "RealVol_20d",      # Realized volatility
    "Volume_Ratio",     # Volume vs 20-day average
    "SectorRelReturn",  # Leave-one-out sector-relative return
]

# LSTM-A features (6 features - includes momentum)
LSTM_A_FEATURE_COLS = [
    "MACD", "RSI_14", "ATR_14",
    "Return_1d", "Return_5d", "Return_21d"
]

# LSTM-B features (8 features - includes momentum)
LSTM_B_FEATURE_COLS = [
    "Return_1d", "Return_5d", "Return_21d",
    "RSI_14", "BB_PctB", "RealVol_20d",
    "Volume_Ratio", "SectorRelReturn"
]

# Baseline models use LSTM-B features
BASELINE_FEATURE_COLS = LSTM_B_FEATURE_COLS

TARGET_COL = 'Target'

# =============================================================================
# 7. MODEL REGISTRY
# =============================================================================
MODELS = ['LR', 'RF', 'XGBoost', 'LSTM-A', 'LSTM-B']
# Also computed: Ensemble (mean of all 5 models)

# =============================================================================
# 8. HYPERPARAMETER GRIDS
# =============================================================================

# LSTM-A: Two-phase tuning (Bhandari §3.3)
LSTM_A_ARCH_GRID = {
    "hidden_size": [16, 32, 64],
    "num_layers":  [1, 2],
    "dropout":     [0.1, 0.2],
}
LSTM_HYPERPARAM_GRID = {
    "optimizer":      ["adam", "adagrad", "nadam"],
    "learning_rate":  [0.1, 0.01, 0.001],
    "batch_size":     [32, 64, 128],
}
LSTM_TUNE_REPLICATES = 3    # Replicate count for tuning
LSTM_TUNE_PATIENCE   = 5    # Early stopping during tuning
LSTM_TUNE_MAX_EPOCHS = 50   # Max epochs during tuning

# LSTM-A: Training settings
LSTM_A_OPTIMIZER     = 'adam'
LSTM_A_LR            = 0.001
LSTM_A_BATCH         = 128
LSTM_A_MAX_EPOCHS    = 200
LSTM_A_PATIENCE      = 15
LSTM_A_VAL_SPLIT     = 0.2

# LSTM-B: Fixed architecture (no tuning)
LSTM_B_HIDDEN_SIZE   = 64
LSTM_B_NUM_LAYERS    = 2
LSTM_B_DROPOUT       = 0.2
LSTM_B_OPTIMIZER     = 'adam'
LSTM_B_LR            = 0.001
LSTM_B_BATCH         = 128
LSTM_B_MAX_EPOCHS    = 200
LSTM_B_PATIENCE      = 15
LSTM_B_LR_PATIENCE   = 7     # ReduceLROnPlateau patience
LSTM_B_LR_FACTOR     = 0.5   # ReduceLROnPlateau factor
LSTM_B_VAL_SPLIT     = 0.2

# Shared LSTM settings
LSTM_WD              = 1e-5  # Weight decay

# XGBoost grid
XGB_PARAM_GRID = {
    'max_depth':  [3, 4, 5],
    'eta':        [0.01],
    'subsample':  [0.6, 0.7],
}
XGB_COLSAMPLE    = 0.5   # Reduced from 0.7 for more regularization
XGB_ROUNDS       = 1000
XGB_EARLY_STOP   = 50
XGB_REG_ALPHA    = 0.1   # L1 regularization
XGB_REG_LAMBDA   = 1.0   # L2 regularization

# Random Forest grid
RF_PARAM_GRID = {
    'n_estimators':     [300, 500],
    'max_depth':        [5, 10, 15],
    'min_samples_leaf': [30, 50],
}

# Logistic Regression
LR_C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# =============================================================================
# 9. WAVELET DENOISING (Bhandari §4.5) — CAUSAL IMPLEMENTATION
# =============================================================================
USE_WAVELET_DENOISING = True     # Set False for ablation study
WAVELET_TYPE          = "haar"   # Paper uses Haar wavelets
WAVELET_LEVEL         = 1        # Decomposition level
WAVELET_MODE          = "soft"   # Soft thresholding (vs 'hard')
WAVELET_WINDOW_SIZE   = 128      # CAUSAL: lookback window for denoising

# =============================================================================
# 10. NORMALIZATION
# =============================================================================
SCALER_TYPE = "standard"   # Options: "standard" (Z-score) | "minmax"

# =============================================================================
# 11. FEATURE SELECTION (Bhandari §4.4)
# =============================================================================
FEATURE_CORR_THRESHOLD = 0.80   # Drop features with |r| > threshold

# After running analysis/feature_correlation.py:
FEATURE_COLS_AFTER_SELECTION = [
    'Return_1d', 'Return_5d', 'Return_21d', 'RSI_14', 'MACD',
    'ATR_14', 'RealVol_20d', 'Volume_Ratio', 'SectorRelReturn'
]

# =============================================================================
# 12. SECTOR MAPPING
# =============================================================================
SECTOR_MAP = {
    # Technology
    'AAPL': 'Tech', 'MSFT': 'Tech', 'GOOGL': 'Tech', 'AMZN': 'Tech', 'NVDA': 'Tech',
    'META': 'Tech', 'ADBE': 'Tech', 'CRM': 'Tech', 'INTC': 'Tech', 'CSCO': 'Tech',
    'NFLX': 'Tech', 'ORCL': 'Tech', 'AVGO': 'Tech', 'QCOM': 'Tech', 'TXN': 'Tech',
    'AMD': 'Tech', 'IBM': 'Tech', 'NOW': 'Tech', 'INTU': 'Tech', 'AMAT': 'Tech',
    # Finance
    'JPM': 'Finance', 'V': 'Finance', 'MA': 'Finance', 'BRK-B': 'Finance', 'GS': 'Finance',
    'BAC': 'Finance', 'WFC': 'Finance', 'MS': 'Finance', 'AXP': 'Finance', 'C': 'Finance',
    'BLK': 'Finance', 'SCHW': 'Finance', 'CME': 'Finance', 'ICE': 'Finance', 'USB': 'Finance',
    # Healthcare
    'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'ABT': 'Healthcare', 'MRK': 'Healthcare',
    'PFE': 'Healthcare', 'LLY': 'Healthcare', 'ABBV': 'Healthcare', 'TMO': 'Healthcare',
    'DHR': 'Healthcare', 'BMY': 'Healthcare', 'AMGN': 'Healthcare', 'GILD': 'Healthcare',
    'MDT': 'Healthcare', 'ISRG': 'Healthcare', 'CVS': 'Healthcare',
    # Consumer Discretionary
    'TSLA': 'Consumer', 'HD': 'Consumer', 'NKE': 'Consumer', 'MCD': 'Consumer',
    'SBUX': 'Consumer', 'TGT': 'Consumer', 'LOW': 'Consumer', 'TJX': 'Consumer',
    'BKNG': 'Consumer', 'MAR': 'Consumer', 'ORLY': 'Consumer', 'CMG': 'Consumer',
    # Consumer Staples
    'PG': 'Staples', 'KO': 'Staples', 'PEP': 'Staples', 'COST': 'Staples',
    'WMT': 'Staples', 'PM': 'Staples', 'MO': 'Staples', 'CL': 'Staples',
    # Communication Services
    'DIS': 'Comm', 'CMCSA': 'Comm', 'TMUS': 'Comm', 'VZ': 'Comm', 'T': 'Comm', 'CHTR': 'Comm',
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
    'EOG': 'Energy', 'MPC': 'Energy', 'PSX': 'Energy', 'VLO': 'Energy',
    # Industrials
    'HON': 'Industrial', 'CAT': 'Industrial', 'UPS': 'Industrial', 'GE': 'Industrial',
    'RTX': 'Industrial', 'BA': 'Industrial', 'DE': 'Industrial', 'LMT': 'Industrial',
    'UNP': 'Industrial', 'FDX': 'Industrial',
    # Utilities
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'D': 'Utilities',
    # Real Estate
    'AMT': 'REIT', 'PLD': 'REIT', 'CCI': 'REIT', 'EQIX': 'REIT',
    # Materials
    'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials',
}
```

---

## Step 1 — Data Download & Cleaning (`pipeline/data_loader.py`)

**Status:** ✅ Implemented

```python
import yfinance as yf
import pandas as pd
from config import TICKERS, START_DATE, END_DATE

def download_and_save():
    raw = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True)
    raw.to_csv('data/raw/ohlcv_raw.csv')

    # Restructure from MultiIndex -> long format
    panels = {}
    for ticker in TICKERS:
        df = raw.xs(ticker, axis=1, level=1).copy()
        df['Ticker'] = ticker
        panels[ticker] = df

    data = pd.concat(panels.values()).reset_index()
    data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']
    data = data.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    # Cleaning rules
    data['Return_1d'] = data.groupby('Ticker')['Close'].pct_change(1)
    data = data[data['Volume'] > 0]                          # remove non-trading days
    data = data.groupby('Ticker').apply(
        lambda g: g.ffill(limit=2)
    ).reset_index(drop=True)
    data.dropna(subset=['Close'], inplace=True)

    data.to_csv('data/raw/ohlcv_long.csv', index=False)
    print(f"Saved {len(data)} rows. Missing: {data.isna().sum().sum()}")
    return data
```

**Data quality rules:**
- `auto_adjust=True` handles splits and dividends automatically.
- Forward-fill gaps <= 2 days; drop rows if streak > 3 days.
- Remove Volume = 0 days (non-trading).
- Flag `|Return_1d| > 20%` for manual review; keep unless clearly erroneous.

---

## Step 2 — Feature Engineering (`pipeline/features.py`)

**Status:** ✅ Implemented with 10 features (including momentum)

### 2a. 10 Technical Features

| Feature | Formula | Economic Intuition |
|---|---|---|
| Return_1d | 1-day simple return | Daily momentum |
| Return_5d | 5-day simple return | **Weekly momentum (NEW)** |
| Return_21d | 21-day simple return | **Monthly momentum (NEW)** |
| RSI_14 | Relative Strength Index, 14-day | Overbought / oversold |
| MACD | 12d EMA - 26d EMA | Trend change momentum |
| ATR_14 | Average True Range, 14d | Daily volatility |
| BB_PctB | (Close-Lower)/(Upper-Lower), 20d | Price position in Bollinger band |
| RealVol_20d | Annualized 20-day realized volatility | Volatility regime |
| Volume_Ratio | Volume / 20d avg Volume | Unusual activity |
| SectorRelReturn | Stock return - leave-one-out sector mean | Cross-sectional relative performance |

### 2b. Causal Wavelet Denoising (Bhandari Extension - CRITICAL FIX)

**Status:** ✅ Implemented with CAUSAL rolling-window denoising

Haar wavelet soft denoising is applied to Close prices using a **rolling window approach**
to prevent look-ahead bias:

```python
def denoise_close_price_causal(close_series: pd.Series, threshold: float,
                                window_size: int = 128,
                                wavelet: str = "haar",
                                level: int = 1,
                                mode: str = "soft") -> pd.Series:
    """
    CAUSAL wavelet denoising: each value at time t uses only a fixed lookback window.

    This prevents look-ahead bias by ensuring the denoised value at time t
    depends only on prices from times [t - window_size, t], not future data.
    """
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
```

**Critical anti-leakage measures:**
1. Thresholds computed from **training data only** per fold
2. Causal denoising: each value uses only **historical data** (rolling window)
3. Applied to each split **independently** (no concatenation)
4. **Target remains based on raw returns** (not denoised) for honest evaluation

**Feature column names (10 total):**
```python
ALL_FEATURE_COLS = [
    'Return_1d', 'Return_5d', 'Return_21d',  # Momentum features
    'RSI_14', 'MACD', 'ATR_14',              # Technical indicators
    'BB_PctB', 'RealVol_20d', 'Volume_Ratio', 'SectorRelReturn'  # Volatility & cross-sectional
]
TARGET_COL = 'Target'
```

---

## Step 3 — Target Variable (`pipeline/targets.py`)

**Status:** ✅ Implemented

Binary label: **1 if stock's next-day return >= cross-sectional median, else 0**.
This is identical to both papers. Creates a ~50/50 balanced dataset by construction.

```python
def create_targets(data: pd.DataFrame, return_col: str = 'Return_1d') -> pd.DataFrame:
    data = data.copy().sort_values(['Date', 'Ticker'])

    # Next-day return for each stock
    data['Return_NextDay'] = data.groupby('Ticker')[return_col].shift(-1)

    # Cross-sectional median per date
    daily_median = data.groupby('Date')['Return_NextDay'].transform('median')

    # Binary label
    data['Target'] = (data['Return_NextDay'] >= daily_median).astype(int)
    data.dropna(subset=['Return_NextDay'], inplace=True)

    print("Class distribution:")
    print(data['Target'].value_counts(normalize=True))  # Expect ~50/50
    return data
```

---

## Step 4 — Walk-Forward Fold Generator (`pipeline/walk_forward.py`)

**Status:** ✅ Implemented

```
Timeline (10 years ~ 2520 trading days):

Fold 1: |---Train 500---|---Val 125---|---Test 125---|
Fold 2:                 |---Train 500---|---Val 125---|---Test 125---|
...
Fold N:                                                                   |---Train 500---|---Val 125---|---Test 125---|

Expect 12-14 folds total (2015-2024 data).
```

```python
def generate_walk_forward_folds(dates_sorted, train_days=500, val_days=125, test_days=125):
    total = len(dates_sorted)
    window = train_days + val_days + test_days
    folds = []
    start = 0
    while start + window <= total:
        train_end = start + train_days
        val_end   = train_end + val_days
        test_end  = val_end + test_days
        folds.append({
            'fold': len(folds) + 1,
            'train': (start, train_end),
            'val':   (train_end, val_end),
            'test':  (val_end, test_end),
            'train_start_date': dates_sorted[start],
            'test_end_date':    dates_sorted[test_end - 1],
        })
        start += test_days   # Roll forward by exactly one test period
    print(f"Generated {len(folds)} folds")
    return folds
```

---

## Step 5 — Feature Standardization (`pipeline/standardizer.py`)

**Status:** ✅ Implemented with dual-scaler approach

**CRITICAL:** Always fit the scaler on training data only. Never use test/val statistics.
Supports both StandardScaler (default) and MinMaxScaler.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def standardize_fold(X_train, X_val, X_test, scaler_type='standard'):
    """Fit scaler on training data, transform all splits."""
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X_train_s = scaler.fit_transform(X_train)   # fit + transform
    X_val_s   = scaler.transform(X_val)         # transform only
    X_test_s  = scaler.transform(X_test)        # transform only
    return X_train_s, X_val_s, X_test_s, scaler
```

**Dual Scaler Approach:** Separate scalers are fit for:
- LSTM-A: 6 features (MACD, RSI_14, ATR_14, Return_1d, Return_5d, Return_21d)
- LSTM-B/Baselines: 8 features (Return_1d, Return_5d, Return_21d, RSI_14, BB_PctB, RealVol_20d, Volume_Ratio, SectorRelReturn)

---

## Step 6 — Models

**Status:** ✅ All 5 models + Ensemble implemented

### Summary Table

| Model | Features | Architecture | Notes |
|---|---|---|---|
| LR | 8 (LSTM-B) | Logistic Regression | L2 regularization, CV grid search |
| RF | 8 (LSTM-B) | Random Forest | Val-based hyperparameter selection |
| XGBoost | 8 (LSTM-B) | Gradient Boosting | Early stopping on val AUC |
| LSTM-A | 6 (with momentum) | Hyperparameter-tuned | Two-phase Bhandari §3.3 tuning |
| LSTM-B | 8 (with momentum) | Fixed (64h, 2L, 0.2d) | No architecture search |
| Ensemble | All 5 | Mean probability | Average of all model predictions |

### 6a. Logistic Regression (`models/baselines.py`)

**Status:** ✅ Implemented

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

def train_logistic(X_train, y_train):
    param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    lr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)
    cv = GridSearchCV(lr, param_grid, cv=tscv, scoring='roc_auc', n_jobs=-1)
    cv.fit(X_train, y_train)
    print(f"Best C: {cv.best_params_['C']:.4f}, CV AUC: {cv.best_score_:.4f}")
    return cv.best_estimator_
```

### 6b. Random Forest (`models/baselines.py`)

**Status:** ✅ Implemented

Validation-based hyperparameter selection (not CV) to preserve time-series order.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def train_random_forest(X_train, y_train, X_val, y_val):
    best_auc, best_model = 0, None
    for n_est in [300, 500]:
        for depth in [5, 10, 15]:
            for min_leaf in [30, 50]:
                rf = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=depth,
                    min_samples_leaf=min_leaf,
                    max_features='sqrt',
                    n_jobs=-1,
                    random_state=42,
                )
                rf.fit(X_train, y_train)
                auc = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
                if auc > best_auc:
                    best_auc, best_model = auc, rf
    print(f"Best RF AUC: {best_auc:.4f}")
    return best_model
```

### 6c. XGBoost (`models/baselines.py`)

**Status:** ✅ Implemented

Grid search with early stopping on validation AUC.

```python
import xgboost as xgb

def train_xgboost(X_train, y_train, X_val, y_val):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)

    best_auc, best_model = 0, None
    for depth in [3, 4, 5]:
        for eta in [0.01]:
            for subsample in [0.6, 0.7]:
                params = {
                    'max_depth':        depth,
                    'eta':              eta,
                    'subsample':        subsample,
                    'colsample_bytree': 0.5,  # Reduced for regularization
                    'objective':        'binary:logistic',
                    'eval_metric':      'auc',
                    'alpha':            0.1,  # L1 regularization
                    'lambda':           1.0,  # L2 regularization
                    'seed':             42,
                }
                model = xgb.train(
                    params, dtrain,
                    num_boost_round=1000,
                    evals=[(dval, 'val')],
                    early_stopping_rounds=50,
                    verbose_eval=False,
                )
                auc = model.best_score
                if auc > best_auc:
                    best_auc, best_model = auc, model
    print(f"Best XGB AUC: {best_auc:.4f}")
    return best_model
```

### 6d. LSTM-A and LSTM-B (`models/lstm_model.py`)

**Status:** ✅ Implemented with temporal split and optional tuning

**Two LSTM variants** following Bhandari et al. (2022):

| Model | Features | Architecture | Tuning |
|---|---|---|---|
| LSTM-A | 6 (MACD, RSI_14, ATR_14, Return_1d, Return_5d, Return_21d) | Hyperparameter-tuned | Two-phase grid search |
| LSTM-B | 8 (Return_1d, Return_5d, Return_21d, RSI_14, BB_PctB, RealVol_20d, Volume_Ratio, SectorRelReturn) | Fixed | 64 hidden, 2 layers, 0.2 dropout |

**Architecture (StockLSTMTunable):**

```python
import torch
import torch.nn as nn

class StockLSTMTunable(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)  # 2-class output (CrossEntropyLoss)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # last timestep only
        out = self.relu(self.fc1(out))
        return self.fc2(out)  # logits
```

**LSTM-A Two-Phase Tuning (Bhandari §3.3):**

1. **Phase 1:** optimizer (adam/adagrad/nadam), learning rate (0.1/0.01/0.001), batch size (32/64/128)
2. **Phase 2:** hidden_size (16/32/64), num_layers (1/2), dropout (0.1/0.2)

**Temporal Split (Critical Fix):**

Instead of ticker-based splitting, we now use **temporal splitting** to ensure proper
train/val/test chronological separation:

```python
def prepare_lstm_a_sequences_temporal_split(df_train_fold, df_test_fold, val_ratio=0.2):
    """
    Build LSTM sequences with TEMPORAL train/val split (not ticker-based).

    This fixes the bug where ticker-based splits could leak future data
    into training when tickers had different date ranges.
    """
    # Get chronological split point for train/val
    dates = sorted(df_train_fold['Date'].unique())
    val_split_idx = int(len(dates) * (1 - val_ratio))
    val_split_date = dates[val_split_idx]

    df_train = df_train_fold[df_train_fold['Date'] < val_split_date]
    df_val = df_train_fold[df_train_fold['Date'] >= val_split_date]

    # Build sequences (with train as historical lookback for val/test)
    ...
```

### 6e. Ensemble Model

**Status:** ✅ Implemented in main.py

The ensemble model averages predictions from all 5 base models:

```python
ensemble_cols = ['Prob_LR', 'Prob_RF', 'Prob_XGB', 'Prob_LSTM_A', 'Prob_LSTM_B']
ensemble_preds['Prob_ENS'] = ensemble_preds[ensemble_cols].mean(axis=1)
```

---

## Step 7 — Trading Signals (`backtest/signals.py`)

**Status:** ✅ Implemented with smoothing and holding constraints

Signal generation with:
- Cross-sectional z-scoring for robust ranking
- EMA smoothing to reduce turnover
- Minimum holding period constraint

```python
def smooth_probabilities(preds_df, prob_col, alpha=0.3):
    """Apply per-ticker exponential smoothing to reduce turnover."""
    preds_df = preds_df.sort_values(['Ticker', 'Date']).copy()
    smoothed_col = f'{prob_col}_Smooth'

    parts = []
    for ticker, group in preds_df.groupby('Ticker'):
        g = group.sort_values('Date').copy()
        g[smoothed_col] = g[prob_col].ewm(alpha=alpha, adjust=False).mean()
        parts.append(g)

    return pd.concat(parts)


def apply_holding_period_constraint(signals_df, min_hold_days=5):
    """Enforce minimum holding period to reduce turnover."""
    # Once a position is entered, must hold for min_hold_days
    # before allowing exit or flip
    ...


def generate_signals(predictions_df, k=10, prob_col=None,
                    use_cross_sectional_z=True):
    """
    Generate Long/Short/Hold signals with optional z-scoring.

    With use_cross_sectional_z=True, probabilities are z-scored
    within each day's cross-section before ranking.
    """
    results = []
    for date, group in predictions_df.groupby('Date'):
        g = group.copy()

        # Compute score (ensemble or single model)
        if prob_col is None:
            prob_cols = [c for c in g.columns if c.startswith('Prob_')]
            g['Score'] = g[prob_cols].mean(axis=1)
        else:
            g['Score'] = g[prob_col]

        # Cross-sectional z-scoring
        if use_cross_sectional_z:
            g['Score'] = (g['Score'] - g['Score'].mean()) / (g['Score'].std() + 1e-10)

        # Rank and assign signals
        g = g.sort_values('Score', ascending=False)
        g['Signal'] = 'Hold'
        g.iloc[:k, g.columns.get_loc('Signal')] = 'Long'
        g.iloc[-k:, g.columns.get_loc('Signal')] = 'Short'

        results.append(g)

    return pd.concat(results)
```

---

## Step 8 — Portfolio & Transaction Costs (`backtest/portfolio.py`)

**Status:** ✅ Implemented

```python
def compute_portfolio_returns(signals_df, tc_bps=5, k=10):
    """
    Daily equal-weighted long-short portfolio return with transaction costs.

    Transaction cost model:
    - TC per half-turn (buy or sell)
    - Long<->Short flip = 2 half-turns
    - Each position change affects 1/(2*k) of portfolio
    """
    tc_per_turn = tc_bps / 10_000
    daily = []
    prev_signals = {}

    for date, group in signals_df.groupby('Date'):
        longs  = group[group['Signal'] == 'Long']
        shorts = group[group['Signal'] == 'Short']

        long_ret  = longs['Return_NextDay'].mean()  if len(longs)  > 0 else 0.0
        short_ret = shorts['Return_NextDay'].mean() if len(shorts) > 0 else 0.0
        gross_ret = long_ret - short_ret

        # Count position changes
        curr = dict(zip(group['Ticker'], group['Signal']))
        turnover = 0
        for ticker, signal in curr.items():
            prev = prev_signals.get(ticker, 'Hold')
            if signal != prev:
                if (signal == 'Long' and prev == 'Short') or \
                   (signal == 'Short' and prev == 'Long'):
                    turnover += 2  # flip = 2 half-turns
                else:
                    turnover += 1  # enter or exit = 1 half-turn

        # TC as fraction of portfolio affected
        tc = turnover * tc_per_turn / (2 * k)
        net_ret = gross_ret - tc
        prev_signals = curr

        daily.append({
            'Date': date,
            'Gross_Return': gross_ret,
            'Net_Return': net_ret,
            'Long_Return': long_ret,
            'Short_Return': short_ret,
            'TC': tc,
            'Turnover': turnover,
        })

    return pd.DataFrame(daily).set_index('Date')
```

---

## Step 9 — Performance Metrics (`backtest/metrics.py`)

**Status:** ✅ Implemented with Daily AUC

```python
def compute_metrics(returns_series, rf_daily=0.00015):
    """Compute risk-return metrics."""
    r = returns_series.dropna()
    mean_d   = r.mean()
    std_d    = r.std()
    excess   = r - rf_daily
    sharpe   = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else 0

    downside = r[r < rf_daily]
    if len(downside) > 1 and downside.std() > 0:
        sortino  = ((mean_d - rf_daily) / downside.std()) * np.sqrt(252)
    else:
        sortino = np.inf if (mean_d - rf_daily) > 0 else 0.0

    cum      = (1 + r).cumprod()
    max_dd   = ((cum - cum.cummax()) / cum.cummax()).min()
    ann_ret  = (1 + mean_d) ** 252 - 1

    return {
        'N Days':                 len(r),
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


def evaluate_classification(y_true, y_prob, threshold=0.5):
    """Binary classification metrics."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        'Accuracy (%)': round(accuracy_score(y_true, y_pred) * 100, 2),
        'AUC-ROC':      round(roc_auc_score(y_true, y_prob), 4),
        'F1 Score':     round(f1_score(y_true, y_pred), 4),
    }


def compute_daily_auc(predictions_df, prob_col, target_col='Target'):
    """
    Compute average AUC-ROC per day (cross-sectional ranking quality).

    This metric is more appropriate for ranking strategies than pooled AUC
    because it measures within-day ranking ability, which is what the
    signal generation actually uses.

    NOTE: Pooled AUC may differ from daily AUC because pooled AUC measures
    global ranking while daily AUC measures within-day ranking.
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

    return {
        'Daily AUC (mean)': round(np.mean(daily_aucs), 4),
        'Daily AUC (std)': round(np.std(daily_aucs), 4),
        'Days with valid AUC': len(daily_aucs),
    }
```

---

## Step 10 — main.py Pipeline Overview

**Status:** ✅ Implemented with ensemble and tuning options

The main pipeline orchestrates the full walk-forward validation process:

```python
# main.py (simplified overview)

from config import *
from pipeline.data_loader import download_and_save
from pipeline.features import (
    build_feature_matrix, compute_wavelet_thresholds,
    apply_wavelet_denoising_causal, recompute_features_from_denoised
)
from pipeline.targets import create_targets
from pipeline.walk_forward import generate_walk_forward_folds
from models.baselines import train_logistic, train_random_forest, train_xgboost
from models.lstm_model import train_lstm_a, train_lstm_b, tune_lstm_hyperparams
from backtest.signals import generate_signals, smooth_probabilities, apply_holding_period_constraint
from backtest.portfolio import compute_portfolio_returns
from backtest.metrics import compute_metrics, evaluate_classification, compute_daily_auc

import torch
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

ENABLE_LSTM_TUNING = False  # Set True for Phase 1+2 hyperparameter tuning

def main(load_cached=True):
    """
    Parameters
    ----------
    load_cached : bool
        If True (default), load from data/processed/features.csv.
        Set False to re-download and recompute.
    """
    set_global_seed(config.RANDOM_SEED)

    # 1. Load data (from cache or download fresh)
    data = pd.read_csv('data/processed/features.csv', parse_dates=['Date'])

    # 2. Generate walk-forward folds
    dates = sorted(data['Date'].unique())
    folds = generate_walk_forward_folds(dates, TRAIN_DAYS, VAL_DAYS, TEST_DAYS)

    # 3. Train models per fold
    all_preds = []
    tuning_results = []

    for fold in folds:
        df_tr = data[data['Date'].isin(dates[fold['train'][0]:fold['train'][1]])]
        df_v  = data[data['Date'].isin(dates[fold['val'][0]:fold['val'][1]])]
        df_ts = data[data['Date'].isin(dates[fold['test'][0]:fold['test'][1]])]

        # CAUSAL wavelet denoising per-fold
        if config.USE_WAVELET_DENOISING:
            thresholds = compute_wavelet_thresholds(df_tr)
            df_tr = apply_wavelet_denoising_causal(df_tr, thresholds)
            df_v  = apply_wavelet_denoising_causal(df_v, thresholds)
            df_ts = apply_wavelet_denoising_causal(df_ts, thresholds)

            # Recompute features (but keep original targets)
            df_tr = recompute_features_from_denoised(df_tr)
            df_v  = recompute_features_from_denoised(df_v)
            df_ts = recompute_features_from_denoised(df_ts)

        # Standardize (dual scaler approach)
        X_tr_b_s, X_v_b_s, X_ts_b_s, _ = standardize_fold(
            df_tr[BASELINE_FEATURE_COLS].values,
            df_v[BASELINE_FEATURE_COLS].values,
            df_ts[BASELINE_FEATURE_COLS].values,
        )

        y_tr = df_tr[TARGET_COL].values
        y_v  = df_v[TARGET_COL].values

        # Train baseline models
        lr_m  = train_logistic(X_tr_b_s, y_tr)
        rf_m  = train_random_forest(X_tr_b_s, y_tr, X_v_b_s, y_v)
        xgb_m = train_xgboost(X_tr_b_s, y_tr, X_v_b_s, y_v)

        # Train LSTM-A (with optional tuning)
        df_train_fold = pd.concat([df_tr, df_v])
        df_test_fold = df_ts.copy()

        X_tr_a, y_tr_a, X_val_a, y_val_a, X_te_a, y_te_a, _, _, keys_te_a = \
            prepare_lstm_a_sequences_temporal_split(df_train_fold, df_test_fold)

        if ENABLE_LSTM_TUNING:
            best_hp_a = tune_lstm_hyperparams(X_tr_a, y_tr_a, X_val_a, y_val_a, device)
            tuning_results.append({'fold': fold['fold'], 'model': 'LSTM-A', **best_hp_a})
            model_a = train_lstm_a(X_tr_a, y_tr_a, X_val_a, y_val_a, device, **best_hp_a)
        else:
            model_a = train_lstm_a(X_tr_a, y_tr_a, X_val_a, y_val_a, device)

        probs_a = predict_lstm(model_a, X_te_a, device)

        # Train LSTM-B (fixed architecture)
        X_tr_b, y_tr_b, X_val_b, y_val_b, X_te_b, y_te_b, _, _, keys_te_b = \
            prepare_lstm_b_sequences_temporal_split(df_train_fold, df_test_fold)

        model_b = train_lstm_b(X_tr_b, y_tr_b, X_val_b, y_val_b, device)
        probs_b = predict_lstm(model_b, X_te_b, device)

        # Collect predictions for this fold
        pred = df_ts.copy()
        pred['Prob_LR'] = lr_m.predict_proba(X_ts_b_s)[:, 1]
        pred['Prob_RF'] = rf_m.predict_proba(X_ts_b_s)[:, 1]
        pred['Prob_XGB'] = xgb_m.predict(xgb.DMatrix(X_ts_b_s))
        pred['Prob_LSTM_A'] = align_predictions_to_df(probs_a, keys_te_a, df_ts)
        pred['Prob_LSTM_B'] = align_predictions_to_df(probs_b, keys_te_b, df_ts)
        pred['Fold'] = fold['fold']

        all_preds.append(pred)

    # 4. Combine folds
    full_preds = pd.concat(all_preds)

    # 5. Backtest each model + Ensemble
    model_cols = {
        'LR': 'Prob_LR',
        'RF': 'Prob_RF',
        'XGBoost': 'Prob_XGB',
        'LSTM-A': 'Prob_LSTM_A',
        'LSTM-B': 'Prob_LSTM_B',
    }

    port_returns_gross = {}
    port_returns_net_5 = {}
    class_metrics = []
    all_signals = []

    for model_name, prob_col in model_cols.items():
        valid_preds = full_preds.dropna(subset=[prob_col])

        # Apply signal smoothing
        smoothed_preds = smooth_probabilities(valid_preds, prob_col, alpha=SIGNAL_SMOOTH_ALPHA)
        smoothed_col = f'{prob_col}_Smooth'

        # Generate signals with z-scoring
        sig_df = generate_signals(smoothed_preds, k=K_STOCKS, prob_col=smoothed_col)

        # Apply minimum holding period
        sig_df = apply_holding_period_constraint(sig_df, min_hold_days=MIN_HOLDING_DAYS)

        sig_df['Model'] = model_name
        all_signals.append(sig_df)

        # Compute portfolio returns
        port_gross = compute_portfolio_returns(sig_df, tc_bps=0, k=K_STOCKS)
        port_net_5 = compute_portfolio_returns(sig_df, tc_bps=5, k=K_STOCKS)

        port_returns_gross[model_name] = port_gross
        port_returns_net_5[model_name] = port_net_5

        # Classification metrics (with Daily AUC)
        y_true = valid_preds[TARGET_COL].values
        y_prob = valid_preds[prob_col].values
        cm = evaluate_classification(y_true, y_prob)
        daily_auc = compute_daily_auc(valid_preds, prob_col, TARGET_COL)
        cm.update(daily_auc)
        cm['Model'] = model_name
        class_metrics.append(cm)

    # 6. Ensemble model
    ensemble_cols = ['Prob_LR', 'Prob_RF', 'Prob_XGB', 'Prob_LSTM_A', 'Prob_LSTM_B']
    ensemble_preds = full_preds.dropna(subset=ensemble_cols).copy()
    ensemble_preds['Prob_ENS'] = ensemble_preds[ensemble_cols].mean(axis=1)

    # Apply smoothing and holding period
    ensemble_smoothed = smooth_probabilities(ensemble_preds, 'Prob_ENS', alpha=SIGNAL_SMOOTH_ALPHA)
    sig_df_ens = generate_signals(ensemble_smoothed, k=K_STOCKS, prob_col='Prob_ENS_Smooth')
    sig_df_ens = apply_holding_period_constraint(sig_df_ens, min_hold_days=MIN_HOLDING_DAYS)
    sig_df_ens['Model'] = 'Ensemble'
    all_signals.append(sig_df_ens)

    # Compute ensemble returns
    port_gross_ens = compute_portfolio_returns(sig_df_ens, tc_bps=0, k=K_STOCKS)
    port_net_5_ens = compute_portfolio_returns(sig_df_ens, tc_bps=5, k=K_STOCKS)

    port_returns_gross['Ensemble'] = port_gross_ens
    port_returns_net_5['Ensemble'] = port_net_5_ens

    # Ensemble classification metrics
    y_true_ens = ensemble_preds[TARGET_COL].values
    y_prob_ens = ensemble_preds['Prob_ENS'].values
    cm_ens = evaluate_classification(y_true_ens, y_prob_ens)
    daily_auc_ens = compute_daily_auc(ensemble_preds, 'Prob_ENS', TARGET_COL)
    cm_ens.update(daily_auc_ens)
    cm_ens['Model'] = 'Ensemble'
    class_metrics.append(cm_ens)

    # 7. Save all results
    save_all_results(
        results_dict={
            'gross': [compute_metrics(port_returns_gross[m]['Gross_Return'])
                     for m in list(model_cols.keys()) + ['Ensemble']],
            'net_5': [compute_metrics(port_returns_net_5[m]['Net_Return'])
                     for m in list(model_cols.keys()) + ['Ensemble']],
            'classification': class_metrics,
            'subperiod': compute_subperiod_metrics(port_returns_net_5['Ensemble']['Net_Return']),
        },
        daily_returns_dict={...},
        signals_dict=pd.concat(all_signals),
        tuning_results=tuning_results,
        reports_dir='reports'
    )

    return {
        'full_preds': full_preds,
        'port_returns': port_returns_net_5,
        'class_metrics': class_metrics,
        'folds': folds,
    }

if __name__ == '__main__':
    main(load_cached=True)
```

**Key Implementation Highlights:**

1. **Causal wavelet denoising** per-fold with training-only thresholds
2. **Dual scaler approach** for LSTM-A (6 features) vs LSTM-B/baselines (8 features)
3. **Temporal LSTM sequence splitting** (not ticker-based) to prevent leakage
4. **Signal smoothing** (EMA) and **minimum holding period** to reduce turnover
5. **Per-model backtest** + **ensemble** for comprehensive comparison
6. **Daily AUC metric** to diagnose within-day ranking quality
7. **Fold-specific seeds** for reproducible per-fold results

---

## Anti-Leakage Rules

| Rule | How to enforce |
|---|---|
| Standardize on training data only | `scaler.fit_transform(X_train)`, then `.transform()` on val/test |
| Wavelet thresholds from train only | `compute_wavelet_thresholds(df_train)`, apply to all splits |
| **CAUSAL wavelet denoising** | **Rolling window: each value uses only historical data** |
| Never shuffle time-series | `shuffle=False` for all val/test DataLoaders |
| Tune hyperparameters on val, not test | All early stopping uses val loss only |
| Features use only data <= t | All pandas rolling windows are causal (no `center=True`) |
| Target uses only realized t+1 returns | `shift(-1)` on actual realized close prices |
| No test period feedback | Test fold is evaluated only, never trained on |
| **Temporal LSTM split** | **Split by date, not by ticker** (prevents leakage across time) |
| **Target unchanged after denoising** | **Targets remain based on raw returns** (honest evaluation) |

---

## Required Thesis Outputs

### Tables

| # | Title | Output File | Status |
|---|---|---|---|
| T1 | Descriptive stats per stock | Manual analysis | 📝 TODO |
| T2 | Walk-forward fold dates | `print_fold_summary()` | ✅ |
| T3 | Hyperparameter configurations | config.py | ✅ |
| T4 | Daily return characteristics | `daily_returns_*.csv` | ✅ |
| T5 | Annualized risk-return metrics | `table_T5_*.csv` | ✅ |
| T6 | Sub-period performance | `table_T6_*.csv` | ✅ |
| T7 | TC sensitivity | `compute_tc_sensitivity()` | ✅ |
| T8 | Classification metrics | `table_T8_*.csv` | ✅ (with Daily AUC) |
| T9 | Feature importance | `rf.feature_importances_` | 📝 TODO |

### Figures

| # | Title | How to create | Status |
|---|---|---|---|
| F1 | Walk-forward timeline | matplotlib gantt chart | 📝 TODO |
| F2 | LSTM architecture diagram | matplotlib / PowerPoint | 📝 TODO |
| F3 | Cumulative equity curves | `(1 + r).cumprod()` | 📝 TODO |
| F4 | Drawdown underwater chart | Rolling max drawdown | 📝 TODO |
| F5 | Feature importance bar chart | `rf.feature_importances_` | 📝 TODO |
| F6 | Return distribution | seaborn.violinplot | 📝 TODO |
| F7 | Sharpe vs TC | Loop over TC values | 📝 TODO |
| F8 | Confusion matrices | sklearn + seaborn | 📝 TODO |
| F9 | Sub-period performance | seaborn.barplot | 📝 TODO |
| F10 | Correlation heatmap | seaborn.heatmap | 📝 TODO |
| F11 | LSTM train/val loss curves | Save epoch losses | 📝 TODO |
| F12 | Feature correlation heatmap | `analysis/feature_correlation.py` | 📝 TODO |

### Generated Reports

| File | Description | Status |
|---|---|---|
| `reports/backtest_summary.txt` | Human-readable performance summary | ✅ |
| `reports/signals_all_models.csv` | All model signals for analysis | ✅ |
| `reports/lstm_tuning_results.csv` | LSTM hyperparameter tuning results | ✅ |
| `outputs/feature_selection_log.txt` | Feature correlation analysis log | ✅ |

---

## Sub-Period Analysis

| Period | Dates | Regime |
|---|---|---|
| Pre-COVID | 2019-01-01 to 2020-02-19 | Normal bull market |
| COVID crash | 2020-02-20 to 2020-04-30 | Extreme volatility / dislocation |
| Recovery / bull | 2020-05-01 to 2021-12-31 | Low rates, momentum-driven |
| 2022 bear market | 2022-01-01 to 2022-12-31 | Rate hikes, drawdowns |
| 2023-2024 AI rally | 2023-01-01 to 2024-12-31 | AI / large-cap concentration |

Compute `compute_subperiod_metrics()` and report in T6 / F9.

---

## Hyperparameter Reference

### LSTM-A (Bhandari-style tuning)

| Parameter | Search Grid | Selected via |
|---|---|---|
| Optimizer | adam, adagrad, nadam | Phase 1 grid search |
| Learning rate | 0.1, 0.01, 0.001 | Phase 1 grid search |
| Batch size | 32, 64, 128 | Phase 1 grid search |
| Hidden size | 16, 32, 64 | Phase 2 grid search |
| Num layers | 1, 2 | Phase 2 grid search |
| Dropout | 0.1, 0.2 | Phase 2 grid search |

### LSTM-B (Fixed architecture)

| Parameter | Value |
|---|---|
| Hidden size | 64 |
| Num layers | 2 |
| Dropout | 0.2 |
| Learning rate | 0.001 |
| Batch size | 128 |
| Optimizer | Adam |
| Sequence length | 60 |
| Max epochs | 200 |
| Early stopping patience | 15 |
| LR scheduler patience | 7 |
| LR scheduler factor | 0.5 |

### XGBoost

| Parameter | Grid | Default |
|---|---|---|
| max_depth | 3, 4, 5 | 4 |
| eta | 0.01 | 0.01 |
| subsample | 0.6, 0.7 | 0.7 |
| colsample_bytree | - | 0.5 |
| alpha (L1) | - | 0.1 |
| lambda (L2) | - | 1.0 |
| num_boost_round | 1000 (early stop) | - |
| early_stopping_rounds | 50 | - |

### Random Forest

| Parameter | Grid |
|---|---|
| n_estimators | 300, 500 |
| max_depth | 5, 10, 15 |
| min_samples_leaf | 30, 50 |
| max_features | sqrt |

### Logistic Regression

| Parameter | Grid |
|---|---|
| C (inverse regularization) | 1e-4, 1e-3, ..., 1e2 |
| penalty | L2 |
| CV | TimeSeriesSplit(5) |

### Trading

| Parameter | Value | Notes |
|---|---|---|
| k (long/short stocks) | 10 | 10 per side out of 105 |
| TC (bps) | 5 | Per half-turn |
| Signal smoothing alpha | 0.3 | EMA for turnover reduction |
| Z-score normalization | True | Cross-sectional |
| Minimum holding days | 5 | Enforce holding period |

---

## Implementation Status Summary

### ✅ Completed

1. **Data Pipeline**
   - Download & cleaning (yfinance)
   - 10 technical features (with multi-day momentum)
   - Causal wavelet denoising (rolling window)
   - Binary target (cross-sectional median)
   - Walk-forward fold generator

2. **Models**
   - Logistic Regression (CV grid search)
   - Random Forest (validation-based selection)
   - XGBoost (early stopping)
   - LSTM-A (with optional 2-phase tuning)
   - LSTM-B (fixed architecture)
   - Ensemble (mean of 5 models)

3. **Backtest**
   - Signal generation (with z-scoring)
   - Probability smoothing (EMA)
   - Minimum holding period constraint
   - Transaction cost model
   - Comprehensive metrics (Sharpe, Sortino, MDD, Calmar, Win Rate, VaR)
   - Daily AUC (within-day ranking quality)
   - Sub-period analysis

4. **Reports**
   - Table T5 (gross/net returns)
   - Table T6 (sub-period performance)
   - Table T8 (classification + Daily AUC)
   - Daily returns CSV
   - Signals CSV
   - LSTM tuning results CSV
   - Human-readable summary

### 📝 TODO

1. **Visualization**
   - All 12 figures (F1-F12)
   - Feature importance plots
   - Cumulative return curves
   - Drawdown charts
   - LSTM loss curves

2. **Analysis**
   - Descriptive statistics (Table T1)
   - Feature importance (Table T9)
   - TC sensitivity analysis plots

3. **Documentation**
   - Thesis write-up
   - Results interpretation
   - Ablation study findings

---

## Realistic Expectations

- **Directional accuracy:** 51-54% is a solid result (random = 50%). Do not expect the papers' 53.8%.
- **Sharpe ratio:** With 105 stocks and k=10, better diversification than originally planned (10 stocks).
  Still expect lower Sharpe than the papers' 5.83 (which used 500 stocks).
- **LSTM-A vs LSTM-B:** LSTM-A with 6 features may perform differently than LSTM-B with 8 features.
  Both include momentum features for improved predictive power.
- **Ensemble advantage:** Ensemble may outperform individual models by averaging away model-specific biases.
- **Wavelet denoising impact:** May improve or harm performance — report both scenarios.
- **Transaction cost sensitivity:** With k=10 positions per side, turnover is more manageable.
  Signal smoothing and minimum holding period further reduce turnover.
- **COVID crash (Feb-Apr 2020):** Expect elevated returns — extreme cross-sectional dispersion
  creates exploitable patterns.
- **2022 bear market:** Expect elevated drawdown. Analyze separately in sub-period section.
- **Sector diversification:** The 105-stock universe spans 10 sectors, providing natural
  diversification. The SectorRelReturn feature captures relative performance within sectors.

---

## Key Technical Improvements

### 1. Causal Wavelet Denoising

**Problem:** Original implementation applied wavelet denoising to entire time series, leaking future
data into training.

**Solution:** Rolling-window denoising where each value at time t uses only data from [t-window_size, t].

### 2. Temporal LSTM Splitting

**Problem:** Ticker-based train/val splits could leak future data when tickers had different date ranges.

**Solution:** Temporal splitting by date ensures chronological separation.

### 3. Daily AUC Metric

**Problem:** Pooled AUC measures global ranking but strategy uses within-day ranking.

**Solution:** Added daily AUC metric to measure cross-sectional ranking quality per day.

### 4. Signal Smoothing & Holding Constraints

**Problem:** High turnover from daily position changes increased transaction costs.

**Solution:** EMA smoothing (α=0.3) + minimum holding period (5 days) to reduce turnover.

### 5. Ensemble Model

**Problem:** Individual models may have biases or overfitting in different regimes.

**Solution:** Ensemble averages 5 model predictions to improve robustness.

---

## Key References

- Fischer, T. & Krauss, C. (2017). *Deep learning with long short-term memory networks for financial market predictions.* FAU Discussion Papers in Economics, No. 11/2017.
- Krauss, C., Do, X.A. & Huck, N. (2017). *Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500.* European Journal of Operational Research, 259, 689-702.
- Bhandari, H.N., et al. (2022). *Predicting stock market index using LSTM.* Machine Learning with Applications.
- Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.
- Fama, E. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work. *Journal of Finance*, 25(2), 383-417.
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.
- Donoho, D.L. & Johnstone, I.M. (1994). Ideal spatial adaptation by wavelet shrinkage. *Biometrika*, 81(3), 425-455.

---

## Changelog

### 2025-01-XX — Major Implementation Update

**Added:**
- Multi-day momentum features (Return_5d, Return_21d)
- Causal wavelet denoising with rolling windows
- Temporal LSTM sequence splitting
- Daily AUC metric for within-day ranking quality
- Signal smoothing (EMA) to reduce turnover
- Minimum holding period constraint (5 days)
- Ensemble model (mean of 5 base models)
- LSTM hyperparameter tuning (optional, Bhandari §3.3)

**Changed:**
- Feature count: 8 → 10 (added momentum)
- LSTM-A features: 4 → 6 (added momentum)
- LSTM-B features: 6 → 8 (added momentum)
- Wavelet: non-causal → causal (rolling window)
- LSTM split: ticker-based → temporal (date-based)

**Fixed:**
- Look-ahead bias in wavelet denoising
- Data leakage in LSTM sequence preparation
- High turnover in signal generation

---

**END OF CLAUDE.MD**
