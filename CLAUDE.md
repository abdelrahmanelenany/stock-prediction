# CLAUDE.md — Neural Networks for Stock Behavior Prediction
> Bachelor's Project · Complete Implementation Reference for IDE Copilot
> Based on Fischer & Krauss (2017), Krauss, Do & Huck (2017), and Bhandari et al. (2022)

---

## Project Summary

Build a **walk-forward validated, backtested long-short trading strategy** using ML models
(Logistic Regression, Random Forest, XGBoost, LSTM-A, LSTM-B) on **105 S&P 500 stocks** across
10 sectors over **2015–2024**. The pipeline predicts each stock's probability of outperforming
the cross-sectional median return the next day, ranks stocks by that probability, and constructs
an equal-weighted long-short portfolio. Features include **8 technical indicators** with optional
**wavelet denoising** (Bhandari extension). All performance is reported net of transaction costs.

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
│   ├── features.py            # Compute 8 technical features + wavelet denoising
│   ├── targets.py             # Binary cross-sectional median target
│   ├── walk_forward.py        # Walk-forward fold generator (train/val/test)
│   └── standardizer.py        # Standard/MinMax scaler (fit on train only)
├── models/
│   ├── baselines.py           # LogisticRegression, RandomForest, XGBoost (with grid search)
│   ├── lstm_model.py          # LSTM-A and LSTM-B architectures + hyperparameter tuning
│   └── calibration.py         # Probability calibration (isotonic/Platt)
├── backtest/
│   ├── signals.py             # Rank -> Long/Short/Hold signals (with z-scoring)
│   ├── portfolio.py           # Daily P&L with transaction costs
│   └── metrics.py             # Sharpe, Sortino, MDD, Calmar, AUC, sub-period analysis
├── analysis/
│   └── feature_correlation.py # Feature correlation analysis and selection
├── outputs/
│   ├── figures/               # Generated plots
│   └── feature_selection_log.txt
├── reports/                   # Output tables and CSVs
│   ├── table_T5_*.csv         # Gross/net returns
│   ├── table_T6_*.csv         # Sub-period performance
│   ├── table_T8_*.csv         # Classification metrics
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
# config.py

# =============================================================================
# 1. UNIVERSE: 105 S&P 500 stocks across 10 sectors
# =============================================================================
TICKERS = [
    # Tech (15)
    'AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'CRM', 'CSCO', 'ACN', 'AMD', 'ADBE',
    'TXN', 'INTC', 'QCOM', 'IBM', 'INTU',
    # Finance (12)
    'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'AXP', 'C', 'BLK', 'SPGI', 'CB',
    # Healthcare (12)
    'UNH', 'JNJ', 'LLY', 'PFE', 'MRK', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN', 'CVS',
    # Consumer (10)
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'CMG',
    # Staples (8)
    'PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'CL',
    # Communication (8)
    'META', 'GOOGL', 'GOOG', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T',
    # Energy (8)
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO',
    # Industrial (12)
    'GE', 'CAT', 'HON', 'UNP', 'UPS', 'RTX', 'BA', 'LMT', 'DE', 'MMM', 'GD', 'FDX',
    # Utilities (8)
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL',
    # REIT (6)
    'PLD', 'AMT', 'EQIX', 'SPG', 'PSA', 'O',
    # Materials (6)
    'LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM',
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

# =============================================================================
# 5. RANDOM SEED
# =============================================================================
RANDOM_SEED = 42

# =============================================================================
# 6. FEATURE SETS
# =============================================================================
ALL_FEATURE_COLS = [
    'Return_1d', 'RSI_14', 'MACD', 'ATR_14',
    'BB_PctB', 'RealVol_20d', 'Volume_Ratio', 'SectorRelReturn'
]

# LSTM-A features (Bhandari-inspired: 4 features)
LSTM_A_FEATURE_COLS = ['MACD', 'RSI_14', 'ATR_14', 'Return_1d']

# LSTM-B features (6 features)
LSTM_B_FEATURE_COLS = [
    'Return_1d', 'RSI_14', 'BB_PctB',
    'RealVol_20d', 'Volume_Ratio', 'SectorRelReturn'
]

# Baseline models use LSTM-B features
BASELINE_FEATURE_COLS = LSTM_B_FEATURE_COLS

TARGET_COL = 'Target'

# =============================================================================
# 7. MODEL REGISTRY
# =============================================================================
MODEL_REGISTRY = ['LR', 'RF', 'XGBoost', 'LSTM-A', 'LSTM-B']

# =============================================================================
# 8. HYPERPARAMETER GRIDS
# =============================================================================

# LSTM-A: Two-phase tuning (Bhandari sect 3.3)
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

# LSTM-B: Fixed architecture
LSTM_B_HIDDEN   = 64
LSTM_B_LAYERS   = 2
LSTM_B_DROPOUT  = 0.2
LSTM_B_LR       = 0.001
LSTM_B_BATCH    = 128
LSTM_MAX_EPOCHS = 200
LSTM_PATIENCE   = 15

# XGBoost grid
XGB_PARAM_GRID = {
    'max_depth':  [3, 4, 5],
    'eta':        [0.01],
    'subsample':  [0.6, 0.7],
}
XGB_ROUNDS     = 1000
XGB_EARLY_STOP = 50

# Random Forest grid
RF_PARAM_GRID = {
    'n_estimators':     [300, 500],
    'max_depth':        [5, 10, 15],
    'min_samples_leaf': [30, 50],
}

# Logistic Regression
LR_C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# =============================================================================
# 9. WAVELET DENOISING (Bhandari sect 4.5)
# =============================================================================
USE_WAVELET_DENOISING = True
WAVELET_TYPE  = "haar"
WAVELET_LEVEL = 1
WAVELET_MODE  = "soft"

# =============================================================================
# 10. SIGNAL GENERATION
# =============================================================================
SIGNAL_SMOOTH_ALPHA = 0.3              # EMA smoothing for turnover reduction
SIGNAL_CONFIDENCE_THRESHOLD = 0.0      # Pure ranking (no threshold)
SIGNAL_USE_ZSCORE = True               # Cross-sectional z-scoring

# =============================================================================
# 11. SECTOR MAPPING (for SectorRelReturn feature)
# =============================================================================
SECTOR_MAP = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech', 'AVGO': 'Tech', ...  # (complete mapping in config.py)
}
```

---

## Step 1 — Data Download & Cleaning (`pipeline/data_loader.py`)

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

### 2a. 8 Technical Features

| Feature | Formula | Economic Intuition |
|---|---|---|
| Return_1d | 1-day simple return | Daily momentum |
| RSI_14 | Relative Strength Index, 14-day | Overbought / oversold |
| MACD | 12d EMA - 26d EMA | Trend change momentum |
| ATR_14 | Average True Range, 14d | Daily volatility |
| BB_PctB | (Close-Lower)/(Upper-Lower), 20d | Price position in Bollinger band |
| RealVol_20d | Annualized 20-day realized volatility | Volatility regime |
| Volume_Ratio | Volume / 20d avg Volume | Unusual activity |
| SectorRelReturn | Stock return - leave-one-out sector mean | Cross-sectional relative performance |

```python
def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values('Date')

    # Return_1d
    df['Return_1d'] = df['Close'].pct_change(1)

    # RSI(14)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI_14'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26

    # ATR(14)
    hl  = df['High'] - df['Low']
    hpc = (df['High'] - df['Close'].shift()).abs()
    lpc = (df['Low']  - df['Close'].shift()).abs()
    df['ATR_14'] = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()

    # Bollinger Band %B
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    df['BB_PctB'] = (df['Close'] - lower) / (upper - lower + 1e-10)

    # Realized Volatility (20-day annualized)
    log_ret = np.log(df['Close'] / df['Close'].shift(1))
    df['RealVol_20d'] = log_ret.rolling(20).std() * np.sqrt(252)

    # Volume Ratio
    df['Volume_Ratio'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1)

    return df


def compute_sector_rel_return(data: pd.DataFrame) -> pd.DataFrame:
    """Compute leave-one-out sector-relative return."""
    data = data.copy()
    data['Sector'] = data['Ticker'].map(SECTOR_MAP)

    for date in data['Date'].unique():
        mask = data['Date'] == date
        for sector in data.loc[mask, 'Sector'].unique():
            sector_mask = mask & (data['Sector'] == sector)
            sector_ret = data.loc[sector_mask, 'Return_1d']
            # Leave-one-out mean
            sector_sum = sector_ret.sum()
            sector_cnt = len(sector_ret)
            if sector_cnt > 1:
                data.loc[sector_mask, 'SectorRelReturn'] = (
                    sector_ret - (sector_sum - sector_ret) / (sector_cnt - 1)
                )
            else:
                data.loc[sector_mask, 'SectorRelReturn'] = 0.0
    return data
```

### 2b. Wavelet Denoising (Bhandari Extension)

Haar wavelet soft denoising is applied to Close prices to reduce noise before computing
Close-dependent features. **Critical:** Wavelet thresholds computed on training data only
(anti-leakage).

```python
import pywt
import numpy as np

def compute_wavelet_threshold(close_series: pd.Series) -> float:
    """Compute universal threshold from training data (Donoho & Johnstone)."""
    coeffs = pywt.wavedec(close_series.values, 'haar', level=1)
    detail = coeffs[-1]
    sigma = np.median(np.abs(detail)) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(close_series)))
    return threshold

def denoise_close_price(close_series: pd.Series, threshold: float) -> pd.Series:
    """Apply soft thresholding to wavelet coefficients."""
    coeffs = pywt.wavedec(close_series.values, 'haar', level=1)
    denoised_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    denoised = pywt.waverec(denoised_coeffs, 'haar')[:len(close_series)]
    return pd.Series(denoised, index=close_series.index)
```

**Feature column names (8 total):**
```python
ALL_FEATURE_COLS = [
    'Return_1d', 'RSI_14', 'MACD', 'ATR_14',
    'BB_PctB', 'RealVol_20d', 'Volume_Ratio', 'SectorRelReturn'
]
TARGET_COL = 'Target'
```

---

## Step 3 — Target Variable (`pipeline/targets.py`)

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

**Dual Scaler Approach:** When using both LSTM-A (4 features) and LSTM-B/baseline (6 features),
separate scalers are fit on each feature set to avoid dimension mismatch.

---

## Step 6 — Models

### 6a. Logistic Regression (`models/baselines.py`)

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
                    'colsample_bytree': 0.7,
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

**Two LSTM variants** following Bhandari et al. (2022):

| Model | Features | Architecture | Tuning |
|---|---|---|---|
| LSTM-A | 4 (MACD, RSI_14, ATR_14, Return_1d) | Hyperparameter-tuned | Two-phase grid search |
| LSTM-B | 6 (Return_1d, RSI_14, BB_PctB, RealVol_20d, Volume_Ratio, SectorRelReturn) | Fixed | 64 hidden, 2 layers, 0.2 dropout |

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

**LSTM-A Two-Phase Tuning (Bhandari sect 3.3):**

1. **Phase 1:** optimizer (adam/adagrad/nadam), learning rate (0.1/0.01/0.001), batch size (32/64/128)
2. **Phase 2:** hidden_size (16/32/64), num_layers (1/2), dropout (0.1/0.2)

```python
def tune_lstm_a(X_train, y_train, X_val, y_val, device):
    best_auc = 0
    best_config = {}

    # Phase 1: training hyperparameters
    for opt_name in ['adam', 'adagrad', 'nadam']:
        for lr in [0.1, 0.01, 0.001]:
            for batch in [32, 64, 128]:
                model = StockLSTMTunable(input_size=4, hidden_size=32, num_layers=1)
                auc = train_and_evaluate(model, X_train, y_train, X_val, y_val,
                                         optimizer=opt_name, lr=lr, batch_size=batch, device=device)
                if auc > best_auc:
                    best_auc = auc
                    best_config['optimizer'] = opt_name
                    best_config['lr'] = lr
                    best_config['batch_size'] = batch

    # Phase 2: architecture (using best Phase 1 config)
    for hidden in [16, 32, 64]:
        for layers in [1, 2]:
            for dropout in [0.1, 0.2]:
                model = StockLSTMTunable(input_size=4, hidden_size=hidden,
                                         num_layers=layers, dropout=dropout)
                auc = train_and_evaluate(model, X_train, y_train, X_val, y_val,
                                         **best_config, device=device)
                if auc > best_auc:
                    best_auc = auc
                    best_config.update({'hidden': hidden, 'layers': layers, 'dropout': dropout})

    return best_config
```

**Sequence Building:**

Training data is used as lookback history when building test sequences to ensure continuous
timeseries without gaps at fold boundaries.

```python
def prepare_lstm_sequences(df_train, df_test, feature_cols, target_col, seq_len=60):
    """Build sequences with train data as lookback for test predictions."""
    # Combine train+test for continuous sequences
    combined = pd.concat([df_train, df_test]).sort_values(['Ticker', 'Date'])

    X_train, y_train, X_test, y_test = [], [], [], []
    keys_train, keys_test = [], []

    for ticker in combined['Ticker'].unique():
        stock = combined[combined['Ticker'] == ticker].reset_index(drop=True)
        X = stock[feature_cols].values.astype(np.float32)
        y = stock[target_col].values.astype(np.float32)
        dates = stock['Date'].values

        train_len = len(df_train[df_train['Ticker'] == ticker])

        for i in range(seq_len, len(X)):
            if i < train_len:
                X_train.append(X[i-seq_len:i])
                y_train.append(y[i])
                keys_train.append((dates[i], ticker))
            else:
                X_test.append(X[i-seq_len:i])
                y_test.append(y[i])
                keys_test.append((dates[i], ticker))

    return (np.array(X_train), np.array(y_train),
            np.array(X_test), np.array(y_test),
            keys_train, keys_test)
```

### 6e. Probability Calibration (`models/calibration.py`)

Post-hoc calibration of model probabilities using isotonic regression or Platt scaling.

```python
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

class ProbabilityCalibrator:
    def __init__(self, method='isotonic'):
        self.method = method
        self.calibrator = None

    def fit(self, y_true, y_prob):
        if self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_prob, y_true)
        else:  # platt
            self.calibrator = LogisticRegression()
            self.calibrator.fit(y_prob.reshape(-1, 1), y_true)

    def transform(self, y_prob):
        if self.method == 'isotonic':
            return self.calibrator.predict(y_prob)
        else:
            return self.calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
```

Calibration is fit on validation probabilities per fold and applied to test predictions.

---

## Step 7 — Trading Signals (`backtest/signals.py`)

Signal generation with optional cross-sectional z-scoring and EMA smoothing.

```python
import pandas as pd
import numpy as np

def generate_signals(
    predictions_df: pd.DataFrame,
    k: int = 10,
    prob_col: str = None,
    use_zscore: bool = True,
    smooth_alpha: float = 0.3
) -> pd.DataFrame:
    """
    Generate Long/Short/Hold signals based on probability ranking.

    Parameters:
        predictions_df: DataFrame with Date, Ticker, and probability columns
        k: Number of long/short positions per day
        prob_col: Column to use for ranking (None = ensemble average)
        use_zscore: Apply cross-sectional z-scoring before ranking
        smooth_alpha: EMA smoothing factor (0 = no smoothing)

    Returns:
        DataFrame with Signal column ('Long', 'Short', 'Hold')
    """
    results = []
    prev_probs = {}

    for date, group in predictions_df.groupby('Date'):
        g = group.copy()

        # Compute score (ensemble or single model)
        if prob_col is None:
            prob_cols = [c for c in g.columns if c.startswith('Prob_')]
            g['Score'] = g[prob_cols].mean(axis=1)
        else:
            g['Score'] = g[prob_col]

        # Optional EMA smoothing to reduce turnover
        if smooth_alpha > 0:
            for i, row in g.iterrows():
                ticker = row['Ticker']
                if ticker in prev_probs:
                    g.loc[i, 'Score'] = (
                        smooth_alpha * g.loc[i, 'Score'] +
                        (1 - smooth_alpha) * prev_probs[ticker]
                    )
                prev_probs[row['Ticker']] = g.loc[i, 'Score']

        # Cross-sectional z-scoring
        if use_zscore:
            g['Score'] = (g['Score'] - g['Score'].mean()) / (g['Score'].std() + 1e-10)

        # Rank and assign signals
        g = g.sort_values('Score', ascending=False).reset_index(drop=True)
        g['Signal'] = 'Hold'
        g.loc[:k - 1, 'Signal'] = 'Long'
        g.loc[len(g) - k:, 'Signal'] = 'Short'

        results.append(g)

    return pd.concat(results)
```

---

## Step 8 — Portfolio & Transaction Costs (`backtest/portfolio.py`)

```python
import pandas as pd
import numpy as np

def compute_portfolio_returns(
    signals_df: pd.DataFrame,
    tc_bps: float = 5,
    k: int = 10
) -> pd.DataFrame:
    """
    Daily equal-weighted long-short portfolio return with transaction costs.

    Transaction cost model:
    - TC per half-turn (buy or sell)
    - Long<->Short flip = 2 half-turns
    - Each position change affects 1/(2*k) of portfolio

    Parameters:
        signals_df: DataFrame with Date, Ticker, Signal, Return_NextDay
        tc_bps: Cost per half-turn in basis points
        k: Number of positions per side (for weight calculation)
    """
    tc_per_turn = tc_bps / 10_000
    daily = []
    prev_signals = {}

    for date, group in signals_df.sort_values('Date').groupby('Date'):
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
        tc = turnover * tc_per_turn / (2 * k) if k > 0 else 0
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

```python
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

def compute_metrics(returns_series: pd.Series, rf_daily: float = 0.00015) -> dict:
    """
    Compute risk-return metrics.
    rf_daily ~ 3.8% annual / 252 trading days.
    """
    r = returns_series.dropna()
    mean_d   = r.mean()
    std_d    = r.std()
    excess   = r - rf_daily
    sharpe   = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else 0
    downside = r[r < rf_daily]
    sortino  = ((mean_d - rf_daily) / downside.std()) * np.sqrt(252) if len(downside) > 0 else 0
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
        'Calmar Ratio':           round(ann_ret / abs(max_dd), 3) if max_dd != 0 else 0,
        'Win Rate (%)':           round((r > 0).mean() * 100, 2),
        'VaR 1% (%)':             round(np.percentile(r, 1) * 100, 4),
    }

def evaluate_classification(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Compute classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'Accuracy (%)': round(accuracy_score(y_true, y_pred) * 100, 2),
        'AUC-ROC':      round(roc_auc_score(y_true, y_prob), 4),
        'F1 Score':     round(f1_score(y_true, y_pred), 4),
    }

def compute_subperiod_metrics(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics for each market regime period."""
    SUBPERIODS = {
        'Pre-COVID':      ('2019-01-01', '2020-02-19'),
        'COVID crash':    ('2020-02-20', '2020-04-30'),
        'Recovery/bull':  ('2020-05-01', '2021-12-31'),
        '2022 bear':      ('2022-01-01', '2022-12-31'),
        '2023-24 AI rally': ('2023-01-01', '2024-12-31'),
    }

    results = []
    for name, (start, end) in SUBPERIODS.items():
        subset = returns_df.loc[start:end]
        if len(subset) > 0:
            metrics = compute_metrics(subset['Net_Return'])
            metrics['Period'] = name
            results.append(metrics)

    return pd.DataFrame(results)

def compute_tc_sensitivity(signals_df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Compute performance across TC levels."""
    TC_GRID = [0, 2, 5, 10, 15, 20, 25, 30]
    results = []

    for tc in TC_GRID:
        from backtest.portfolio import compute_portfolio_returns
        port = compute_portfolio_returns(signals_df, tc_bps=tc, k=k)
        metrics = compute_metrics(port['Net_Return'])
        metrics['TC (bps)'] = tc
        results.append(metrics)

    return pd.DataFrame(results)
```

---

## Step 10 — main.py Pipeline Overview

The main pipeline orchestrates the full walk-forward validation process:

```python
# main.py (simplified overview)

from config import *
from pipeline.data_loader import download_and_save
from pipeline.features import (
    build_feature_matrix, compute_wavelet_thresholds,
    apply_wavelet_denoising, recompute_features_from_denoised
)
from pipeline.targets import create_targets
from pipeline.walk_forward import generate_walk_forward_folds
from pipeline.standardizer import standardize_fold
from models.baselines import train_logistic, train_random_forest, train_xgboost
from models.lstm_model import train_lstm_a, train_lstm_b, prepare_lstm_a_sequences, prepare_lstm_b_sequences
from backtest.signals import generate_signals
from backtest.portfolio import compute_portfolio_returns
from backtest.metrics import compute_metrics, evaluate_classification, compute_subperiod_metrics

import torch
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

ENABLE_LSTM_TUNING = False  # Set True for Phase 1+2 hyperparameter tuning

def main():
    # 1. Load data (from cache or download fresh)
    data = pd.read_csv('data/processed/features.csv', parse_dates=['Date'])

    # 2. Generate walk-forward folds
    dates = sorted(data['Date'].unique())
    folds = generate_walk_forward_folds(dates, TRAIN_DAYS, VAL_DAYS, TEST_DAYS)

    all_preds = []
    for fold in folds:
        print(f"\n=== FOLD {fold['fold']} ===")
        df_tr = data[data['Date'].isin(dates[fold['train'][0]:fold['train'][1]])]
        df_v  = data[data['Date'].isin(dates[fold['val'][0]:fold['val'][1]])]
        df_ts = data[data['Date'].isin(dates[fold['test'][0]:fold['test'][1]])]

        # 3. Wavelet denoising (if enabled)
        if USE_WAVELET_DENOISING:
            thresholds = compute_wavelet_thresholds(df_tr)
            df_tr = apply_wavelet_denoising(df_tr, thresholds)
            df_v  = apply_wavelet_denoising(df_v, thresholds)
            df_ts = apply_wavelet_denoising(df_ts, thresholds)
            # Recompute Close-dependent features and targets
            df_tr = recompute_features_from_denoised(df_tr)
            df_v  = recompute_features_from_denoised(df_v)
            df_ts = recompute_features_from_denoised(df_ts)

        # 4. Standardize (fit on train only)
        X_tr, X_v, X_ts, scaler = standardize_fold(
            df_tr[BASELINE_FEATURE_COLS].values,
            df_v[BASELINE_FEATURE_COLS].values,
            df_ts[BASELINE_FEATURE_COLS].values,
        )

        # 5. Train baseline models
        y_tr, y_v = df_tr[TARGET_COL].values, df_v[TARGET_COL].values
        lr_m  = train_logistic(X_tr, y_tr)
        rf_m  = train_random_forest(X_tr, y_tr, X_v, y_v)
        xgb_m = train_xgboost(X_tr, y_tr, X_v, y_v)

        # 6. Train LSTM-A (4 features)
        Xa_tr, ya_tr, Xa_ts, ya_ts, _, keys_a = prepare_lstm_a_sequences(
            df_tr, df_v, df_ts, LSTM_A_FEATURE_COLS, TARGET_COL, SEQ_LEN)
        lstm_a = train_lstm_a(Xa_tr, ya_tr, Xa_ts, device,
                              tune=ENABLE_LSTM_TUNING)

        # 7. Train LSTM-B (6 features)
        Xb_tr, yb_tr, Xb_ts, yb_ts, _, keys_b = prepare_lstm_b_sequences(
            df_tr, df_v, df_ts, LSTM_B_FEATURE_COLS, TARGET_COL, SEQ_LEN)
        lstm_b = train_lstm_b(Xb_tr, yb_tr, Xb_ts, device)

        # 8. Collect predictions
        pred = df_ts.copy()
        pred['Prob_LR']     = lr_m.predict_proba(X_ts)[:, 1]
        pred['Prob_RF']     = rf_m.predict_proba(X_ts)[:, 1]
        pred['Prob_XGBoost'] = xgb_m.predict(xgb.DMatrix(X_ts))
        # LSTM predictions aligned via keys_a, keys_b
        pred['Prob_LSTM-A'] = align_lstm_predictions(lstm_a, Xa_ts, keys_a, pred)
        pred['Prob_LSTM-B'] = align_lstm_predictions(lstm_b, Xb_ts, keys_b, pred)

        all_preds.append(pred)

    # 9. Backtest
    full_preds = pd.concat(all_preds)
    for model in MODEL_REGISTRY:
        prob_col = f'Prob_{model}'
        signals = generate_signals(full_preds, k=K_STOCKS, prob_col=prob_col)
        port = compute_portfolio_returns(signals, tc_bps=TC_BPS, k=K_STOCKS)
        print(f"\n=== {model} Performance ===")
        print(compute_metrics(port['Net_Return']))

    # 10. Save reports
    # - table_T5_*.csv (gross/net returns)
    # - table_T6_*.csv (sub-period performance)
    # - table_T8_*.csv (classification metrics)
    # - daily_returns_*.csv
    # - signals_all_models.csv
    # - backtest_summary.txt

if __name__ == '__main__':
    main()
```

**Key Implementation Notes:**

1. **Wavelet denoising** is applied per-fold with thresholds computed from training data only.
2. **Dual scaler approach:** LSTM-A uses 4 features, LSTM-B/baselines use 6 features.
3. **LSTM sequence alignment:** Train data is used as lookback history for test predictions.
4. **Per-model backtest:** Each model's performance is computed separately for comparison.

---

## Anti-Leakage Rules

| Rule | How to enforce |
|---|---|
| Standardize on training data only | `scaler.fit_transform(X_train)`, then `.transform()` on val/test |
| Wavelet thresholds from train only | `compute_wavelet_thresholds(df_train)`, apply to all splits |
| Never shuffle time-series | `shuffle=False` for all val/test DataLoaders |
| Tune hyperparameters on val, not test | All early stopping uses val loss only |
| Features use only data <= t | All pandas rolling windows are causal (no `center=True`) |
| Target uses only realized t+1 returns | `shift(-1)` on actual realized close prices |
| No test period feedback | Test fold is evaluated only, never trained on |
| LSTM lookback from train | Use train data as history when building test sequences |

---

## Required Thesis Outputs

### Tables

| # | Title | Output File |
|---|---|---|
| T1 | Descriptive stats per stock (mean, std, min, max, skew, kurt) | Manual analysis |
| T2 | Walk-forward fold dates | Printed during run |
| T3 | Hyperparameter configurations (all models) | config.py |
| T4 | Daily return characteristics — gross vs net | `daily_returns_*.csv` |
| T5 | Annualized risk-return metrics (all models) | `table_T5_gross_returns.csv`, `table_T5_net_returns_5bps.csv` |
| T6 | Sub-period performance breakdown | `table_T6_subperiod_performance.csv` |
| T7 | TC sensitivity (0-30 bps grid) | `compute_tc_sensitivity()` |
| T8 | Classification metrics: AUC, F1, Accuracy | `table_T8_classification_metrics.csv` |
| T9 | Top-15 most important features (RF/XGB) | `rf.feature_importances_` |

### Figures

| # | Title | How to create |
|---|---|---|
| F1 | Walk-forward timeline diagram | matplotlib gantt-style bar chart |
| F2 | LSTM architecture diagram | matplotlib / PowerPoint |
| F3 | Cumulative equity curves (all models + buy-and-hold) | `(1 + r).cumprod()` |
| F4 | Drawdown underwater chart | Rolling max drawdown, fill_between |
| F5 | Feature importance bar chart (top 8) | `rf.feature_importances_` / XGB SHAP |
| F6 | Return distribution (violin or box) | seaborn.violinplot per model |
| F7 | Sharpe ratio vs transaction cost (0-30 bps) | Loop over TC values |
| F8 | Confusion matrices (2x2 per model) | sklearn + seaborn heatmap |
| F9 | Sub-period performance bar chart | seaborn.barplot by sub-period |
| F10 | Correlation heatmap of stock returns | seaborn.heatmap(returns.corr()) |
| F11 | LSTM train/val loss curves per fold | Save epoch losses during training |
| F12 | Feature correlation heatmap | `analysis/feature_correlation.py` |

### Generated Reports

| File | Description |
|---|---|
| `reports/backtest_summary.txt` | Human-readable performance summary |
| `reports/signals_all_models.csv` | All model signals for analysis |
| `outputs/feature_selection_log.txt` | Feature correlation analysis log |

---

## Sub-Period Analysis

| Period | Dates | Regime |
|---|---|---|
| Pre-COVID | 2019-01-01 to 2020-02-19 | Normal bull market |
| COVID crash | 2020-02-20 to 2020-04-30 | Extreme volatility / dislocation |
| Recovery / bull | 2020-05-01 to 2021-12-31 | Low rates, momentum-driven |
| 2022 bear market | 2022-01-01 to 2022-12-31 | Rate hikes, drawdowns |
| 2023-2024 AI rally | 2023-01-01 to 2024-12-31 | AI / large-cap concentration |

Note: 2015-2018 data is used for initial training folds but not explicitly analyzed as a
sub-period since the walk-forward test periods begin in 2017.

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

### XGBoost

| Parameter | Grid | Default |
|---|---|---|
| max_depth | 3, 4, 5 | 4 |
| eta | 0.01 | 0.01 |
| subsample | 0.6, 0.7 | 0.7 |
| colsample_bytree | 0.7 | 0.7 |
| alpha (L1) | 0.1 | 0.1 |
| lambda (L2) | 1.0 | 1.0 |
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

---

## Realistic Expectations

- **Directional accuracy:** 51-54% is a solid result (random = 50%). Do not expect the papers' 53.8%.
- **Sharpe ratio:** With 105 stocks and k=10, better diversification than originally planned (10 stocks).
  Still expect lower Sharpe than the papers' 5.83 (which used 500 stocks).
- **LSTM-A vs LSTM-B:** LSTM-A with 4 features may underperform LSTM-B with 6 features due to
  limited information. Both may underperform ensemble models. This is a valid finding.
- **Wavelet denoising impact:** May improve or harm performance — report both scenarios.
- **Transaction cost sensitivity:** With k=10 positions per side, turnover is more manageable than
  with k=2. TC sensitivity analysis (T7, F7) remains important.
- **COVID crash (Feb-Apr 2020):** Expect elevated returns — extreme cross-sectional dispersion
  creates exploitable patterns.
- **2022 bear market:** Expect elevated drawdown. Analyze separately in sub-period section.
- **Sector diversification:** The 105-stock universe spans 10 sectors, providing natural
  diversification. The SectorRelReturn feature captures relative performance within sectors.

---

## Key References

- Fischer, T. & Krauss, C. (2017). *Deep learning with long short-term memory networks for financial market predictions.* FAU Discussion Papers in Economics, No. 11/2017.
- Krauss, C., Do, X.A. & Huck, N. (2017). *Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500.* European Journal of Operational Research, 259, 689-702.
- Bhandari, H.N., et al. (2022). *Predicting stock market index using LSTM.* Machine Learning with Applications.
- Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.
- Fama, E. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work. *Journal of Finance*, 25(2), 383-417.
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.
- Avellaneda, M. & Lee, J.H. (2010). Statistical arbitrage in the US equities market. *Quantitative Finance*, 10(7), 761-782.
- Donoho, D.L. & Johnstone, I.M. (1994). Ideal spatial adaptation by wavelet shrinkage. *Biometrika*, 81(3), 425-455.