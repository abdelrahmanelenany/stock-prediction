# CLAUDE.md — Neural Networks for Stock Behavior Prediction
> Bachelor's Project · Complete Implementation Reference for IDE Copilot
> Based on Fischer & Krauss (2017) and Krauss, Do & Huck (2017)

---

## Project Summary

Build a **walk-forward validated, backtested long-short trading strategy** using ML models
(Logistic Regression, Random Forest, XGBoost, LSTM) on the top 10 S&P 500 stocks over
2019–2024. The pipeline predicts each stock's probability of outperforming the cross-sectional
median return the next day, ranks stocks by that probability, and constructs an equal-weighted
long-short portfolio. All performance is reported net of transaction costs.

**Target hardware:** MacBook Air M4 (CPU / MPS backend for PyTorch).

---

## Repository Layout

```
stock_prediction/
├── config.py                  # All hyperparameters and constants (single source of truth)
├── main.py                    # Orchestrates the full pipeline end-to-end
├── data/
│   ├── raw/
│   │   ├── ohlcv_raw.csv      # Multi-index download from yfinance
│   │   └── ohlcv_long.csv     # Restructured: Date, Ticker, OHLCV
│   └── processed/
│       └── features.csv       # All features + target, ready for fold splitting
├── pipeline/
│   ├── data_loader.py         # Download + clean + save raw data
│   ├── features.py            # Compute 31 lagged returns + 10 technicals
│   ├── targets.py             # Binary cross-sectional median target
│   ├── walk_forward.py        # Fold generator (train/val/test splits)
│   └── standardizer.py       # Fit-on-train-only StandardScaler helper
├── models/
│   ├── baselines.py           # LogisticRegression, RandomForest, XGBoost
│   ├── lstm_model.py          # PyTorch LSTM architecture + Dataset class
│   └── ensemble.py            # Equal-weighted probability averaging
├── backtest/
│   ├── signals.py             # Rank -> Long/Short/Hold signals
│   ├── portfolio.py           # Daily P&L with transaction costs
│   └── metrics.py             # Sharpe, Sortino, MDD, Calmar, AUC, Accuracy
├── notebooks/
│   └── results.ipynb          # Visualization + reporting
└── reports/
    └── figures/               # All saved .png figures for the thesis
```

---

## Environment Setup

```bash
# Core data & ML
pip install yfinance pandas numpy scikit-learn xgboost

# PyTorch (Apple Silicon MPS support)
pip install torch torchvision

# Technical indicators (choose one)
pip install pandas-ta        # Preferred: pure Python, no C dependency
# pip install ta-lib-easy TA-Lib  # Alternative if TA-Lib C is available

# Visualization & stats
pip install matplotlib seaborn plotly scipy statsmodels

# Portfolio analytics (optional but useful)
pip install pyfolio-reloaded
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
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JPM', 'V']
START_DATE = '2019-01-01'
END_DATE   = '2024-12-31'

# Walk-forward fold structure
TRAIN_DAYS = 500   # ~2 years
VAL_DAYS   = 125   # ~6 months (hyperparameter tuning)
TEST_DAYS  = 125   # ~6 months (out-of-sample evaluation)

# Feature config
SEQ_LEN       = 60    # LSTM lookback window (trading days)
LAGGED_RETURN_PERIODS = list(range(1, 21)) + [40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
N_RETURN_FEATURES     = 31
N_TECH_FEATURES       = 10
N_TOTAL_FEATURES      = N_RETURN_FEATURES + N_TECH_FEATURES  # 41

# Trading
K_STOCKS = 2   # Number of long / short positions per day (use 3 if n_stocks >= 9)
TC_BPS   = 5   # Transaction cost per half-turn in basis points (0.0005)

# LSTM hyperparameters
LSTM_HIDDEN      = 64
LSTM_LAYERS      = 2
LSTM_DROPOUT     = 0.2
LSTM_LR          = 0.001
LSTM_BATCH       = 128
LSTM_MAX_EPOCHS  = 200
LSTM_PATIENCE    = 15

# Random Forest
RF_N_ESTIMATORS = 500
RF_MAX_DEPTH    = 20

# XGBoost
XGB_MAX_DEPTH    = 4
XGB_ETA          = 0.05
XGB_SUBSAMPLE    = 0.7
XGB_COLSAMPLE    = 0.5
XGB_ROUNDS       = 500
XGB_EARLY_STOP   = 30

RANDOM_SEED = 42
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

### 2a. 31 Lagged Cumulative Returns (from both papers)

```python
import pandas as pd
import numpy as np
from config import LAGGED_RETURN_PERIODS

def add_lagged_returns(data: pd.DataFrame) -> pd.DataFrame:
    """R(m) = P_t / P_{t-m} - 1 (simple cumulative return over m days)."""
    data = data.sort_values(['Ticker', 'Date'])
    for m in LAGGED_RETURN_PERIODS:
        data[f'Return_{m}d'] = data.groupby('Ticker')['Close'].pct_change(m)
    return data
```

Period list: `[1,2,...,20, 40,60,80,100,120,140,160,180,200,220,240]` — **31 features**.

### 2b. 10 Technical Indicators (your extension over the papers)

| Feature | Formula | Economic Intuition |
|---|---|---|
| RSI_14 | Relative Strength Index, 14-day | Overbought / oversold |
| MACD | 12d EMA - 26d EMA | Trend change momentum |
| MACD_Signal | 9d EMA of MACD | Entry/exit timing |
| BB_Width | (Upper-Lower)/Middle, 20d | Volatility regime |
| BB_PctB | (Close-Lower)/(Upper-Lower) | Price in band |
| ATR_14 | Average True Range, 14d | Daily volatility |
| OBV | On-Balance Volume cumsum | Volume-price trend |
| Volume_Ratio | Volume / 20d avg Volume | Unusual activity |
| HL_Pct_5d | Rolling 5d avg of (High-Low)/Close | Range expansion |
| Mom_10d | Close/Close(10) - 1 | Raw 10-day momentum |

```python
def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values('Date')

    # RSI(14)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI_14'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    df['BB_Width'] = (upper - lower) / sma20
    df['BB_PctB'] = (df['Close'] - lower) / (upper - lower + 1e-10)

    # ATR(14)
    hl  = df['High'] - df['Low']
    hpc = (df['High'] - df['Close'].shift()).abs()
    lpc = (df['Low']  - df['Close'].shift()).abs()
    df['ATR_14'] = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()

    # OBV
    direction = np.sign(df['Close'].diff()).fillna(0)
    df['OBV'] = (direction * df['Volume']).cumsum()

    # Volume Ratio
    df['Volume_Ratio'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1)

    # High-Low %
    df['HL_Pct_5d'] = ((df['High'] - df['Low']) / df['Close']).rolling(5).mean()

    # 10-day Momentum
    df['Mom_10d'] = df['Close'].pct_change(10)

    return df


def build_feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
    data = add_lagged_returns(data)
    enriched = []
    for ticker, group in data.groupby('Ticker'):
        enriched.append(compute_technical_features(group))
    result = pd.concat(enriched).sort_values(['Date', 'Ticker'])
    result.dropna(inplace=True)
    result.to_csv('data/processed/features.csv', index=False)
    return result
```

**Feature column names (41 total):**
```python
FEATURE_COLS = (
    [f'Return_{m}d' for m in LAGGED_RETURN_PERIODS] +
    ['RSI_14', 'MACD', 'MACD_Signal', 'BB_Width', 'BB_PctB',
     'ATR_14', 'OBV', 'Volume_Ratio', 'HL_Pct_5d', 'Mom_10d']
)
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
Timeline (5 years ~ 1260 trading days):

Fold 1: |---Train 500---|---Val 125---|---Test 125---|
Fold 2:                 |---Train 500---|---Val 125---|---Test 125---|
Fold 3:                                |---Train 500---|---Val 125---|---Test 125---|
Fold 4:                                               |---Train 500---|---Val 125---|---Test 125---|

Expect 3-4 folds total.
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

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

def standardize_fold(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)   # fit + transform
    X_val_s   = scaler.transform(X_val)         # transform only
    X_test_s  = scaler.transform(X_test)        # transform only
    return X_train_s, X_val_s, X_test_s, scaler
```

---

## Step 6 — Models

### 6a. Logistic Regression (`models/baselines.py`)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def train_logistic(X_train, y_train):
    param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    lr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
    cv = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    cv.fit(X_train, y_train)
    print(f"Best C: {cv.best_params_['C']:.4f}, CV AUC: {cv.best_score_:.4f}")
    return cv.best_estimator_
```

### 6b. Random Forest (`models/baselines.py`)

```python
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=500,     # paper uses 1000; 500 is fine for 10 stocks
        max_depth=20,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    return rf
```

### 6c. XGBoost (`models/baselines.py`)

```python
import xgboost as xgb

def train_xgboost(X_train, y_train, X_val, y_val):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    params = {
        'max_depth':        4,
        'eta':              0.05,
        'subsample':        0.7,
        'colsample_bytree': 0.5,
        'objective':        'binary:logistic',
        'eval_metric':      'auc',
        'seed':             42,
    }
    model = xgb.train(
        params, dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=30,
        verbose_eval=100,
    )
    return model
```

### 6d. LSTM (`models/lstm_model.py`)

**Architecture:**

| Layer | Config |
|---|---|
| Input | 41 features x 60 timesteps |
| LSTM Layer 1 | 64 hidden units, dropout=0.2 |
| LSTM Layer 2 | 32 hidden units, dropout=0.2 |
| Dense | 16 units, ReLU |
| Output | 1 unit, Sigmoid |
| Loss | Binary Cross-Entropy |
| Optimizer | Adam (lr=0.001, weight_decay=1e-5) |
| Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |

```python
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

class StockSequenceDataset(Dataset):
    def __init__(self, data_df, feature_cols, target_col, seq_len=60, tickers=None):
        self.sequences, self.labels = [], []
        tickers = tickers or data_df['Ticker'].unique()
        for ticker in tickers:
            stock = (data_df[data_df['Ticker'] == ticker]
                     .sort_values('Date').reset_index(drop=True))
            X = stock[feature_cols].values.astype(np.float32)
            y = stock[target_col].values.astype(np.float32)
            for i in range(seq_len, len(X)):
                self.sequences.append(X[i - seq_len:i])   # (60, 41)
                self.labels.append(y[i])
        self.sequences = torch.tensor(np.array(self.sequences))
        self.labels    = torch.tensor(np.array(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class StockLSTM(nn.Module):
    def __init__(self, input_size=41, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(hidden_size, 32)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])   # last timestep only
        out = self.relu(self.fc1(out))
        return self.sigmoid(self.fc2(out)).squeeze(1)


def train_lstm(model, train_loader, val_loader, device,
               max_epochs=200, patience=15, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)
    best_val_loss, best_weights, patience_ctr = float('inf'), None, 0

    model.to(device)
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                val_loss += criterion(model(X_b.to(device)), y_b.to(device)).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stop @ epoch {epoch + 1}")
                break

        if (epoch + 1) % 20 == 0:
            tl = train_loss / len(train_loader)
            print(f"Epoch {epoch+1:3d} | Train: {tl:.4f} | Val: {val_loss:.4f}")

    model.load_state_dict(best_weights)
    return model
```

### 6e. Ensemble (`models/ensemble.py`)

```python
import numpy as np
import xgboost as xgb

def ensemble_predict(lr_model, rf_model, xgb_model, X_np):
    """Equal-weighted average probability of class 1 across all three baseline models."""
    p_lr  = lr_model.predict_proba(X_np)[:, 1]
    p_rf  = rf_model.predict_proba(X_np)[:, 1]
    p_xgb = xgb_model.predict(xgb.DMatrix(X_np))
    return np.mean([p_lr, p_rf, p_xgb], axis=0)
```

---

## Step 7 — Trading Signals (`backtest/signals.py`)

```python
import pandas as pd

def generate_signals(predictions_df: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    """
    predictions_df columns:
        Date, Ticker, Prob_LR, Prob_RF, Prob_XGB, Prob_LSTM,
        Return_NextDay, Target
    Returns df with Signal = 'Long' / 'Short' / 'Hold' per ticker per day.
    """
    results = []
    for date, group in predictions_df.groupby('Date'):
        g = group.copy()
        g['Prob_ENS'] = (
            g['Prob_LR'] + g['Prob_RF'] + g['Prob_XGB'] + g['Prob_LSTM']
        ) / 4
        g = g.sort_values('Prob_ENS', ascending=False).reset_index(drop=True)
        g['Signal'] = 'Hold'
        g.loc[:k - 1, 'Signal'] = 'Long'            # top-k -> long
        g.loc[len(g) - k:, 'Signal'] = 'Short'      # bottom-k -> short
        results.append(g)
    return pd.concat(results)
```

---

## Step 8 — Portfolio & Transaction Costs (`backtest/portfolio.py`)

```python
import pandas as pd

def compute_portfolio_returns(
    signals_df: pd.DataFrame,
    tc_bps: float = 5
) -> pd.DataFrame:
    """
    Daily equal-weighted long-short portfolio return.
    tc_bps: cost per half-turn in basis points (paper baseline = 5 bps).
    """
    tc = tc_bps / 10_000
    daily = []
    prev_signals = {}

    for date, group in signals_df.sort_values('Date').groupby('Date'):
        longs  = group[group['Signal'] == 'Long']
        shorts = group[group['Signal'] == 'Short']

        long_ret  = longs['Return_NextDay'].mean()  if len(longs)  > 0 else 0.0
        short_ret = shorts['Return_NextDay'].mean() if len(shorts) > 0 else 0.0
        gross_ret = long_ret - short_ret

        curr = dict(zip(group['Ticker'], group['Signal']))
        turnover = sum(1 for t in curr if curr[t] != prev_signals.get(t))
        net_ret  = gross_ret - turnover * tc
        prev_signals = curr

        daily.append({
            'Date': date, 'Gross_Return': gross_ret,
            'Net_Return': net_ret, 'Long_Return': long_ret,
            'Short_Return': short_ret, 'TC': turnover * tc,
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
    """rf_daily ~ 3.8% annual / 252 trading days."""
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
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'Accuracy (%)': round(accuracy_score(y_true, y_pred) * 100, 2),
        'AUC-ROC':       round(roc_auc_score(y_true, y_prob), 4),
        'F1 Score':      round(f1_score(y_true, y_pred), 4),
    }
```

---

## Step 10 — main.py Skeleton

```python
from pipeline.data_loader   import download_and_save
from pipeline.features      import build_feature_matrix, FEATURE_COLS
from pipeline.targets       import create_targets
from pipeline.walk_forward  import generate_walk_forward_folds
from pipeline.standardizer  import standardize_fold
from models.baselines       import train_logistic, train_random_forest, train_xgboost
from models.lstm_model      import StockLSTM, StockSequenceDataset, train_lstm
from models.ensemble        import ensemble_predict
from backtest.signals       import generate_signals
from backtest.portfolio     import compute_portfolio_returns
from backtest.metrics       import compute_metrics, evaluate_classification
from config import *
import pandas as pd, numpy as np, xgboost as xgb, torch
from torch.utils.data import DataLoader

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
TARGET_COL = 'Target'

def main():
    data = download_and_save()
    data = build_feature_matrix(data)
    data = create_targets(data)

    dates = sorted(data['Date'].unique())
    folds = generate_walk_forward_folds(dates, TRAIN_DAYS, VAL_DAYS, TEST_DAYS)

    all_preds = []
    for fold in folds:
        print(f"\n=== FOLD {fold['fold']} ===")
        df_tr = data[data['Date'].isin(dates[fold['train'][0]:fold['train'][1]])]
        df_v  = data[data['Date'].isin(dates[fold['val'][0]:fold['val'][1]])]
        df_ts = data[data['Date'].isin(dates[fold['test'][0]:fold['test'][1]])]

        X_tr, X_v, X_ts, scaler = standardize_fold(
            df_tr[FEATURE_COLS].values,
            df_v[FEATURE_COLS].values,
            df_ts[FEATURE_COLS].values,
        )
        y_tr = df_tr[TARGET_COL].values

        lr_m  = train_logistic(X_tr, y_tr)
        rf_m  = train_random_forest(X_tr, y_tr)
        xgb_m = train_xgboost(X_tr, y_tr, X_v, df_v[TARGET_COL].values)

        def scaled_df(df_orig, X_s):
            d = df_orig.copy(); d[FEATURE_COLS] = X_s; return d

        lstm_m = StockLSTM(input_size=N_TOTAL_FEATURES)
        lstm_m = train_lstm(
            lstm_m,
            DataLoader(StockSequenceDataset(scaled_df(df_tr, X_tr), FEATURE_COLS, TARGET_COL, SEQ_LEN), batch_size=LSTM_BATCH, shuffle=True),
            DataLoader(StockSequenceDataset(scaled_df(df_v, X_v),   FEATURE_COLS, TARGET_COL, SEQ_LEN), batch_size=256),
            device,
        )

        # Collect test predictions (align LSTM output with df_ts rows per ticker)
        pred = df_ts.copy()
        pred['Prob_LR']  = lr_m.predict_proba(X_ts)[:, 1]
        pred['Prob_RF']  = rf_m.predict_proba(X_ts)[:, 1]
        pred['Prob_XGB'] = xgb_m.predict(xgb.DMatrix(X_ts))
        # LSTM: sequences skip first SEQ_LEN rows per ticker - handle alignment here
        # See StockSequenceDataset for index mapping
        pred['Prob_LSTM'] = np.nan  # TODO: fill with aligned LSTM predictions
        all_preds.append(pred)

    full_preds   = pd.concat(all_preds)
    signals_df   = generate_signals(full_preds.dropna(subset=['Prob_LSTM']), k=K_STOCKS)
    port_returns = compute_portfolio_returns(signals_df, tc_bps=TC_BPS)

    print("\n=== Final Portfolio Performance ===")
    print(compute_metrics(port_returns['Net_Return']))

if __name__ == '__main__':
    main()
```

> **LSTM sequence alignment note:** `StockSequenceDataset` drops the first `SEQ_LEN` rows per ticker
> (no lookback available). Store `(Date, Ticker)` keys alongside sequences in the dataset, then map
> predictions back to the correct rows in `df_ts`.

---

## Anti-Leakage Rules

| Rule | How to enforce |
|---|---|
| Standardize on training data only | `scaler.fit_transform(X_train)`, then `.transform()` on val/test |
| Never shuffle time-series | `shuffle=False` for all val/test DataLoaders |
| Tune hyperparameters on val, not test | All early stopping uses val loss only |
| Features use only data <= t | All pandas rolling windows are causal (no `center=True`) |
| Target uses only realized t+1 returns | `shift(-1)` on actual realized close prices |
| No test period feedback | Test fold is evaluated only, never trained on |

---

## Required Thesis Outputs

### Tables

| # | Title | Mirrors |
|---|---|---|
| T1 | Descriptive stats per stock (mean, std, min, max, skew, kurt) | Krauss 2017 Table 1 |
| T2 | Walk-forward fold dates | Both papers Section 3.1 |
| T3 | Hyperparameter configurations (all models) | Both papers Section 3.3 |
| T4 | Daily return characteristics — gross vs net | Fischer Table 3 / Krauss Table 2 |
| T5 | Annualized risk-return metrics (all models) | Fischer Table 3 / Krauss Table 3 |
| T6 | Sub-period performance breakdown | Krauss Table 5 |
| T7 | TC sensitivity (0-30 bps grid) | New contribution |
| T8 | Classification metrics: AUC, F1, Accuracy | Fischer Table 2 Panel B |
| T9 | Top-15 most important features (RF/XGB) | Krauss Figure 3 |

### Figures

| # | Title | How to create |
|---|---|---|
| F1 | Walk-forward timeline diagram | matplotlib gantt-style bar chart |
| F2 | LSTM architecture diagram | matplotlib / PowerPoint |
| F3 | Cumulative equity curves (all models + buy-and-hold) | `(1 + r).cumprod()` |
| F4 | Drawdown underwater chart | Rolling max drawdown, fill_between |
| F5 | Feature importance bar chart (top 15) | `rf.feature_importances_` / XGB SHAP |
| F6 | Return distribution (violin or box) | seaborn.violinplot per model |
| F7 | Sharpe ratio vs transaction cost (0-30 bps) | Loop over TC values |
| F8 | Confusion matrices (2x2 per model) | sklearn + seaborn heatmap |
| F9 | Sub-period performance bar chart | seaborn.barplot by sub-period |
| F10 | Correlation heatmap of stock returns | seaborn.heatmap(returns.corr()) |
| F11 | LSTM train/val loss curves per fold | Save epoch losses during training |
| F12 | Slippage sensitivity heatmap | Sharpe vs. (TC, slippage) grid |

---

## Sub-Period Analysis

| Period | Dates | Regime |
|---|---|---|
| Pre-COVID | 2019-01-01 to 2020-02-19 | Normal bull market |
| COVID crash | 2020-02-20 to 2020-04-30 | Extreme volatility / dislocation |
| Recovery / bull | 2020-05-01 to 2021-12-31 | Low rates, momentum-driven |
| 2022 bear market | 2022-01-01 to 2022-12-31 | Rate hikes, drawdowns |
| 2023-2024 AI rally | 2023-01-01 to 2024-12-31 | AI / large-cap concentration |

Compute `compute_metrics()` for each sub-period separately and report in T6 / F9.

---

## Hyperparameter Reference

| Parameter | Recommended | Paper Value | Range to Try |
|---|---|---|---|
| LSTM hidden units | 64 | 25 | 32, 64, 128 |
| LSTM layers | 2 | 1 | 1, 2 |
| LSTM dropout | 0.2 | 0.16 | 0.1, 0.2, 0.3 |
| Sequence length | 60 | 240 | 40, 60, 90 |
| Batch size | 128 | N/A | 64, 128, 256 |
| Learning rate | 0.001 | RMSprop default | 0.01, 0.001, 0.0001 |
| Early stopping patience | 15 | 10 | 10, 15, 20 |
| RF n_estimators | 500 | 1000 | 300, 500, 1000 |
| RF max_depth | 20 | 20 | 10, 20, None |
| XGB n_rounds | 500 (early stop) | 100 | 200-1000 |
| XGB eta | 0.05 | 0.1 | 0.01, 0.05, 0.1 |
| XGB max_depth | 4 | 3 | 3, 4, 5, 6 |
| LR regularization C | CV-selected | CV-selected | 1e-4 to 1e4 |
| k (long/short stocks) | 2 | 10 (500 stocks) | 2, 3 |

---

## Realistic Expectations

- **Directional accuracy:** 51-54% is a solid result (random = 50%). Do not expect the papers' 53.8%.
- **Sharpe ratio:** Positive but well below the papers' 5.83 — that number is driven by diversification across 500 stocks.
- **LSTM vs RF:** On a small dataset (10 stocks), RF may outperform LSTM. This is itself a valid, discussable finding.
- **Transaction cost sensitivity:** With only 10 stocks, turnover is high and TC erodes returns quickly. The TC sensitivity analysis (T7, F7) is one of your most important results.
- **COVID crash (Feb-Apr 2020):** Expect a spike in returns — extreme cross-sectional dispersion creates exploitable patterns.
- **2022 bear market:** Expect elevated drawdown. Analyze separately in sub-period section.

---

## Key References

- Fischer, T. & Krauss, C. (2017). *Deep learning with long short-term memory networks for financial market predictions.* FAU Discussion Papers in Economics, No. 11/2017.
- Krauss, C., Do, X.A. & Huck, N. (2017). *Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500.* European Journal of Operational Research, 259, 689-702.
- Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.
- Fama, E. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work. *Journal of Finance*, 25(2), 383-417.
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.
- Avellaneda, M. & Lee, J.H. (2010). Statistical arbitrage in the US equities market. *Quantitative Finance*, 10(7), 761-782.