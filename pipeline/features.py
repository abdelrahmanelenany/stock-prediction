"""
pipeline/features.py
Step 2: Compute 31 lagged cumulative returns + 10 technical indicators.
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import LAGGED_RETURN_PERIODS

# ── Feature column names (41 total) ────────────────────────────────────────
FEATURE_COLS = (
    [f'Return_{m}d' for m in LAGGED_RETURN_PERIODS] +
    ['RSI_14', 'MACD', 'MACD_Signal', 'BB_Width', 'BB_PctB',
     'ATR_14', 'OBV', 'Volume_Ratio', 'HL_Pct_5d', 'Mom_10d']
)
TARGET_COL = 'Target'


def add_lagged_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    R(m) = P_t / P_{t-m} - 1  (simple cumulative return over m days).
    Computed per ticker to avoid look-across-ticker contamination.
    Overwrites Return_1d (already present from data_loader) for consistency.
    """
    data = data.sort_values(['Ticker', 'Date'])
    for m in LAGGED_RETURN_PERIODS:
        data[f'Return_{m}d'] = data.groupby('Ticker')['Close'].pct_change(m)
    return data


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 10 technical indicators for a single ticker's DataFrame.
    All windows are causal (no center=True) to prevent look-ahead leakage.
    """
    df = df.copy().sort_values('Date')

    # ── RSI(14) ────────────────────────────────────────────────────────────
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI_14'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # ── MACD & Signal ──────────────────────────────────────────────────────
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ── Bollinger Bands (20-day) ────────────────────────────────────────────
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    df['BB_Width'] = (upper - lower) / sma20
    df['BB_PctB'] = (df['Close'] - lower) / (upper - lower + 1e-10)

    # ── ATR(14) ────────────────────────────────────────────────────────────
    hl  = df['High'] - df['Low']
    hpc = (df['High'] - df['Close'].shift()).abs()
    lpc = (df['Low']  - df['Close'].shift()).abs()
    df['ATR_14'] = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()

    # ── OBV ────────────────────────────────────────────────────────────────
    direction = np.sign(df['Close'].diff()).fillna(0)
    df['OBV'] = (direction * df['Volume']).cumsum()

    # ── Volume Ratio ───────────────────────────────────────────────────────
    df['Volume_Ratio'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1)

    # ── High-Low % (5-day rolling avg) ─────────────────────────────────────
    df['HL_Pct_5d'] = ((df['High'] - df['Low']) / df['Close']).rolling(5).mean()

    # ── 10-day Momentum ────────────────────────────────────────────────────
    df['Mom_10d'] = df['Close'].pct_change(10)

    return df


def build_feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the full feature pipeline on the long-format OHLCV DataFrame:
      1. Adds 31 lagged returns
      2. Adds 10 technical indicators per ticker
      3. Drops rows with any NaN in feature columns
      4. Saves to data/processed/features.csv

    Returns the enriched DataFrame (Date, Ticker, OHLCV, features...).
    """
    print("Computing lagged returns...")
    data = add_lagged_returns(data)

    print("Computing technical indicators per ticker...")
    enriched = []
    for ticker, group in data.groupby('Ticker'):
        enriched.append(compute_technical_features(group))
    result = pd.concat(enriched).sort_values(['Date', 'Ticker']).reset_index(drop=True)

    before = len(result)
    result.dropna(subset=FEATURE_COLS, inplace=True)
    result.reset_index(drop=True, inplace=True)
    print(f"Dropped {before - len(result)} rows with NaN features "
          f"(expected: ~240 per ticker = {240 * len(result['Ticker'].unique())})")

    result.to_csv('data/processed/features.csv', index=False)
    print(f"\nSaved {len(result)} rows to data/processed/features.csv")
    print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"Date range: {result['Date'].min()} → {result['Date'].max()}")
    print(f"Rows per ticker:\n{result['Ticker'].value_counts().sort_index()}")
    return result


if __name__ == '__main__':
    data = pd.read_csv('data/raw/ohlcv_long.csv', parse_dates=['Date'])
    build_feature_matrix(data)
