"""
pipeline/data_loader.py
Step 1: Download raw OHLCV data from yfinance and clean it.
"""
import yfinance as yf
import pandas as pd
import sys
import os
import config

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import TICKERS, START_DATE, END_DATE

FEATURES_CACHE = f'data/processed/features_{config.UNIVERSE_MODE}.csv'


def download_and_save() -> pd.DataFrame:
    """
    Downloads OHLCV data for all tickers via yfinance, restructures from
    MultiIndex to long format, applies data quality rules, and saves:
      - data/raw/ohlcv_raw.csv  (raw MultiIndex download)
      - data/raw/ohlcv_long.csv (cleaned long format)

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        Date, Ticker, Open, High, Low, Close, Volume, Return_1d
    """
    print(f"Downloading {len(TICKERS)} tickers from {START_DATE} to {END_DATE}...")
    raw = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True)
    print(
        f"Universe mode: {config.UNIVERSE_MODE} | Tickers: {len(TICKERS)} | "
        f"Date range: {START_DATE} to {END_DATE}"
    )
    raw.to_csv('data/raw/ohlcv_raw.csv')
    print(f"Raw data saved: {raw.shape[0]} rows x {raw.shape[1]} columns")

    # Restructure from MultiIndex -> long format, one ticker at a time.
    # yfinance MultiIndex level 0 = price type (alphabetical), level 1 = ticker.
    # We extract each ticker's OHLCV and explicitly select columns by name.
    panels = {}
    for ticker in TICKERS:
        df = raw.xs(ticker, axis=1, level=1).copy()
        # Explicitly select and order the columns we need (defence vs col-order assumptions)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df['Ticker'] = ticker
        panels[ticker] = df

    data = pd.concat(panels.values()).reset_index()
    # At this point columns are: Date, Open, High, Low, Close, Volume, Ticker
    data = data.rename(columns={'Price': 'Date'}) if 'Price' in data.columns else data
    data = data.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    # ---- Data quality rules (per CLAUDE.md) ----

    # 1. Compute 1-day return (kept as utility column, used by targets later)
    data['Return_1d'] = data.groupby('Ticker')['Close'].pct_change(1)

    # 2. Remove non-trading days (zero volume)
    before = len(data)
    data = data[data['Volume'] > 0].copy()
    print(f"Removed {before - len(data)} zero-volume rows")

    # 3. Forward-fill short gaps (<= 2 days) within each ticker.
    # Use column-by-column ffill to avoid pandas 3.0 groupby.apply include_groups issue.
    for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Return_1d']:
        data[col] = data.groupby('Ticker')[col].ffill(limit=2)

    # 4. Drop rows with missing Close (gap > 3 days or data truly absent)
    before = len(data)
    data.dropna(subset=['Close'], inplace=True)
    print(f"Dropped {before - len(data)} rows with missing Close")

    # 5. Flag large intraday moves for awareness (|Return_1d| > 20%)
    large_moves = data[data['Return_1d'].abs() > 0.20]
    if len(large_moves) > 0:
        print(f"WARNING: {len(large_moves)} rows with |Return_1d| > 20% — review manually:")
        print(large_moves[['Date', 'Ticker', 'Close', 'Return_1d']].to_string(index=False))

    data.to_csv('data/raw/ohlcv_long.csv', index=False)
    print(f"\nSaved {len(data)} rows to data/raw/ohlcv_long.csv")
    print(f"Missing values remaining: {data.isna().sum().sum()}")
    print(f"Date range: {data['Date'].min()} → {data['Date'].max()}")
    print(f"Tickers: {sorted(data['Ticker'].unique())}")
    return data


if __name__ == '__main__':
    download_and_save()
