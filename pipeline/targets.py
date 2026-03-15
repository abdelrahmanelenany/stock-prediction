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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


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
    data = pd.read_csv('data/processed/features.csv', parse_dates=['Date'])
    result = create_targets(data)

    # Quick sanity: per-ticker class balance
    print("\nPer-ticker class balance:")
    print(result.groupby('Ticker')['Target'].mean().round(3))

    # Save augmented feature+target file
    result.to_csv('data/processed/features.csv', index=False)
    print("\nUpdated data/processed/features.csv with Target and Return_NextDay columns.")
