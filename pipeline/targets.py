"""
pipeline/targets.py
Step 3: Create binary cross-sectional median target variable.

Label = 1 if stock's next-day return >= cross-sectional median return that day, else 0.
This mirrors Fischer & Krauss (2017) and Krauss et al. (2017) exactly,
and produces a ~50/50 balanced dataset by construction.
"""
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def create_targets(data: pd.DataFrame, return_col: str = 'Return_1d') -> pd.DataFrame:
    """
    Adds two columns to the feature DataFrame:
      - Return_NextDay : realized t+1 return for each stock (used in backtesting)
      - Target         : 1 if Return_NextDay >= cross-sectional daily median, else 0

    The last trading day per ticker is dropped (no realized next-day return available).

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

    # Cross-sectional median across all tickers for that date
    daily_median = data.groupby('Date')['Return_NextDay'].transform('median')

    # Binary label
    data['Target'] = (data['Return_NextDay'] >= daily_median).astype(int)

    # Drop the last day per ticker (Return_NextDay is NaN — no future return exists)
    before = len(data)
    data.dropna(subset=['Return_NextDay'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    print(f"Dropped {before - len(data)} rows (last trading day per ticker, no future return)")

    print(f"\nClass distribution (expect ~50/50):")
    print(data['Target'].value_counts(normalize=True).rename({0: 'Short (0)', 1: 'Long (1)'}).round(4))
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
