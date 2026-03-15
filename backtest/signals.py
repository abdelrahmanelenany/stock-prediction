"""
backtest/signals.py
Step 7: Convert per-stock daily probabilities to Long / Short / Hold signals.

Ranking logic (Fischer & Krauss 2017 / Krauss et al. 2017):
  - Compute ensemble probability Prob_ENS = mean(Prob_LR, Prob_RF, Prob_XGB, Prob_LSTM)
  - Sort all stocks by Prob_ENS descending each day
  - Top-k → Long   (predicted to outperform the median)
  - Bottom-k → Short (predicted to underperform the median)
  - Remaining → Hold
"""
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import K_STOCKS


def generate_signals(
    predictions_df: pd.DataFrame,
    k: int = K_STOCKS,
    prob_col: str = None,
) -> pd.DataFrame:
    """
    Generate Long / Short / Hold signals for each (Date, Ticker) row.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Must contain: Date, Ticker, Return_NextDay, Target, and either
        all four probability columns (Prob_LR, Prob_RF, Prob_XGB, Prob_LSTM)
        or a single column specified via `prob_col`.
    k : int
        Number of long and short positions per day (default K_STOCKS).
    prob_col : str or None
        If provided, use this single column as the ranking signal instead
        of computing the 4-model ensemble average. Useful for per-model
        performance attribution (e.g. prob_col='Prob_RF').

    Returns
    -------
    pd.DataFrame
        Input DataFrame with two added columns:
          Prob_ENS : ensemble probability (or copy of prob_col if specified)
          Signal   : 'Long', 'Short', or 'Hold'
    """
    ensemble_cols = ['Prob_LR', 'Prob_RF', 'Prob_XGB', 'Prob_LSTM']
    results = []

    for date, group in predictions_df.sort_values('Date').groupby('Date'):
        g = group.copy()
        n = len(g)

        # ── Ensemble probability ────────────────────────────────────────────
        if prob_col is not None:
            g['Prob_ENS'] = g[prob_col]
        else:
            g['Prob_ENS'] = g[ensemble_cols].mean(axis=1)

        # ── Rank by ensemble probability (descending); stable sort for ties ─
        g = g.sort_values('Prob_ENS', ascending=False, kind='stable').reset_index(drop=True)

        # ── Assign signals: top-k Long, bottom-k Short, rest Hold ──────────
        g['Signal'] = 'Hold'

        long_k  = min(k, n // 2)
        short_k = min(k, n // 2)

        g.loc[:long_k - 1, 'Signal'] = 'Long'
        g.loc[n - short_k:, 'Signal'] = 'Short'

        results.append(g)

    out = pd.concat(results).reset_index(drop=True)

    # Sanity summary
    counts = out['Signal'].value_counts()
    print(f"Signal counts — Long: {counts.get('Long', 0)}  "
          f"Short: {counts.get('Short', 0)}  "
          f"Hold: {counts.get('Hold', 0)}")
    return out


if __name__ == '__main__':
    # Quick demo with synthetic data to verify signal assignment
    import numpy as np
    np.random.seed(42)
    dates = pd.date_range('2022-01-03', periods=5, freq='B')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JPM', 'V']
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                'Date': d, 'Ticker': t,
                'Prob_LR':   np.random.rand(),
                'Prob_RF':   np.random.rand(),
                'Prob_XGB':  np.random.rand(),
                'Prob_LSTM': np.random.rand(),
                'Return_NextDay': np.random.randn() * 0.01,
                'Target': np.random.randint(0, 2),
            })
    df = pd.DataFrame(rows)
    signals = generate_signals(df, k=2)
    print('\nSample output (first date):')
    print(signals[signals['Date'] == signals['Date'].iloc[0]]
          [['Date', 'Ticker', 'Prob_ENS', 'Signal']].to_string(index=False))
    assert (signals.groupby('Date')['Signal'].apply(lambda x: (x == 'Long').sum()) == 2).all()
    assert (signals.groupby('Date')['Signal'].apply(lambda x: (x == 'Short').sum()) == 2).all()
    print('\nAll signal counts verified (2 long, 2 short per day).')
