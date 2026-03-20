"""
backtest/signals.py
Step 7: Convert per-stock daily probabilities to Long / Short / Hold signals.

Ranking logic (Fischer & Krauss 2017 / Krauss et al. 2017):
  - Compute ensemble probability Prob_ENS = mean(Prob_LR, Prob_RF, Prob_XGB, Prob_LSTM)
  - Sort all stocks by Prob_ENS descending each day
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
from config import K_STOCKS, SIGNAL_CONFIDENCE_THRESHOLD, SIGNAL_USE_ZSCORE


def smooth_probabilities(
    preds_df: pd.DataFrame,
    prob_col: str,
    alpha: float = 0.3,
) -> pd.DataFrame:
    """
    Apply per-ticker exponential smoothing to predicted probabilities.

    Reduces daily position turnover by making signals "stickier".
    Lower alpha = more smoothing (more weight on history).

    Returns a copy of preds_df with a new column '{prob_col}_Smooth'.
    """
    preds_df = preds_df.sort_values(['Ticker', 'Date']).copy()
    smoothed_col = f'{prob_col}_Smooth'

    parts = []
    for ticker, group in preds_df.groupby('Ticker'):
        g = group.sort_values('Date').copy()
        g[smoothed_col] = g[prob_col].ewm(alpha=alpha, adjust=False).mean()
        parts.append(g)

    return pd.concat(parts).sort_values(['Date', 'Ticker']).reset_index(drop=True)


def generate_signals(
    predictions_df: pd.DataFrame,
    k: int = K_STOCKS,
    prob_col: str = None,
    confidence_threshold: float = SIGNAL_CONFIDENCE_THRESHOLD,
    use_cross_sectional_z: bool = SIGNAL_USE_ZSCORE,
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
    confidence_threshold : float
        Minimum distance from 0.5 (or z-score if use_cross_sectional_z=True)
        required to generate a Long or Short signal.
        Set to 0.0 for pure-ranking behavior.
    use_cross_sectional_z : bool
        If True, z-score probabilities within each day's cross-section
        before applying confidence threshold. This makes the threshold
        work on standard deviations rather than raw probability units,
        which is more stable across different probability calibrations.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns:
          Prob_ENS : ensemble probability (or copy of prob_col if specified)
          Prob_Z   : cross-sectional z-score (if use_cross_sectional_z=True)
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
