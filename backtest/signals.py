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
from __future__ import annotations

import sys
import os
from typing import Any

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    K_STOCKS,
    SIGNAL_CONFIDENCE_THRESHOLD,
    SIGNAL_USE_ZSCORE,
    MIN_HOLDING_DAYS,
    SIGNAL_SMOOTH_ALPHA,
    SIGNAL_EMA_METHOD,
    SIGNAL_EMA_SPAN,
)


def smooth_probabilities(
    preds_df: pd.DataFrame,
    prob_col: str,
    alpha: float | None = None,
    ema_method: str | None = None,
    ema_span: float | None = None,
) -> pd.DataFrame:
    """
    Apply causal per-ticker EMA smoothing to predicted probabilities.

    Uses config SIGNAL_EMA_METHOD: 'alpha' (ewm alpha) or 'span' (ewm span).
    """
    preds_df = preds_df.sort_values(['Ticker', 'Date']).copy()
    smoothed_col = f'{prob_col}_Smooth'
    method = (ema_method or SIGNAL_EMA_METHOD or 'alpha').lower()
    span = ema_span if ema_span is not None else SIGNAL_EMA_SPAN
    a = SIGNAL_SMOOTH_ALPHA if alpha is None else alpha

    parts = []
    for ticker, group in preds_df.groupby('Ticker'):
        g = group.sort_values('Date').copy()
        if method == 'span' and span is not None and float(span) > 1:
            g[smoothed_col] = g[prob_col].ewm(span=float(span), adjust=False).mean()
        else:
            g[smoothed_col] = g[prob_col].ewm(alpha=a, adjust=False).mean()
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

    entry_dates = {}

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

                if new_signal != old_signal and days_held < min_hold_days:
                    day_signals.loc[idx, 'Signal'] = old_signal
                else:
                    if new_signal != old_signal:
                        if new_signal in ('Long', 'Short'):
                            entry_dates[ticker] = (new_signal, date_idx)
                        else:
                            del entry_dates[ticker]
            else:
                if new_signal in ('Long', 'Short'):
                    entry_dates[ticker] = (new_signal, date_idx)

        results.append(day_signals)

    return pd.concat(results).sort_values(['Date', 'Ticker']).reset_index(drop=True)


def generate_signals(
    predictions_df: pd.DataFrame,
    k: int = K_STOCKS,
    prob_col: str | None = None,
    confidence_threshold: float = SIGNAL_CONFIDENCE_THRESHOLD,
    use_cross_sectional_z: bool = SIGNAL_USE_ZSCORE,
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """
    Generate Long / Short / Hold signals for each (Date, Ticker) row.

    Returns
    -------
    pd.DataFrame or (DataFrame, diagnostics dict) if return_diagnostics=True.
    """
    ensemble_cols = ['Prob_LR', 'Prob_RF', 'Prob_XGB', 'Prob_LSTM_A', 'Prob_LSTM_B']
    results = []
    long_candidate_slots = 0
    long_after_threshold = 0
    short_candidate_slots = 0
    short_after_threshold = 0

    for date, group in predictions_df.sort_values('Date').groupby('Date'):
        g = group.copy()
        n = len(g)

        if prob_col is not None:
            g['Prob_ENS'] = g[prob_col]
        else:
            g['Prob_ENS'] = g[ensemble_cols].mean(axis=1)

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

        g = g.sort_values(sort_col, ascending=False, kind='stable').reset_index(drop=True)

        g['Signal'] = 'Hold'

        long_k = min(k, n // 2)
        short_k = min(k, n // 2)

        if confidence_threshold > 0.0:
            long_candidates = g.head(long_k)
            short_candidates = g.tail(short_k)
            long_candidate_slots += long_k
            short_candidate_slots += short_k

            if use_cross_sectional_z:
                long_mask = long_candidates['Prob_Z'] > confidence_threshold
                short_mask = short_candidates['Prob_Z'] < -confidence_threshold
            else:
                long_mask = long_candidates['Prob_ENS'] > (0.5 + confidence_threshold)
                short_mask = short_candidates['Prob_ENS'] < (0.5 - confidence_threshold)

            n_l = long_mask.sum()
            n_s = short_mask.sum()
            long_after_threshold += int(n_l)
            short_after_threshold += int(n_s)

            g.loc[long_candidates[long_mask].index, 'Signal'] = 'Long'
            g.loc[short_candidates[short_mask].index, 'Signal'] = 'Short'
        else:
            g.loc[:long_k - 1, 'Signal'] = 'Long'
            g.loc[n - short_k:, 'Signal'] = 'Short'
            long_candidate_slots += long_k
            short_candidate_slots += short_k
            long_after_threshold += long_k
            short_after_threshold += short_k

        results.append(g)

    out = pd.concat(results).reset_index(drop=True)

    counts = out['Signal'].value_counts()
    print(
        f"Signal counts — Long: {counts.get('Long', 0)}  "
        f"Short: {counts.get('Short', 0)}  "
        f"Hold: {counts.get('Hold', 0)}"
    )

    diagnostics: dict[str, Any] = {
        'n_rows': len(out),
        'n_days': out['Date'].nunique(),
        'k': k,
        'confidence_threshold': confidence_threshold,
        'use_cross_sectional_z': use_cross_sectional_z,
        'long_candidate_slots': long_candidate_slots,
        'long_positions_assigned': int(counts.get('Long', 0)),
        'short_candidate_slots': short_candidate_slots,
        'short_positions_assigned': int(counts.get('Short', 0)),
        'long_slots_filtered_by_threshold': max(0, long_candidate_slots - long_after_threshold),
        'short_slots_filtered_by_threshold': max(0, short_candidate_slots - short_after_threshold),
    }

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
) -> dict[str, Any]:
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
                'Prob_LSTM_A': np.random.rand(),
                'Prob_LSTM_B': np.random.rand(),
                'Return_NextDay': np.random.randn() * 0.01,
                'Target': np.random.randint(0, 2),
            })
    df = pd.DataFrame(rows)
    signals, diag = generate_signals(df, k=2, confidence_threshold=0.0, return_diagnostics=True)
    print('diag keys', sorted(diag.keys()))
    print('\nSample output (first date):')
    print(
        signals[signals['Date'] == signals['Date'].iloc[0]][
            ['Date', 'Ticker', 'Prob_ENS', 'Signal']
        ].to_string(index=False)
    )
    assert (signals.groupby('Date')['Signal'].apply(lambda x: (x == 'Long').sum()) == 2).all()
    assert (signals.groupby('Date')['Signal'].apply(lambda x: (x == 'Short').sum()) == 2).all()
    print('\nAll signal counts verified (2 long, 2 short per day).')
