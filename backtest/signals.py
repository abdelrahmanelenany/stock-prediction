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
from config import K_STOCKS, SIGNAL_CONFIDENCE_THRESHOLD, SIGNAL_USE_ZSCORE, MIN_HOLDING_DAYS


def smooth_probabilities(
    preds_df: pd.DataFrame,
    prob_col: str,
    alpha: float = 0.3,
    ema_method: str = 'alpha',
    ema_span: int | None = None,
) -> pd.DataFrame:
    """
    Apply per-ticker exponential smoothing to predicted probabilities.

    Reduces daily position turnover by making signals "stickier".
    Lower alpha = more smoothing (more weight on history).
    Supports pandas EWM smoothing configured either by alpha or span.

    Returns a copy of preds_df with a new column '{prob_col}_Smooth'.
    """
    preds_df = preds_df.sort_values(['Ticker', 'Date']).copy()
    smoothed_col = f'{prob_col}_Smooth'

    if ema_method not in ('alpha', 'span'):
        raise ValueError(f"Unsupported ema_method='{ema_method}'. Expected 'alpha' or 'span'.")
    if ema_method == 'span':
        if ema_span is None or ema_span <= 0:
            raise ValueError('ema_span must be a positive integer when ema_method="span".')
    else:
        # Allow alpha <= 0 as an explicit "no smoothing" mode.
        if alpha is None or alpha <= 0:
            alpha = None
        elif alpha > 1:
            raise ValueError('alpha must satisfy 0 < alpha <= 1 when ema_method="alpha".')

    parts = []
    for ticker, group in preds_df.groupby('Ticker'):
        g = group.sort_values('Date').copy()
        if ema_method == 'span':
            g[smoothed_col] = g[prob_col].ewm(span=ema_span, adjust=False).mean()
        elif alpha is None:
            g[smoothed_col] = g[prob_col]
        else:
            g[smoothed_col] = g[prob_col].ewm(alpha=alpha, adjust=False).mean()
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

    # Track entry dates per ticker
    entry_dates = {}  # {ticker: (signal, date_idx)}

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

                # Check if we're trying to change position before min hold
                if new_signal != old_signal and days_held < min_hold_days:
                    # Force keep old signal
                    day_signals.loc[idx, 'Signal'] = old_signal
                else:
                    # Allow change, update entry date
                    if new_signal != old_signal:
                        if new_signal in ('Long', 'Short'):
                            entry_dates[ticker] = (new_signal, date_idx)
                        else:
                            # Exiting to Hold, remove tracking
                            del entry_dates[ticker]
            else:
                # New position entry
                if new_signal in ('Long', 'Short'):
                    entry_dates[ticker] = (new_signal, date_idx)

        results.append(day_signals)

    return pd.concat(results).sort_values(['Date', 'Ticker']).reset_index(drop=True)


def generate_signals(
    predictions_df: pd.DataFrame,
    k: int = K_STOCKS,
    prob_col: str = None,
    confidence_threshold: float = SIGNAL_CONFIDENCE_THRESHOLD,
    use_cross_sectional_z: bool = SIGNAL_USE_ZSCORE,
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, int | float | bool | str | None]]:
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
    return_diagnostics : bool
        If True, return a `(signals_df, diagnostics)` tuple. Otherwise
        return only the signals DataFrame for backward compatibility.

    Returns
    -------
    pd.DataFrame or tuple[pd.DataFrame, dict]
        Input DataFrame with added columns:
          Prob_ENS : ensemble probability (or copy of prob_col if specified)
          Prob_Z   : cross-sectional z-score (if use_cross_sectional_z=True)
          Signal   : 'Long', 'Short', or 'Hold'
        If `return_diagnostics=True`, also returns summary diagnostics about
        requested versus assigned long/short slots.
    """
    ensemble_cols = ['Prob_LR', 'Prob_RF', 'Prob_XGB', 'Prob_LSTM_A', 'Prob_LSTM_B']
    results = []
    long_slots_requested = 0
    short_slots_requested = 0
    long_slots_assigned = 0
    short_slots_assigned = 0

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
        long_slots_requested += int(long_k)
        short_slots_requested += int(short_k)

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

        long_slots_assigned += int((g['Signal'] == 'Long').sum())
        short_slots_assigned += int((g['Signal'] == 'Short').sum())
        results.append(g)

    out = pd.concat(results).reset_index(drop=True)
    diagnostics = {
        'prob_col_used': prob_col,
        'confidence_threshold': float(confidence_threshold),
        'use_cross_sectional_z': bool(use_cross_sectional_z),
        'n_dates': int(out['Date'].nunique()) if not out.empty else 0,
        'n_rows': int(len(out)),
        'long_slots_requested': int(long_slots_requested),
        'short_slots_requested': int(short_slots_requested),
        'long_slots_assigned': int(long_slots_assigned),
        'short_slots_assigned': int(short_slots_assigned),
        'long_slots_filtered_by_threshold': int(long_slots_requested - long_slots_assigned),
        'short_slots_filtered_by_threshold': int(short_slots_requested - short_slots_assigned),
    }

    # Sanity summary
    counts = out['Signal'].value_counts()
    print(f"Signal counts — Long: {counts.get('Long', 0)}  "
          f"Short: {counts.get('Short', 0)}  "
          f"Hold: {counts.get('Hold', 0)}")
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
):
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
