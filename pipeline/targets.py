"""
pipeline/targets.py
Step 3: Create binary cross-sectional median-based target variable.

On each day, compute the cross-sectional median of the N-day forward return
(where N = config.TARGET_HORIZON_DAYS).
Target = 1 if stock's forward return >= median, else 0.

Key design decisions
--------------------
* Return_NextDay always = the 1-day close-to-close return (used by portfolio.py
  for P&L calculation). It is NEVER changed by TARGET_HORIZON_DAYS.
* Target is derived from a configurable N-day forward return so that the model
  predicts a less noisy, more persistent signal:
    - N=1  (original) : near-zero predictability for large-cap liquid stocks
    - N=5  (default)  : weekly momentum — well-documented in academic literature
    - N=21            : monthly momentum — Jegadeesh & Titman (1993)
* Rows where the N-day forward return cannot be computed (last N rows per
  ticker) are dropped.  This is slightly more wasteful than N=1 but necessary.

Mirrors Fischer & Krauss (2017) cross-sectional labelling strategy.
"""
import pandas as pd
import numpy as np
import sys
import os
import config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

FEATURES_CACHE = f'data/processed/features_{config.UNIVERSE_MODE}.csv'


def create_targets(
    data: pd.DataFrame,
    return_col: str = 'Return_1d',
    horizon: int | None = None,
) -> pd.DataFrame:
    """
    Adds two columns to the feature DataFrame:

    * ``Return_NextDay`` — 1-day forward return for each stock (close t → close
      t+1).  Always 1-day regardless of *horizon*.  Used by portfolio.py for
      actual P&L; do **not** use it as the target input.

    * ``Target`` — binary label: 1 if the stock's *horizon*-day forward return
      is >= the cross-sectional median of that day's cohort, else 0.

    Parameters
    ----------
    data : pd.DataFrame
        Output of build_feature_matrix() — must contain 'Date', 'Ticker',
        return_col, and 'Close'.
    return_col : str
        Column to use as the basis for forward-return computation.
        Default 'Return_1d' (or 'Close'-based pct_change when horizon > 1).
    horizon : int, optional
        Number of trading days ahead for the Target.  Defaults to
        ``config.TARGET_HORIZON_DAYS`` (recommended: 5 for large-cap).

    Returns
    -------
    pd.DataFrame
        Original DataFrame plus 'Return_NextDay' and 'Target' columns.
    """
    if horizon is None:
        horizon = getattr(config, 'TARGET_HORIZON_DAYS', 1)

    data = data.copy().sort_values(['Date', 'Ticker']).reset_index(drop=True)

    # ── 1. Return_NextDay — always 1-day forward return (portfolio P&L) ───────
    # Shift Return_1d backward by 1 within each ticker: today's row gets
    # tomorrow's realised return.
    data['Return_NextDay'] = data.groupby('Ticker')[return_col].shift(-1)

    # ── 2. Forward return for Target ──────────────────────────────────────────
    if horizon == 1:
        # Optimisation: reuse Return_NextDay to avoid a redundant groupby
        forward_return_col = 'Return_NextDay'
        data['_FwdReturn'] = data['Return_NextDay']
    else:
        # Compute the true N-day compound return using Close prices.
        # Close is already in the DataFrame (raw OHLCV preserved through the
        # feature cache pipeline).  We compute:
        #   fwd_return(t) = Close(t+N) / Close(t) - 1
        # This is causal at inference time (data available after close t).
        if 'Close' in data.columns:
            data['_FwdReturn'] = data.groupby('Ticker')['Close'].transform(
                lambda x: x.shift(-horizon) / x - 1
            )
        else:
            # Fallback: compound the 1-day returns (less accurate but functional)
            print(
                f"[targets] WARNING: 'Close' column not found. "
                f"Approximating {horizon}-day forward return by summing Return_1d shifts."
            )
            # Sum of next horizon daily log-returns ≈ compound return for small |r|
            data['_FwdReturn'] = sum(
                data.groupby('Ticker')[return_col].shift(-d)
                for d in range(1, horizon + 1)
            )

    # ── 3. Drop rows where forward returns are unavailable ───────────────────
    # Last `horizon` rows per ticker have NaN _FwdReturn.
    # Last 1 row per ticker has NaN Return_NextDay.
    # Drop whichever is the wider constraint.
    before = len(data)
    data.dropna(subset=['Return_NextDay', '_FwdReturn'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    print(
        f"Dropped {before - len(data)} rows "
        f"(last {horizon} day(s) per ticker — no future return available)"
    )

    # ── 4. Cross-sectional median target ─────────────────────────────────────
    # Each day, rank stocks by their N-day forward return.
    # Target=1 if >= cross-sectional median (outperformer), else 0.
    daily_median = data.groupby('Date')['_FwdReturn'].transform('median')
    data['Target'] = (data['_FwdReturn'] >= daily_median).astype(int)

    # Drop the helper column — not needed downstream
    data.drop(columns=['_FwdReturn'], inplace=True)

    # ── 5. Diagnostics ───────────────────────────────────────────────────────
    n_pos = (data['Target'] == 1).sum()
    n_neg = (data['Target'] == 0).sum()
    print(f"\nTarget horizon : {horizon} day(s)")
    print(f"Class distribution:")
    print(f"  Target=1 (>= {horizon}d median): {n_pos:,} ({n_pos / len(data) * 100:.1f}%)")
    print(f"  Target=0 (<  {horizon}d median): {n_neg:,} ({n_neg / len(data) * 100:.1f}%)")
    print(f"\nTotal samples : {len(data):,}")
    print(f"Date range    : {data['Date'].min()} \u2192 {data['Date'].max()}")
    return data


if __name__ == '__main__':
    data = pd.read_csv(FEATURES_CACHE, parse_dates=['Date'])
    result = create_targets(data)

    # Quick sanity: per-ticker class balance
    print("\nPer-ticker class balance (should be ~0.50 for each ticker):")
    print(result.groupby('Ticker')['Target'].mean().round(3))

    # Save augmented feature+target file
    result.to_csv(FEATURES_CACHE, index=False)
    print(f"\nUpdated {FEATURES_CACHE} with Target and Return_NextDay columns.")
