"""
backtest/metrics.py
Step 9: Portfolio performance and classification metrics.

Covers all thesis table requirements:
  T4/T5 — compute_metrics()           (Sharpe, Sortino, MDD, Calmar, etc.)
  T6     — compute_subperiod_metrics() (per market-regime breakdown)
  T7     — compute_tc_sensitivity()    (Sharpe vs transaction cost grid)
  T8     — evaluate_classification()   (AUC, Accuracy, F1)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ~3.8% annual risk-free rate / 252 trading days
RF_DAILY = 0.00015

# Sub-period definitions for T6 / F9 (CLAUDE.md Section "Sub-Period Analysis")
SUBPERIODS = {
    'Pre-COVID':        ('2019-01-01', '2020-02-19'),
    'COVID crash':      ('2020-02-20', '2020-04-30'),
    'Recovery/bull':    ('2020-05-01', '2021-12-31'),
    '2022 bear':        ('2022-01-01', '2022-12-31'),
    '2023-24 AI rally': ('2023-01-01', '2024-12-31'),
}


def compute_metrics(
    returns_series: pd.Series,
    rf_daily: float = RF_DAILY,
) -> dict:
    """
    Annualised risk-return metrics for a daily return series.

    Parameters
    ----------
    returns_series : pd.Series
        Daily net (or gross) portfolio returns, indexed by date.
    rf_daily : float
        Daily risk-free rate (default ≈ 3.8% p.a. / 252).

    Returns
    -------
    dict with keys matching thesis Table T5 columns.
    """
    r = returns_series.dropna()
    if len(r) == 0:
        return {}

    mean_d = r.mean()
    std_d  = r.std()
    excess = r - rf_daily

    sharpe = (
        (excess.mean() / excess.std()) * np.sqrt(252)
        if excess.std() > 0 else 0.0
    )

    downside = r[r < rf_daily]
    sortino  = (
        ((mean_d - rf_daily) / downside.std()) * np.sqrt(252)
        if len(downside) > 1 else 0.0
    )

    cum     = (1 + r).cumprod()
    max_dd  = ((cum - cum.cummax()) / cum.cummax()).min()
    ann_ret = (1 + mean_d) ** 252 - 1

    return {
        'N Days':                 int(len(r)),
        'Mean Daily Return (%)':  round(mean_d * 100, 4),
        'Annualized Return (%)':  round(ann_ret * 100, 2),
        'Annualized Std Dev (%)': round(std_d * np.sqrt(252) * 100, 2),
        'Sharpe Ratio':           round(sharpe, 3),
        'Sortino Ratio':          round(sortino, 3),
        'Max Drawdown (%)':       round(max_dd * 100, 2),
        'Calmar Ratio':           round(ann_ret / abs(max_dd), 3) if max_dd != 0 else 0.0,
        'Win Rate (%)':           round((r > 0).mean() * 100, 2),
        'VaR 1% (%)':             round(np.percentile(r, 1) * 100, 4),
    }


def evaluate_classification(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Binary classification metrics for thesis Table T8.

    Parameters
    ----------
    y_true    : array-like of 0/1 ground-truth labels
    y_prob    : array-like of predicted probabilities for class 1
    threshold : decision boundary (default 0.5)
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        'Accuracy (%)': round(accuracy_score(y_true, y_pred) * 100, 2),
        'AUC-ROC':      round(roc_auc_score(y_true, y_prob), 4),
        'F1 Score':     round(f1_score(y_true, y_pred), 4),
    }


def compute_subperiod_metrics(
    returns_series: pd.Series,
    rf_daily: float = RF_DAILY,
    subperiods: dict = None,
) -> pd.DataFrame:
    """
    Compute compute_metrics() for each market-regime sub-period (thesis T6 / F9).

    Parameters
    ----------
    returns_series : pd.Series indexed by date (datetime or string).
    subperiods     : dict mapping label → (start_str, end_str).
                     Defaults to the five regimes defined in CLAUDE.md.

    Returns
    -------
    pd.DataFrame with one row per sub-period and metric columns.
    """
    if subperiods is None:
        subperiods = SUBPERIODS

    idx = pd.to_datetime(returns_series.index)
    rows = {}
    for label, (start, end) in subperiods.items():
        mask = (idx >= start) & (idx <= end)
        sub  = returns_series[mask]
        rows[label] = compute_metrics(sub, rf_daily) if len(sub) > 0 else {}

    return pd.DataFrame(rows).T


def compute_tc_sensitivity(
    signals_df: pd.DataFrame,
    tc_grid: list = None,
) -> pd.DataFrame:
    """
    Compute Sharpe ratio and annualised return for a range of TC values (thesis T7 / F7).

    Parameters
    ----------
    signals_df : output of generate_signals() — must contain Date, Ticker,
                 Signal, Return_NextDay.
    tc_grid    : list of TC values in basis points to evaluate
                 (default: 0, 2, 5, 10, 15, 20, 25, 30).

    Returns
    -------
    pd.DataFrame indexed by TC (bps) with Sharpe and Annualized Return columns.
    """
    from backtest.portfolio import compute_portfolio_returns

    if tc_grid is None:
        tc_grid = [0, 2, 5, 10, 15, 20, 25, 30]

    rows = {}
    for tc in tc_grid:
        port = compute_portfolio_returns(signals_df, tc_bps=tc)
        m    = compute_metrics(port['Net_Return'])
        rows[tc] = {
            'Sharpe Ratio':          m.get('Sharpe Ratio', np.nan),
            'Annualized Return (%)':  m.get('Annualized Return (%)', np.nan),
            'Max Drawdown (%)':       m.get('Max Drawdown (%)', np.nan),
        }

    df = pd.DataFrame(rows).T
    df.index.name = 'TC (bps)'
    return df
