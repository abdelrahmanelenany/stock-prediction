"""
backtest/portfolio.py
Step 8: Daily equal-weighted long-short P&L with transaction costs.

Transaction cost model (Fischer & Krauss 2017):
  - tc_bps basis points charged per half-turn
  - Each position change affects 1/(2*k) of the total portfolio
    (2 legs × k positions per leg = 2k active positions)
  - Day 1: all positions are new → full turnover cost on all active positions
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import TC_BPS, K_STOCKS


def compute_portfolio_returns(
    signals_df: pd.DataFrame,
    tc_bps: float = TC_BPS,
    k: int = K_STOCKS,
) -> pd.DataFrame:
    """
    Compute daily gross and net portfolio returns for a long-short strategy.

    Portfolio construction per day:
      long_ret  = equal-weighted mean return of Long positions
      short_ret = equal-weighted mean return of Short positions
      gross_ret = long_ret - short_ret
      net_ret   = gross_ret - turnover * tc / (2 * k)

    Each position change only affects 1/(2*k) of the portfolio, since
    we have 2 legs (long + short) with k equal-weight positions each.

    Parameters
    ----------
    signals_df : pd.DataFrame
        Output of backtest.signals.generate_signals() — must contain:
        Date, Ticker, Signal ('Long'/'Short'/'Hold'), Return_NextDay.
    tc_bps : float
        Transaction cost per half-turn in basis points (default 5 bps = 0.0005).
    k : int
        Number of long (and short) positions per day.

    Returns
    -------
    pd.DataFrame indexed by Date with columns:
        Gross_Return, Net_Return, Long_Return, Short_Return,
        TC, Turnover (number of position changes that day)
    """
    tc = tc_bps / 10_000
    daily = []
    prev_signals: dict[str, str] = {}   # ticker → last signal

    for date, group in signals_df.sort_values('Date').groupby('Date'):
        longs  = group[group['Signal'] == 'Long']
        shorts = group[group['Signal'] == 'Short']

        long_ret  = longs['Return_NextDay'].mean()  if len(longs)  > 0 else 0.0
        short_ret = shorts['Return_NextDay'].mean() if len(shorts) > 0 else 0.0
        gross_ret = long_ret - short_ret

        # Count position changes vs previous day (each change = one half-turn)
        curr = dict(zip(group['Ticker'], group['Signal']))
        turnover = sum(1 for t, sig in curr.items() if sig != prev_signals.get(t, 'Hold'))

        # Each position change affects 1/(2*k) of the portfolio
        tc_cost = turnover * tc / (2 * k)
        net_ret = gross_ret - tc_cost
        prev_signals = curr

        daily.append({
            'Date':         date,
            'Gross_Return': gross_ret,
            'Net_Return':   net_ret,
            'Long_Return':  long_ret,
            'Short_Return': short_ret,
            'TC':           tc_cost,
            'Turnover':     turnover,
        })

    return pd.DataFrame(daily).set_index('Date')


if __name__ == '__main__':
    # Smoke test with synthetic signals
    import numpy as np
    from backtest.signals import generate_signals

    np.random.seed(42)
    dates   = pd.date_range('2022-01-03', periods=10, freq='B')
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
    pred_df  = pd.DataFrame(rows)
    sig_df   = generate_signals(pred_df, k=2)
    port     = compute_portfolio_returns(sig_df, tc_bps=5, k=2)

    print('\nPortfolio daily returns (10 days):')
    print(port.round(6).to_string())
    print(f'\nMean gross return : {port["Gross_Return"].mean()*100:.4f}%')
    print(f'Mean net return   : {port["Net_Return"].mean()*100:.4f}%')
    print(f'Mean TC per day   : {port["TC"].mean()*100:.4f}%')
    print(f'Mean turnover/day : {port["Turnover"].mean():.1f} position changes')

    assert len(port) == len(dates), 'Wrong number of rows'
    assert (port['Net_Return'] <= port['Gross_Return']).all(), 'Net > Gross — TC bug'
    print('\nAll portfolio checks passed.')
