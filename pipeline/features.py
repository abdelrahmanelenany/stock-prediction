"""
pipeline/features.py
Compute only the 8 active features used across all models (Section 6 of IMPLEMENTATION_EXTENSIONS.md).
Includes sector-relative return computation.

Features computed:
    Return_1d, RSI_14, MACD, ATR_14, BB_PctB, RealVol_20d, Volume_Ratio, SectorRelReturn
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config

# ── Feature column names (8 total per Section 6) ──────────────────────────────
FEATURE_COLS = config.ALL_FEATURE_COLS
TARGET_COL = 'Target'


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 10 features actively used across all models.
    Includes multi-day momentum features (Return_5d, Return_21d) for
    improved predictive power.

    ANTI-LEAKAGE: All indicators are causal — each value at time t
    uses only data from t and earlier.
    """
    df = df.copy().sort_values("Date")

    # ── Return_1d ─────────────────────────────────────────────────────
    df["Return_1d"] = df["Close"].pct_change(1)

    # ── Multi-day Momentum (Fischer & Krauss 2017 style) ───────────────
    df["Return_5d"] = df["Close"].pct_change(5)    # Weekly momentum
    df["Return_21d"] = df["Close"].pct_change(21)  # Monthly momentum

    # ── RSI_14 (Bhandari §4.3) ────────────────────────────────────────
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI_14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # ── MACD (Bhandari §4.3 — 12/26 EMA difference) ───────────────────
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    # Note: MACD_Signal is NOT computed — not in any active feature set

    # ── ATR_14 (Bhandari §4.3) ────────────────────────────────────────
    hl = df["High"] - df["Low"]
    hpc = (df["High"] - df["Close"].shift()).abs()
    lpc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()

    # ── BB_PctB (%B position within Bollinger Bands) ───────────────────
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    df["BB_PctB"] = (df["Close"] - lower) / (upper - lower + 1e-10)
    # Note: BB_Width is NOT computed — not in any active feature set

    # ── RealVol_20d (annualised 20-day realised volatility) ───────────
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    df["RealVol_20d"] = log_ret.rolling(20).std() * np.sqrt(252)

    # ── Volume_Ratio ──────────────────────────────────────────────────
    df["Volume_Ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1)

    # Note: OBV, HL_Pct, HL_Pct_5d, Mom_10d are NOT computed

    return df


def compute_sector_rel_return(df: pd.DataFrame, sector_map: dict, 
                              sector_min_size: int = 5, 
                              sector_winsorize: bool = True,
                              sector_winsorize_pct: float = 0.05) -> pd.DataFrame:
    """
    Compute SectorRelReturn: each stock's Return_1d minus the equal-weighted
    mean Return_1d of all stocks in the same sector on that date.

    ANTI-LEAKAGE: Uses only same-day cross-sectional returns — no future data.
    The sector mean at time t is the realised (not forecast) mean of t, which is
    legitimate: it is a feature known after market close on day t, used to predict
    day t+1.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with columns [Date, Ticker, Return_1d].
        Must already have Return_1d computed.
    sector_map : dict
        {ticker: sector_name} mapping from config.py.
    sector_min_size : int
        Minimum number of stocks required in a sector to compute relative return.
        If fewer, SectorRelReturn is set to NaN.
    sector_winsorize : bool
        If True, winsorizes the resulting feature at sector_winsorize_pct and 1-sector_winsorize_pct cross-sectionally per date.
    sector_winsorize_pct : float
        Percentile for winsorization.

    Returns
    -------
    pd.DataFrame with added column 'SectorRelReturn'.
    """
    df = df.copy()
    df['Sector'] = df['Ticker'].map(sector_map)

    # Handle tickers not in sector_map (assign to 'Unknown')
    df['Sector'] = df['Sector'].fillna('Unknown')

    # Vectorised implementation (faster than apply for large DataFrames)
    # Step 1: compute sector mean per (Date, Sector) over all tickers
    sector_means = (
        df.groupby(['Date', 'Sector'])['Return_1d']
          .transform('mean')
    )
    # Step 2: subtract self-contribution to get leave-one-out mean
    sector_counts = (
        df.groupby(['Date', 'Sector'])['Return_1d']
          .transform('count')
    )
    # Leave-one-out sector mean:
    # mean_excl_self = (sum - self_value) / (count - 1)
    sector_sums = sector_means * sector_counts
    sr_return = (
        df['Return_1d']
        - (sector_sums - df['Return_1d']) / (sector_counts - 1).clip(lower=1)
    )

    # Set to NaN if sector size < sector_min_size
    sr_return = sr_return.where(sector_counts >= sector_min_size, np.nan)

    if sector_winsorize:
        # Winsorize at sector_winsorize_pct and 1-sector_winsorize_pct percentile within each date
        def winsorize_group(x):
            if x.dropna().empty:
                return x
            lower_q = x.quantile(sector_winsorize_pct)
            upper_q = x.quantile(1.0 - sector_winsorize_pct)
            return x.clip(lower=lower_q, upper=upper_q)
            
        sr_return = df.assign(sr=sr_return).groupby('Date')['sr'].transform(winsorize_group)

    df['SectorRelReturn'] = sr_return

    df.drop(columns=['Sector'], inplace=True)
    return df


def compute_market_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Market-wide features: cross-sectional mean returns (same day, PIT at t),
    rolling market vol on the market return series, excess returns vs market,
    rolling CAPM-style beta vs market (past window only per ticker).
    """
    df = df.copy()
    vol_windows = getattr(config, 'MARKET_VOL_WINDOWS', (20, 60))

    agg_cols = ['Return_1d', 'Return_5d']
    if 'Return_21d' in df.columns:
        agg_cols.append('Return_21d')

    market_daily = df.groupby('Date')[agg_cols].mean().reset_index()
    rename = {'Return_1d': 'Market_Return_1d', 'Return_5d': 'Market_Return_5d'}
    if 'Return_21d' in agg_cols:
        rename['Return_21d'] = 'Market_Return_21d'
    market_daily.rename(columns=rename, inplace=True)
    market_daily = market_daily.sort_values('Date')

    for w in vol_windows:
        col = f'Market_Vol_{w}d'
        market_daily[col] = (
            market_daily['Market_Return_1d'].rolling(int(w)).std() * np.sqrt(252)
        )

    df = df.merge(market_daily, on='Date', how='left')

    df['RelToMarket_1d'] = df['Return_1d'] - df['Market_Return_1d']
    df['RelToMarket_5d'] = df['Return_5d'] - df['Market_Return_5d']
    if 'Market_Return_21d' in df.columns and 'Return_21d' in df.columns:
        df['RelToMarket_21d'] = df['Return_21d'] - df['Market_Return_21d']

    beta_w = int(getattr(config, 'BETA_WINDOW', 60))
    beta_col = f'Beta_{beta_w}d'
    beta_parts = []
    for _, g in df.groupby('Ticker'):
        g = g.sort_values('Date').copy()
        rm = g['Market_Return_1d']
        rs = g['Return_1d']
        c = rs.rolling(beta_w).cov(rm)
        v = rm.rolling(beta_w).var() + 1e-12
        g[beta_col] = c / v
        beta_parts.append(g)
    df = pd.concat(beta_parts).sort_values(['Date', 'Ticker'])
    return df


def compute_sector_context_features(df: pd.DataFrame, sector_map: dict) -> pd.DataFrame:
    """
    Leave-one-out sector mean returns, rolling sector vol on sector mean return,
    within (Date, Sector) z-score of returns (same-day peers only).
    """
    df = df.copy()
    if 'Sector' not in df.columns:
        df['Sector'] = df['Ticker'].map(sector_map).fillna('Unknown')

    def leave_one_out_mean(col_name: str, new_col_name: str) -> None:
        sector_means = df.groupby(['Date', 'Sector'])[col_name].transform('mean')
        sector_counts = df.groupby(['Date', 'Sector'])[col_name].transform('count')
        sector_sums = sector_means * sector_counts
        df[new_col_name] = np.where(
            sector_counts > 1,
            (sector_sums - df[col_name]) / (sector_counts - 1),
            0.0,
        )

    leave_one_out_mean('Return_1d', 'Sector_Return_1d')
    leave_one_out_mean('Return_5d', 'Sector_Return_5d')
    if 'Return_21d' in df.columns:
        leave_one_out_mean('Return_21d', 'Sector_Return_21d')

    extra_vol_w = getattr(config, 'SECTOR_VOL_EXTRA_WINDOWS', (60,))
    sector_daily = df.groupby(['Date', 'Sector'])['Return_1d'].mean().reset_index()
    sector_daily = sector_daily.sort_values('Date')
    sector_daily['Sector_Vol_20d'] = sector_daily.groupby('Sector')['Return_1d'].transform(
        lambda x: x.rolling(20).std() * np.sqrt(252)
    )
    for w in extra_vol_w:
        if int(w) == 20:
            continue
        sector_daily[f'Sector_Vol_{int(w)}d'] = sector_daily.groupby('Sector')['Return_1d'].transform(
            lambda x, ww=int(w): x.rolling(ww).std() * np.sqrt(252)
        )

    merge_cols = ['Date', 'Sector'] + [c for c in sector_daily.columns if c.startswith('Sector_Vol')]
    df = df.merge(sector_daily[merge_cols], on=['Date', 'Sector'], how='left')

    df['SectorRelZ_Return_1d'] = df.groupby(['Date', 'Sector'])['Return_1d'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-12)
    )

    df.drop(columns=['Sector'], inplace=True)
    return df


def build_feature_matrix(data: pd.DataFrame, sector_min_size: int = 5, sector_winsorize: bool = True, sector_winsorize_pct: float = 0.05) -> pd.DataFrame:
    """
    Runs the full feature pipeline on the long-format OHLCV DataFrame:
      1. Compute 8 technical features per ticker (from RAW Close prices)
      2. Compute SectorRelReturn cross-sectional feature
      3. Drops rows with any NaN in feature columns
    4. Saves to data/processed/features_{UNIVERSE_MODE}.csv
      
    Returns the enriched DataFrame (Date, Ticker, OHLCV, features...).
    """
    # Step 1 — Compute technical indicators per ticker (from raw Close)
    print("Computing technical indicators per ticker (raw Close)...")
    enriched = []
    for ticker, group in data.groupby('Ticker'):
        enriched.append(compute_technical_features(group))
    result = pd.concat(enriched).sort_values(['Date', 'Ticker']).reset_index(drop=True)

    # Step 2 — Compute SectorRelReturn and context features
    print("Computing SectorRelReturn and Context Features...")
    result = compute_sector_rel_return(
        result, sector_map=config.SECTOR_MAP,
        sector_min_size=sector_min_size,
        sector_winsorize=sector_winsorize,
        sector_winsorize_pct=sector_winsorize_pct
    )
    
    if getattr(config, 'MARKET_FEATURES_ENABLED', False):
        result = compute_market_context_features(result)
        
    if getattr(config, 'SECTOR_FEATURES_ENABLED', False):
        result = compute_sector_context_features(result, sector_map=config.SECTOR_MAP)

    print("Computed dataframe columns:")
    print(result.columns.tolist())

    # Step 3 — Drop NaN rows
    before = len(result)
    result.dropna(subset=FEATURE_COLS, inplace=True)
    result.reset_index(drop=True, inplace=True)
    print(f"Dropped {before - len(result)} rows with NaN features "
          f"(expected: ~26 per ticker for MACD warmup)")

    # Step 4 — Save
    features_cache = f'data/processed/features_{config.UNIVERSE_MODE}.csv'
    result.to_csv(features_cache, index=False)
    print(f"\nSaved {len(result)} rows to {features_cache}")
    print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"Date range: {result['Date'].min()} -> {result['Date'].max()}")
    print(f"Tickers: {len(result['Ticker'].unique())}")
    return result


if __name__ == '__main__':
    data = pd.read_csv('data/raw/ohlcv_long.csv', parse_dates=['Date'])
    build_feature_matrix(data)
