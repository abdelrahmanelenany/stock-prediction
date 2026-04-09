"""
pipeline/features.py
Compute only the 8 active features used across all models (Section 6 of IMPLEMENTATION_EXTENSIONS.md).
Includes wavelet denoising (Bhandari §4.5) and sector-relative return computation.

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


def compute_wavelet_threshold(close_series: pd.Series, wavelet: str = "haar",
                               level: int = 1) -> float:
    """
    Compute the wavelet denoising threshold from a training price series.
    
    ANTI-LEAKAGE: This must only be called on training data. The returned
    threshold is then applied to denoise val/test data without recomputation.

    Parameters
    ----------
    close_series : pd.Series
        Raw daily close prices for training window only.
    wavelet : str
        Wavelet family. 'haar' is the paper's choice.
    level : int
        Decomposition level.

    Returns
    -------
    float
        The universal threshold value computed from training data.
    """
    try:
        import pywt
    except ImportError:
        return 0.0

    prices = close_series.values.astype(float)
    if len(prices) < 4:
        return 0.0

    coeffs = pywt.wavedec(prices, wavelet=wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(prices)))
    return threshold


def denoise_close_price(close_series: pd.Series, wavelet: str = "haar",
                        level: int = 1, mode: str = "soft",
                        threshold: float = None) -> pd.Series:
    """
    Bhandari §4.5 — Haar wavelet soft denoising of close price series.

    Uses PyWavelets (pywt) — a pure-Python library requiring no GPU.
    Install: pip install PyWavelets

    Parameters
    ----------
    close_series : pd.Series
        Raw daily close prices for a single ticker (chronological order).
    wavelet : str
        Wavelet family. 'haar' is the paper's choice.
    level : int
        Decomposition level. Level 1 is sufficient for daily data.
    mode : str
        Thresholding mode: 'soft' (paper) or 'hard'.
    threshold : float, optional
        Pre-computed threshold from training data. If None, threshold is
        computed from close_series itself (use only for training data).

    Returns
    -------
    pd.Series
        Denoised close prices with the same index.
    """
    try:
        import pywt
    except ImportError:
        print("Warning: PyWavelets not installed. Skipping wavelet denoising.")
        print("Install with: pip install PyWavelets")
        return close_series

    prices = close_series.values.astype(float)

    # Handle edge case of too few data points
    if len(prices) < 4:
        return close_series

    # Wavelet decomposition
    coeffs = pywt.wavedec(prices, wavelet=wavelet, level=level)

    # Use provided threshold or compute from this series (training only)
    if threshold is None:
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(prices)))

    # Apply soft thresholding to all detail coefficient levels
    coeffs_thresholded = [coeffs[0]]  # keep approximation coefficients unchanged
    for detail in coeffs[1:]:
        coeffs_thresholded.append(pywt.threshold(detail, threshold, mode=mode))

    # Reconstruct denoised signal
    denoised = pywt.waverec(coeffs_thresholded, wavelet=wavelet)

    # waverec may produce one extra sample due to odd-length inputs — trim
    denoised = denoised[:len(prices)]

    return pd.Series(denoised, index=close_series.index, name=close_series.name)


def denoise_close_price_causal(close_series: pd.Series, threshold: float,
                                window_size: int = 128,
                                wavelet: str = "haar",
                                level: int = 1,
                                mode: str = "soft") -> pd.Series:
    """
    CAUSAL wavelet denoising: each value at time t uses only a fixed lookback window.

    This prevents look-ahead bias by ensuring the denoised value at time t
    depends only on prices from times [t - window_size, t], not future data.

    Parameters
    ----------
    close_series : pd.Series
        Raw daily close prices (chronological order).
    threshold : float
        Pre-computed threshold from training data (required).
    window_size : int
        Number of historical points to use for each denoising operation.
        Must be >= 4 for wavelet decomposition.
    wavelet : str
        Wavelet family ('haar' is the paper's choice).
    level : int
        Decomposition level.
    mode : str
        Thresholding mode: 'soft' (paper) or 'hard'.

    Returns
    -------
    pd.Series
        Causally denoised close prices (same index as input).
    """
    try:
        import pywt
    except ImportError:
        print("Warning: PyWavelets not installed. Returning raw prices.")
        return close_series

    prices = close_series.values.astype(float)
    denoised = np.full(len(prices), np.nan)

    # For each time point >= window_size, apply wavelet denoising to lookback window
    for t in range(window_size, len(prices)):
        # Fixed lookback window ending at t (inclusive)
        window = prices[t - window_size:t + 1]

        # Wavelet decomposition
        coeffs = pywt.wavedec(window, wavelet=wavelet, level=level)

        # Apply thresholding to detail coefficients
        coeffs_thresh = [coeffs[0]]  # Keep approximation unchanged
        for detail in coeffs[1:]:
            coeffs_thresh.append(pywt.threshold(detail, threshold, mode=mode))

        # Reconstruct and take only the LAST value (current time t)
        reconstructed = pywt.waverec(coeffs_thresh, wavelet=wavelet)
        denoised[t] = reconstructed[-1]

    # For early points (warm-up period), use raw prices
    denoised[:window_size] = prices[:window_size]

    return pd.Series(denoised, index=close_series.index, name=close_series.name)


def compute_wavelet_thresholds(df_train: pd.DataFrame) -> dict:
    """
    Compute wavelet denoising thresholds from training data only.
    
    ANTI-LEAKAGE: These thresholds are computed once per fold on df_train,
    then reused to denoise df_val and df_test without recomputation.
    
    Parameters
    ----------
    df_train : pd.DataFrame
        Training fold data with columns [Date, Ticker, Close, ...].
        
    Returns
    -------
    dict
        {ticker: threshold} mapping for each ticker in the training set.
    """
    thresholds = {}
    for ticker, grp in df_train.groupby("Ticker"):
        grp = grp.sort_values("Date")
        thresholds[ticker] = compute_wavelet_threshold(
            grp["Close"],
            wavelet=config.WAVELET_TYPE,
            level=config.WAVELET_LEVEL,
        )
    return thresholds


def apply_wavelet_denoising(df: pd.DataFrame, 
                            thresholds: dict = None) -> pd.DataFrame:
    """
    Apply denoise_close_price() to each ticker independently.
    Replaces the 'Close' column with the denoised version.
    Call this BEFORE compute_technical_features() and compute_returns().
    
    ANTI-LEAKAGE: If thresholds dict is provided, uses pre-computed thresholds
    (for val/test data). If None, computes thresholds from the data itself
    (only appropriate for training data).
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with columns [Date, Ticker, Close, ...].
    thresholds : dict, optional
        {ticker: threshold} pre-computed from training data.
        If None, thresholds are computed per-ticker from df itself.
        
    Returns
    -------
    pd.DataFrame
        Copy of df with 'Close' column replaced by denoised values.
    """
    df = df.copy()
    denoised_rows = []
    for ticker, grp in df.groupby("Ticker"):
        grp = grp.sort_values("Date").copy()
        # Use pre-computed threshold if available, else compute from this series
        thresh = thresholds.get(ticker) if thresholds else None
        grp["Close"] = denoise_close_price(
            grp["Close"],
            wavelet=config.WAVELET_TYPE,
            level=config.WAVELET_LEVEL,
            mode=config.WAVELET_MODE,
            threshold=thresh,
        )
        denoised_rows.append(grp)
    return pd.concat(denoised_rows).sort_values(["Date", "Ticker"])


def apply_wavelet_denoising_causal(df: pd.DataFrame,
                                    thresholds: dict,
                                    window_size: int = None) -> pd.DataFrame:
    """
    Apply CAUSAL wavelet denoising to each ticker independently.

    CRITICAL FIX: This function uses rolling-window denoising to prevent
    look-ahead bias. Each denoised value at time t only uses historical data
    from [t - window_size, t].

    Unlike apply_wavelet_denoising(), this function:
    1. REQUIRES pre-computed thresholds (no auto-computation)
    2. Uses denoise_close_price_causal() instead of denoise_close_price()
    3. Applies denoising per-split independently (no concatenation)

    Parameters
    ----------
    df : pd.DataFrame
        Data with columns [Date, Ticker, Close, ...].
    thresholds : dict
        {ticker: threshold} pre-computed from training data (REQUIRED).
    window_size : int, optional
        Lookback window size. Defaults to config.WAVELET_WINDOW_SIZE.

    Returns
    -------
    pd.DataFrame
        Copy of df with 'Close' column replaced by causally denoised values.
    """
    if window_size is None:
        window_size = getattr(config, 'WAVELET_WINDOW_SIZE', 128)

    df = df.copy()
    denoised_rows = []

    for ticker, grp in df.groupby("Ticker"):
        grp = grp.sort_values("Date").copy()

        # Get pre-computed threshold (required for causal denoising)
        thresh = thresholds.get(ticker)
        if thresh is None:
            # Ticker not in training set - use raw prices
            denoised_rows.append(grp)
            continue

        grp["Close"] = denoise_close_price_causal(
            grp["Close"],
            threshold=thresh,
            window_size=window_size,
            wavelet=config.WAVELET_TYPE,
            level=config.WAVELET_LEVEL,
            mode=config.WAVELET_MODE,
        )
        denoised_rows.append(grp)

    return pd.concat(denoised_rows).sort_values(["Date", "Ticker"])


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

    if 'NegReturn_1d' not in df.columns:
        df['NegReturn_1d'] = -df['Return_1d']

    if 'NegReturn_5d' not in df.columns:
        df['NegReturn_5d'] = -df['Return_5d']

    if 'RSI_Reversal' not in df.columns:
        df['RSI_Reversal'] = 100 - df['RSI_14']

    if 'NegMACD' not in df.columns:
        df['NegMACD'] = -df['MACD']

    if 'BB_Reversal' not in df.columns:
        df['BB_Reversal'] = 1 - df['BB_PctB']

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


def apply_reversal_orientation(df: pd.DataFrame, universe_cfg) -> pd.DataFrame:
    """
    Add reversal-oriented feature aliases for mean-reversion universes.

    Original columns are preserved for compatibility across universe configs.
    """
    if not getattr(universe_cfg, "invert_features", False):
        return df

    if 'NegReturn_1d' not in df.columns:
        df['NegReturn_1d'] = -df['Return_1d']

    if 'NegReturn_5d' not in df.columns:
        df['NegReturn_5d'] = -df['Return_5d']

    if 'RSI_Reversal' not in df.columns:
        df['RSI_Reversal'] = 100 - df['RSI_14']

    if 'NegMACD' not in df.columns:
        df['NegMACD'] = -df['MACD']

    if 'BB_Reversal' not in df.columns:
        df['BB_Reversal'] = 1 - df['BB_PctB']
    return df


def recompute_features_from_denoised(df: pd.DataFrame, sector_min_size: int = 5, sector_winsorize: bool = True, sector_winsorize_pct: float = 0.05) -> pd.DataFrame:
    """
    Recompute all Close-dependent features from denoised Close prices.
    
    This function is called per-fold AFTER wavelet denoising has been applied
    with training-derived thresholds. It recomputes:
      - Return_1d, RSI_14, MACD, ATR_14, BB_PctB, RealVol_20d
      
    Volume_Ratio uses only Volume (no Close dependency) so is unchanged.
    SectorRelReturn depends on Return_1d, so must be recomputed after this.
    
    ANTI-LEAKAGE: Called on train/val/test splits separately after denoising
    with training-only thresholds.
    
    Parameters
    ----------
    df : pd.DataFrame
        Fold data with denoised 'Close' column and OHLCV.
        
    Returns
    -------
    pd.DataFrame
        Copy with recomputed feature columns.
    """
    result_parts = []
    for ticker, grp in df.groupby('Ticker'):
        grp = grp.copy().sort_values('Date')
        
        # ── Return_1d ─────────────────────────────────────────────────────
        grp["Return_1d"] = grp["Close"].pct_change(1)

        # ── RSI_14 (Bhandari §4.3) ────────────────────────────────────────
        delta = grp["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        grp["RSI_14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

        # ── MACD (Bhandari §4.3 — 12/26 EMA difference) ───────────────────
        ema12 = grp["Close"].ewm(span=12, adjust=False).mean()
        ema26 = grp["Close"].ewm(span=26, adjust=False).mean()
        grp["MACD"] = ema12 - ema26

        # ── ATR_14 (Bhandari §4.3) ────────────────────────────────────────
        hl = grp["High"] - grp["Low"]
        hpc = (grp["High"] - grp["Close"].shift()).abs()
        lpc = (grp["Low"] - grp["Close"].shift()).abs()
        tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
        grp["ATR_14"] = tr.rolling(14).mean()

        # ── BB_PctB (%B position within Bollinger Bands) ───────────────────
        sma20 = grp["Close"].rolling(20).mean()
        std20 = grp["Close"].rolling(20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        grp["BB_PctB"] = (grp["Close"] - lower) / (upper - lower + 1e-10)

        # ── RealVol_20d (annualised 20-day realised volatility) ───────────
        log_ret = np.log(grp["Close"] / grp["Close"].shift(1))
        grp["RealVol_20d"] = log_ret.rolling(20).std() * np.sqrt(252)

        # Volume_Ratio is unchanged (no Close dependency)
        
        result_parts.append(grp)
    
    result = pd.concat(result_parts).sort_values(['Date', 'Ticker'])
    
    # Recompute SectorRelReturn (depends on Return_1d)
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

    universe_cfg = config.LARGE_CAP_CONFIG if config.UNIVERSE_MODE == "large_cap" else config.SMALL_CAP_CONFIG
    result = apply_reversal_orientation(result, universe_cfg)
        
    return result


def build_feature_matrix(data: pd.DataFrame, sector_min_size: int = 5, sector_winsorize: bool = True, sector_winsorize_pct: float = 0.05) -> pd.DataFrame:
    """
    Runs the full feature pipeline on the long-format OHLCV DataFrame:
      1. Compute 8 technical features per ticker (from RAW Close prices)
      2. Compute SectorRelReturn cross-sectional feature
      3. Drops rows with any NaN in feature columns
    4. Saves to data/processed/features_{UNIVERSE_MODE}.csv
      
    NOTE: Wavelet denoising is intentionally NOT applied here to prevent
    look-ahead bias. Denoising must be done per fold inside the walk-forward
    loop in main.py, with thresholds computed only on training data.
    The cached CSV preserves raw OHLCV columns for per-fold recomputation.

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

    universe_cfg = config.LARGE_CAP_CONFIG if config.UNIVERSE_MODE == "large_cap" else config.SMALL_CAP_CONFIG
    result = apply_reversal_orientation(result, universe_cfg)

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
    print(f"NOTE: Wavelet denoising is applied per-fold in main.py (not cached)")
    return result


if __name__ == '__main__':
    data = pd.read_csv('data/raw/ohlcv_long.csv', parse_dates=['Date'])
    build_feature_matrix(data)
