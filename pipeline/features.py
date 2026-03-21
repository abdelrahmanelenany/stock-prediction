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


def denoise_close_price(close_series: pd.Series, wavelet: str = "haar",
                        level: int = 1, mode: str = "soft") -> pd.Series:
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

    # Estimate noise sigma from detail coefficients (median absolute deviation)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745   # universal threshold formula

    # Universal threshold (Donoho & Johnstone 1994)
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


def apply_wavelet_denoising(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply denoise_close_price() to each ticker independently.
    Replaces the 'Close' column with the denoised version.
    Call this BEFORE compute_technical_features() and compute_returns().
    """
    df = df.copy()
    denoised_rows = []
    for ticker, grp in df.groupby("Ticker"):
        grp = grp.sort_values("Date").copy()
        grp["Close"] = denoise_close_price(
            grp["Close"],
            wavelet=config.WAVELET_TYPE,
            level=config.WAVELET_LEVEL,
            mode=config.WAVELET_MODE,
        )
        denoised_rows.append(grp)
    return pd.concat(denoised_rows).sort_values(["Date", "Ticker"])


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute only the 8 features actively used across all models.
    All other previously computed features (lagged returns, MACD_Signal,
    BB_Width, OBV, HL_Pct, Mom_10d) are intentionally omitted.

    ANTI-LEAKAGE: All indicators are causal — each value at time t
    uses only data from t and earlier.
    """
    df = df.copy().sort_values("Date")

    # ── Return_1d ─────────────────────────────────────────────────────
    df["Return_1d"] = df["Close"].pct_change(1)

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


def compute_sector_rel_return(df: pd.DataFrame, sector_map: dict) -> pd.DataFrame:
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
    df['SectorRelReturn'] = (
        df['Return_1d']
        - (sector_sums - df['Return_1d']) / (sector_counts - 1).clip(lower=1)
    )

    df.drop(columns=['Sector'], inplace=True)
    return df


def build_feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the full feature pipeline on the long-format OHLCV DataFrame:
      1. (Optional) Apply wavelet denoising to Close prices
      2. Compute 8 technical features per ticker
      3. Compute SectorRelReturn cross-sectional feature
      4. Drops rows with any NaN in feature columns
      5. Saves to data/processed/features.csv

    Returns the enriched DataFrame (Date, Ticker, OHLCV, features...).
    """
    # Step 1 — Optional wavelet denoising (Bhandari §4.5)
    if config.USE_WAVELET_DENOISING:
        print("Applying wavelet denoising to Close prices...")
        data = apply_wavelet_denoising(data)

    # Step 2 — Compute technical indicators per ticker
    print("Computing technical indicators per ticker...")
    enriched = []
    for ticker, group in data.groupby('Ticker'):
        enriched.append(compute_technical_features(group))
    result = pd.concat(enriched).sort_values(['Date', 'Ticker']).reset_index(drop=True)

    # Step 3 — Compute SectorRelReturn
    print("Computing SectorRelReturn...")
    result = compute_sector_rel_return(result, sector_map=config.SECTOR_MAP)

    # Step 4 — Drop NaN rows
    before = len(result)
    result.dropna(subset=FEATURE_COLS, inplace=True)
    result.reset_index(drop=True, inplace=True)
    print(f"Dropped {before - len(result)} rows with NaN features "
          f"(expected: ~26 per ticker for MACD warmup)")

    # Step 5 — Save
    result.to_csv('data/processed/features.csv', index=False)
    print(f"\nSaved {len(result)} rows to data/processed/features.csv")
    print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"Date range: {result['Date'].min()} -> {result['Date'].max()}")
    print(f"Tickers: {len(result['Ticker'].unique())}")
    return result


if __name__ == '__main__':
    data = pd.read_csv('data/raw/ohlcv_long.csv', parse_dates=['Date'])
    build_feature_matrix(data)
