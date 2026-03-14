"""
main.py — Full pipeline orchestrator
Walk-forward validated long-short strategy using LR, RF, XGBoost, LSTM + Ensemble.

Run:
    .venv/bin/python3 main.py
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'    # Prevent XGBoost/PyTorch dual-OpenMP segfault

import time
import pandas as pd
import numpy as np
import xgboost as xgb
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pipeline.data_loader  import download_and_save
from pipeline.features     import build_feature_matrix, FEATURE_COLS
from pipeline.targets      import create_targets
from pipeline.walk_forward import generate_walk_forward_folds, print_fold_summary
from pipeline.standardizer import standardize_fold
from models.baselines      import train_logistic, train_random_forest, train_xgboost
from models.lstm_model     import StockLSTM, StockSequenceDataset, train_lstm, lstm_predict
from models.ensemble       import ensemble_predict
from backtest.signals      import generate_signals
from backtest.portfolio    import compute_portfolio_returns
from backtest.metrics      import (
    compute_metrics, evaluate_classification,
    compute_subperiod_metrics, compute_tc_sensitivity,
)
from config import (
    TRAIN_DAYS, VAL_DAYS, TEST_DAYS,
    SEQ_LEN, N_TOTAL_FEATURES,
    LSTM_BATCH, K_STOCKS, TC_BPS,
)

TARGET_COL = 'Target'

device = (
    torch.device('mps') if torch.backends.mps.is_available()
    else torch.device('cpu')
)
# MPS is slower than CPU for small LSTM (hidden=64): Metal kernel dispatch
# overhead dominates for tiny matrix ops. Always train LSTM on CPU.
lstm_device = torch.device('cpu')
print(f'Using device: {device}  (LSTM will use: {lstm_device})')


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def scaled_df(df_orig: pd.DataFrame, X_scaled: np.ndarray) -> pd.DataFrame:
    """Return a copy of df_orig with FEATURE_COLS replaced by scaled values."""
    d = df_orig.copy()
    d[FEATURE_COLS] = X_scaled
    return d


def align_lstm_predictions(
    probs: np.ndarray,
    keys: list,
    df_ts: pd.DataFrame,
) -> np.ndarray:
    """
    Map LSTM output (aligned to dataset keys) back to all rows of df_ts.

    Rows that had no LSTM sequence (first SEQ_LEN rows per ticker per fold)
    receive np.nan and are dropped before signal generation.
    """
    prob_map = {
        (pd.Timestamp(date).strftime('%Y-%m-%d'), ticker): float(prob)
        for (date, ticker), prob in zip(keys, probs)
    }

    result = np.full(len(df_ts), np.nan)
    for i, (_, row) in enumerate(df_ts.iterrows()):
        key = (pd.Timestamp(row['Date']).strftime('%Y-%m-%d'), row['Ticker'])
        if key in prob_map:
            result[i] = prob_map[key]
    return result


# ────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ────────────────────────────────────────────────────────────────────────────

def main(load_cached: bool = True):
    """
    Parameters
    ----------
    load_cached : bool
        If True (default), load data from data/processed/features.csv instead
        of re-downloading and recomputing. Set False to run from scratch.
    """

    # ── Steps 1-3: Data ─────────────────────────────────────────────────────
    if load_cached:
        print('Loading cached features.csv ...')
        data = pd.read_csv('data/processed/features.csv', parse_dates=['Date'])
        print(f'Loaded {len(data)} rows, {len(data.columns)} columns.')
    else:
        raw  = download_and_save()
        data = build_feature_matrix(raw)
        data = create_targets(data)

    required = FEATURE_COLS + [TARGET_COL, 'Return_NextDay', 'Date', 'Ticker']
    missing  = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f'Missing columns in features.csv: {missing}')

    # ── Step 4: Walk-forward folds ───────────────────────────────────────────
    dates = sorted(data['Date'].unique())
    folds = generate_walk_forward_folds(dates, TRAIN_DAYS, VAL_DAYS, TEST_DAYS)
    print()
    print_fold_summary(folds)

    # ── Steps 5-6: Train models per fold ────────────────────────────────────
    all_preds = []

    # ── LSTM pre-training diagnostic (runs once before any fold) ─────────────
    print('\n-- LSTM Pre-training Diagnostic --')
    _f0      = folds[0]
    _df_diag = data[data['Date'].isin(dates[_f0['train'][0]:_f0['train'][1]])]
    _X_diag, _, _, _ = standardize_fold(
        _df_diag[FEATURE_COLS].values,
        _df_diag[FEATURE_COLS].values,
        _df_diag[FEATURE_COLS].values,
    )
    _diag_ds     = StockSequenceDataset(
        scaled_df(_df_diag, _X_diag), FEATURE_COLS, TARGET_COL, SEQ_LEN
    )
    _diag_loader = DataLoader(_diag_ds, batch_size=LSTM_BATCH, shuffle=False,
                              num_workers=0)
    _X_batch, _  = next(iter(_diag_loader))
    print(f'  LSTM device   : {lstm_device}')
    print(f'  Sequence dtype: {_X_batch.dtype}')
    _diag_model = StockLSTM(input_size=N_TOTAL_FEATURES).to(lstm_device)
    _dummy_in   = torch.randn(64, SEQ_LEN, N_TOTAL_FEATURES).to(lstm_device)
    _t0_diag    = time.perf_counter()
    with torch.no_grad():
        _diag_model(_dummy_in)
    print(f'  Dummy fwd pass: {(time.perf_counter() - _t0_diag)*1000:.2f} ms (batch=64)')
    del _diag_model, _dummy_in, _diag_loader, _diag_ds, _df_diag, _X_diag
    print()

    for fold in tqdm(folds, desc="Walk-Forward Folds", unit="fold"):
        print(f'\n{"="*60}')
        print(f'FOLD {fold["fold"]}  '
              f'train: {str(fold["train_start_date"])[:10]} → '
              f'{str(fold["train_end_date"])[:10]}  '
              f'test: {str(fold["test_start_date"])[:10]} → '
              f'{str(fold["test_end_date"])[:10]}')
        print('='*60)

        df_tr = data[data['Date'].isin(dates[fold['train'][0]:fold['train'][1]])]
        df_v  = data[data['Date'].isin(dates[fold['val'][0]:fold['val'][1]])]
        df_ts = data[data['Date'].isin(dates[fold['test'][0]:fold['test'][1]])]

        print(f'  Train: {len(df_tr):>6} rows | '
              f'Val: {len(df_v):>5} rows | '
              f'Test: {len(df_ts):>5} rows')

        X_tr, X_v, X_ts, _ = standardize_fold(
            df_tr[FEATURE_COLS].values,
            df_v[FEATURE_COLS].values,
            df_ts[FEATURE_COLS].values,
        )
        y_tr = df_tr[TARGET_COL].values
        y_v  = df_v[TARGET_COL].values

        # ── Baselines ────────────────────────────────────────────────────────
        t0 = time.time()
        print('\n-- Logistic Regression --')
        lr_m = train_logistic(X_tr, y_tr)
        print(f'  done in {time.time()-t0:.1f}s')

        t0 = time.time()
        print('-- Random Forest --')
        rf_m = train_random_forest(X_tr, y_tr)
        print(f'  done in {time.time()-t0:.1f}s')

        t0 = time.time()
        print('-- XGBoost --')
        xgb_m = train_xgboost(X_tr, y_tr, X_v, y_v)
        print(f'  done in {time.time()-t0:.1f}s')

        # ── LSTM ─────────────────────────────────────────────────────────────
        t0 = time.time()
        print('-- LSTM (CPU) --')
        lstm_m = StockLSTM(input_size=N_TOTAL_FEATURES)

        train_loader = DataLoader(
            StockSequenceDataset(scaled_df(df_tr, X_tr), FEATURE_COLS, TARGET_COL, SEQ_LEN),
            batch_size=LSTM_BATCH, shuffle=True, num_workers=0,
        )
        val_loader = DataLoader(
            StockSequenceDataset(scaled_df(df_v, X_v), FEATURE_COLS, TARGET_COL, SEQ_LEN),
            batch_size=256, shuffle=False, num_workers=0,
        )

        lstm_m, train_losses, val_losses = train_lstm(
            lstm_m, train_loader, val_loader, lstm_device
        )
        print(f'  done in {time.time()-t0:.1f}s')

        # Store loss curves for Figure F11
        fold['train_losses'] = train_losses
        fold['val_losses']   = val_losses

        # ── LSTM inference + alignment ────────────────────────────────────────
        test_ds = StockSequenceDataset(
            scaled_df(df_ts, X_ts), FEATURE_COLS, TARGET_COL, SEQ_LEN
        )
        lstm_probs, lstm_keys = lstm_predict(lstm_m, test_ds, lstm_device)

        # ── Collect all predictions for this fold ────────────────────────────
        pred = df_ts.copy().reset_index(drop=True)
        pred['Prob_LR']   = lr_m.predict_proba(X_ts)[:, 1]
        pred['Prob_RF']   = rf_m.predict_proba(X_ts)[:, 1]
        pred['Prob_XGB']  = xgb_m.predict(xgb.DMatrix(X_ts))
        pred['Prob_LSTM'] = align_lstm_predictions(lstm_probs, lstm_keys, df_ts)
        pred['Fold']      = fold['fold']

        n_valid = pred['Prob_LSTM'].notna().sum()
        print(f'  LSTM aligned: {n_valid}/{len(pred)} rows '
              f'({len(pred) - n_valid} skipped — no lookback window)')

        all_preds.append(pred)

    # ── Combine folds ─────────────────────────────────────────────────────────
    full_preds  = pd.concat(all_preds).reset_index(drop=True)
    valid_preds = full_preds.dropna(subset=['Prob_LSTM']).copy()
    print(f'\nTotal predictions: {len(full_preds)} | '
          f'With LSTM: {len(valid_preds)}')

    # ── Steps 7-8: Signals → Portfolio returns (per model + ensemble) ─────────
    print('\n' + '='*60)
    print('BACKTEST RESULTS')
    print('='*60)

    model_cols = {
        'LR':       'Prob_LR',
        'RF':       'Prob_RF',
        'XGBoost':  'Prob_XGB',
        'LSTM':     'Prob_LSTM',
        'Ensemble': None,   # None → average of all four columns
    }

    port_returns  = {}
    class_metrics = {}

    for model_name, prob_col in model_cols.items():
        print(f'\n--- {model_name} ---')
        sig_df = generate_signals(valid_preds, k=K_STOCKS, prob_col=prob_col)
        port   = compute_portfolio_returns(sig_df, tc_bps=TC_BPS)
        port_returns[model_name] = port

        m = compute_metrics(port['Net_Return'])
        print(f'  Sharpe={m["Sharpe Ratio"]}  '
              f'Ann.Ret={m["Annualized Return (%)"]:.2f}%  '
              f'MDD={m["Max Drawdown (%)"]:.2f}%')

        if prob_col is not None:
            y_prob = valid_preds[prob_col].values
        else:
            y_prob = (valid_preds[['Prob_LR', 'Prob_RF', 'Prob_XGB', 'Prob_LSTM']]
                      .mean(axis=1).values)
        class_metrics[model_name] = evaluate_classification(
            valid_preds[TARGET_COL].values, y_prob
        )

    # ── Step 9: Summary tables ────────────────────────────────────────────────
    print('\n\n' + '='*60)
    print('TABLE T5 — Risk-Return Metrics (Net Returns)')
    print('='*60)
    t5 = pd.DataFrame({n: compute_metrics(p['Net_Return'])
                       for n, p in port_returns.items()}).T
    display_cols = ['Sharpe Ratio', 'Sortino Ratio', 'Annualized Return (%)',
                    'Annualized Std Dev (%)', 'Max Drawdown (%)',
                    'Calmar Ratio', 'Win Rate (%)', 'VaR 1% (%)']
    print(t5[display_cols].to_string())

    print('\n\n' + '='*60)
    print('TABLE T8 — Classification Metrics')
    print('='*60)
    print(pd.DataFrame(class_metrics).T.to_string())

    print('\n\n' + '='*60)
    print('TABLE T6 — Sub-Period Performance (Ensemble Net Returns)')
    print('='*60)
    t6 = compute_subperiod_metrics(port_returns['Ensemble']['Net_Return'])
    print(t6[['N Days', 'Sharpe Ratio', 'Annualized Return (%)',
              'Max Drawdown (%)']].to_string())

    print('\n\nPipeline complete.')
    return {
        'full_preds':    full_preds,
        'valid_preds':   valid_preds,
        'port_returns':  port_returns,
        'class_metrics': class_metrics,
        'folds':         folds,
    }


if __name__ == '__main__':
    main(load_cached=True)
