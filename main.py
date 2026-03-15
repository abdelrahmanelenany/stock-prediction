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
from backtest.signals      import generate_signals
from backtest.portfolio    import compute_portfolio_returns
from backtest.metrics      import (
    compute_metrics, evaluate_classification,
    compute_subperiod_metrics, compute_tc_sensitivity,
)
from sklearn.metrics import roc_auc_score
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

    With context_df provided to StockSequenceDataset, all test rows should
    have predictions. Any remaining NaN indicates a data gap.
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

        # ── Standardize (fit on train only) ─────────────────────────────────
        X_tr_s, X_v_s, X_ts_s, _ = standardize_fold(
            df_tr[FEATURE_COLS].values,
            df_v[FEATURE_COLS].values,
            df_ts[FEATURE_COLS].values,
        )

        y_tr = df_tr[TARGET_COL].values.astype(int)
        y_v  = df_v[TARGET_COL].values.astype(int)

        # ── Baselines ────────────────────────────────────────────────────────
        t0 = time.time()
        print('\n-- Logistic Regression --')
        lr_m = train_logistic(X_tr_s, y_tr)
        print(f'  done in {time.time()-t0:.1f}s')

        t0 = time.time()
        print('-- Random Forest --')
        rf_m = train_random_forest(X_tr_s, y_tr, X_v_s, y_v)
        print(f'  done in {time.time()-t0:.1f}s')

        t0 = time.time()
        print('-- XGBoost --')
        xgb_m = train_xgboost(X_tr_s, y_tr, X_v_s, y_v)
        print(f'  done in {time.time()-t0:.1f}s')

        # ── LSTM ─────────────────────────────────────────────────────────────
        t0 = time.time()
        print('-- LSTM (CPU) --')
        lstm_m = StockLSTM(input_size=N_TOTAL_FEATURES)

        df_tr_scaled = scaled_df(df_tr, X_tr_s)
        df_v_scaled  = scaled_df(df_v, X_v_s)
        df_ts_scaled = scaled_df(df_ts, X_ts_s)

        train_loader = DataLoader(
            StockSequenceDataset(df_tr_scaled, FEATURE_COLS, TARGET_COL, SEQ_LEN),
            batch_size=LSTM_BATCH, shuffle=True, num_workers=0,
        )
        val_loader = DataLoader(
            StockSequenceDataset(
                df_v_scaled, FEATURE_COLS, TARGET_COL, SEQ_LEN,
                context_df=df_tr_scaled,
            ),
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
        # Prepend the last SEQ_LEN days of validation data as context so that
        # even the first test day has a full lookback window.
        test_ds = StockSequenceDataset(
            df_ts_scaled, FEATURE_COLS, TARGET_COL, SEQ_LEN,
            skip_nan_targets=False,
            context_df=df_v_scaled,
        )
        lstm_probs, lstm_keys = lstm_predict(lstm_m, test_ds, lstm_device)

        # ── Compute validation AUC for ensemble weighting ──────────────
        val_auc = {}
        val_auc['LR']  = roc_auc_score(y_v, lr_m.predict_proba(X_v_s)[:, 1])
        val_auc['RF']  = roc_auc_score(y_v, rf_m.predict_proba(X_v_s)[:, 1])
        val_auc['XGB'] = roc_auc_score(y_v, xgb_m.predict(xgb.DMatrix(X_v_s)))

        # LSTM validation AUC
        val_ds_for_auc = StockSequenceDataset(
            df_v_scaled, FEATURE_COLS, TARGET_COL, SEQ_LEN,
            context_df=df_tr_scaled,
        )
        lstm_val_probs, lstm_val_keys = lstm_predict(lstm_m, val_ds_for_auc, lstm_device)
        lstm_val_aligned = align_lstm_predictions(lstm_val_probs, lstm_val_keys, df_v)
        valid_mask = ~np.isnan(lstm_val_aligned)
        if valid_mask.sum() > 0:
            val_auc['LSTM'] = roc_auc_score(
                y_v[valid_mask], lstm_val_aligned[valid_mask]
            )
        else:
            val_auc['LSTM'] = 0.5

        # Weights: only models with AUC > 0.5 contribute
        raw_weights = {m: max(auc - 0.5, 0.0) for m, auc in val_auc.items()}
        total_w = sum(raw_weights.values())
        if total_w > 0:
            ens_weights = {m: w / total_w for m, w in raw_weights.items()}
        else:
            ens_weights = {'LR': 0.25, 'RF': 0.25, 'XGB': 0.25, 'LSTM': 0.25}

        print(f'  Val AUC: { {m: round(a, 4) for m, a in val_auc.items()} }')
        print(f'  Ensemble weights: { {m: round(w, 3) for m, w in ens_weights.items()} }')

        # ── Collect all predictions for this fold ────────────────────────────
        pred = df_ts.copy().reset_index(drop=True)
        pred['Prob_LR']   = lr_m.predict_proba(X_ts_s)[:, 1]
        pred['Prob_RF']   = rf_m.predict_proba(X_ts_s)[:, 1]
        pred['Prob_XGB']  = xgb_m.predict(xgb.DMatrix(X_ts_s))
        pred['Prob_LSTM'] = align_lstm_predictions(lstm_probs, lstm_keys, df_ts)
        pred['Fold']      = fold['fold']
        pred['Fold_W_LR']   = ens_weights['LR']
        pred['Fold_W_RF']   = ens_weights['RF']
        pred['Fold_W_XGB']  = ens_weights['XGB']
        pred['Fold_W_LSTM'] = ens_weights['LSTM']

        n_valid = pred['Prob_LSTM'].notna().sum()
        print(f'  LSTM aligned: {n_valid}/{len(pred)} rows '
              f'({len(pred) - n_valid} skipped — no lookback window)')

        all_preds.append(pred)

    # ── Combine folds ─────────────────────────────────────────────────────────
    full_preds = pd.concat(all_preds).reset_index(drop=True)
    # With context_df, LSTM should cover all test rows
    n_lstm_valid = full_preds['Prob_LSTM'].notna().sum()
    print(f'\nTotal predictions: {len(full_preds)} | '
          f'With LSTM: {n_lstm_valid}')

    # Use rows that have all predictions (should be all with context fix)
    valid_preds = full_preds.dropna(subset=['Prob_LSTM']).copy()

    # Compute validation-weighted ensemble probabilities
    valid_preds['Prob_ENS_Weighted'] = (
        valid_preds['Prob_LR']   * valid_preds['Fold_W_LR'] +
        valid_preds['Prob_RF']   * valid_preds['Fold_W_RF'] +
        valid_preds['Prob_XGB']  * valid_preds['Fold_W_XGB'] +
        valid_preds['Prob_LSTM'] * valid_preds['Fold_W_LSTM']
    )

    # ── Steps 7-8: Signals → Portfolio returns (per model + ensemble) ─────────
    print('\n' + '='*60)
    print('BACKTEST RESULTS')
    print('='*60)

    model_cols = {
        'LR':                  'Prob_LR',
        'RF':                  'Prob_RF',
        'XGBoost':             'Prob_XGB',
        'LSTM':                'Prob_LSTM',
        'Ensemble (equal)':    None,               # equal-weight average
        'Ensemble (weighted)': 'Prob_ENS_Weighted', # validation-AUC-weighted
    }

    port_returns  = {}
    class_metrics = {}

    for model_name, prob_col in model_cols.items():
        print(f'\n--- {model_name} ---')
        sig_df = generate_signals(valid_preds, k=K_STOCKS, prob_col=prob_col)
        port   = compute_portfolio_returns(sig_df, tc_bps=TC_BPS, k=K_STOCKS)
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
    print('TABLE T6 — Sub-Period Performance (Weighted Ensemble Net Returns)')
    print('='*60)
    t6 = compute_subperiod_metrics(port_returns['Ensemble (weighted)']['Net_Return'])
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
    main(load_cached=False)
