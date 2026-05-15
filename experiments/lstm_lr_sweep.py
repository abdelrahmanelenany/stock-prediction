"""
Learning rate × hidden size ablation sweep for LSTM on fold 0 (quick diagnostic).

Sweeps over:
  - learning_rate : LSTM_LR_GRID          (default [0.0005, 0.001, 0.003, 0.005])
  - hidden_size   : LSTM_HIDDEN_SWEEP_GRID (default [32, 64])

Uses a capped epoch budget for speed (LSTM_LR_SWEEP_MAX_EPOCHS in config).
Output: reports/{universe}_lstm_hp_sweep.csv

Usage:
    python experiments/lstm_lr_sweep.py
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
from main import set_global_seed, CACHE_FEATURES_PATH
from pipeline.walk_forward import generate_walk_forward_folds
from models.lstm_model import (
    prepare_lstm_b_sequences_temporal_split,
    train_lstm_b,
    predict_lstm,
    align_predictions_to_df,
)
from backtest.signals import (
    smooth_probabilities,
    generate_signals,
    apply_holding_period_constraint,
)
from backtest.portfolio import compute_portfolio_returns
from backtest.metrics import compute_metrics
from evaluation.metrics_utils import binary_auc_safe


def _score_on_val(df_v: pd.DataFrame, probs_val, keys_val, prob_col: str) -> float:
    """Compute net Sharpe on the walk-forward val window using the full signals pipeline."""
    universe_cfg = (
        config.LARGE_CAP_CONFIG
        if config.UNIVERSE_MODE == 'large_cap'
        else config.SMALL_CAP_CONFIG
    )
    preds = df_v.copy().reset_index(drop=True)
    preds[prob_col] = align_predictions_to_df(probs_val, keys_val, df_v)
    valid = preds.dropna(subset=[prob_col]).copy()
    if len(valid) == 0:
        return float('nan')
    smoothed = smooth_probabilities(
        valid, prob_col,
        alpha=getattr(config, 'SIGNAL_SMOOTH_ALPHA', 0.0),
        ema_method=getattr(config, 'SIGNAL_EMA_METHOD', 'alpha'),
        ema_span=getattr(config, 'SIGNAL_EMA_SPAN', None),
    )
    sig_df, _ = generate_signals(
        smoothed,
        k=config.K_STOCKS,
        prob_col=f'{prob_col}_Smooth',
        return_diagnostics=True,
    )
    sig_df = apply_holding_period_constraint(
        sig_df, min_hold_days=getattr(config, 'MIN_HOLDING_DAYS', 5)
    )
    port = compute_portfolio_returns(
        sig_df,
        tc_bps=config.TC_BPS,
        k=config.K_STOCKS,
        slippage_bps=getattr(config, 'SLIPPAGE_BPS', 0.0),
        invert_signals=universe_cfg.invert_signals,
    )
    met = compute_metrics(port['Net_Return'])
    return float(met['Sharpe Ratio'])


def main() -> None:
    max_ep = getattr(config, 'LSTM_LR_SWEEP_MAX_EPOCHS', 40)
    lr_grid = getattr(config, 'LSTM_LR_GRID', [0.0005, 0.001, 0.003, 0.005])
    hidden_grid = getattr(config, 'LSTM_HIDDEN_SWEEP_GRID', [32, 64])
    prefix = config.UNIVERSE_MODE

    old_max = config.LSTM_B_MAX_EPOCHS
    config.LSTM_B_MAX_EPOCHS = min(int(max_ep), int(old_max))

    data = pd.read_csv(CACHE_FEATURES_PATH, parse_dates=['Date'])
    dates = sorted(data['Date'].unique())
    folds = generate_walk_forward_folds(
        dates,
        config.TRAIN_DAYS,
        config.VAL_DAYS,
        config.TEST_DAYS,
        stride_days=getattr(config, 'WALK_FORWARD_STRIDE', None),
        train_window_mode=getattr(config, 'TRAIN_WINDOW_MODE', 'rolling'),
    )
    fold = folds[0]
    df_tr = data[data['Date'].isin(dates[fold['train'][0]:fold['train'][1]])]
    df_v  = data[data['Date'].isin(dates[fold['val'][0]:fold['val'][1]])]
    df_ts = data[data['Date'].isin(dates[fold['test'][0]:fold['test'][1]])]
    df_train_fold = pd.concat([df_tr, df_v]).sort_values(['Ticker', 'Date'])
    df_test_fold  = df_ts.copy()

    device = (
        torch.device('mps') if torch.backends.mps.is_available()
        else torch.device('cpu')
    )

    rows = []
    total = len(lr_grid) * len(hidden_grid)
    done  = 0
    print(f'Sweeping {total} combinations (max_epochs={max_ep}) …\n')

    try:
        for lr in lr_grid:
            for hs in hidden_grid:
                done += 1
                set_global_seed(config.RANDOM_SEED)

                X_tr, y_tr, X_val, y_val, X_te, y_te, _, keys_val, _ = (
                    prepare_lstm_b_sequences_temporal_split(
                        df_train_fold, df_test_fold,
                        val_ratio=config.LSTM_B_VAL_SPLIT,
                    )
                )

                t0 = time.time()
                model = train_lstm_b(
                    X_tr, y_tr, X_val, y_val, device,
                    seed=config.RANDOM_SEED,
                    fold_idx=0,
                    learning_rate=float(lr),
                    hidden_size=int(hs),
                )
                elapsed = time.time() - t0

                probs_val = predict_lstm(model, X_val, device)
                probs_te  = predict_lstm(model, X_te,  device)
                val_auc   = binary_auc_safe(y_val, probs_val, log_on_fail=False)
                test_auc  = binary_auc_safe(y_te,  probs_te,  log_on_fail=False)
                val_sharpe = _score_on_val(df_v, probs_val, keys_val, 'Prob_LSTM_B')

                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                row = {
                    'lr':           lr,
                    'hidden_size':  hs,
                    'val_sharpe':   round(val_sharpe, 3),
                    'val_auc':      round(val_auc,  4),
                    'test_auc':     round(test_auc, 4),
                    'n_params':     n_params,
                    'train_time_s': round(elapsed,  2),
                    'max_epochs':   config.LSTM_B_MAX_EPOCHS,
                }
                rows.append(row)
                print(
                    f'[{done:>2}/{total}] lr={lr}  hs={hs:<3}  '
                    f'val_sharpe={val_sharpe:.3f}  val_auc={val_auc:.4f}  '
                    f'test_auc={test_auc:.4f}  t={elapsed:.1f}s  params={n_params:,}'
                )
    finally:
        config.LSTM_B_MAX_EPOCHS = old_max

    os.makedirs('reports', exist_ok=True)
    path = os.path.join('reports', f'{prefix}_lstm_hp_sweep.csv')
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f'\nSaved {path}')

    df_res = pd.DataFrame(rows).sort_values('val_sharpe', ascending=False)
    print('\nAll combinations sorted by val_sharpe:')
    print(df_res.to_string(index=False))


if __name__ == '__main__':
    main()
