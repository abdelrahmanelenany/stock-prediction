"""
Kernel × levels × channels architecture sweep for TCN on fold 0 (quick diagnostic).

Uses a capped epoch budget for speed (see TCN_SWEEP_MAX_EPOCHS in config).

Usage:
    python experiments/tcn_arch_sweep.py
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
from models.tcn_model import (
    TCNModel,
    prepare_tcn_sequences_temporal_split,
    train_tcn,
    predict_tcn,
    tcn_receptive_field,
)
from evaluation.metrics_utils import binary_auc_safe


def _count_params(model: TCNModel) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    max_ep = getattr(config, 'TCN_SWEEP_MAX_EPOCHS', 40)
    kernel_grid = getattr(config, 'TCN_SWEEP_KERNEL_GRID', [2, 3, 5])
    levels_grid = getattr(config, 'TCN_SWEEP_LEVELS_GRID', [3, 4, 5])
    channel_grid = getattr(config, 'TCN_SWEEP_CHANNEL_GRID', [32, 64])
    feature_cols = list(getattr(config, 'TCN_FEATURE_COLS_FULL', config.LSTM_B_FEATURE_COLS))

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
    df_v = data[data['Date'].isin(dates[fold['val'][0]:fold['val'][1]])]
    df_ts = data[data['Date'].isin(dates[fold['test'][0]:fold['test'][1]])]
    df_train_fold = pd.concat([df_tr, df_v]).sort_values(['Ticker', 'Date'])
    df_test_fold = df_ts.copy()

    val_ratio = getattr(config, 'TCN_VAL_SPLIT', 0.2)
    X_tr, y_tr, X_val, y_val, X_te, y_te, _, _, _ = prepare_tcn_sequences_temporal_split(
        df_train_fold, df_test_fold,
        feature_cols=feature_cols,
        val_ratio=val_ratio,
        seq_len=getattr(config, 'TCN_SEQ_LEN', config.SEQ_LEN),
    )

    device = (
        torch.device('mps') if torch.backends.mps.is_available()
        else torch.device('cpu')
    )

    rows = []
    total = len(kernel_grid) * len(levels_grid) * len(channel_grid)
    done = 0
    print(f'Sweeping {total} combinations (max_epochs={max_ep}) …\n')

    for kernel in kernel_grid:
        for levels in levels_grid:
            for ch in channel_grid:
                done += 1
                num_channels = [ch] * levels
                rf = tcn_receptive_field(kernel, levels)
                viable = rf >= getattr(config, 'TCN_SEQ_LEN', config.SEQ_LEN)

                set_global_seed(config.RANDOM_SEED)
                t0 = time.time()
                model = train_tcn(
                    X_tr, y_tr, X_val, y_val, device,
                    seed=config.RANDOM_SEED,
                    fold_idx=0,
                    num_channels=num_channels,
                    kernel_size=kernel,
                    dropout=getattr(config, 'TCN_DROPOUT', 0.1),
                    use_weight_norm=getattr(config, 'TCN_USE_WEIGHT_NORM', True),
                    learning_rate=getattr(config, 'TCN_LR', 1e-3),
                    batch_size=getattr(config, 'TCN_BATCH', 256),
                    max_epochs=max_ep,
                    patience=getattr(config, 'TCN_TUNE_PATIENCE', 4),
                )
                elapsed = time.time() - t0

                probs_val = predict_tcn(model, X_val, device)
                probs_te = predict_tcn(model, X_te, device)
                vauc = binary_auc_safe(y_val, probs_val, log_on_fail=False)
                tauc = binary_auc_safe(y_te, probs_te, log_on_fail=False)
                n_params = _count_params(model)

                row = {
                    'kernel': kernel,
                    'levels': levels,
                    'channels': ch,
                    'receptive_field': rf,
                    'viable': viable,
                    'params': n_params,
                    'val_auc': vauc,
                    'test_auc': tauc,
                    'train_time_s': round(elapsed, 2),
                }
                rows.append(row)
                viable_tag = '✓' if viable else '✗ (RF<seq_len)'
                print(
                    f'[{done:>2}/{total}] k={kernel} L={levels} ch={ch:<3}  '
                    f'RF={rf:<4} {viable_tag:<18}  '
                    f'val_auc={vauc:.4f}  test_auc={tauc:.4f}  '
                    f't={elapsed:.1f}s  params={n_params:,}'
                )

    os.makedirs('reports', exist_ok=True)
    path = os.path.join('reports', 'tcn_arch_sweep.csv')
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f'\nSaved {path}')

    # Print summary of viable combinations sorted by val_auc
    df_res = pd.DataFrame(rows)
    viable_df = df_res[df_res['viable']].sort_values('val_auc', ascending=False)
    if len(viable_df):
        print('\nTop viable combinations by val_auc:')
        print(viable_df[['kernel', 'levels', 'channels', 'receptive_field', 'params', 'val_auc', 'test_auc', 'train_time_s']].head(10).to_string(index=False))
    else:
        print('\nNo viable combinations found (all RF < seq_len).')


if __name__ == '__main__':
    main()
