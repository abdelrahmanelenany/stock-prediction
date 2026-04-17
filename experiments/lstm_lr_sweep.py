"""
Learning-rate grid for LSTM on the first walk-forward fold (quick diagnostic).

Uses a capped epoch budget for speed (see LSTM_LR_SWEEP_MAX_EPOCHS in config).

Usage:
    python experiments/lstm_lr_sweep.py
"""
from __future__ import annotations

import os
import sys

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
)
from evaluation.metrics_utils import binary_auc_safe


def main() -> None:
    max_ep = getattr(config, 'LSTM_LR_SWEEP_MAX_EPOCHS', 40)
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
    df_v = data[data['Date'].isin(dates[fold['val'][0]:fold['val'][1]])]
    df_ts = data[data['Date'].isin(dates[fold['test'][0]:fold['test'][1]])]
    df_train_fold = pd.concat([df_tr, df_v]).sort_values(['Ticker', 'Date'])
    df_test_fold = df_ts.copy()

    X_tr, y_tr, X_val, y_val, X_te, y_te, _, _, _ = prepare_lstm_b_sequences_temporal_split(
        df_train_fold, df_test_fold, val_ratio=config.LSTM_B_VAL_SPLIT
    )

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    rows = []
    try:
        for lr in config.LSTM_LR_GRID:
            set_global_seed(config.RANDOM_SEED)
            model = train_lstm_b(
                X_tr, y_tr, X_val, y_val, device,
                seed=config.RANDOM_SEED + 99,
                fold_idx=0,
                learning_rate=float(lr),
            )
            probs_val = predict_lstm(model, X_val, device)
            probs_te = predict_lstm(model, X_te, device)
            vauc = binary_auc_safe(y_val, probs_val, log_on_fail=False)
            tauc = binary_auc_safe(y_te, probs_te, log_on_fail=False)
            rows.append({
                'lr': lr,
                'val_auc': vauc,
                'test_auc': tauc,
                'max_epochs_used': config.LSTM_B_MAX_EPOCHS,
            })
            print(f'lr={lr}  val_auc={vauc}  test_auc={tauc}')
    finally:
        config.LSTM_B_MAX_EPOCHS = old_max

    os.makedirs('reports', exist_ok=True)
    path = os.path.join('reports', 'lstm_lr_sweep.csv')
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f'Saved {path}')


if __name__ == '__main__':
    main()
