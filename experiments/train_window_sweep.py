"""
Reproducible comparison of walk-forward train-window lengths.

Writes reports/train_window_comparison.csv (aggregated) and per-setting outputs
under reports_exp/train_<days>/ when run from repo root.

Usage:
    python experiments/train_window_sweep.py
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
from main import run_walk_forward_pipeline
from backtest.metrics import compute_metrics


def main() -> None:
    os.makedirs('reports', exist_ok=True)
    rows = []
    for td in config.TRAIN_DAYS_CANDIDATES:
        subdir = os.path.join('reports_exp', f'train_{td}')
        t0 = time.time()
        out = run_walk_forward_pipeline(
            load_cached=True,
            train_days=int(td),
            reports_dir=subdir,
        )
        elapsed = time.time() - t0
        cm_by = {d['Model']: d for d in out['class_metrics']}
        row = {
            'train_days': td,
            'n_folds': len(out['folds']),
            'wall_seconds': round(elapsed, 1),
            'reports_dir': subdir,
        }
        for name in ('LR', 'RF', 'XGBoost', 'LSTM', 'Ensemble'):
            if name in cm_by:
                row[f'{name}_pooled_auc'] = cm_by[name].get('AUC-ROC')
                row[f'{name}_daily_auc_mean'] = cm_by[name].get('Daily AUC (mean)')
            if name in out['port_returns']:
                m = compute_metrics(out['port_returns'][name]['Net_Return'])
                row[f'{name}_sharpe_net'] = m.get('Sharpe Ratio')
                row[f'{name}_ann_ret_net_pct'] = m.get('Annualized Return (%)')
                row[f'{name}_max_dd_pct'] = m.get('Max Drawdown (%)')
        rows.append(row)

    out_path = os.path.join('reports', 'train_window_comparison.csv')
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
