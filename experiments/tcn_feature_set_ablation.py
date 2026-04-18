"""
Controlled TCN feature-set ablation across four configurations:
1) core
2) core + market
3) core + sector
4) core + market + sector

This script runs walk-forward pipeline four times with only TCN enabled and writes:
- per-run artifacts under reports_exp/tcn_feature_ablation/<set_name>/
- aggregated summary to reports/tcn_feature_set_ablation.csv

Usage:
    python experiments/tcn_feature_set_ablation.py
    python experiments/tcn_feature_set_ablation.py --enable-tuning
    python experiments/tcn_feature_set_ablation.py --enable-tuning --tune-every-fold
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
import main as pipeline_main
from backtest.metrics import compute_metrics


def _unique_keep_order(cols: list[str]) -> list[str]:
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _market_cols_from_full(full_cols: list[str]) -> list[str]:
    return [
        c for c in full_cols
        if c.startswith('Market_') or c.startswith('RelToMarket_') or c.startswith('Beta_')
    ]


def _sector_cols_from_full(full_cols: list[str]) -> list[str]:
    return [
        c for c in full_cols
        if c.startswith('Sector_') or c.startswith('SectorRelZ_')
    ]


def _extract_tcn_metrics(out: dict) -> dict:
    row = {}

    # Trading metrics (net) from returned portfolio outputs
    if 'TCN' in out.get('port_returns', {}):
        m = compute_metrics(out['port_returns']['TCN']['Net_Return'])
        row['tcn_sharpe_net'] = m.get('Sharpe Ratio')
        row['tcn_sortino_net'] = m.get('Sortino Ratio')
        row['tcn_ann_ret_net_pct'] = m.get('Annualized Return (%)')
        row['tcn_max_dd_pct'] = m.get('Max Drawdown (%)')

    # Classification metrics from pipeline output
    class_rows = out.get('class_metrics', [])
    tcn_cls = next((r for r in class_rows if r.get('Model') == 'TCN'), None)
    if tcn_cls is not None:
        row['tcn_auc_roc'] = tcn_cls.get('AUC-ROC')
        row['tcn_accuracy'] = tcn_cls.get('Accuracy')
        row['tcn_f1'] = tcn_cls.get('F1 Score')
        row['tcn_daily_auc_mean'] = tcn_cls.get('Daily AUC (mean)')
        row['tcn_daily_auc_std'] = tcn_cls.get('Daily AUC (std)')

    return row


def main(enable_tuning: bool = False, tune_every_fold: bool = False) -> None:
    os.makedirs('reports', exist_ok=True)

    core_cols = list(getattr(config, 'TCN_FEATURE_COLS_CORE', config.LSTM_B_FEATURE_COLS))
    full_cols = list(getattr(config, 'TCN_FEATURE_COLS_FULL', config.LSTM_B_FEATURE_COLS))
    market_cols = _market_cols_from_full(full_cols)
    sector_cols = _sector_cols_from_full(full_cols)

    feature_sets = {
        'core': _unique_keep_order(core_cols),
        'core_market': _unique_keep_order(core_cols + market_cols),
        'core_sector': _unique_keep_order(core_cols + sector_cols),
        'core_market_sector': _unique_keep_order(core_cols + market_cols + sector_cols),
    }

    # Snapshot mutable globals/config so this experiment has no side effects after completion.
    old_run_baselines = pipeline_main.RUN_BASELINES
    old_run_lstms = pipeline_main.RUN_LSTMS
    old_run_tcns = pipeline_main.RUN_TCNS

    old_tcn_enable_tuning = config.TCN_ENABLE_TUNING
    old_tcn_tune_once = config.TCN_TUNE_ON_FIRST_FOLD_ONLY
    old_tcn_arch_grid = dict(getattr(config, 'TCN_ARCH_GRID', {}))
    old_tcn_feature_sets = dict(getattr(config, 'TCN_FEATURE_SETS', {'full': full_cols}))
    old_tcn_feature_default = getattr(config, 'TCN_FEATURE_SET_DEFAULT', 'full')

    rows = []
    base_reports_dir = os.path.join(
        'reports_exp',
        'tcn_feature_ablation_tuned' if enable_tuning else 'tcn_feature_ablation',
    )

    try:
        # Controlled ablation: keep architecture/training fixed and isolate TCN.
        pipeline_main.RUN_BASELINES = False
        pipeline_main.RUN_LSTMS = False
        pipeline_main.RUN_TCNS = True

        # Keep per-run feature sets fixed. Optionally enable TCN tuning.
        config.TCN_ENABLE_TUNING = bool(enable_tuning)
        config.TCN_TUNE_ON_FIRST_FOLD_ONLY = bool(enable_tuning and not tune_every_fold)

        for set_name, cols in feature_sets.items():
            out_dir = os.path.join(base_reports_dir, set_name)
            t0 = time.time()

            config.TCN_FEATURE_SETS = {set_name: cols}
            config.TCN_FEATURE_SET_DEFAULT = set_name
            # Keep tuning candidate feature_set aligned with the active ablation run.
            # Without this, Phase 2 may reference unavailable set names and produce no candidates.
            if enable_tuning:
                arch_grid = dict(getattr(config, 'TCN_ARCH_GRID', {}))
                arch_grid['feature_set'] = [set_name]
                config.TCN_ARCH_GRID = arch_grid

            print(f'\n=== TCN ablation run: {set_name} ({len(cols)} features) ===')
            out = pipeline_main.run_walk_forward_pipeline(
                load_cached=True,
                reports_dir=out_dir,
            )
            elapsed = time.time() - t0

            row = {
                'universe_mode': config.UNIVERSE_MODE,
                'feature_set': set_name,
                'n_features': len(cols),
                'features': '|'.join(cols),
                'n_folds': len(out.get('folds', [])),
                'wall_seconds': round(elapsed, 1),
                'reports_dir': out_dir,
                'tcn_tuning_enabled': bool(enable_tuning),
                'tcn_tune_on_first_fold_only': bool(enable_tuning and not tune_every_fold),
            }
            row.update(_extract_tcn_metrics(out))
            rows.append(row)

    finally:
        pipeline_main.RUN_BASELINES = old_run_baselines
        pipeline_main.RUN_LSTMS = old_run_lstms
        pipeline_main.RUN_TCNS = old_run_tcns

        config.TCN_ENABLE_TUNING = old_tcn_enable_tuning
        config.TCN_TUNE_ON_FIRST_FOLD_ONLY = old_tcn_tune_once
        config.TCN_ARCH_GRID = old_tcn_arch_grid
        config.TCN_FEATURE_SETS = old_tcn_feature_sets
        config.TCN_FEATURE_SET_DEFAULT = old_tcn_feature_default

    out_path = os.path.join(
        'reports',
        'tcn_feature_set_ablation_tuned.csv' if enable_tuning else 'tcn_feature_set_ablation.csv',
    )
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TCN feature-set ablation runner')
    parser.add_argument(
        '--enable-tuning',
        action='store_true',
        help='Enable TCN tuning inside each feature-set run.',
    )
    parser.add_argument(
        '--tune-every-fold',
        action='store_true',
        help='When tuning is enabled, retune on every fold instead of tuning once on fold 1.',
    )
    args = parser.parse_args()
    main(enable_tuning=args.enable_tuning, tune_every_fold=args.tune_every_fold)
