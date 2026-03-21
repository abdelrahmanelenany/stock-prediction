"""
main.py — Full pipeline orchestrator
Walk-forward validated long-short strategy using LR, RF, XGBoost, LSTM-A, LSTM-B.

Models after refactor:
  - LR:      Logistic Regression (baseline) — 6 features
  - RF:      Random Forest (baseline) — 6 features
  - XGBoost: XGBoost (baseline) — 6 features
  - LSTM-A:  Bhandari-inspired (4 technical features, tuned architecture)
  - LSTM-B:  Extended ablation (6 features, fixed architecture)

Implements Bhandari et al. (2022) extensions:
  - Dual scalers: separate feature normalization for LSTM-A (4 features) and
    LSTM-B/baselines (6 features)
  - Optional LSTM hyperparameter tuning (Phase 1 + Phase 2 for LSTM-A)
  - Configurable scaler type (standard/minmax)

Run:
    .venv/bin/python3 main.py
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'  # Prevent XGBoost/PyTorch dual-OpenMP segfault

import time
import pandas as pd
import numpy as np
import xgboost as xgb
import torch
from tqdm import tqdm

from pipeline.data_loader import download_and_save
from pipeline.features import build_feature_matrix, FEATURE_COLS
from pipeline.targets import create_targets
from pipeline.walk_forward import generate_walk_forward_folds, print_fold_summary
from pipeline.standardizer import standardize_fold
from models.baselines import train_logistic, train_random_forest, train_xgboost
from models.lstm_model import (
    prepare_lstm_a_sequences, prepare_lstm_b_sequences,
    train_lstm_a, train_lstm_b,
    predict_lstm, align_predictions_to_df,
    tune_lstm_hyperparams,
)
from backtest.signals import generate_signals
from backtest.portfolio import compute_portfolio_returns
from backtest.metrics import (
    compute_metrics, evaluate_classification,
    compute_subperiod_metrics,
)
import config
from config import (
    TRAIN_DAYS, VAL_DAYS, TEST_DAYS,
    K_STOCKS, TC_BPS, MODELS,
    LSTM_A_VAL_SPLIT, LSTM_B_VAL_SPLIT,
    LSTM_A_FEATURES, LSTM_B_FEATURES,
    BASELINE_FEATURE_COLS,
)

TARGET_COL = 'Target'

# ── Pipeline options ─────────────────────────────────────────────────────────
ENABLE_LSTM_TUNING = False  # Set True to run Bhandari §3.3 hyperparameter tuning
                            # (computationally expensive — ~2-3x slower per fold)

device = (
    torch.device('mps') if torch.backends.mps.is_available()
    else torch.device('cpu')
)
print(f'Using device: {device}')


# ────────────────────────────────────────────────────────────────────────────
# Results saving functions
# ────────────────────────────────────────────────────────────────────────────

def save_all_results(
    results_dict: dict,
    daily_returns_dict: dict,
    signals_dict: pd.DataFrame,
    tuning_results: list = None,
    reports_dir: str = 'reports'
):
    """
    Saves all backtest results to the /reports folder.

    Args:
        results_dict: {
            'gross':   list of metric dicts (one per model),
            'net_5':   list of metric dicts,
            'net_10':  list of metric dicts,
            'classification': list of classification metric dicts,
            'subperiod': DataFrame of sub-period metrics,
        }
        daily_returns_dict: {
            'gross':  pd.DataFrame columns=[Date, LR, RF, XGBoost, LSTM-A, LSTM-B],
            'net_5':  pd.DataFrame same columns,
        }
        signals_dict: pd.DataFrame with all signals
        tuning_results: list of dicts with tuning results per fold (optional)
        reports_dir: output directory path
    """
    os.makedirs(reports_dir, exist_ok=True)

    # Table T5: Risk-Return Metrics
    pd.DataFrame(results_dict['gross']).to_csv(
        f'{reports_dir}/table_T5_gross_returns.csv', index=False
    )
    pd.DataFrame(results_dict['net_5']).to_csv(
        f'{reports_dir}/table_T5_net_returns_5bps.csv', index=False
    )
    pd.DataFrame(results_dict['net_10']).to_csv(
        f'{reports_dir}/table_T5_net_returns_10bps.csv', index=False
    )

    # Table T8: Classification Metrics
    pd.DataFrame(results_dict['classification']).to_csv(
        f'{reports_dir}/table_T8_classification_metrics.csv', index=False
    )

    # Table T6: Sub-Period Performance
    if results_dict['subperiod'] is not None:
        results_dict['subperiod'].to_csv(
            f'{reports_dir}/table_T6_subperiod_performance.csv', index=False
        )

    # LSTM Tuning Results (Bhandari §3.3 Tables)
    if tuning_results and len(tuning_results) > 0:
        pd.DataFrame(tuning_results).to_csv(
            f'{reports_dir}/lstm_tuning_results.csv', index=False
        )

    # Raw daily returns
    daily_returns_dict['gross'].to_csv(
        f'{reports_dir}/daily_returns_gross.csv', index=False
    )
    daily_returns_dict['net_5'].to_csv(
        f'{reports_dir}/daily_returns_net_5bps.csv', index=False
    )

    # Signals
    signals_dict.to_csv(f'{reports_dir}/signals_all_models.csv', index=False)

    # Human-readable summary
    with open(f'{reports_dir}/backtest_summary.txt', 'w') as f:
        f.write(_format_summary(results_dict))

    print(f"\nAll results saved to /{reports_dir}/")


def _format_summary(results_dict: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("BACKTEST RESULTS SUMMARY")
    lines.append("=" * 60)

    for label, key in [
        ("GROSS RETURNS (0 bps TC)", 'gross'),
        ("NET RETURNS  (5 bps TC)", 'net_5'),
        ("NET RETURNS (10 bps TC)", 'net_10'),
    ]:
        lines.append(f"\n{'─' * 60}")
        lines.append(label)
        lines.append(f"{'─' * 60}")
        for row in results_dict[key]:
            lines.append(
                f"  {row['Model']:<12}  "
                f"Sharpe={row['Sharpe Ratio']:>6.3f}  "
                f"Ann.Ret={row['Annualized Return (%)']:>6.2f}%  "
                f"MDD={row['Max Drawdown (%)']:>6.2f}%"
            )

    lines.append("\n" + "=" * 60)
    lines.append("CLASSIFICATION METRICS")
    lines.append("=" * 60)
    for row in results_dict['classification']:
        lines.append(
            f"  {row['Model']:<12}  "
            f"Acc={row['Accuracy (%)']:>5.2f}%  "
            f"AUC={row['AUC-ROC']:.4f}  "
            f"F1={row['F1 Score']:.4f}"
        )

    return "\n".join(lines)


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
    print("=" * 60)
    print(f"BACKTEST — {len(MODELS)} MODELS × N FOLDS")
    print(f"LSTM Tuning: {'ENABLED' if ENABLE_LSTM_TUNING else 'DISABLED'}")
    print(f"Scaler Type: {config.SCALER_TYPE}")
    print("=" * 60)

    # ── Steps 1-3: Data ─────────────────────────────────────────────────────
    if load_cached:
        print('\nLoading cached features.csv ...')
        data = pd.read_csv('data/processed/features.csv', parse_dates=['Date'])
        print(f'Loaded {len(data)} rows, {len(data.columns)} columns.')
    else:
        raw = download_and_save()
        data = build_feature_matrix(raw)
        data = create_targets(data)

    # Verify required columns
    required = FEATURE_COLS + [TARGET_COL, 'Return_NextDay', 'Date', 'Ticker']
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f'Missing columns in features.csv: {missing}')

    # Verify LSTM-A features are available (4 features)
    lstm_a_missing = [c for c in LSTM_A_FEATURES if c not in data.columns]
    if lstm_a_missing:
        raise ValueError(f'Missing LSTM-A features: {lstm_a_missing}')

    # Verify LSTM-B features are available (6 features)
    lstm_b_missing = [c for c in LSTM_B_FEATURES if c not in data.columns]
    if lstm_b_missing:
        raise ValueError(f'Missing LSTM-B features: {lstm_b_missing}')

    print(f'\nFeature sets:')
    print(f'  LSTM-A: {LSTM_A_FEATURES} ({len(LSTM_A_FEATURES)} features)')
    print(f'  LSTM-B: {LSTM_B_FEATURES} ({len(LSTM_B_FEATURES)} features)')
    print(f'  Baselines (LR/RF/XGB): {BASELINE_FEATURE_COLS} ({len(BASELINE_FEATURE_COLS)} features)')

    # ── Step 4: Walk-forward folds ───────────────────────────────────────────
    dates = sorted(data['Date'].unique())
    folds = generate_walk_forward_folds(dates, TRAIN_DAYS, VAL_DAYS, TEST_DAYS)
    print()
    print_fold_summary(folds)

    # ── Steps 5-6: Train models per fold ────────────────────────────────────
    all_preds = []
    tuning_results = []  # Store tuning results for thesis reporting

    for fold in tqdm(folds, desc="Walk-Forward Folds", unit="fold"):
        print(f'\n{"=" * 60}')
        print(f'=== Fold {fold["fold"]}/{len(folds)} ===')
        print('=' * 60)

        df_tr = data[data['Date'].isin(dates[fold['train'][0]:fold['train'][1]])]
        df_v = data[data['Date'].isin(dates[fold['val'][0]:fold['val'][1]])]
        df_ts = data[data['Date'].isin(dates[fold['test'][0]:fold['test'][1]])]

        print(f'  Train: {len(df_tr):>6} rows | '
              f'Val: {len(df_v):>5} rows | '
              f'Test: {len(df_ts):>5} rows')

        # ── Scaler B: fit on LSTM-B / baseline features (6 features) ──────
        # Baselines use the same scaler as LSTM-B for fair comparison
        X_tr_b_s, X_v_b_s, X_ts_b_s, _ = standardize_fold(
            df_tr[BASELINE_FEATURE_COLS].values,
            df_v[BASELINE_FEATURE_COLS].values,
            df_ts[BASELINE_FEATURE_COLS].values,
        )

        y_tr = df_tr[TARGET_COL].values.astype(int)
        y_v = df_v[TARGET_COL].values.astype(int)

        # ── Baseline Models ──────────────────────────────────────────────────
        t0 = time.time()
        print('\n  [LR]      fitting...')
        lr_m = train_logistic(X_tr_b_s, y_tr)
        print(f'  [LR]      fit done in {time.time()-t0:.1f}s')

        t0 = time.time()
        print('  [RF]      fitting...')
        rf_m = train_random_forest(X_tr_b_s, y_tr, X_v_b_s, y_v)
        print(f'  [RF]      fit done in {time.time()-t0:.1f}s')

        t0 = time.time()
        print('  [XGBoost] fitting...')
        xgb_m = train_xgboost(X_tr_b_s, y_tr, X_v_b_s, y_v)
        print(f'  [XGBoost] fit done in {time.time()-t0:.1f}s')

        # ── LSTM-A: Bhandari-inspired (4 technical features) ─────────────────
        t0 = time.time()
        print('  [LSTM-A]  building sequences & training...')

        # Combine train and validation for LSTM data preparation
        df_train_fold = pd.concat([df_tr, df_v]).sort_values(['Ticker', 'Date'])
        df_test_fold = df_ts.copy()

        X_tr_a, y_tr_a, X_te_a, y_te_a, keys_tr_a, keys_te_a = prepare_lstm_a_sequences(
            df_train_fold, df_test_fold
        )

        print(f'    LSTM-A sequences: train={len(X_tr_a)}, test={len(X_te_a)}')

        # Optional: hyperparameter tuning for LSTM-A
        best_hp_a = None
        if ENABLE_LSTM_TUNING:
            print('    [LSTM-A Tuning] Running Phase 1 + Phase 2...')
            val_split = int(len(X_tr_a) * (1 - LSTM_A_VAL_SPLIT))
            best_hp_a = tune_lstm_hyperparams(
                X_tr_a[:val_split], y_tr_a[:val_split],
                X_tr_a[val_split:], y_tr_a[val_split:],
                input_size=len(LSTM_A_FEATURES),
                device=device,
                arch_grid=config.LSTM_A_ARCH_GRID,  # Phase 2 architecture search
            )
            tuning_results.append({
                'fold': fold['fold'],
                'model': 'LSTM-A',
                **best_hp_a
            })
            print(f'    [LSTM-A Tuning] Best: {best_hp_a}')

        # 80/20 split of training sequences into train/val
        val_split = int(len(X_tr_a) * (1 - LSTM_A_VAL_SPLIT))

        # Train with tuned or default hyperparameters
        if best_hp_a:
            model_a = train_lstm_a(
                X_tr_a[:val_split], y_tr_a[:val_split],
                X_tr_a[val_split:], y_tr_a[val_split:],
                device,
                optimizer_name=best_hp_a['optimizer'],
                lr=best_hp_a['lr'],
                hidden_size=best_hp_a['hidden_size'],
                num_layers=best_hp_a['num_layers'],
                dropout=best_hp_a['dropout'],
                batch_size=best_hp_a['batch_size'],
            )
        else:
            model_a = train_lstm_a(
                X_tr_a[:val_split], y_tr_a[:val_split],
                X_tr_a[val_split:], y_tr_a[val_split:],
                device
            )
        print(f'  [LSTM-A]  fit done in {time.time()-t0:.1f}s')

        # LSTM-A inference
        probs_a = predict_lstm(model_a, X_te_a, device)

        # Free LSTM-A memory before training LSTM-B
        del model_a, X_tr_a, y_tr_a, X_te_a, y_te_a
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # ── LSTM-B: Extended (6 features, fixed architecture) ────────────────
        t0 = time.time()
        print('  [LSTM-B]  building sequences & training...')

        X_tr_b, y_tr_b, X_te_b, y_te_b, keys_tr_b, keys_te_b = prepare_lstm_b_sequences(
            df_train_fold, df_test_fold
        )

        print(f'    LSTM-B sequences: train={len(X_tr_b)}, test={len(X_te_b)}')

        val_split = int(len(X_tr_b) * (1 - LSTM_B_VAL_SPLIT))
        model_b = train_lstm_b(
            X_tr_b[:val_split], y_tr_b[:val_split],
            X_tr_b[val_split:], y_tr_b[val_split:],
            device
        )
        print(f'  [LSTM-B]  fit done in {time.time()-t0:.1f}s')

        # LSTM-B inference
        probs_b = predict_lstm(model_b, X_te_b, device)

        # Free LSTM-B memory for next fold
        del model_b, X_tr_b, y_tr_b, X_te_b, y_te_b
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # ── Collect predictions for this fold ────────────────────────────────
        pred = df_ts.copy().reset_index(drop=True)
        pred['Prob_LR'] = lr_m.predict_proba(X_ts_b_s)[:, 1]
        pred['Prob_RF'] = rf_m.predict_proba(X_ts_b_s)[:, 1]
        pred['Prob_XGB'] = xgb_m.predict(xgb.DMatrix(X_ts_b_s))
        pred['Prob_LSTM_A'] = align_predictions_to_df(probs_a, keys_te_a, df_ts)
        pred['Prob_LSTM_B'] = align_predictions_to_df(probs_b, keys_te_b, df_ts)
        pred['Fold'] = fold['fold']

        # Report coverage
        n_lstm_a = pred['Prob_LSTM_A'].notna().sum()
        n_lstm_b = pred['Prob_LSTM_B'].notna().sum()
        print(f'  Predictions: LR/RF/XGB={len(pred)}, '
              f'LSTM-A={n_lstm_a}, LSTM-B={n_lstm_b}')

        all_preds.append(pred)

    # ── Combine folds ─────────────────────────────────────────────────────────
    full_preds = pd.concat(all_preds).reset_index(drop=True)
    print(f'\nTotal predictions: {len(full_preds)}')
    print(f'  LSTM-A valid: {full_preds["Prob_LSTM_A"].notna().sum()}')
    print(f'  LSTM-B valid: {full_preds["Prob_LSTM_B"].notna().sum()}')

    # ── Backtest each model independently ─────────────────────────────────────
    print('\n' + '=' * 60)
    print('RESULTS — GROSS (0 bps)')
    print('=' * 60)

    model_cols = {
        'LR': 'Prob_LR',
        'RF': 'Prob_RF',
        'XGBoost': 'Prob_XGB',
        'LSTM-A': 'Prob_LSTM_A',
        'LSTM-B': 'Prob_LSTM_B',
    }

    port_returns_gross = {}
    port_returns_net_5 = {}
    port_returns_net_10 = {}
    class_metrics = []
    all_signals = []
    daily_returns_gross = {'Date': None}
    daily_returns_net_5 = {'Date': None}

    for model_name, prob_col in model_cols.items():
        # Filter to rows that have predictions for this model
        valid_preds = full_preds.dropna(subset=[prob_col]).copy()

        if len(valid_preds) == 0:
            print(f'  {model_name:<12}  [SKIPPED - no valid predictions]')
            continue

        # Generate signals for this model
        sig_df = generate_signals(valid_preds, k=K_STOCKS, prob_col=prob_col)
        sig_df['Model'] = model_name
        all_signals.append(sig_df)

        # Compute portfolio returns at different TC levels
        port_gross = compute_portfolio_returns(sig_df, tc_bps=0, k=K_STOCKS)
        port_net_5 = compute_portfolio_returns(sig_df, tc_bps=5, k=K_STOCKS)
        port_net_10 = compute_portfolio_returns(sig_df, tc_bps=10, k=K_STOCKS)

        port_returns_gross[model_name] = port_gross
        port_returns_net_5[model_name] = port_net_5
        port_returns_net_10[model_name] = port_net_10

        # Store daily returns for export
        if daily_returns_gross['Date'] is None:
            daily_returns_gross['Date'] = port_gross.index
            daily_returns_net_5['Date'] = port_net_5.index
        daily_returns_gross[model_name] = port_gross['Gross_Return'].values
        daily_returns_net_5[model_name] = port_net_5['Net_Return'].values

        # Classification metrics
        y_true = valid_preds[TARGET_COL].values
        y_prob = valid_preds[prob_col].values
        cm = evaluate_classification(y_true, y_prob)
        cm['Model'] = model_name
        class_metrics.append(cm)

        # Print gross metrics
        m = compute_metrics(port_gross['Gross_Return'])
        print(f'  {model_name:<12}  '
              f'Sharpe={m["Sharpe Ratio"]:>6.3f}  '
              f'Ann.Ret={m["Annualized Return (%)"]:.2f}%  '
              f'MDD={m["Max Drawdown (%)"]:.2f}%')

    # ── Print net results ─────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('RESULTS — NET (5 bps)')
    print('=' * 60)

    results_gross = []
    results_net_5 = []
    results_net_10 = []

    for model_name in model_cols.keys():
        if model_name not in port_returns_gross:
            continue

        # Gross
        m = compute_metrics(port_returns_gross[model_name]['Gross_Return'])
        m['Model'] = model_name
        results_gross.append(m)

        # Net 5 bps
        m = compute_metrics(port_returns_net_5[model_name]['Net_Return'])
        m['Model'] = model_name
        results_net_5.append(m)
        print(f'  {model_name:<12}  '
              f'Sharpe={m["Sharpe Ratio"]:>6.3f}  '
              f'Ann.Ret={m["Annualized Return (%)"]:.2f}%  '
              f'MDD={m["Max Drawdown (%)"]:.2f}%')

        # Net 10 bps
        m = compute_metrics(port_returns_net_10[model_name]['Net_Return'])
        m['Model'] = model_name
        results_net_10.append(m)

    # ── Sub-period analysis (using the best performing model) ─────────────────
    subperiod_metrics = None
    if 'LSTM-A' in port_returns_net_5:
        try:
            subperiod_metrics = compute_subperiod_metrics(
                port_returns_net_5['LSTM-A']['Net_Return']
            )
        except Exception as e:
            print(f'Warning: Could not compute sub-period metrics: {e}')

    # ── Save all results ─────────────────────────────────────────────────────
    results_dict = {
        'gross': results_gross,
        'net_5': results_net_5,
        'net_10': results_net_10,
        'classification': class_metrics,
        'subperiod': subperiod_metrics,
    }

    daily_returns_gross_df = pd.DataFrame(daily_returns_gross)
    daily_returns_net_5_df = pd.DataFrame(daily_returns_net_5)

    signals_df = pd.concat(all_signals).reset_index(drop=True) if all_signals else pd.DataFrame()

    save_all_results(
        results_dict=results_dict,
        daily_returns_dict={
            'gross': daily_returns_gross_df,
            'net_5': daily_returns_net_5_df,
        },
        signals_dict=signals_df,
        tuning_results=tuning_results,
        reports_dir='reports'
    )

    print('\n\nPipeline complete.')
    return {
        'full_preds': full_preds,
        'port_returns': port_returns_net_5,
        'class_metrics': class_metrics,
        'folds': folds,
    }


if __name__ == '__main__':
    main(load_cached=False)
