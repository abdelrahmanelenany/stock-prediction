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
import random
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger('pipeline')

from pipeline.data_loader import download_and_save
from pipeline.features import (
    build_feature_matrix, FEATURE_COLS,
    compute_wavelet_thresholds, apply_wavelet_denoising, apply_wavelet_denoising_causal,
    recompute_features_from_denoised,
)
from pipeline.targets import create_targets
from pipeline.walk_forward import generate_walk_forward_folds, print_fold_summary
from pipeline.standardizer import standardize_fold, winsorize_fold
from pipeline.fold_reporting import save_fold_report
from evaluation.metrics_utils import binary_auc_safe, classification_sanity_checks, log_split_balance
from models.baselines import train_logistic, train_random_forest, train_xgboost
from models.lstm_model import (
    prepare_lstm_a_sequences, prepare_lstm_b_sequences,
    prepare_lstm_a_sequences_temporal_split, prepare_lstm_b_sequences_temporal_split,
    train_lstm_a, train_lstm_b,
    predict_lstm, align_predictions_to_df,
    tune_lstm_hyperparams,
)
from backtest.signals import (
    generate_signals,
    smooth_probabilities,
    apply_holding_period_constraint,
    compute_turnover_and_holding_stats,
)
from backtest.portfolio import compute_portfolio_returns
from backtest.metrics import (
    compute_metrics, evaluate_classification,
    compute_subperiod_metrics, compute_daily_auc,
)
import config
from config import (
    TRAIN_DAYS, VAL_DAYS, TEST_DAYS,
    K_STOCKS, TC_BPS, MODELS,
    LSTM_A_VAL_SPLIT, LSTM_B_VAL_SPLIT,
    LSTM_A_FEATURES, LSTM_B_FEATURES,
    BASELINE_FEATURE_COLS,
    SIGNAL_SMOOTH_ALPHA, MIN_HOLDING_DAYS,
    SLIPPAGE_BPS,
    WINSORIZE_ENABLED, WINSORIZE_LOWER_Q, WINSORIZE_UPPER_Q,
    TRAIN_WINDOW_MODE,
    RUN_SIGNAL_ABLATION,
    SAVE_FOLD_REPORTS,
    SIGNAL_EMA_METHOD, SIGNAL_EMA_SPAN,
)

TARGET_COL = 'Target'
CACHE_FEATURES_PATH = 'data/processed/features.csv'

# ── Pipeline options ─────────────────────────────────────────────────────────
ENABLE_LSTM_TUNING = False  # Set True to run Bhandari §3.3 hyperparameter tuning
                            # (computationally expensive — ~2-3x slower per fold)

RUN_BASELINES = True        # Set False to skip LR, RF, XGB
RUN_LSTMS = True            # Set False to skip LSTM-A, LSTM-B

device = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('mps') if torch.backends.mps.is_available()
    else torch.device('cpu')
)
print(f'Using device: {device}')


def set_global_seed(seed: int):
    """Set deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Keep deterministic mode best-effort to avoid runtime failures on unsupported ops.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


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
    ]:
        lines.append(f"\n{'─' * 60}")
        lines.append(label)
        lines.append(f"{'─' * 60}")
        for row in results_dict[key]:
            lines.append(
                f"  {row['Model']:<12}  "
                f"Sharpe={row['Sharpe Ratio']:>6.3f}  "
                f"Sortino={row['Sortino Ratio']:>6.3f}  "
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

def run_walk_forward_pipeline(
    load_cached: bool = True,
    train_days: int | None = None,
    reports_dir: str = 'reports',
):
    """
    Parameters
    ----------
    load_cached : bool
        If True (default), load data from data/processed/features.csv instead
        of re-downloading and recomputing. Set False to run from scratch.
    train_days : int, optional
        Override config TRAIN_DAYS for walk-forward train window length.
    reports_dir : str
        Directory for tables and fold reports.
    """
    print("=" * 60)
    print(f"BACKTEST — {len(MODELS)} MODELS × N FOLDS")
    print(f"LSTM Tuning: {'ENABLED' if ENABLE_LSTM_TUNING else 'DISABLED'}")
    print(f"Scaler Type: {config.SCALER_TYPE}")
    print("=" * 60)

    # Ensure run-to-run reproducibility for all stochastic components.
    set_global_seed(config.RANDOM_SEED)

    # ── Steps 1-3: Data ─────────────────────────────────────────────────────
    if load_cached:
        print('\nLoading cached features.csv ...')
        data = pd.read_csv(CACHE_FEATURES_PATH, parse_dates=['Date'])
        print(f'Loaded {len(data)} rows, {len(data.columns)} columns.')

        # Backward compatibility: older caches may contain features but not targets.
        # If enough columns exist, rebuild targets from cached Return_1d and persist.
        missing_target_cols = [c for c in ['Return_NextDay', TARGET_COL] if c not in data.columns]
        if missing_target_cols:
            print(f"Cached file missing target columns: {missing_target_cols}")
            if {'Date', 'Ticker', 'Return_1d'}.issubset(data.columns):
                print('Recomputing Return_NextDay/Target from cached features...')
                data = create_targets(data)
                data.to_csv(CACHE_FEATURES_PATH, index=False)
                print(f'Repaired and saved cache to {CACHE_FEATURES_PATH}')
            else:
                raise ValueError(
                    'Cached features.csv is missing target columns and lacks the columns '
                    "required to rebuild them ('Date', 'Ticker', 'Return_1d'). "
                    'Run once with load_cached=False to regenerate cache.'
                )
    else:
        raw = download_and_save()
        data = build_feature_matrix(raw)
        data = create_targets(data)
        # Persist full dataset (features + Return_NextDay + Target) for future cached runs.
        data.to_csv(CACHE_FEATURES_PATH, index=False)
        print(f'Saved full cache (with targets) to {CACHE_FEATURES_PATH}')

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

    # Step 4: Walk-forward folds
    dates = sorted(data['Date'].unique())
    td = train_days if train_days is not None else TRAIN_DAYS
    wf_stride = getattr(config, 'WALK_FORWARD_STRIDE', None)
    folds = generate_walk_forward_folds(
        dates, td, VAL_DAYS, getattr(config, 'TEST_DAYS', TEST_DAYS),
        stride_days=wf_stride,
        train_window_mode=TRAIN_WINDOW_MODE,
    )
    print()
    print_fold_summary(folds)

    ablation_rows: list[dict] = []

    # ── Steps 5-6: Train models per fold ────────────────────────────────────
    all_preds = []
    tuning_results = []  # Store tuning results for thesis reporting

    for fold in tqdm(folds, desc="Walk-Forward Folds", unit="fold"):
        print(f'\n{"=" * 60}')
        print(f'=== Fold {fold["fold"]}/{len(folds)} ===')
        print('=' * 60)

        # Use deterministic fold-specific seeds so reruns are stable while
        # keeping each fold/model independent.
        fold_seed_base = config.RANDOM_SEED + (fold['fold'] * 1000)
        set_global_seed(fold_seed_base)

        df_tr = data[data['Date'].isin(dates[fold['train'][0]:fold['train'][1]])]
        df_v = data[data['Date'].isin(dates[fold['val'][0]:fold['val'][1]])]
        df_ts = data[data['Date'].isin(dates[fold['test'][0]:fold['test'][1]])]

        print(f'  Train: {len(df_tr):>6} rows | '
              f'Val: {len(df_v):>5} rows | '
              f'Test: {len(df_ts):>5} rows')

        # ── Per-fold wavelet denoising (CAUSAL - NO LEAKAGE) ─────────────────────
        # 1. Threshold computed from training data only
        # 2. CAUSAL denoising: each value uses only historical data (rolling window)
        # 3. Apply to each split INDEPENDENTLY (no concatenation)
        # 4. CRITICAL: Do NOT recompute Target - it must remain based on raw returns
        if config.USE_WAVELET_DENOISING:
            print('  [Wavelet] Computing thresholds from training data...')
            wavelet_thresholds = compute_wavelet_thresholds(df_tr)

            # Apply CAUSAL denoising to each split INDEPENDENTLY
            # This prevents future data from leaking into training
            print('  [Wavelet] Applying causal denoising per split...')
            df_tr = apply_wavelet_denoising_causal(df_tr, thresholds=wavelet_thresholds)
            df_v = apply_wavelet_denoising_causal(df_v, thresholds=wavelet_thresholds)
            df_ts = apply_wavelet_denoising_causal(df_ts, thresholds=wavelet_thresholds)

            # Recompute Close-dependent FEATURES only (RSI, MACD, BB, etc.)
            # CRITICAL: Do NOT recompute Return_NextDay or Target - they must
            # remain based on RAW realized returns for honest evaluation
            print('  [Wavelet] Recomputing features from denoised Close...')

            # Save original targets before feature recomputation
            tr_target = df_tr['Target'].copy()
            tr_return_next = df_tr['Return_NextDay'].copy()
            v_target = df_v['Target'].copy()
            v_return_next = df_v['Return_NextDay'].copy()
            ts_target = df_ts['Target'].copy()
            ts_return_next = df_ts['Return_NextDay'].copy()

            # Recompute features from denoised Close
            df_tr = recompute_features_from_denoised(df_tr)
            df_v = recompute_features_from_denoised(df_v)
            df_ts = recompute_features_from_denoised(df_ts)

            # Restore original targets (based on raw returns, not denoised)
            df_tr['Target'] = tr_target.values
            df_tr['Return_NextDay'] = tr_return_next.values
            df_v['Target'] = v_target.values
            df_v['Return_NextDay'] = v_return_next.values
            df_ts['Target'] = ts_target.values
            df_ts['Return_NextDay'] = ts_return_next.values

            # Drop rows with NaN from warm-up period after recomputation
            df_tr = df_tr.dropna(subset=FEATURE_COLS).reset_index(drop=True)
            df_v = df_v.dropna(subset=FEATURE_COLS).reset_index(drop=True)
            df_ts = df_ts.dropna(subset=FEATURE_COLS).reset_index(drop=True)

            print(f'  [Wavelet] After denoising: Train={len(df_tr)}, '
                  f'Val={len(df_v)}, Test={len(df_ts)}')

        y_tr = df_tr[TARGET_COL].values.astype(int)
        y_v = df_v[TARGET_COL].values.astype(int)
        y_ts = df_ts[TARGET_COL].values.astype(int)

        log_split_balance(y_tr, f'fold{fold["fold"]} train', logger)
        log_split_balance(y_v, f'fold{fold["fold"]} val', logger)
        log_split_balance(y_ts, f'fold{fold["fold"]} test', logger)

        Xb_tr = df_tr[BASELINE_FEATURE_COLS].values
        Xb_v = df_v[BASELINE_FEATURE_COLS].values
        Xb_ts = df_ts[BASELINE_FEATURE_COLS].values
        if WINSORIZE_ENABLED:
            Xb_tr, Xb_v, Xb_ts = winsorize_fold(
                Xb_tr, Xb_v, Xb_ts,
                lower_q=WINSORIZE_LOWER_Q, upper_q=WINSORIZE_UPPER_Q,
            )

        X_tr_b_s, X_v_b_s, X_ts_b_s, _ = standardize_fold(Xb_tr, Xb_v, Xb_ts)

        # Baseline Models
        if RUN_BASELINES:
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
        else:
            print('\n  [Baselines] Skipping LR, RF, XGBoost (RUN_BASELINES=False)')
            lr_m = rf_m = xgb_m = None

        # ── LSTM-A: Bhandari-inspired (4 technical features) ─────────────────
        probs_a = None
        keys_te_a = None
        probs_b = None
        keys_te_b = None
        
        if RUN_LSTMS:
            t0 = time.time()
            print('  [LSTM-A]  building sequences & training...')

            # Combine train and validation for LSTM data preparation
            df_train_fold = pd.concat([df_tr, df_v]).sort_values(['Ticker', 'Date'])
            df_test_fold = df_ts.copy()

            # Use TEMPORAL split (splits by date, not index) - FIX for ticker-based split bug
            X_tr_a, y_tr_a, X_val_a, y_val_a, X_te_a, y_te_a, keys_tr_a, keys_val_a, keys_te_a = \
                prepare_lstm_a_sequences_temporal_split(df_train_fold, df_test_fold, val_ratio=LSTM_A_VAL_SPLIT)

            print(f'    LSTM-A sequences: train={len(X_tr_a)}, val={len(X_val_a)}, test={len(X_te_a)}')

            # Optional: hyperparameter tuning for LSTM-A (using temporal val split)
            best_hp_a = None
            if ENABLE_LSTM_TUNING:
                print('    [LSTM-A Tuning] Running Phase 1 + Phase 2...')
                best_hp_a = tune_lstm_hyperparams(
                    X_tr_a, y_tr_a,
                    X_val_a, y_val_a,
                    input_size=len(LSTM_A_FEATURES),
                    device=device,
                    arch_grid=config.LSTM_A_ARCH_GRID,  # Phase 2 architecture search
                    seed=fold_seed_base + 10,
                )
                tuning_results.append({
                    'fold': fold['fold'],
                    'model': 'LSTM-A',
                    **best_hp_a
                })
                print(f'    [LSTM-A Tuning] Best: {best_hp_a}')

            # Train with tuned or default hyperparameters (using temporal val split)
            if best_hp_a:
                model_a = train_lstm_a(
                    X_tr_a, y_tr_a,
                    X_val_a, y_val_a,
                    device,
                    optimizer_name=best_hp_a['optimizer'],
                    lr=best_hp_a['lr'],
                    hidden_size=best_hp_a['hidden_size'],
                    num_layers=best_hp_a['num_layers'],
                    dropout=best_hp_a['dropout'],
                    batch_size=best_hp_a['batch_size'],
                    seed=fold_seed_base + 20,
                    fold_idx=fold['fold'],
                )
            else:
                model_a = train_lstm_a(
                    X_tr_a, y_tr_a,
                    X_val_a, y_val_a,
                    device,
                    seed=fold_seed_base + 20,
                    fold_idx=fold['fold'],
                )
            print(f'  [LSTM-A]  fit done in {time.time()-t0:.1f}s')

            # LSTM-A inference
            probs_a = predict_lstm(model_a, X_te_a, device)

            # Free LSTM-A memory before training LSTM-B
            del model_a, X_tr_a, y_tr_a, X_val_a, y_val_a, X_te_a, y_te_a
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

            # ── LSTM-B: Extended (6 features, fixed architecture) ────────────────
            t0 = time.time()
            print('  [LSTM-B]  building sequences & training...')

            # Use TEMPORAL split (splits by date, not index) - FIX for ticker-based split bug
            X_tr_b, y_tr_b, X_val_b, y_val_b, X_te_b, y_te_b, keys_tr_b, keys_val_b, keys_te_b = \
                prepare_lstm_b_sequences_temporal_split(df_train_fold, df_test_fold, val_ratio=LSTM_B_VAL_SPLIT)

            print(f'    LSTM-B sequences: train={len(X_tr_b)}, val={len(X_val_b)}, test={len(X_te_b)}')

            model_b = train_lstm_b(
                X_tr_b, y_tr_b,
                X_val_b, y_val_b,
                device,
                seed=fold_seed_base + 30,
                fold_idx=fold['fold'],
            )
            print(f'  [LSTM-B]  fit done in {time.time()-t0:.1f}s')

            # LSTM-B inference
            probs_b = predict_lstm(model_b, X_te_b, device)

            # Free LSTM-B memory for next fold
            del model_b, X_tr_b, y_tr_b, X_val_b, y_val_b, X_te_b, y_te_b
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        else:
            print('\n  [LSTMs]     Skipping LSTM-A and LSTM-B (RUN_LSTMS=False)')

        # ── Collect predictions for this fold ────────────────────────────────
        pred = df_ts.copy().reset_index(drop=True)
        pred['Prob_LR'] = lr_m.predict_proba(X_ts_b_s)[:, 1] if RUN_BASELINES else np.nan
        pred['Prob_RF'] = rf_m.predict_proba(X_ts_b_s)[:, 1] if RUN_BASELINES else np.nan
        pred['Prob_XGB'] = xgb_m.predict(xgb.DMatrix(X_ts_b_s)) if RUN_BASELINES else np.nan
        pred['Prob_LSTM_A'] = align_predictions_to_df(probs_a, keys_te_a, df_ts) if RUN_LSTMS else np.nan
        pred['Prob_LSTM_B'] = align_predictions_to_df(probs_b, keys_te_b, df_ts) if RUN_LSTMS else np.nan
        pred['Fold'] = fold['fold']

        if RUN_BASELINES:
            classification_sanity_checks(
                y_ts, pred['Prob_LR'].values, name=f"fold{fold['fold']} test LR",
            )
            val_auc_lr = binary_auc_safe(y_v, lr_m.predict_proba(X_v_b_s)[:, 1], log_on_fail=False)
            test_auc_lr = binary_auc_safe(y_ts, pred['Prob_LR'].values, log_on_fail=False)
        else:
            val_auc_lr = test_auc_lr = float('nan')

        if SAVE_FOLD_REPORTS:
            fr_extra = {
                'val_auc_lr': val_auc_lr,
                'test_auc_lr': test_auc_lr,
            }
            path_fr = save_fold_report(
                fold, df_tr, df_v, df_ts, TARGET_COL,
                extra=fr_extra,
                reports_dir=os.path.join(reports_dir, 'fold_reports'),
            )
            print(f'  Fold report: {path_fr}')

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

        smoothed_preds = smooth_probabilities(
            valid_preds, prob_col,
            alpha=SIGNAL_SMOOTH_ALPHA,
            ema_method=SIGNAL_EMA_METHOD,
            ema_span=SIGNAL_EMA_SPAN,
        )
        smoothed_col = f'{prob_col}_Smooth'

        sig_df, sig_diag = generate_signals(
            smoothed_preds, k=K_STOCKS, prob_col=smoothed_col,
            return_diagnostics=True,
        )
        sig_df = apply_holding_period_constraint(sig_df, min_hold_days=MIN_HOLDING_DAYS)
        hold_st = compute_turnover_and_holding_stats(sig_df, k=K_STOCKS)
        print(f'  [{model_name}] turnover~{hold_st["mean_daily_turnover_half_turns"]:.2f}  '
              f'avg_hold~{hold_st["avg_holding_period_trading_days"]:.1f}  '
              f'threshold_filtered(L/S)={sig_diag["long_slots_filtered_by_threshold"]}/'
              f'{sig_diag["short_slots_filtered_by_threshold"]}')

        sig_df['Model'] = model_name
        all_signals.append(sig_df)

        port_gross = compute_portfolio_returns(
            sig_df, tc_bps=0, k=K_STOCKS, slippage_bps=SLIPPAGE_BPS,
        )
        port_net_5 = compute_portfolio_returns(
            sig_df, tc_bps=TC_BPS, k=K_STOCKS, slippage_bps=SLIPPAGE_BPS,
        )

        if RUN_SIGNAL_ABLATION:
            sig_raw, _ = generate_signals(
                valid_preds, k=K_STOCKS, prob_col=prob_col,
                confidence_threshold=0.0, return_diagnostics=True,
            )
            sig_raw = apply_holding_period_constraint(sig_raw, min_hold_days=1)
            port_raw = compute_portfolio_returns(
                sig_raw, tc_bps=TC_BPS, k=K_STOCKS, slippage_bps=SLIPPAGE_BPS,
            )
            st_raw = compute_turnover_and_holding_stats(sig_raw, k=K_STOCKS)
            ablation_rows.append({
                'Model': model_name,
                'variant': 'raw_rank_min_hold_1',
                'Sharpe Net': compute_metrics(port_raw['Net_Return'])['Sharpe Ratio'],
                'mean_turnover': st_raw['mean_daily_turnover_half_turns'],
                'avg_hold': st_raw['avg_holding_period_trading_days'],
            })
            ablation_rows.append({
                'Model': model_name,
                'variant': 'ema_threshold_min_hold',
                'Sharpe Net': compute_metrics(port_net_5['Net_Return'])['Sharpe Ratio'],
                'mean_turnover': hold_st['mean_daily_turnover_half_turns'],
                'avg_hold': hold_st['avg_holding_period_trading_days'],
            })

        port_returns_gross[model_name] = port_gross
        port_returns_net_5[model_name] = port_net_5

        # Store daily returns for export
        if daily_returns_gross['Date'] is None:
            daily_returns_gross['Date'] = port_gross.index
            daily_returns_net_5['Date'] = port_net_5.index
        daily_returns_gross[model_name] = port_gross['Gross_Return'].values
        daily_returns_net_5[model_name] = port_net_5['Net_Return'].values

        # Classification metrics (pooled + daily AUC for diagnostic)
        y_true = valid_preds[TARGET_COL].values
        y_prob = valid_preds[prob_col].values
        cm = evaluate_classification(y_true, y_prob)

        # Add daily AUC to diagnose pooled vs within-day ranking
        daily_auc = compute_daily_auc(valid_preds, prob_col, TARGET_COL)
        cm['Daily AUC (mean)'] = daily_auc['Daily AUC (mean)']
        cm['Daily AUC (std)'] = daily_auc['Daily AUC (std)']

        cm['Model'] = model_name
        class_metrics.append(cm)

        # Print gross metrics
        m = compute_metrics(port_gross['Gross_Return'])
        print(f'  {model_name:<12}  '
              f'Sharpe={m["Sharpe Ratio"]:>6.3f}  '
              f'Sortino={m["Sortino Ratio"]:>6.3f}  '
              f'Ann.Ret={m["Annualized Return (%)"]:.2f}%  '
              f'MDD={m["Max Drawdown (%)"]:.2f}%')

    # ── Ensemble Model Evaluation ──────────────────────────────────────────────
    print('\n  [Ensemble] Computing ensemble...')
    ensemble_cols = ['Prob_LR', 'Prob_RF', 'Prob_XGB', 'Prob_LSTM_A', 'Prob_LSTM_B']

    # Sub-select only columns that actually have valid predictions
    actual_ensemble_cols = [c for c in ensemble_cols if c in full_preds.columns and full_preds[c].notna().any()]
    
    if len(actual_ensemble_cols) >= 2:
        # Compute ensemble probability as mean of available models
        ensemble_preds = full_preds.dropna(subset=actual_ensemble_cols).copy()
        ensemble_preds['Prob_ENS'] = ensemble_preds[actual_ensemble_cols].mean(axis=1)

        # Apply smoothing and holding period
        ensemble_smoothed = smooth_probabilities(
            ensemble_preds, 'Prob_ENS',
            alpha=SIGNAL_SMOOTH_ALPHA,
            ema_method=SIGNAL_EMA_METHOD,
            ema_span=SIGNAL_EMA_SPAN,
        )
        sig_df_ens, _ = generate_signals(
            ensemble_smoothed, k=K_STOCKS, prob_col='Prob_ENS_Smooth',
            return_diagnostics=True,
        )
        sig_df_ens = apply_holding_period_constraint(sig_df_ens, min_hold_days=MIN_HOLDING_DAYS)
        sig_df_ens['Model'] = 'Ensemble'
        all_signals.append(sig_df_ens)

        port_gross_ens = compute_portfolio_returns(
            sig_df_ens, tc_bps=0, k=K_STOCKS, slippage_bps=SLIPPAGE_BPS,
        )
        port_net_5_ens = compute_portfolio_returns(
            sig_df_ens, tc_bps=TC_BPS, k=K_STOCKS, slippage_bps=SLIPPAGE_BPS,
        )

        port_returns_gross['Ensemble'] = port_gross_ens
        port_returns_net_5['Ensemble'] = port_net_5_ens

        # Store daily returns
        daily_returns_gross['Ensemble'] = port_gross_ens['Gross_Return'].values
        daily_returns_net_5['Ensemble'] = port_net_5_ens['Net_Return'].values

        # Classification metrics for ensemble
        y_true_ens = ensemble_preds[TARGET_COL].values
        y_prob_ens = ensemble_preds['Prob_ENS'].values
        cm_ens = evaluate_classification(y_true_ens, y_prob_ens)
        daily_auc_ens = compute_daily_auc(ensemble_preds, 'Prob_ENS', TARGET_COL)
        cm_ens['Daily AUC (mean)'] = daily_auc_ens['Daily AUC (mean)']
        cm_ens['Daily AUC (std)'] = daily_auc_ens['Daily AUC (std)']
        cm_ens['Model'] = 'Ensemble'
        class_metrics.append(cm_ens)

        # Print gross metrics
        m = compute_metrics(port_gross_ens['Gross_Return'])
        print(f'  {"Ensemble":<12}  '
              f'Sharpe={m["Sharpe Ratio"]:>6.3f}  '
              f'Sortino={m["Sortino Ratio"]:>6.3f}  '
              f'Ann.Ret={m["Annualized Return (%)"]:.2f}%  '
              f'MDD={m["Max Drawdown (%)"]:.2f}%')

    # ── Print net results ─────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print(f'RESULTS — NET ({TC_BPS:g} bps TC + {SLIPPAGE_BPS:g} bps slippage per half-turn)')
    print('=' * 60)

    results_gross = []
    results_net_5 = []

    # Include Ensemble in the list of models to report
    all_model_names = list(model_cols.keys()) + ['Ensemble']

    for model_name in all_model_names:
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
              f'Sortino={m["Sortino Ratio"]:>6.3f}  '
              f'Ann.Ret={m["Annualized Return (%)"]:.2f}%  '
              f'MDD={m["Max Drawdown (%)"]:.2f}%')

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
        'classification': class_metrics,
        'subperiod': subperiod_metrics,
    }

    daily_returns_gross_df = pd.DataFrame(daily_returns_gross)
    daily_returns_net_5_df = pd.DataFrame(daily_returns_net_5)

    signals_df = pd.concat(all_signals).reset_index(drop=True) if all_signals else pd.DataFrame()

    # Save full predictions so Baseline/LSTM runs can be stitched together later
    full_preds_path = os.path.join(reports_dir, 'full_predictions.csv')
    full_preds.to_csv(full_preds_path, index=False)
    print(f'  Saved raw probabilities to {full_preds_path}')

    save_all_results(
        results_dict=results_dict,
        daily_returns_dict={
            'gross': daily_returns_gross_df,
            'net_5': daily_returns_net_5_df,
        },
        signals_dict=signals_df,
        tuning_results=tuning_results,
        reports_dir=reports_dir,
    )

    if ablation_rows:
        ab_path = os.path.join(reports_dir, 'signal_ablation_summary.csv')
        pd.DataFrame(ablation_rows).to_csv(ab_path, index=False)
        print(f'Signal ablation summary: {ab_path}')

    print('\n\nPipeline complete.')
    return {
        'full_preds': full_preds,
        'port_returns': port_returns_net_5,
        'class_metrics': class_metrics,
        'folds': folds,
    }


def main(load_cached: bool = True):
    """CLI entry: delegates to run_walk_forward_pipeline with default reports dir."""
    return run_walk_forward_pipeline(load_cached=load_cached, reports_dir='reports')


if __name__ == '__main__':
    main(load_cached=False)
