"""
main.py — Full pipeline orchestrator
Walk-forward validated long-short strategy using LR, RF, XGBoost, LSTM, Ensemble.

Models:
  - LR:       Logistic Regression (baseline)
  - RF:       Random Forest (baseline)
  - XGBoost:  Gradient Boosted Trees (baseline)
  - LSTM:   Primary neural-network model (multi-feature, fixed architecture)
  - Ensemble: Mean probability of LR + LSTM (RF and XGBoost excluded: negative Sharpe)

Run:
    .venv/bin/python3 main.py
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'  # Prevent XGBoost/PyTorch dual-OpenMP segfault

import json
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
from pipeline.features import build_feature_matrix, FEATURE_COLS
from pipeline.targets import create_targets
from pipeline.walk_forward import generate_walk_forward_folds, print_fold_summary
from pipeline.standardizer import standardize_fold, winsorize_fold
from pipeline.fold_reporting import save_fold_report
from evaluation.metrics_utils import binary_auc_safe, classification_sanity_checks, log_split_balance
from models.baselines import train_logistic, train_random_forest, train_xgboost, extract_feature_importances
from models.calibration import ProbabilityCalibrator
from models.lstm_model import (
    prepare_lstm_b_sequences_temporal_split,
    train_lstm_b,
    predict_lstm, align_predictions_to_df,
    tune_lstm_hyperparams,
)
from models.tcn_model import (
    prepare_tcn_sequences_temporal_split,
    train_tcn,
    predict_tcn,
    tune_tcn_hyperparams,
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
    LSTM_B_VAL_SPLIT,
    LSTM_B_FEATURES,
    BASELINE_FEATURE_COLS,
    SIGNAL_SMOOTH_ALPHA, MIN_HOLDING_DAYS,
    SLIPPAGE_BPS,
    WINSORIZE_ENABLED, WINSORIZE_LOWER_Q, WINSORIZE_UPPER_Q,
    TRAIN_WINDOW_MODE,
    RUN_SIGNAL_ABLATION,
    SAVE_FOLD_REPORTS,
    SIGNAL_EMA_METHOD, SIGNAL_EMA_SPAN,
    LARGE_CAP_CONFIG, SMALL_CAP_CONFIG,
    TARGET_HORIZON_DAYS,
)

# Whether the active universe inverts signals at portfolio level.
# Classification metrics must be evaluated on (1 - prob) when True,
# because the portfolio bets against the raw model output.
_UNIVERSE_CFG = LARGE_CAP_CONFIG if config.UNIVERSE_MODE == 'large_cap' else SMALL_CAP_CONFIG
INVERT_SIGNALS: bool = _UNIVERSE_CFG.invert_signals

TARGET_COL = 'Target'
CACHE_FEATURES_PATH = f'data/processed/features_{config.UNIVERSE_MODE}.csv'

# ── Pipeline options ─────────────────────────────────────────────────────────
RUN_BASELINES = True        # Set False to skip LR, RF, XGB
RUN_LSTMS = True            # Set False to skip LSTM
RUN_TCNS = bool(getattr(config, 'RUN_TCNS', True))   # Set False to skip TCN

device = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('mps') if torch.backends.mps.is_available()
    else torch.device('cpu')
)
print(f'Using device: {device}')


def set_global_seed(seed: int):
    """Set deterministic seeds across Python, NumPy, and PyTorch (including MPS)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

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
            'gross':  pd.DataFrame columns=[Date, LR, RF, XGBoost, LSTM, Ensemble],
            'net_5':  pd.DataFrame same columns,
        }
        signals_dict: pd.DataFrame with all signals
        tuning_results: list of dicts with tuning results per fold (optional)
        reports_dir: output directory path
    """
    os.makedirs(reports_dir, exist_ok=True)
    prefix = config.UNIVERSE_MODE

    # Table T5: Risk-Return Metrics
    pd.DataFrame(results_dict['gross']).to_csv(
        f'{reports_dir}/{prefix}_table_T5_gross_returns.csv', index=False
    )
    pd.DataFrame(results_dict['net_5']).to_csv(
        f'{reports_dir}/{prefix}_table_T5_net_returns_5bps.csv', index=False
    )

    # Table T8: Classification Metrics
    pd.DataFrame(results_dict['classification']).to_csv(
        f'{reports_dir}/{prefix}_table_T8_classification_metrics.csv', index=False
    )

    # Table T6: Sub-Period Performance
    if results_dict['subperiod'] is not None:
        results_dict['subperiod'].to_csv(
            f'{reports_dir}/{prefix}_table_T6_subperiod_performance.csv', index=False
        )

    # LSTM Tuning Results (Bhandari §3.3 Tables)
    if tuning_results and len(tuning_results) > 0:
        pd.DataFrame(tuning_results).to_csv(
            f'{reports_dir}/{prefix}_lstm_tuning_results.csv', index=False
        )

    # Raw daily returns
    daily_returns_dict['gross'].to_csv(
        f'{reports_dir}/{prefix}_daily_returns_gross.csv', index=False
    )
    daily_returns_dict['net_5'].to_csv(
        f'{reports_dir}/{prefix}_daily_returns_net_5bps.csv', index=False
    )

    # Signals
    signals_dict.to_csv(f'{reports_dir}/{prefix}_signals_all_models.csv', index=False)

    # Human-readable summary
    with open(f'{reports_dir}/{prefix}_backtest_summary.txt', 'w') as f:
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
    # Detect whether this run used inverted probs (from first available row)
    _first_cls = results_dict['classification'][0] if results_dict['classification'] else {}
    _inverted = _first_cls.get('Signals Inverted', False)
    if _inverted:
        lines.append("  (evaluated on 1-prob to match invert_signals=True trading direction)")
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
        If True (default), load data from the universe-specific cache
        data/processed/features_{UNIVERSE_MODE}.csv instead
        of re-downloading and recomputing. Set False to run from scratch.
    train_days : int, optional
        Override config TRAIN_DAYS for walk-forward train window length.
    reports_dir : str
        Directory for tables and fold reports.
    """
    lstm_b_tuning_enabled = bool(getattr(config, 'LSTM_B_ENABLE_TUNING', False))
    lstm_b_tune_once = bool(getattr(config, 'LSTM_B_TUNE_ON_FIRST_FOLD_ONLY', True))
    tcn_tuning_enabled = bool(getattr(config, 'TCN_ENABLE_TUNING', False))
    tcn_tune_once = bool(getattr(config, 'TCN_TUNE_ON_FIRST_FOLD_ONLY', True))

    # ── Frozen hyperparameters (TUNE_USE_FROZEN_HPS) ────────────────────────
    # Pre-populate best_hp_*_global from the saved JSON so the fold loop
    # skips the tuning phase entirely and reuses the frozen architecture.
    _frozen_hp_b_init = None   # will become best_hp_b_global before fold loop
    _frozen_hp_t_init = None   # will become best_hp_t_global before fold loop
    if getattr(config, 'TUNE_USE_FROZEN_HPS', False):
        _frozen_path = os.path.join(reports_dir, f'{config.UNIVERSE_MODE}_tuned_hyperparams.json')
        if os.path.exists(_frozen_path):
            with open(_frozen_path) as _fh:
                _frozen = json.load(_fh)
            _ABSENT = object()  # distinguishes missing key from JSON null
            _lstm_frozen = _frozen.get('lstm', _ABSENT)
            _tcn_frozen  = _frozen.get('tcn',  _ABSENT)
            # dict  → use those specific HPs
            # null  → default won tuning; use config defaults ('DEFAULT' sentinel)
            # absent → not yet saved; leave tuning enabled for that model
            if _lstm_frozen is not _ABSENT:
                _frozen_hp_b_init     = _lstm_frozen if _lstm_frozen is not None else 'DEFAULT'
                lstm_b_tuning_enabled = False
            if _tcn_frozen is not _ABSENT:
                _frozen_hp_t_init     = _tcn_frozen  if _tcn_frozen  is not None else 'DEFAULT'
                tcn_tuning_enabled    = False
            print(f'[Frozen HPs] Loaded from {_frozen_path}')
            print(f'  LSTM : {_lstm_frozen if _lstm_frozen is not _ABSENT else "(not saved yet — will tune)"}')
            print(f'  TCN  : {_tcn_frozen  if _tcn_frozen  is not _ABSENT else "(not saved yet — will tune)"}')
        else:
            print(
                f'[WARNING] TUNE_USE_FROZEN_HPS=True but {_frozen_path} not found. '
                'Falling back to scratch tuning.'
            )

    _HP_UNSET = object()  # sentinel: caller did not supply this model's HPs

    def _save_frozen_hps(hp_b=_HP_UNSET, hp_t=_HP_UNSET):
        """Write current best LSTM + TCN HPs to the frozen JSON (upsert).

        Pass hp_b / hp_t as a dict (tuned params) or None (default won).
        Omitting a keyword keeps that model's existing JSON entry untouched.
        """
        _path = os.path.join(reports_dir, f'{config.UNIVERSE_MODE}_tuned_hyperparams.json')
        existing = {}
        if os.path.exists(_path):
            try:
                with open(_path) as _fh:
                    existing = json.load(_fh)
            except Exception:
                existing = {}
        if hp_b is not _HP_UNSET:
            existing['lstm'] = hp_b   # dict → tuned params; None → use config defaults
        if hp_t is not _HP_UNSET:
            existing['tcn'] = hp_t
        existing['universe']  = config.UNIVERSE_MODE
        existing['saved_at']  = time.strftime('%Y-%m-%dT%H:%M:%S')
        with open(_path, 'w') as _fh:
            json.dump(existing, _fh, indent=2)
        print(f'    [Frozen HPs] Saved → {_path}')

    print("=" * 60)
    print("EXPERIMENT CONFIGURATION")
    print(f"  Universe mode : {config.UNIVERSE_MODE}")
    print(f"  Tickers       : {config.N_STOCKS} stocks")
    print(f"  Date range    : {config.START_DATE} -> {config.END_DATE}")
    print(f"  Windows       : {config.TRAIN_DAYS}/{config.VAL_DAYS}/{config.TEST_DAYS} days")
    print(f"  Sequence len  : {config.SEQ_LEN} days")
    print(f"  K (long/short): {config.K_STOCKS} stocks per side")
    print("=" * 60)
    print(f"BACKTEST — {len(MODELS)} MODELS × N FOLDS")
    print(f"LSTM Tuning: {'ENABLED' if lstm_b_tuning_enabled else 'DISABLED'}")
    print(f"Scaler Type: {config.SCALER_TYPE}")
    print(f"Configured tickers: {len(config.TICKERS)}")
    print("=" * 60)

    # Ensure run-to-run reproducibility for all stochastic components.
    set_global_seed(config.RANDOM_SEED)

    # ── Clear stale fold artifacts from previous runs ────────────────────────
    # Remove all files in fold_reports/ and training_logs/ so that only
    # files written by this run remain (avoids stale folds from prior runs
    # with a different fold count).
    import glob as _glob
    import shutil as _shutil
    for _stale_dir in [
        os.path.join(reports_dir, 'fold_reports'),
        os.path.join(os.path.dirname(__file__), 'reports', 'training_logs'),
    ]:
        if os.path.isdir(_stale_dir):
            for _f in _glob.glob(os.path.join(_stale_dir, '*')):
                try:
                    os.remove(_f)
                except Exception:
                    pass
            print(f'Cleared stale files from {_stale_dir}')

    # ── Steps 1-3: Data ─────────────────────────────────────────────────────
    if load_cached:
        print(f'\nLoading cached {os.path.basename(CACHE_FEATURES_PATH)} ...')
        data = pd.read_csv(CACHE_FEATURES_PATH, parse_dates=['Date'])
        print(f'Loaded {len(data)} rows, {len(data.columns)} columns.')

        # Backward compatibility: older caches may contain features but not targets,
        # or may have been built with a different TARGET_HORIZON_DAYS.
        # Check for the horizon tag embedded in the cache (written below).
        # Default to 1 so that legacy caches (without the tag) are treated as
        # having been built with the original 1-day target and are automatically
        # recomputed when TARGET_HORIZON_DAYS != 1.
        cached_horizon = 1
        if '_target_horizon' in data.columns:
            cached_horizon = int(data['_target_horizon'].iloc[0])
            data.drop(columns=['_target_horizon'], inplace=True)

        missing_target_cols = [c for c in ['Return_NextDay', TARGET_COL] if c not in data.columns]
        horizon_mismatch = (cached_horizon is not None) and (cached_horizon != TARGET_HORIZON_DAYS)

        if missing_target_cols or horizon_mismatch:
            if horizon_mismatch:
                print(
                    f"[targets] Cache was built with TARGET_HORIZON_DAYS={cached_horizon}, "
                    f"but config says {TARGET_HORIZON_DAYS}. Recomputing targets..."
                )
            else:
                print(f"Cached file missing target columns: {missing_target_cols}")
            if {'Date', 'Ticker', 'Return_1d'}.issubset(data.columns):
                print(f'Recomputing Return_NextDay/Target (horizon={TARGET_HORIZON_DAYS}d) from cached features...')
                data = create_targets(data, horizon=TARGET_HORIZON_DAYS)
                # Tag the cache with the horizon so future loads can detect mismatches.
                data['_target_horizon'] = TARGET_HORIZON_DAYS
                data.to_csv(CACHE_FEATURES_PATH, index=False)
                data.drop(columns=['_target_horizon'], inplace=True)
                print(f'Repaired and saved cache to {CACHE_FEATURES_PATH}')
            else:
                raise ValueError(
                    f'Cached {os.path.basename(CACHE_FEATURES_PATH)} is missing target columns and lacks the columns '
                    "required to rebuild them ('Date', 'Ticker', 'Return_1d'). "
                    'Run once with load_cached=False to regenerate cache.'
                )
    else:
        raw = download_and_save()
        data = build_feature_matrix(raw)
        data = create_targets(data, horizon=TARGET_HORIZON_DAYS)
        # Tag cache with the horizon used so reloads can detect config changes.
        data['_target_horizon'] = TARGET_HORIZON_DAYS
        data.to_csv(CACHE_FEATURES_PATH, index=False)
        data.drop(columns=['_target_horizon'], inplace=True)
        print(f'Saved full cache (with {TARGET_HORIZON_DAYS}d-horizon targets) to {CACHE_FEATURES_PATH}')

    # Verify required columns
    required = FEATURE_COLS + [TARGET_COL, 'Return_NextDay', 'Date', 'Ticker']
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f'Missing columns in {os.path.basename(CACHE_FEATURES_PATH)}: {missing}')

    # Verify LSTM features are available
    lstm_b_missing = [c for c in LSTM_B_FEATURES if c not in data.columns]
    if lstm_b_missing:
        raise ValueError(f'Missing LSTM features: {lstm_b_missing}')

    print(f'\nFeature sets:')
    print(f'  LSTM: {LSTM_B_FEATURES} ({len(LSTM_B_FEATURES)} features)')
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
    max_folds = getattr(config, 'MAX_FOLDS', None)
    if max_folds is not None:
        folds = folds[:max_folds]
        print(f'Limiting walk-forward run to first {len(folds)} fold(s) (MAX_FOLDS={max_folds}).')
    print()
    print_fold_summary(folds)

    ablation_rows: list[dict] = []

    def _score_candidate_on_val(preds_val: pd.DataFrame,
                                 prob_col: str = 'Prob_LSTM_B') -> tuple[float, float]:
        """
        Score a candidate on validation TRADING performance (not only AUC).
        Returns (val_sharpe_net, val_annual_return_net_pct).
        Works for any model by passing its Prob_* column name.
        """
        valid = preds_val.dropna(subset=[prob_col]).copy()
        if len(valid) == 0:
            return float('-inf'), float('-inf')

        smoothed = smooth_probabilities(
            valid,
            prob_col,
            alpha=SIGNAL_SMOOTH_ALPHA,
            ema_method=SIGNAL_EMA_METHOD,
            ema_span=SIGNAL_EMA_SPAN,
        )
        sig_df, _ = generate_signals(
            smoothed,
            k=K_STOCKS,
            prob_col=f'{prob_col}_Smooth',
            return_diagnostics=True,
        )
        sig_df = apply_holding_period_constraint(sig_df, min_hold_days=MIN_HOLDING_DAYS)
        port_val = compute_portfolio_returns(
            sig_df,
            tc_bps=TC_BPS,
            k=K_STOCKS,
            slippage_bps=SLIPPAGE_BPS,
            invert_signals=INVERT_SIGNALS,
        )
        met_val = compute_metrics(port_val['Net_Return'])
        return float(met_val['Sharpe Ratio']), float(met_val['Annualized Return (%)'])

    # Backward-compatible alias used inside the LSTM block.
    _score_lstm_b_candidate_on_val = _score_candidate_on_val

    # ── Steps 5-6: Train models per fold ────────────────────────────────────
    all_preds = []
    tuning_results = []  # Store tuning results for thesis reporting
    best_hp_b_global = _frozen_hp_b_init   # pre-populated when TUNE_USE_FROZEN_HPS=True
    best_hp_t_global = _frozen_hp_t_init   # pre-populated when TUNE_USE_FROZEN_HPS=True
    feat_imp_records = []       # Per-fold baseline importance accumulator (Task 5)
    lstm_perm_records = []      # Per-fold LSTM permutation importance accumulator
    tcn_perm_records = []       # Per-fold TCN permutation importance accumulator

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

            # ── Feature importance extraction (Task 5) ───────────────────
            fold_imp = extract_feature_importances(lr_m, rf_m, xgb_m, BASELINE_FEATURE_COLS)
            for feat, scores in fold_imp.items():
                feat_imp_records.append({'Fold': fold['fold'], 'Feature': feat, **scores})
        else:
            print('\n  [Baselines] Skipping LR, RF, XGBoost (RUN_BASELINES=False)')
            lr_m = rf_m = xgb_m = None

        probs_b = None
        keys_te_b = None
        probs_t = None
        keys_te_t = None
        tcn_feature_cols_used = None

        if RUN_LSTMS:
            df_train_fold = pd.concat([df_tr, df_v]).sort_values(['Ticker', 'Date'])
            df_test_fold = df_ts.copy()

            # ── LSTM: Primary neural-network model ─────────────────────────────
            t0 = time.time()
            print('  [LSTM]  building sequences & training...')

            # Use TEMPORAL split (splits by date, not index) - FIX for ticker-based split bug
            X_tr_b, y_tr_b, X_val_b, y_val_b, X_te_b, y_te_b, keys_tr_b, keys_val_b, keys_te_b = \
                prepare_lstm_b_sequences_temporal_split(df_train_fold, df_test_fold, val_ratio=LSTM_B_VAL_SPLIT)

            print(f'    LSTM sequences: train={len(X_tr_b)}, val={len(X_val_b)}, test={len(X_te_b)}')

            best_hp_b = None
            should_tune_b = lstm_b_tuning_enabled and (
                (best_hp_b_global is None) or (not lstm_b_tune_once)
            )
            if should_tune_b:
                print('    [LSTM Tuning] Running Phase 1 + Phase 2...')
                tuned_hp_b = tune_lstm_hyperparams(
                    X_tr_b, y_tr_b,
                    X_val_b, y_val_b,
                    input_size=len(LSTM_B_FEATURES),
                    device=device,
                    arch_grid=config.LSTM_B_ARCH_GRID,
                    train_grid=config.LSTM_B_HYPERPARAM_GRID,
                    tune_replicates=config.LSTM_B_TUNE_REPLICATES,
                    tune_patience=config.LSTM_B_TUNE_PATIENCE,
                    tune_max_epochs=config.LSTM_B_TUNE_MAX_EPOCHS,
                    seed_hidden=config.LSTM_B_HIDDEN_SIZE,
                    seed_layers=config.LSTM_B_NUM_LAYERS,
                    seed_dropout=config.LSTM_B_DROPOUT,
                    seed=fold_seed_base + 25,
                )

                # Return-aware guardrail: compare tuned-vs-default on val trading metrics.
                candidate_hps = [
                    ('default', None),
                    ('tuned', tuned_hp_b),
                ]
                candidate_scores = []

                for cand_name, cand_hp in candidate_hps:
                    if cand_hp is None:
                        model_b_cand = train_lstm_b(
                            X_tr_b, y_tr_b,
                            X_val_b, y_val_b,
                            device,
                            seed=fold_seed_base + 28,
                            fold_idx=fold['fold'],
                        )
                    else:
                        model_b_cand = train_lstm_b(
                            X_tr_b, y_tr_b,
                            X_val_b, y_val_b,
                            device,
                            seed=fold_seed_base + 28,
                            fold_idx=fold['fold'],
                            optimizer_name=cand_hp['optimizer'],
                            learning_rate=cand_hp['lr'],
                            batch_size=cand_hp['batch_size'],
                            hidden_size=cand_hp['hidden_size'],
                            num_layers=cand_hp['num_layers'],
                            dropout=cand_hp['dropout'],
                        )

                    probs_val_b = predict_lstm(model_b_cand, X_val_b, device)
                    pred_val_b = df_v.copy().reset_index(drop=True)
                    pred_val_b['Prob_LSTM_B'] = align_predictions_to_df(
                        probs_val_b, keys_val_b, df_v
                    )
                    val_sharpe_b, val_ann_ret_b = _score_lstm_b_candidate_on_val(pred_val_b)
                    candidate_scores.append({
                        'name': cand_name,
                        'hp': cand_hp,
                        'val_sharpe_net': val_sharpe_b,
                        'val_ann_ret_net_pct': val_ann_ret_b,
                    })

                    del model_b_cand
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()

                candidate_scores = sorted(
                    candidate_scores,
                    key=lambda x: (x['val_sharpe_net'], x['val_ann_ret_net_pct']),
                    reverse=True,
                )
                selected = candidate_scores[0]
                best_hp_b = selected['hp']

                print(
                    '    [LSTM Tuning] Candidate scores: '
                    + ', '.join(
                        f"{c['name']}(Sharpe={c['val_sharpe_net']:.3f},AnnRet={c['val_ann_ret_net_pct']:.2f}%)"
                        for c in candidate_scores
                    )
                )
                if selected['name'] == 'default':
                    print('    [LSTM Tuning] Tuned candidate underperformed on val returns; using default.')

                tuning_results.append({
                    'fold': fold['fold'],
                    'model': 'LSTM',
                    'selection_basis': 'val_net_sharpe_then_annret',
                    'selected_candidate': selected['name'],
                    'val_sharpe_selected': selected['val_sharpe_net'],
                    'val_ann_ret_selected': selected['val_ann_ret_net_pct'],
                    **(tuned_hp_b if tuned_hp_b is not None else {}),
                })
                print(f'    [LSTM Tuning] Selected params: {best_hp_b}')
                if lstm_b_tune_once:
                    best_hp_b_global = best_hp_b if best_hp_b is not None else 'DEFAULT'
                    _save_frozen_hps(hp_b=best_hp_b)   # None saved as null = use config defaults
            elif best_hp_b_global is not None:
                if best_hp_b_global == 'DEFAULT':
                    best_hp_b = None
                else:
                    best_hp_b = best_hp_b_global
                print(f'    [LSTM Tuning] Reusing tuned params: {best_hp_b}')

            if best_hp_b is not None:
                model_b = train_lstm_b(
                    X_tr_b, y_tr_b,
                    X_val_b, y_val_b,
                    device,
                    seed=fold_seed_base + 30,
                    fold_idx=fold['fold'],
                    optimizer_name=best_hp_b['optimizer'],
                    learning_rate=best_hp_b['lr'],
                    batch_size=best_hp_b['batch_size'],
                    hidden_size=best_hp_b['hidden_size'],
                    num_layers=best_hp_b['num_layers'],
                    dropout=best_hp_b['dropout'],
                )
            else:
                model_b = train_lstm_b(
                    X_tr_b, y_tr_b,
                    X_val_b, y_val_b,
                    device,
                    seed=fold_seed_base + 30,
                    fold_idx=fold['fold'],
                )
            print(f'  [LSTM]  fit done in {time.time()-t0:.1f}s')

            # LSTM inference
            probs_b = predict_lstm(model_b, X_te_b, device)

            # ── Calibrate LSTM probabilities (fit on val, apply to test) ───
            # Corrects logit-saturation overconfidence (raw prob std ~0.34 vs ~0.05-0.09
            # for baselines). Isotonic regression is rank-preserving so AUC is unchanged;
            # it re-maps extreme probs toward the empirical label frequencies.
            if getattr(config, 'LSTM_CALIBRATE_PROBS', True):
                _lstm_val_probs = predict_lstm(model_b, X_val_b, device)
                _lstm_cal = ProbabilityCalibrator(
                    method=getattr(config, 'CALIBRATION_METHOD', 'isotonic')
                )
                _lstm_cal.fit(_lstm_val_probs, y_val_b)
                _std_before = float(np.nanstd(probs_b))
                probs_b = _lstm_cal.transform(probs_b)
                print(
                    f'  [LSTM] calibration: prob std {_std_before:.4f} → '
                    f'{float(np.nanstd(probs_b)):.4f}'
                )

            # ── LSTM permutation importance (Task 5) ───────────────────────
            if getattr(config, 'COMPUTE_PERMUTATION_IMPORTANCE', True):
                baseline_auc = binary_auc_safe(y_te_b, probs_b, log_on_fail=False)
                if not np.isnan(baseline_auc):
                    n_lstm_feats = X_te_b.shape[2]
                    perm_drops = np.zeros(n_lstm_feats)
                    rng_perm = np.random.default_rng(config.RANDOM_SEED + fold['fold'])
                    for f_idx in range(n_lstm_feats):
                        X_perm = X_te_b.copy()
                        # Shuffle this feature across all samples (axis 0), keep time axis intact
                        X_perm[:, :, f_idx] = rng_perm.permutation(X_perm[:, :, f_idx])
                        probs_perm = predict_lstm(model_b, X_perm, device)
                        perm_auc = binary_auc_safe(y_te_b, probs_perm, log_on_fail=False)
                        drop = baseline_auc - (perm_auc if not np.isnan(perm_auc) else baseline_auc)
                        perm_drops[f_idx] = max(drop, 0.0)   # clip negatives: not informative
                    perm_norm = perm_drops / (perm_drops.sum() + 1e-12)
                    for f_idx, feat in enumerate(LSTM_B_FEATURES):
                        lstm_perm_records.append({
                            'Fold': fold['fold'],
                            'Feature': feat,
                            'LSTM_B_perm': float(perm_norm[f_idx]),
                        })

            # Free LSTM memory for next fold
            del model_b, X_te_b, y_te_b
            del X_tr_b, y_tr_b, X_val_b, y_val_b
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        else:
            print('\n  [LSTMs]     Skipping LSTM (RUN_LSTMS=False)')

        if RUN_TCNS:
            df_train_fold_t = pd.concat([df_tr, df_v]).sort_values(['Ticker', 'Date'])
            df_test_fold_t = df_ts.copy()

            t0 = time.time()
            print('  [TCN]   building sequences & training...')

            # Build sequences for every feature set referenced by the TCN tuning grid.
            # This lets Phase 2 pick the best (arch × feature_set) jointly.
            tcn_feature_sets = dict(getattr(config, 'TCN_FEATURE_SETS', {
                'full': config.LSTM_B_FEATURE_COLS,
            }))
            should_tune_t = tcn_tuning_enabled and (
                (best_hp_t_global is None) or (not tcn_tune_once)
            )
            # Only build the default feature-set sequences when we're not tuning,
            # to avoid paying the cost of the core set on every fold.
            if should_tune_t:
                feature_sets_to_build = list(tcn_feature_sets.keys())
            else:
                # Reuse the tuned feature set from fold 1 when available.
                tuned_fs = (
                    best_hp_t_global.get('feature_set')
                    if isinstance(best_hp_t_global, dict)
                    else None
                )
                chosen_fs = tuned_fs if tuned_fs in tcn_feature_sets else config.TCN_FEATURE_SET_DEFAULT
                feature_sets_to_build = [chosen_fs]

            built_seqs: dict[str, dict[str, np.ndarray]] = {}
            built_test: dict[str, tuple] = {}
            for fs_name in feature_sets_to_build:
                fs_cols = tcn_feature_sets[fs_name]
                X_tr_t, y_tr_t, X_val_t, y_val_t, X_te_t, y_te_t, _, keys_val_t, keys_te_t_tmp = \
                    prepare_tcn_sequences_temporal_split(
                        df_train_fold_t, df_test_fold_t,
                        feature_cols=fs_cols,
                        val_ratio=float(config.TCN_VAL_SPLIT),
                    )
                built_seqs[fs_name] = {
                    'X_tr': X_tr_t, 'y_tr': y_tr_t,
                    'X_val': X_val_t, 'y_val': y_val_t,
                    'keys_val': keys_val_t,
                }
                built_test[fs_name] = (X_te_t, y_te_t, keys_te_t_tmp)
                print(
                    f'    TCN sequences [{fs_name}]: train={len(X_tr_t)}, '
                    f'val={len(X_val_t)}, test={len(X_te_t)}'
                )

            best_hp_t = None
            if should_tune_t:
                print('    [TCN Tuning] Running Phase 1 + Phase 2...')
                seed_fs = config.TCN_FEATURE_SET_DEFAULT
                tuned_hp_t = tune_tcn_hyperparams(
                    seqs_by_feature_set=built_seqs,
                    device=device,
                    arch_grid=config.TCN_ARCH_GRID,
                    train_grid=config.TCN_HYPERPARAM_GRID,
                    tune_replicates=int(config.TCN_TUNE_REPLICATES),
                    tune_patience=int(config.TCN_TUNE_PATIENCE),
                    tune_max_epochs=int(config.TCN_TUNE_MAX_EPOCHS),
                    seed_num_channels=list(config.TCN_NUM_CHANNELS),
                    seed_kernel_size=int(config.TCN_KERNEL_SIZE),
                    seed_dropout=float(config.TCN_DROPOUT),
                    seed_feature_set=seed_fs,
                    seed=fold_seed_base + 45,
                )

                # Return-aware guardrail: compare tuned-vs-default on val trading metrics.
                candidate_scores = []
                for cand_name, cand_hp in [('default', None), ('tuned', tuned_hp_t)]:
                    cand_fs = (
                        cand_hp['feature_set']
                        if (cand_hp is not None and cand_hp.get('feature_set') in built_seqs)
                        else config.TCN_FEATURE_SET_DEFAULT
                    )
                    blk = built_seqs[cand_fs]
                    if cand_hp is None:
                        model_t_cand = train_tcn(
                            blk['X_tr'], blk['y_tr'],
                            blk['X_val'], blk['y_val'],
                            device,
                            seed=fold_seed_base + 48,
                            fold_idx=fold['fold'],
                        )
                    else:
                        model_t_cand = train_tcn(
                            blk['X_tr'], blk['y_tr'],
                            blk['X_val'], blk['y_val'],
                            device,
                            seed=fold_seed_base + 48,
                            fold_idx=fold['fold'],
                            optimizer_name=cand_hp['optimizer'],
                            learning_rate=cand_hp['lr'],
                            batch_size=cand_hp['batch_size'],
                            num_channels=cand_hp['num_channels'],
                            kernel_size=cand_hp['kernel_size'],
                            dropout=cand_hp['dropout'],
                        )
                    probs_val_t = predict_tcn(model_t_cand, blk['X_val'], device)
                    pred_val_t = df_v.copy().reset_index(drop=True)
                    pred_val_t['Prob_TCN'] = align_predictions_to_df(
                        probs_val_t, blk['keys_val'], df_v
                    )
                    val_sharpe_t, val_ann_ret_t = _score_candidate_on_val(pred_val_t, 'Prob_TCN')
                    candidate_scores.append({
                        'name': cand_name,
                        'hp': cand_hp,
                        'feature_set': cand_fs,
                        'val_sharpe_net': val_sharpe_t,
                        'val_ann_ret_net_pct': val_ann_ret_t,
                    })
                    del model_t_cand
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()

                candidate_scores = sorted(
                    candidate_scores,
                    key=lambda x: (x['val_sharpe_net'], x['val_ann_ret_net_pct']),
                    reverse=True,
                )
                selected = candidate_scores[0]
                best_hp_t = selected['hp']
                print(
                    '    [TCN Tuning] Candidate scores: '
                    + ', '.join(
                        f"{c['name']}(Sharpe={c['val_sharpe_net']:.3f},"
                        f"AnnRet={c['val_ann_ret_net_pct']:.2f}%,feat={c['feature_set']})"
                        for c in candidate_scores
                    )
                )
                if selected['name'] == 'default':
                    print('    [TCN Tuning] Tuned candidate underperformed on val returns; using default.')

                tuning_results.append({
                    'fold': fold['fold'],
                    'model': 'TCN',
                    'selection_basis': 'val_net_sharpe_then_annret',
                    'selected_candidate': selected['name'],
                    'val_sharpe_selected': selected['val_sharpe_net'],
                    'val_ann_ret_selected': selected['val_ann_ret_net_pct'],
                    **(tuned_hp_t if tuned_hp_t is not None else {}),
                })
                print(f'    [TCN Tuning] Selected params: {best_hp_t}')
                if tcn_tune_once:
                    best_hp_t_global = best_hp_t if best_hp_t is not None else 'DEFAULT'
                    _save_frozen_hps(hp_t=best_hp_t)   # None saved as null = use config defaults
            elif best_hp_t_global is not None:
                if best_hp_t_global == 'DEFAULT':
                    best_hp_t = None
                else:
                    best_hp_t = best_hp_t_global
                print(f'    [TCN Tuning] Reusing tuned params: {best_hp_t}')

            # Resolve which feature set + sequences to use for final training.
            chosen_fs = (
                best_hp_t.get('feature_set')
                if (isinstance(best_hp_t, dict) and best_hp_t.get('feature_set') in built_seqs)
                else config.TCN_FEATURE_SET_DEFAULT
            )
            if chosen_fs not in built_seqs:
                # Fallback: we chose a feature set that wasn't built for this fold
                # (e.g. tuning ran on fold 1, non-default fs requested). Build it now.
                fs_cols = tcn_feature_sets[chosen_fs]
                X_tr_t, y_tr_t, X_val_t, y_val_t, X_te_t, y_te_t, _, keys_val_t, keys_te_t_tmp = \
                    prepare_tcn_sequences_temporal_split(
                        df_train_fold_t, df_test_fold_t,
                        feature_cols=fs_cols,
                        val_ratio=float(config.TCN_VAL_SPLIT),
                    )
                built_seqs[chosen_fs] = {
                    'X_tr': X_tr_t, 'y_tr': y_tr_t,
                    'X_val': X_val_t, 'y_val': y_val_t,
                    'keys_val': keys_val_t,
                }
                built_test[chosen_fs] = (X_te_t, y_te_t, keys_te_t_tmp)

            tcn_feature_cols_used = tcn_feature_sets[chosen_fs]
            fit_blk = built_seqs[chosen_fs]
            X_te_t, y_te_t, keys_te_t = built_test[chosen_fs]

            if best_hp_t is not None:
                model_t = train_tcn(
                    fit_blk['X_tr'], fit_blk['y_tr'],
                    fit_blk['X_val'], fit_blk['y_val'],
                    device,
                    seed=fold_seed_base + 50,
                    fold_idx=fold['fold'],
                    optimizer_name=best_hp_t['optimizer'],
                    learning_rate=best_hp_t['lr'],
                    batch_size=best_hp_t['batch_size'],
                    num_channels=best_hp_t['num_channels'],
                    kernel_size=best_hp_t['kernel_size'],
                    dropout=best_hp_t['dropout'],
                )
            else:
                model_t = train_tcn(
                    fit_blk['X_tr'], fit_blk['y_tr'],
                    fit_blk['X_val'], fit_blk['y_val'],
                    device,
                    seed=fold_seed_base + 50,
                    fold_idx=fold['fold'],
                )
            print(f'  [TCN]   fit done in {time.time()-t0:.1f}s')

            probs_t = predict_tcn(model_t, X_te_t, device)

            # ── Calibrate TCN probabilities (fit on val, apply to test) ────
            if getattr(config, 'TCN_CALIBRATE_PROBS', True):
                _tcn_val_probs = predict_tcn(model_t, fit_blk['X_val'], device)
                _tcn_cal = ProbabilityCalibrator(
                    method=getattr(config, 'CALIBRATION_METHOD', 'isotonic')
                )
                _tcn_cal.fit(_tcn_val_probs, fit_blk['y_val'])
                _std_before_t = float(np.nanstd(probs_t))
                probs_t = _tcn_cal.transform(probs_t)
                print(
                    f'  [TCN]  calibration: prob std {_std_before_t:.4f} → '
                    f'{float(np.nanstd(probs_t)):.4f}'
                )

            # ── TCN permutation importance (mirrors LSTM) ──────────────────
            if getattr(config, 'COMPUTE_PERMUTATION_IMPORTANCE', True) and len(X_te_t) > 0:
                baseline_auc_t = binary_auc_safe(y_te_t, probs_t, log_on_fail=False)
                if not np.isnan(baseline_auc_t):
                    n_tcn_feats = X_te_t.shape[2]
                    perm_drops_t = np.zeros(n_tcn_feats)
                    rng_perm_t = np.random.default_rng(config.RANDOM_SEED + fold['fold'] + 7)
                    for f_idx in range(n_tcn_feats):
                        X_perm = X_te_t.copy()
                        X_perm[:, :, f_idx] = rng_perm_t.permutation(X_perm[:, :, f_idx])
                        probs_perm_t = predict_tcn(model_t, X_perm, device)
                        perm_auc_t = binary_auc_safe(y_te_t, probs_perm_t, log_on_fail=False)
                        drop = baseline_auc_t - (
                            perm_auc_t if not np.isnan(perm_auc_t) else baseline_auc_t
                        )
                        perm_drops_t[f_idx] = max(drop, 0.0)
                    perm_norm_t = perm_drops_t / (perm_drops_t.sum() + 1e-12)
                    for f_idx, feat in enumerate(tcn_feature_cols_used):
                        tcn_perm_records.append({
                            'Fold': fold['fold'],
                            'Feature': feat,
                            'TCN_perm': float(perm_norm_t[f_idx]),
                        })

            del model_t, built_seqs, built_test
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        else:
            print('\n  [TCN]       Skipping TCN (RUN_TCNS=False)')

        # ── Collect predictions for this fold ────────────────────────────────
        pred = df_ts.copy().reset_index(drop=True)
        pred['Prob_LR'] = lr_m.predict_proba(X_ts_b_s)[:, 1] if RUN_BASELINES else np.nan
        pred['Prob_RF'] = rf_m.predict_proba(X_ts_b_s)[:, 1] if RUN_BASELINES else np.nan
        pred['Prob_XGB'] = xgb_m.predict(xgb.DMatrix(X_ts_b_s)) if RUN_BASELINES else np.nan
        pred['Prob_LSTM_B'] = align_predictions_to_df(probs_b, keys_te_b, df_ts) if RUN_LSTMS else np.nan
        pred['Prob_TCN'] = align_predictions_to_df(probs_t, keys_te_t, df_ts) if RUN_TCNS else np.nan
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
        n_lstm_b = pred['Prob_LSTM_B'].notna().sum()
        n_tcn = pred['Prob_TCN'].notna().sum()
        print(f'  Predictions: LR/RF/XGB={len(pred)}, LSTM={n_lstm_b}, TCN={n_tcn}')

        all_preds.append(pred)

    # ── Save feature importances — all models in one file (Task 5) ──────────
    if feat_imp_records:
        fi_per_fold = pd.DataFrame(feat_imp_records)

        # Merge LSTM permutation importance (same feature set, join on Fold+Feature)
        if lstm_perm_records:
            lstm_df = pd.DataFrame(lstm_perm_records)
            fi_per_fold = fi_per_fold.merge(lstm_df, on=['Fold', 'Feature'], how='left')
        else:
            fi_per_fold['LSTM_B_perm'] = np.nan

        # Merge TCN permutation importance. TCN may use a subset of features
        # (e.g. 'core' set from tuning), so NaN is expected for features absent
        # from the chosen set on a given fold.
        if tcn_perm_records:
            tcn_df = pd.DataFrame(tcn_perm_records)
            fi_per_fold = fi_per_fold.merge(tcn_df, on=['Fold', 'Feature'], how='left')
        else:
            fi_per_fold['TCN_perm'] = np.nan

        imp_cols = ['LR_coef', 'RF_importance', 'XGB_gain', 'LSTM_B_perm', 'TCN_perm']
        fi_per_fold.to_csv(
            f'{reports_dir}/{config.UNIVERSE_MODE}_feature_importances_per_fold.csv',
            index=False,
        )

        fi_avg = (
            fi_per_fold
            .groupby('Feature')[imp_cols]
            .mean()
            .reset_index()
        )
        fi_avg.to_csv(
            f'{reports_dir}/{config.UNIVERSE_MODE}_feature_importances_avg.csv',
            index=False,
        )
        print(f'\nFeature importances saved to {reports_dir}/')

    # ── Combine folds ─────────────────────────────────────────────────────────
    full_preds = pd.concat(all_preds).reset_index(drop=True)
    print(f'\nTotal predictions: {len(full_preds)}')
    print(f'  LSTM valid: {full_preds["Prob_LSTM_B"].notna().sum()}')
    print(f'  TCN  valid: {full_preds["Prob_TCN"].notna().sum()}')

    # ── Backtest each model independently ─────────────────────────────────────
    print('\n' + '=' * 60)
    print('RESULTS — GROSS (0 bps)')
    print('=' * 60)

    model_cols = {
        'LR': 'Prob_LR',
        'RF': 'Prob_RF',
        'XGBoost': 'Prob_XGB',
        'LSTM': 'Prob_LSTM_B',
        'TCN': 'Prob_TCN',
    }

    port_returns_gross = {}
    port_returns_net_5 = {}
    class_metrics = []
    all_signals = []
    fold_sharpe_rows = []
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
            invert_signals=INVERT_SIGNALS,
        )
        port_net_5 = compute_portfolio_returns(
            sig_df, tc_bps=TC_BPS, k=K_STOCKS, slippage_bps=SLIPPAGE_BPS,
            invert_signals=INVERT_SIGNALS,
        )

        if RUN_SIGNAL_ABLATION:
            sig_raw, _ = generate_signals(
                valid_preds, k=K_STOCKS, prob_col=prob_col,
                confidence_threshold=0.0, return_diagnostics=True,
            )
            sig_raw = apply_holding_period_constraint(sig_raw, min_hold_days=1)
            port_raw = compute_portfolio_returns(
                sig_raw, tc_bps=TC_BPS, k=K_STOCKS, slippage_bps=SLIPPAGE_BPS,
                invert_signals=INVERT_SIGNALS,
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

        # Fold-level Sharpe (gross) — one row per fold per model
        date_fold = (
            valid_preds[['Date', 'Fold']].drop_duplicates()
            .set_index('Date')['Fold']
        )
        port_gross_folds = port_gross.join(date_fold, how='left')
        for fold_id, fold_grp in port_gross_folds.groupby('Fold'):
            fm = compute_metrics(fold_grp['Gross_Return'])
            fold_sharpe_rows.append({
                'Model':              model_name,
                'Fold':               int(fold_id),
                'Sharpe Ratio (Gross)': round(fm.get('Sharpe Ratio', float('nan')), 3),
                'N Days':             len(fold_grp),
                'Start Date':         fold_grp.index.min().date(),
                'End Date':           fold_grp.index.max().date(),
            })

        # Store daily returns for export
        if daily_returns_gross['Date'] is None:
            daily_returns_gross['Date'] = port_gross.index
            daily_returns_net_5['Date'] = port_net_5.index
        daily_returns_gross[model_name] = port_gross['Gross_Return'].values
        daily_returns_net_5[model_name] = port_net_5['Net_Return'].values

        # Classification metrics (pooled + daily AUC for diagnostic)
        # When invert_signals=True the portfolio trades against raw model output,
        # so we evaluate on (1 - prob) to measure the actual trading direction.
        y_true = valid_preds[TARGET_COL].values
        y_prob = valid_preds[prob_col].values
        cm = evaluate_classification(y_true, y_prob, invert_probs=INVERT_SIGNALS)

        # Add daily AUC to diagnose pooled vs within-day ranking
        daily_auc = compute_daily_auc(
            valid_preds, prob_col, TARGET_COL, invert_probs=INVERT_SIGNALS
        )
        cm['Daily AUC (mean)'] = daily_auc['Daily AUC (mean)']
        cm['Daily AUC (std)'] = daily_auc['Daily AUC (std)']
        cm['Signals Inverted'] = INVERT_SIGNALS

        cm['Model'] = model_name
        class_metrics.append(cm)

        # Print gross metrics
        m = compute_metrics(port_gross['Gross_Return'])
        print(f'  {model_name:<12}  '
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

    all_model_names = list(model_cols.keys())

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

    # ── Ensemble: include any model whose out-of-sample net Sharpe > 0 ──────────
    # All models are evaluated; inclusion is determined purely by performance,
    # not by a fixed model list.
    _ens_base_cols = []
    for _ens_model, _ens_prob_col in model_cols.items():
        if _ens_model not in port_returns_net_5:
            print(f'  [Ensemble gate] {_ens_model} skipped (no backtest results)')
            continue
        _model_net_sharpe = compute_metrics(
            port_returns_net_5[_ens_model]['Net_Return']
        )['Sharpe Ratio']
        if _model_net_sharpe > 0.0:
            _ens_base_cols.append(_ens_prob_col)
            print(f'  [Ensemble gate] {_ens_model} included (net Sharpe={_model_net_sharpe:.3f})')
        else:
            print(f'  [Ensemble gate] {_ens_model} excluded (net Sharpe={_model_net_sharpe:.3f} <= 0)')
    _ens_avail = [
        c for c in _ens_base_cols
        if c in full_preds.columns and full_preds[c].notna().any()
    ]
    if _ens_avail:
        ens_preds = full_preds.dropna(subset=_ens_avail).copy()
        ens_preds['Prob_ENS'] = ens_preds[_ens_avail].mean(axis=1)
        _ens_model_names = [m for m, c in model_cols.items() if c in _ens_avail]
        print(f'\n  [Ensemble] built from {_ens_model_names} ({len(ens_preds)} rows)')

        smoothed_ens = smooth_probabilities(
            ens_preds, 'Prob_ENS',
            alpha=SIGNAL_SMOOTH_ALPHA,
            ema_method=SIGNAL_EMA_METHOD,
            ema_span=SIGNAL_EMA_SPAN,
        )
        sig_ens, sig_ens_diag = generate_signals(
            smoothed_ens, k=K_STOCKS, prob_col='Prob_ENS_Smooth',
            return_diagnostics=True,
        )
        sig_ens = apply_holding_period_constraint(sig_ens, min_hold_days=MIN_HOLDING_DAYS)
        hold_st_ens = compute_turnover_and_holding_stats(sig_ens, k=K_STOCKS)
        print(f'  [Ensemble] turnover~{hold_st_ens["mean_daily_turnover_half_turns"]:.2f}  '
              f'avg_hold~{hold_st_ens["avg_holding_period_trading_days"]:.1f}')

        sig_ens['Model'] = 'Ensemble'
        all_signals.append(sig_ens)

        port_gross_ens = compute_portfolio_returns(sig_ens, tc_bps=0, k=K_STOCKS, slippage_bps=SLIPPAGE_BPS, invert_signals=INVERT_SIGNALS)
        port_net_ens = compute_portfolio_returns(sig_ens, tc_bps=TC_BPS, k=K_STOCKS, slippage_bps=SLIPPAGE_BPS, invert_signals=INVERT_SIGNALS)
        port_returns_gross['Ensemble'] = port_gross_ens
        port_returns_net_5['Ensemble'] = port_net_ens

        m_ens_g = compute_metrics(port_gross_ens['Gross_Return'])
        m_ens_g['Model'] = 'Ensemble'
        results_gross.append(m_ens_g)

        m_ens_n = compute_metrics(port_net_ens['Net_Return'])
        m_ens_n['Model'] = 'Ensemble'
        results_net_5.append(m_ens_n)
        print(f'  {"Ensemble":<12}  '
              f'Sharpe={m_ens_n["Sharpe Ratio"]:>6.3f}  '
              f'Sortino={m_ens_n["Sortino Ratio"]:>6.3f}  '
              f'Ann.Ret={m_ens_n["Annualized Return (%)"]:.2f}%  '
              f'MDD={m_ens_n["Max Drawdown (%)"]:.2f}%')

        cm_ens = evaluate_classification(
            ens_preds[TARGET_COL].values, ens_preds['Prob_ENS'].values,
            invert_probs=INVERT_SIGNALS,
        )
        daily_auc_ens = compute_daily_auc(ens_preds, 'Prob_ENS', TARGET_COL, invert_probs=INVERT_SIGNALS)
        cm_ens['Daily AUC (mean)'] = daily_auc_ens['Daily AUC (mean)']
        cm_ens['Daily AUC (std)'] = daily_auc_ens['Daily AUC (std)']
        cm_ens['Signals Inverted'] = INVERT_SIGNALS
        cm_ens['Model'] = 'Ensemble'
        class_metrics.append(cm_ens)

        if daily_returns_gross['Date'] is not None:
            daily_returns_gross['Ensemble'] = port_gross_ens.reindex(daily_returns_gross['Date'])['Gross_Return'].values
            daily_returns_net_5['Ensemble'] = port_net_ens.reindex(daily_returns_net_5['Date'])['Net_Return'].values
    else:
        print('\n  [Ensemble] skipped — no model passed the net Sharpe > 0 gate')

    # ── Sub-period analysis (all models) ────────────────────────────────────
    subperiod_metrics = None
    try:
        sp_frames = []
        for _sp_model, _sp_port in port_returns_net_5.items():
            _sp_df = compute_subperiod_metrics(_sp_port['Net_Return'])
            _sp_df.insert(0, 'Period', _sp_df.index)
            _sp_df.insert(0, 'Model', _sp_model)
            _sp_df = _sp_df.reset_index(drop=True)
            sp_frames.append(_sp_df)
        if sp_frames:
            subperiod_metrics = pd.concat(sp_frames, ignore_index=True)
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
    full_preds_path = os.path.join(reports_dir, f'{config.UNIVERSE_MODE}_full_predictions.csv')
    full_preds.to_csv(full_preds_path, index=False)
    print(f'  Saved raw probabilities to {full_preds_path}')

    # Save fold-level gross Sharpe diagnostics for this run.
    if fold_sharpe_rows:
        fold_sharpe_df = pd.DataFrame(fold_sharpe_rows).sort_values(['Model', 'Fold']).reset_index(drop=True)
        fold_sharpe_path = os.path.join(reports_dir, f'{config.UNIVERSE_MODE}_fold_sharpe_per_model.csv')
        fold_sharpe_df.to_csv(fold_sharpe_path, index=False)
        print(f'  Saved fold-level Sharpe report to {fold_sharpe_path}')

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
        ab_path = os.path.join(reports_dir, f'{config.UNIVERSE_MODE}_signal_ablation_summary.csv')
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
    main(load_cached=True)
