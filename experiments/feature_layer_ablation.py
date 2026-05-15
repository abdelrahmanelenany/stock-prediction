# ==============================================================
# EDIT THESE TWO LINES TO SELECT WHAT TO RUN
# ==============================================================
ABLATION_CONDITION = "L1_L2"   # "L1_only" | "L1_L2" | "L1_L2_L3"
UNIVERSE_MODE      = "small_cap" # "large_cap" | "small_cap"
# ==============================================================

import os
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import random
import time
import logging
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import xgboost as xgb
import torch

import config
from pipeline.walk_forward import generate_walk_forward_folds
from pipeline.standardizer import standardize_fold
from models.baselines import train_logistic, train_random_forest, train_xgboost
from models.lstm_model import (
    prepare_lstm_b_sequences_temporal_split,
    train_lstm_b,
    predict_lstm,
    align_predictions_to_df,
)
from models.tcn_model import (
    prepare_tcn_sequences_temporal_split,
    train_tcn,
    predict_tcn,
)
from backtest.signals import (
    generate_signals,
    smooth_probabilities,
    apply_holding_period_constraint,
)
from backtest.portfolio import compute_portfolio_returns
from backtest.metrics import compute_metrics
from evaluation.metrics_utils import binary_auc_safe

logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')


# ── Helpers ──────────────────────────────────────────────────────────────────

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _classify_features(feature_cols):
    """Split a feature column list into (L1, L2, L3) by naming convention."""
    L2_PREFIXES = ("Market_", "RelToMarket_", "Beta_")
    L3_PREFIXES = ("Sector_", "SectorRelZ_")
    L3_EXCLUDE  = {"SectorRelReturn"}
    l1, l2, l3 = [], [], []
    for col in feature_cols:
        if any(col.startswith(p) for p in L2_PREFIXES):
            l2.append(col)
        elif any(col.startswith(p) for p in L3_PREFIXES) and col not in L3_EXCLUDE:
            l3.append(col)
        else:
            l1.append(col)
    return l1, l2, l3


def _sharpe_or_nan(returns_series):
    if len(returns_series) <= 1 or returns_series.std() == 0:
        return float('nan')
    return compute_metrics(returns_series)['Sharpe Ratio']


def _print_aligned_table(rows, headers):
    col_widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
                  for i, h in enumerate(headers)]
    sep = "  ".join("-" * w for w in col_widths)
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print(sep)
    for r in rows:
        print(fmt.format(*[str(v) for v in r]))


# ── Validate flags ────────────────────────────────────────────────────────────

VALID_CONDITIONS = ("L1_only", "L1_L2", "L1_L2_L3")
if ABLATION_CONDITION not in VALID_CONDITIONS:
    raise ValueError(f"ABLATION_CONDITION must be one of {VALID_CONDITIONS}, got {ABLATION_CONDITION!r}")
if UNIVERSE_MODE not in ("large_cap", "small_cap"):
    raise ValueError(f"UNIVERSE_MODE must be 'large_cap' or 'small_cap', got {UNIVERSE_MODE!r}")

BASELINE_MODELS = ["LR", "RF", "XGBoost"]
SEQ_MODELS      = ["LSTM", "TCN"]

if ABLATION_CONDITION == "L1_L2_L3":
    print(
        "\n[WARNING] ABLATION_CONDITION='L1_L2_L3': LR, RF, XGBoost are NOT run — "
        "baselines already reach their maximum feature set at L1_L2 (no sector features). "
        "Only LSTM and TCN will run.\n"
    )
    MODELS_TO_RUN = list(SEQ_MODELS)
else:
    MODELS_TO_RUN = BASELINE_MODELS + SEQ_MODELS

RUN_BASELINES = any(m in MODELS_TO_RUN for m in BASELINE_MODELS)
RUN_SEQ       = any(m in MODELS_TO_RUN for m in SEQ_MODELS)

# ── Universe config ───────────────────────────────────────────────────────────

universe_cfg   = config.LARGE_CAP_CONFIG if UNIVERSE_MODE == "large_cap" else config.SMALL_CAP_CONFIG
INVERT_SIGNALS = universe_cfg.invert_signals
K_STOCKS       = universe_cfg.k_stocks

# ── Feature layer derivation (programmatic from config, no hardcoding) ───────
# Use the full L1+L2+L3 pool for both model types so that every condition
# (L1_only / L1_L2 / L1_L2_L3) resolves to a distinct feature set regardless
# of what the universe config restricts baselines to by default.
# This does NOT affect the main pipeline — only this ablation script reads these lists.

baseline_all_cols = (
    list(config._CORE_FEATURE_COLS)
    + list(config._MARKET_FEATURE_COLS)
    + list(config._SECTOR_FEATURE_COLS)
)
lstm_all_cols = (
    list(config._CORE_FEATURE_COLS)
    + list(config._MARKET_FEATURE_COLS)
    + list(config._SECTOR_FEATURE_COLS)
)

baseline_l1, baseline_l2, baseline_l3 = _classify_features(baseline_all_cols)
lstm_l1,     lstm_l2,     lstm_l3     = _classify_features(lstm_all_cols)

def _resolve_cols(condition, l1, l2, l3):
    if condition == "L1_only":
        return l1
    if condition == "L1_L2":
        return l1 + l2
    return l1 + l2 + l3  # L1_L2_L3

baseline_cols = _resolve_cols(ABLATION_CONDITION, baseline_l1, baseline_l2, baseline_l3)
seq_cols      = _resolve_cols(ABLATION_CONDITION, lstm_l1,     lstm_l2,     lstm_l3)

# ── Device ────────────────────────────────────────────────────────────────────

device = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('mps') if torch.backends.mps.is_available()
    else torch.device('cpu')
)

# ── Startup summary ───────────────────────────────────────────────────────────

print("=" * 70)
print("FEATURE-LAYER ABLATION EXPERIMENT")
print("=" * 70)
print(f"  Condition   : {ABLATION_CONDITION}")
print(f"  Universe    : {UNIVERSE_MODE}  (invert_signals={INVERT_SIGNALS})")
print(f"  Models      : {', '.join(MODELS_TO_RUN)}")
print(f"  Device      : {device}")
print()
print(f"  Baseline feature cols ({len(baseline_cols)}):")
print(f"    L1 ({len(baseline_l1)}): {baseline_l1}")
print(f"    L2 ({len(baseline_l2)}): {baseline_l2}")
if baseline_l3:
    print(f"    L3 ({len(baseline_l3)}): {baseline_l3}")
print(f"  => Active for baselines: {baseline_cols}")
print()
print(f"  LSTM/TCN feature cols ({len(seq_cols)}):")
print(f"    L1 ({len(lstm_l1)}): {lstm_l1}")
print(f"    L2 ({len(lstm_l2)}): {lstm_l2}")
print(f"    L3 ({len(lstm_l3)}): {lstm_l3}")
print(f"  => Active for LSTM/TCN: {seq_cols}")
print("=" * 70)

# ── Load cached feature CSV ───────────────────────────────────────────────────

cache_path = os.path.join(ROOT, f"data/processed/features_{UNIVERSE_MODE}.csv")
if not os.path.exists(cache_path):
    raise FileNotFoundError(
        f"Feature cache not found: {cache_path}\n"
        f"Run main.py with load_cached=False first to generate it."
    )

print(f"\nLoading: {cache_path}")
data = pd.read_csv(cache_path, parse_dates=['Date'])
print(f"  Loaded {len(data):,} rows, {data['Date'].nunique()} dates, "
      f"{data['Ticker'].nunique()} tickers")

# ── Walk-forward folds ────────────────────────────────────────────────────────

dates  = sorted(data['Date'].unique())
folds  = generate_walk_forward_folds(
    dates,
    train_days=252,
    val_days=63,
    test_days=63,
    stride_days=63,
    train_window_mode="rolling",
)
n_folds = len(folds)
print(f"  Walk-forward folds: {n_folds}")
print()

# ── Walk-forward loop ─────────────────────────────────────────────────────────

all_preds_by_model = {m: [] for m in MODELS_TO_RUN}
fold_val_metrics   = {m: [] for m in MODELS_TO_RUN}

_set_seed(config.RANDOM_SEED)

for fold in folds:
    fold_num  = fold['fold']
    fold_seed = config.RANDOM_SEED + fold_num * 1000
    _set_seed(fold_seed)

    df_tr = data[data['Date'].isin(dates[fold['train'][0]:fold['train'][1]])]
    df_v  = data[data['Date'].isin(dates[fold['val'][0]:fold['val'][1]])]
    df_ts = data[data['Date'].isin(dates[fold['test'][0]:fold['test'][1]])]

    y_tr = df_tr['Target'].values.astype(int)
    y_v  = df_v['Target'].values.astype(int)

    n_val_days  = int(df_v['Date'].nunique())
    n_test_days = int(df_ts['Date'].nunique())
    fold_meta   = dict(
        fold       = fold_num,
        fold_start = fold['test_start_date'],
        fold_end   = fold['test_end_date'],
        n_val_days = n_val_days,
        n_test_days= n_test_days,
    )

    print(f"Fold {fold_num}/{n_folds} | train={len(df_tr):,} val={len(df_v):,} test={len(df_ts):,}")

    # ── Baseline models ───────────────────────────────────────────────────────
    if RUN_BASELINES:
        Xb_tr = df_tr[baseline_cols].values
        Xb_v  = df_v[baseline_cols].values
        Xb_ts = df_ts[baseline_cols].values
        X_tr_s, X_v_s, X_ts_s, _ = standardize_fold(Xb_tr, Xb_v, Xb_ts)

        t0 = time.time()
        lr_m  = train_logistic(X_tr_s, y_tr)
        rf_m  = train_random_forest(X_tr_s, y_tr, X_v_s, y_v)
        xgb_m = train_xgboost(X_tr_s, y_tr, X_v_s, y_v)
        print(f"  [Baselines] fit done in {time.time()-t0:.1f}s")

        val_preds_map = {
            "LR":      lr_m.predict_proba(X_v_s)[:, 1],
            "RF":      rf_m.predict_proba(X_v_s)[:, 1],
            "XGBoost": xgb_m.predict(xgb.DMatrix(X_v_s)),
        }
        test_preds_map = {
            "LR":      lr_m.predict_proba(X_ts_s)[:, 1],
            "RF":      rf_m.predict_proba(X_ts_s)[:, 1],
            "XGBoost": xgb_m.predict(xgb.DMatrix(X_ts_s)),
        }

        for model_name in BASELINE_MODELS:
            val_probs  = val_preds_map[model_name]
            test_probs = test_preds_map[model_name]

            val_auc = binary_auc_safe(y_v, val_probs, log_on_fail=False)
            val_auc = float(val_auc) if val_auc is not None else float('nan')
            val_acc = float(((val_probs >= 0.5).astype(int) == y_v).mean())

            fold_pred = df_ts.copy().reset_index(drop=True)
            fold_pred[f'Prob_{model_name}'] = test_probs
            fold_pred['Fold'] = fold_num
            all_preds_by_model[model_name].append(fold_pred)
            fold_val_metrics[model_name].append({**fold_meta, 'val_auc': val_auc, 'val_accuracy': val_acc})
            print(f"  Fold {fold_num}/{n_folds} | {model_name:<8s} | val_auc={val_auc:.4f}")

    # ── Sequential models (LSTM / TCN) ────────────────────────────────────────
    if RUN_SEQ:
        df_train_fold = pd.concat([df_tr, df_v]).sort_values(['Ticker', 'Date'])
        df_test_fold  = df_ts.copy()

        # ── LSTM ──────────────────────────────────────────────────────────────
        t0 = time.time()
        (X_tr_b, y_tr_b, X_val_b, y_val_b,
         X_te_b, y_te_b, _, keys_val_b, keys_te_b) = \
            prepare_lstm_b_sequences_temporal_split(
                df_train_fold, df_test_fold,
                val_ratio=config.LSTM_B_VAL_SPLIT,
                feature_cols=seq_cols,
            )

        model_lstm = train_lstm_b(
            X_tr_b, y_tr_b, X_val_b, y_val_b,
            device, seed=fold_seed + 30, fold_idx=fold_num,
        )
        print(f"  [LSTM]    fit done in {time.time()-t0:.1f}s")

        val_probs_lstm  = predict_lstm(model_lstm, X_val_b, device)
        val_auc_lstm = binary_auc_safe(y_val_b, val_probs_lstm, log_on_fail=False)
        val_auc_lstm = float(val_auc_lstm) if val_auc_lstm is not None else float('nan')
        val_acc_lstm = float(((val_probs_lstm >= 0.5).astype(int) == y_val_b).mean())

        test_probs_lstm = predict_lstm(model_lstm, X_te_b, device)
        fold_pred_lstm  = df_ts.copy().reset_index(drop=True)
        fold_pred_lstm['Prob_LSTM'] = align_predictions_to_df(test_probs_lstm, keys_te_b, df_ts)
        fold_pred_lstm['Fold'] = fold_num
        all_preds_by_model['LSTM'].append(fold_pred_lstm)
        fold_val_metrics['LSTM'].append({**fold_meta, 'val_auc': val_auc_lstm, 'val_accuracy': val_acc_lstm})
        print(f"  Fold {fold_num}/{n_folds} | {'LSTM':<8s} | val_auc={val_auc_lstm:.4f}")

        del model_lstm, X_te_b, y_te_b, X_tr_b, y_tr_b, X_val_b, y_val_b
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # ── TCN ───────────────────────────────────────────────────────────────
        t0 = time.time()
        (X_tr_t, y_tr_t, X_val_t, y_val_t,
         X_te_t, y_te_t, _, keys_val_t, keys_te_t) = \
            prepare_tcn_sequences_temporal_split(
                df_train_fold, df_test_fold,
                feature_cols=seq_cols,
                val_ratio=float(config.TCN_VAL_SPLIT),
            )

        model_tcn = train_tcn(
            X_tr_t, y_tr_t, X_val_t, y_val_t,
            device, seed=fold_seed + 50, fold_idx=fold_num,
        )
        print(f"  [TCN]     fit done in {time.time()-t0:.1f}s")

        val_probs_tcn  = predict_tcn(model_tcn, X_val_t, device)
        val_auc_tcn = binary_auc_safe(y_val_t, val_probs_tcn, log_on_fail=False)
        val_auc_tcn = float(val_auc_tcn) if val_auc_tcn is not None else float('nan')
        val_acc_tcn = float(((val_probs_tcn >= 0.5).astype(int) == y_val_t).mean())

        test_probs_tcn = predict_tcn(model_tcn, X_te_t, device)
        fold_pred_tcn  = df_ts.copy().reset_index(drop=True)
        fold_pred_tcn['Prob_TCN'] = align_predictions_to_df(test_probs_tcn, keys_te_t, df_ts)
        fold_pred_tcn['Fold'] = fold_num
        all_preds_by_model['TCN'].append(fold_pred_tcn)
        fold_val_metrics['TCN'].append({**fold_meta, 'val_auc': val_auc_tcn, 'val_accuracy': val_acc_tcn})
        print(f"  Fold {fold_num}/{n_folds} | {'TCN':<8s} | val_auc={val_auc_tcn:.4f}")

        del model_tcn, X_te_t, y_te_t, X_tr_t, y_tr_t, X_val_t, y_val_t
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

# ── Post-loop: signal pipeline + per-fold Sharpe ─────────────────────────────

print("\n" + "=" * 70)
print("POST-LOOP: computing signals and per-fold Sharpe...")
print("=" * 70)

fold_rows = []

for model_name in MODELS_TO_RUN:
    preds_list = all_preds_by_model[model_name]
    if not preds_list:
        continue

    full_preds = pd.concat(preds_list).reset_index(drop=True)
    prob_col   = f'Prob_{model_name}'
    valid      = full_preds.dropna(subset=[prob_col]).copy()
    if len(valid) == 0:
        print(f"  {model_name}: no valid predictions — skipping")
        continue

    smoothed   = smooth_probabilities(
        valid, prob_col, alpha=0.0, ema_method="alpha", ema_span=None,
    )
    smooth_col = f'{prob_col}_Smooth'

    sig_df, _ = generate_signals(
        smoothed, k=K_STOCKS, prob_col=smooth_col, return_diagnostics=True,
    )
    sig_df = apply_holding_period_constraint(sig_df, min_hold_days=config.MIN_HOLDING_DAYS)

    port_gross = compute_portfolio_returns(
        sig_df, tc_bps=0, k=K_STOCKS,
        slippage_bps=config.SLIPPAGE_BPS, invert_signals=INVERT_SIGNALS,
    )
    port_net = compute_portfolio_returns(
        sig_df, tc_bps=config.TC_BPS, k=K_STOCKS,
        slippage_bps=config.SLIPPAGE_BPS, invert_signals=INVERT_SIGNALS,
    )

    for vm in fold_val_metrics[model_name]:
        start = pd.Timestamp(vm['fold_start'])
        end   = pd.Timestamp(vm['fold_end'])

        gr = port_gross.loc[
            (port_gross.index >= start) & (port_gross.index <= end), 'Gross_Return'
        ]
        nr = port_net.loc[
            (port_net.index >= start) & (port_net.index <= end), 'Net_Return'
        ]

        gross_sharpe = _sharpe_or_nan(gr)
        net_sharpe   = _sharpe_or_nan(nr)

        fold_rows.append({
            'condition':         ABLATION_CONDITION,
            'model':             model_name,
            'fold':              vm['fold'],
            'fold_start':        str(vm['fold_start'])[:10],
            'fold_end':          str(vm['fold_end'])[:10],
            'n_val_days':        vm['n_val_days'],
            'n_test_days':       vm['n_test_days'],
            'val_auc':           round(vm['val_auc'],        4),
            'val_accuracy':      round(vm['val_accuracy'],   4),
            'test_gross_sharpe': round(gross_sharpe,         4) if not np.isnan(gross_sharpe) else float('nan'),
            'test_net_sharpe':   round(net_sharpe,           4) if not np.isnan(net_sharpe)   else float('nan'),
        })
        print(f"  Fold {vm['fold']}/{n_folds} | {model_name:<8s} | "
              f"val_auc={vm['val_auc']:.4f} | test_net_sharpe={net_sharpe:+.2f}")

# ── Write outputs ─────────────────────────────────────────────────────────────

ablation_dir = os.path.join(ROOT, "reports", "ablation")
os.makedirs(ablation_dir, exist_ok=True)

fold_df  = pd.DataFrame(fold_rows)

# File 1 — per-fold detail (overwrite)
fold_csv = os.path.join(ablation_dir, f"{UNIVERSE_MODE}_{ABLATION_CONDITION}_fold_metrics.csv")
fold_df.to_csv(fold_csv, index=False)
print(f"\nWrote: {fold_csv}")

# File 2 — aggregated summary (append)
summary_cols = [
    'condition', 'model',
    'mean_val_auc', 'std_val_auc',
    'mean_test_gross_sharpe', 'std_test_gross_sharpe',
    'mean_test_net_sharpe', 'std_test_net_sharpe',
    'n_folds', 'run_timestamp',
]
summary_csv = os.path.join(ablation_dir, f"{UNIVERSE_MODE}_ablation_summary.csv")
run_ts = datetime.now().isoformat()

new_rows = []
for model_name in MODELS_TO_RUN:
    subset = fold_df[fold_df['model'] == model_name]
    if len(subset) == 0:
        continue
    new_rows.append({
        'condition':               ABLATION_CONDITION,
        'model':                   model_name,
        'mean_val_auc':            round(float(subset['val_auc'].mean()),            4),
        'std_val_auc':             round(float(subset['val_auc'].std()),             4),
        'mean_test_gross_sharpe':  round(float(subset['test_gross_sharpe'].mean()),  4),
        'std_test_gross_sharpe':   round(float(subset['test_gross_sharpe'].std()),   4),
        'mean_test_net_sharpe':    round(float(subset['test_net_sharpe'].mean()),    4),
        'std_test_net_sharpe':     round(float(subset['test_net_sharpe'].std()),     4),
        'n_folds':                 len(subset),
        'run_timestamp':           run_ts,
    })

new_df = pd.DataFrame(new_rows, columns=summary_cols)
if os.path.exists(summary_csv):
    existing = pd.read_csv(summary_csv)
    pd.concat([existing, new_df], ignore_index=True).to_csv(summary_csv, index=False)
else:
    new_df.to_csv(summary_csv, index=False)
print(f"Wrote: {summary_csv}")

# File 3 — T10 thesis table (always rebuilt from the full summary CSV)
T10_ROW_ORDER = [
    ("LR",       "L1_only"), ("LR",       "L1_L2"),
    ("RF",       "L1_only"), ("RF",       "L1_L2"),
    ("XGBoost",  "L1_only"), ("XGBoost",  "L1_L2"),
    ("LSTM",     "L1_only"), ("LSTM",     "L1_L2"), ("LSTM",     "L1_L2_L3"),
    ("TCN",      "L1_only"), ("TCN",      "L1_L2"), ("TCN",      "L1_L2_L3"),
]

summary_full = pd.read_csv(summary_csv)
latest = (
    summary_full
    .sort_values('run_timestamp')
    .groupby(['condition', 'model'], as_index=False)
    .last()
)

t10_rows = []
for model, cond in T10_ROW_ORDER:
    match = latest[(latest['model'] == model) & (latest['condition'] == cond)]
    if len(match) == 0:
        continue
    r = match.iloc[0]
    t10_rows.append({
        'model':                model,
        'condition':            cond,
        'mean_val_auc':         round(float(r['mean_val_auc']),         4),
        'mean_test_net_sharpe': round(float(r['mean_test_net_sharpe']), 4),
    })

t10_df  = pd.DataFrame(t10_rows)
t10_csv = os.path.join(ablation_dir, f"{UNIVERSE_MODE}_ablation_T10.csv")
t10_df.to_csv(t10_csv, index=False)
print(f"Wrote: {t10_csv}")

# ── Final summary tables ──────────────────────────────────────────────────────

print("\n" + "=" * 70)
print(f"CONDITION: {ABLATION_CONDITION}  |  UNIVERSE: {UNIVERSE_MODE}")
print("=" * 70)

print("\nPer-model summary for this run:")
summary_rows = []
for model_name in MODELS_TO_RUN:
    subset = fold_df[fold_df['model'] == model_name]
    if len(subset) == 0:
        continue
    summary_rows.append((
        model_name,
        f"{subset['val_auc'].mean():.4f}",
        f"{subset['val_auc'].std():.4f}",
        f"{subset['test_net_sharpe'].mean():.4f}",
        f"{subset['test_net_sharpe'].std():.4f}",
        str(len(subset)),
    ))
_print_aligned_table(
    summary_rows,
    ["Model", "mean_val_auc", "std_val_auc", "mean_net_sharpe", "std_net_sharpe", "n_folds"],
)

if len(t10_rows) > 0:
    print("\nT10 table (all conditions seen so far):")
    _print_aligned_table(
        [(r['model'], r['condition'], r['mean_val_auc'], r['mean_test_net_sharpe'])
         for r in t10_rows],
        ["Model", "Condition", "mean_val_auc", "mean_test_net_sharpe"],
    )
