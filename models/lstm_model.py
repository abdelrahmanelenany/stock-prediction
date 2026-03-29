"""
models/lstm_model.py
LSTM-A: Bhandari-inspired technical indicator LSTM (4 features, tuned architecture)
LSTM-B: Extended ablation — 6 curated features, fixed architecture (64 units, 2 layers)

Both models output raw logits (2 classes) — use CrossEntropyLoss in training.
Inference applies softmax to get class probabilities.

Implements Bhandari §3.3 hyperparameter tuning (Section 1 of IMPLEMENTATION_EXTENSIONS.md):
- Phase 1: tune optimizer, learning rate, batch size
- Phase 2 (LSTM-A only): tune architecture (hidden_size, num_layers, dropout)
"""
import itertools
import logging
import os
import random
import sys
import csv
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from evaluation.metrics_utils import binary_auc_safe

logger = logging.getLogger(__name__)


def _seed_everything(seed: int):
    """Seed all RNGs used by LSTM training for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_torch_generator(seed: int) -> torch.Generator:
    """Create a seeded CPU generator for deterministic DataLoader shuffling."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


# ────────────────────────────────────────────────────────────────────────────
# Hyperparameter Tuning (Bhandari §3.3)
# ────────────────────────────────────────────────────────────────────────────

def _build_optimizer(model, name: str, lr: float):
    """Helper: instantiate optimizer by name string."""
    params = model.parameters()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=config.LSTM_WD)
    elif name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr, weight_decay=config.LSTM_WD)
    elif name == "nadam":
        return torch.optim.NAdam(params, lr=lr, weight_decay=config.LSTM_WD)
    elif name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=config.LSTM_WD)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def _run_tuning_replicates(
    X_train, y_train, X_val, y_val, device,
    opt_name, lr, bs, input_size, hidden_size, num_layers, dropout,
    max_epochs, patience, seed,
):
    """
    Run cfg.LSTM_TUNE_REPLICATES independent training runs for one hyperparameter
    combination. Returns a list of validation AUC scores (one per replicate).
    """
    n_replicates = config.LSTM_TUNE_REPLICATES
    auc_scores = []

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

    for rep in range(n_replicates):
        rep_seed = seed + rep
        _seed_everything(rep_seed)
        train_gen = _make_torch_generator(rep_seed)

        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, generator=train_gen)
        val_dl = DataLoader(val_ds, batch_size=bs * 2, shuffle=False)

        model = StockLSTMTunable(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)

        optimizer = _build_optimizer(model, opt_name, lr)
        criterion = nn.CrossEntropyLoss()

        best_val_loss, patience_ctr = float("inf"), 0
        for epoch in range(max_epochs):
            model.train()
            for Xb, yb in train_dl:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for Xb, yb in val_dl:
                    val_loss += criterion(model(Xb.to(device)), yb.to(device)).item()
            val_loss /= max(len(val_dl), 1)

            if val_loss < best_val_loss:
                best_val_loss, patience_ctr = val_loss, 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    break

        # Compute AUC on validation set
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for Xb, yb in val_dl:
                logits = model(Xb.to(device))
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(yb.numpy())

        if len(set(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_preds)
        else:
            auc = 0.5
        auc_scores.append(auc)

    return auc_scores


def tune_lstm_hyperparams(
    X_train, y_train, X_val, y_val,
    input_size, device,
    arch_grid=None,
    seed_hidden=64, seed_layers=2, seed_dropout=0.2,
    seed=None,
):
    """
    Bhandari §3.3 Algorithm 1 + Algorithm 2 — adapted for classification
    (AUC replaces RMSE).

    Phase 1 (always): sweeps (optimizer, lr, batch_size) from config.LSTM_HYPERPARAM_GRID.
    Phase 2 (when arch_grid is provided): sweeps (hidden_size, num_layers, dropout)
             from arch_grid, using the best Phase-1 hyperparameters as fixed context.
             This mirrors Bhandari Algorithm 2, which tunes architecture after fixing
             the training hyperparameters.

    LSTM-A calls this function with arch_grid=config.LSTM_A_ARCH_GRID.
    LSTM-B calls this function with arch_grid=None (architecture stays fixed).

    Parameters
    ----------
    X_train : np.ndarray
        Training sequences, shape (N, seq_len, n_features)
    y_train : np.ndarray
        Training labels
    X_val : np.ndarray
        Validation sequences
    y_val : np.ndarray
        Validation labels
    input_size : int
        Number of input features
    device : torch.device
        Device to train on
    arch_grid : dict or None
        If provided, contains hidden_size, num_layers, dropout options to search
    seed_hidden, seed_layers, seed_dropout : int/float
        Default architecture to use if arch_grid is None (Phase 1 only)

    Returns
    -------
    dict with keys: optimizer, lr, batch_size, hidden_size, num_layers, dropout
    """
    if seed is None:
        seed = config.RANDOM_SEED

    # ── Phase 1: tune training hyperparameters ────────────────────────────────
    grid = config.LSTM_HYPERPARAM_GRID
    combos = list(itertools.product(
        grid["optimizer"], grid["learning_rate"], grid["batch_size"]
    ))

    # For Phase 1, use a fixed architecture seed
    if arch_grid is not None:
        p1_hidden = arch_grid["hidden_size"][0]
        p1_layers = arch_grid["num_layers"][0]
        p1_drop = arch_grid["dropout"][0]
    else:
        p1_hidden = seed_hidden
        p1_layers = seed_layers
        p1_drop = seed_dropout

    print(f"[LSTM Tuning - Phase 1] {len(combos)} training combos x "
          f"{config.LSTM_TUNE_REPLICATES} replicates")

    phase1_results = []
    for opt_name, lr, bs in combos:
        auc_scores = _run_tuning_replicates(
            X_train, y_train, X_val, y_val, device,
            opt_name, lr, bs, input_size,
            hidden_size=p1_hidden, num_layers=p1_layers, dropout=p1_drop,
            max_epochs=config.LSTM_TUNE_MAX_EPOCHS,
            patience=config.LSTM_TUNE_PATIENCE,
            seed=seed,
        )
        avg_auc = sum(auc_scores) / len(auc_scores)
        phase1_results.append({
            "optimizer": opt_name, "lr": lr, "batch_size": bs, "avg_val_auc": avg_auc
        })
        print(f"  opt={opt_name:7s}  lr={lr:.4f}  bs={bs:3d}  -> avg AUC={avg_auc:.4f}")

    best_p1 = max(phase1_results, key=lambda x: x["avg_val_auc"])
    print(f"[Phase 1 best] {best_p1}")

    # If no arch grid, return Phase 1 results with fixed architecture
    if arch_grid is None:
        return {
            "optimizer": best_p1["optimizer"],
            "lr": best_p1["lr"],
            "batch_size": best_p1["batch_size"],
            "hidden_size": seed_hidden,
            "num_layers": seed_layers,
            "dropout": seed_dropout,
        }

    # ── Phase 2: tune architecture (LSTM-A only) ──────────────────────────────
    arch_combos = list(itertools.product(
        arch_grid["hidden_size"], arch_grid["num_layers"], arch_grid["dropout"]
    ))
    print(f"\n[LSTM Tuning - Phase 2] {len(arch_combos)} architecture combos x "
          f"{config.LSTM_TUNE_REPLICATES} replicates")

    phase2_results = []
    for hidden_size, num_layers, dropout in arch_combos:
        auc_scores = _run_tuning_replicates(
            X_train, y_train, X_val, y_val, device,
            best_p1["optimizer"], best_p1["lr"], best_p1["batch_size"], input_size,
            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
            max_epochs=config.LSTM_TUNE_MAX_EPOCHS,
            patience=config.LSTM_TUNE_PATIENCE,
            seed=seed + 10_000,
        )
        avg_auc = sum(auc_scores) / len(auc_scores)
        phase2_results.append({
            "hidden_size": hidden_size, "num_layers": num_layers,
            "dropout": dropout, "avg_val_auc": avg_auc
        })
        print(f"  h={hidden_size:3d}  layers={num_layers}  drop={dropout:.1f}"
              f"  -> avg AUC={avg_auc:.4f}")

    best_p2 = max(phase2_results, key=lambda x: x["avg_val_auc"])
    print(f"[Phase 2 best] {best_p2}")

    return {
        "optimizer": best_p1["optimizer"],
        "lr": best_p1["lr"],
        "batch_size": best_p1["batch_size"],
        "hidden_size": best_p2["hidden_size"],
        "num_layers": best_p2["num_layers"],
        "dropout": best_p2["dropout"],
    }


# ────────────────────────────────────────────────────────────────────────────
# Sequence Building Functions
# ────────────────────────────────────────────────────────────────────────────

def _build_sequences(df: pd.DataFrame, seq_len: int, feature_col: str):
    """
    Constructs overlapping sequences per ticker for single-feature input.
    df must have columns: [Date, Ticker, {feature_col}, Target]
    sorted by [Ticker, Date] ascending.

    Returns:
        X: np.array shape (N, seq_len, 1)
        y: np.array shape (N,)
        keys: list of (date, ticker) tuples for alignment
    """
    X_list, y_list, keys = [], [], []
    for ticker, grp in df.sort_values(['Ticker', 'Date']).groupby('Ticker'):
        grp = grp.reset_index(drop=True)
        vals = grp[feature_col].values
        labels = grp['Target'].values
        dates = grp['Date'].values
        for i in range(seq_len, len(vals)):
            X_list.append(vals[i - seq_len:i])
            y_list.append(labels[i])
            keys.append((dates[i], ticker))
    if len(X_list) == 0:
        # No sequences could be built (not enough data points per ticker)
        X = np.zeros((0, seq_len, 1), dtype=np.float32)
        y = np.zeros((0,), dtype=np.int64)
        return X, y, keys
    X = np.array(X_list)[:, :, np.newaxis].astype(np.float32)  # (N, seq_len, 1)
    y = np.array(y_list).astype(np.int64)
    return X, y, keys


def _build_sequences_multi(df: pd.DataFrame, seq_len: int, feature_cols: list):
    """
    Constructs overlapping multi-feature sequences per ticker.
    df must have columns: [Date, Ticker, *feature_cols, Target]
    sorted by [Ticker, Date] ascending.

    Returns:
        X: np.array shape (N, seq_len, n_feat)
        y: np.array shape (N,)
        keys: list of (date, ticker) tuples for alignment
    """
    X_list, y_list, keys = [], [], []
    n_feat = len(feature_cols)
    for ticker, grp in df.sort_values(['Ticker', 'Date']).groupby('Ticker'):
        grp = grp.reset_index(drop=True)
        vals = grp[feature_cols].values
        labels = grp['Target'].values
        dates = grp['Date'].values
        for i in range(seq_len, len(vals)):
            X_list.append(vals[i - seq_len:i])  # shape (seq_len, n_feat)
            y_list.append(labels[i])
            keys.append((dates[i], ticker))
    if len(X_list) == 0:
        # No sequences could be built (not enough data points per ticker)
        X = np.zeros((0, seq_len, n_feat), dtype=np.float32)
        y = np.zeros((0,), dtype=np.int64)
        return X, y, keys
    X = np.array(X_list).astype(np.float32)  # (N, seq_len, n_feat)
    y = np.array(y_list).astype(np.int64)
    return X, y, keys


def _build_sequences_multi_with_lookback(df_combined: pd.DataFrame, seq_len: int,
                                          feature_cols: list, test_dates: set):
    """
    Build multi-feature sequences from combined train+test data, but only output
    sequences where the target day is in test_dates.

    Args:
        df_combined: DataFrame with train+test data
        seq_len: lookback window length
        feature_cols: list of column names for features
        test_dates: set of dates that define the test period

    Returns:
        X, y, keys for test period only
    """
    X_list, y_list, keys = [], [], []
    n_feat = len(feature_cols)
    for ticker, grp in df_combined.sort_values(['Ticker', 'Date']).groupby('Ticker'):
        grp = grp.reset_index(drop=True)
        vals = grp[feature_cols].values
        labels = grp['Target'].values
        dates = grp['Date'].values
        for i in range(seq_len, len(vals)):
            date_str = pd.Timestamp(dates[i]).strftime('%Y-%m-%d')
            if date_str in test_dates:
                X_list.append(vals[i - seq_len:i])
                y_list.append(labels[i])
                keys.append((dates[i], ticker))

    if len(X_list) == 0:
        X = np.zeros((0, seq_len, n_feat), dtype=np.float32)
        y = np.zeros((0,), dtype=np.int64)
        return X, y, keys
    X = np.array(X_list).astype(np.float32)
    y = np.array(y_list).astype(np.int64)
    return X, y, keys


def _build_sequences_with_lookback(df_combined: pd.DataFrame, seq_len: int,
                                    feature_col: str, test_dates: set):
    """
    Build sequences from combined train+test data, but only output sequences
    where the target day is in test_dates. This allows using training data
    as lookback history for test predictions.

    Args:
        df_combined: DataFrame with train+test data, sorted by [Ticker, Date]
        seq_len: lookback window length
        feature_col: column name for the feature
        test_dates: set of dates (as strings or Timestamps) that define the test period

    Returns:
        X, y, keys for test period only
    """
    X_list, y_list, keys = [], [], []
    for ticker, grp in df_combined.sort_values(['Ticker', 'Date']).groupby('Ticker'):
        grp = grp.reset_index(drop=True)
        vals = grp[feature_col].values
        labels = grp['Target'].values
        dates = grp['Date'].values
        for i in range(seq_len, len(vals)):
            date_str = pd.Timestamp(dates[i]).strftime('%Y-%m-%d')
            # Only include if target day is in test period
            if date_str in test_dates:
                X_list.append(vals[i - seq_len:i])
                y_list.append(labels[i])
                keys.append((dates[i], ticker))

    if len(X_list) == 0:
        X = np.zeros((0, seq_len, 1), dtype=np.float32)
        y = np.zeros((0,), dtype=np.int64)
        return X, y, keys
    X = np.array(X_list)[:, :, np.newaxis].astype(np.float32)
    y = np.array(y_list).astype(np.int64)
    return X, y, keys


def prepare_lstm_a_sequences(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    Builds overlapping multi-feature sequences for LSTM-A.
    Uses 4 features: MACD, RSI_14, ATR_14, Return_1d (Bhandari §4.3 inspired).

    Standardization is fit on training data ONLY, then applied to both splits.

    For test sequences, training data is used as lookback history so that
    predictions can be made even when test period < seq_len.

    Args:
        df_train: DataFrame with columns [Date, Ticker, *LSTM_A_FEATURES, Target]
        df_test:  DataFrame with same columns

    Returns:
        X_train: np.array shape (N_train, seq_len, n_feat)
        y_train: np.array shape (N_train,)
        X_test:  np.array shape (N_test, seq_len, n_feat)
        y_test:  np.array shape (N_test,)
        keys_train: list of (date, ticker)
        keys_test:  list of (date, ticker)
    """
    seq_len = config.LSTM_A_SEQ_LEN
    feature_cols = config.LSTM_A_FEATURES  # 4 features

    # Fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols].values)

    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train[feature_cols] = scaler.transform(df_train[feature_cols].values)
    df_test[feature_cols] = scaler.transform(df_test[feature_cols].values)

    # Build training sequences from train data only
    X_train, y_train, keys_train = _build_sequences_multi(df_train, seq_len, feature_cols)

    # Build test sequences using train data as lookback history
    test_dates = set(df_test['Date'].apply(lambda d: pd.Timestamp(d).strftime('%Y-%m-%d')))
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    X_test, y_test, keys_test = _build_sequences_multi_with_lookback(
        df_combined, seq_len, feature_cols, test_dates
    )

    return X_train, y_train, X_test, y_test, keys_train, keys_test


def prepare_lstm_b_sequences(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    Builds overlapping multi-feature sequences for LSTM-B.
    Scaler is fit on training fold only and applied to both splits.

    For test sequences, training data is used as lookback history so that
    predictions can be made even when test period < seq_len.

    Args:
        df_train: DataFrame with columns [Date, Ticker, *LSTM_B_FEATURES, Target]
        df_test:  DataFrame with same columns

    Returns:
        X_train: np.array shape (N_train, seq_len, n_feat)
        y_train: np.array shape (N_train,)
        X_test:  np.array shape (N_test, seq_len, n_feat)
        y_test:  np.array shape (N_test,)
        keys_train: list of (date, ticker)
        keys_test:  list of (date, ticker)
    """
    seq_len = config.LSTM_B_SEQ_LEN
    feature_cols = config.LSTM_B_FEATURES

    # Fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols].values)

    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train[feature_cols] = scaler.transform(df_train[feature_cols].values)
    df_test[feature_cols] = scaler.transform(df_test[feature_cols].values)

    # Build training sequences from train data only
    X_train, y_train, keys_train = _build_sequences_multi(df_train, seq_len, feature_cols)

    # Build test sequences using train data as lookback history
    test_dates = set(df_test['Date'].apply(lambda d: pd.Timestamp(d).strftime('%Y-%m-%d')))
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    X_test, y_test, keys_test = _build_sequences_multi_with_lookback(
        df_combined, seq_len, feature_cols, test_dates
    )

    return X_train, y_train, X_test, y_test, keys_train, keys_test


def prepare_lstm_a_sequences_temporal_split(df_train: pd.DataFrame, df_test: pd.DataFrame,
                                             val_ratio: float = 0.2):
    """
    Build LSTM-A sequences with TEMPORAL train/val split.

    FIX: Instead of splitting by index (which splits by ticker), this function
    splits by DATE - the last val_ratio of training DATES become validation.

    Args:
        df_train: DataFrame with columns [Date, Ticker, *LSTM_A_FEATURES, Target]
        df_test:  DataFrame with same columns
        val_ratio: Fraction of training dates to use for validation (default 0.2)

    Returns:
        X_train, y_train: Training sequences
        X_val, y_val: Validation sequences (last 20% of dates)
        X_test, y_test: Test sequences
        keys_train, keys_val, keys_test: (date, ticker) tuples for alignment
    """
    seq_len = config.LSTM_A_SEQ_LEN
    feature_cols = config.LSTM_A_FEATURES

    # Split training dates temporally
    train_dates_sorted = sorted(df_train['Date'].unique())
    n_dates = len(train_dates_sorted)
    val_start_idx = int(n_dates * (1 - val_ratio))

    train_date_set = set(pd.Timestamp(d).strftime('%Y-%m-%d')
                         for d in train_dates_sorted[:val_start_idx])
    val_date_set = set(pd.Timestamp(d).strftime('%Y-%m-%d')
                       for d in train_dates_sorted[val_start_idx:])

    # Fit scaler on TRUE training data only (excluding validation dates)
    df_true_train = df_train[df_train['Date'].isin(train_dates_sorted[:val_start_idx])]
    scaler = StandardScaler()
    scaler.fit(df_true_train[feature_cols].values)

    # Apply scaler to all data
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train[feature_cols] = scaler.transform(df_train[feature_cols].values)
    df_test[feature_cols] = scaler.transform(df_test[feature_cols].values)

    # Build ALL sequences from training data
    X_all, y_all, keys_all = _build_sequences_multi(df_train, seq_len, feature_cols)

    # Split sequences by DATE (not by index!)
    X_train_list, y_train_list, keys_train = [], [], []
    X_val_list, y_val_list, keys_val = [], [], []

    for i, (date, ticker) in enumerate(keys_all):
        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        if date_str in train_date_set:
            X_train_list.append(X_all[i])
            y_train_list.append(y_all[i])
            keys_train.append((date, ticker))
        elif date_str in val_date_set:
            X_val_list.append(X_all[i])
            y_val_list.append(y_all[i])
            keys_val.append((date, ticker))

    X_train = np.array(X_train_list, dtype=np.float32) if X_train_list else np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int64) if y_train_list else np.zeros(0, dtype=np.int64)
    X_val = np.array(X_val_list, dtype=np.float32) if X_val_list else np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.int64) if y_val_list else np.zeros(0, dtype=np.int64)

    # Build test sequences using full train data as lookback history
    test_dates = set(df_test['Date'].apply(lambda d: pd.Timestamp(d).strftime('%Y-%m-%d')))
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    X_test, y_test, keys_test = _build_sequences_multi_with_lookback(
        df_combined, seq_len, feature_cols, test_dates
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, keys_train, keys_val, keys_test


def prepare_lstm_b_sequences_temporal_split(df_train: pd.DataFrame, df_test: pd.DataFrame,
                                             val_ratio: float = 0.2):
    """
    Build LSTM-B sequences with TEMPORAL train/val split.

    FIX: Instead of splitting by index (which splits by ticker), this function
    splits by DATE - the last val_ratio of training DATES become validation.

    Args:
        df_train: DataFrame with columns [Date, Ticker, *LSTM_B_FEATURES, Target]
        df_test:  DataFrame with same columns
        val_ratio: Fraction of training dates to use for validation (default 0.2)

    Returns:
        X_train, y_train: Training sequences
        X_val, y_val: Validation sequences (last 20% of dates)
        X_test, y_test: Test sequences
        keys_train, keys_val, keys_test: (date, ticker) tuples for alignment
    """
    seq_len = config.LSTM_B_SEQ_LEN
    feature_cols = config.LSTM_B_FEATURES

    # Split training dates temporally
    train_dates_sorted = sorted(df_train['Date'].unique())
    n_dates = len(train_dates_sorted)
    val_start_idx = int(n_dates * (1 - val_ratio))

    train_date_set = set(pd.Timestamp(d).strftime('%Y-%m-%d')
                         for d in train_dates_sorted[:val_start_idx])
    val_date_set = set(pd.Timestamp(d).strftime('%Y-%m-%d')
                       for d in train_dates_sorted[val_start_idx:])

    # Fit scaler on TRUE training data only (excluding validation dates)
    df_true_train = df_train[df_train['Date'].isin(train_dates_sorted[:val_start_idx])]
    scaler = StandardScaler()
    scaler.fit(df_true_train[feature_cols].values)

    # Apply scaler to all data
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train[feature_cols] = scaler.transform(df_train[feature_cols].values)
    df_test[feature_cols] = scaler.transform(df_test[feature_cols].values)

    # Build ALL sequences from training data
    X_all, y_all, keys_all = _build_sequences_multi(df_train, seq_len, feature_cols)

    # Split sequences by DATE (not by index!)
    X_train_list, y_train_list, keys_train = [], [], []
    X_val_list, y_val_list, keys_val = [], [], []

    for i, (date, ticker) in enumerate(keys_all):
        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        if date_str in train_date_set:
            X_train_list.append(X_all[i])
            y_train_list.append(y_all[i])
            keys_train.append((date, ticker))
        elif date_str in val_date_set:
            X_val_list.append(X_all[i])
            y_val_list.append(y_all[i])
            keys_val.append((date, ticker))

    X_train = np.array(X_train_list, dtype=np.float32) if X_train_list else np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int64) if y_train_list else np.zeros(0, dtype=np.int64)
    X_val = np.array(X_val_list, dtype=np.float32) if X_val_list else np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.int64) if y_val_list else np.zeros(0, dtype=np.int64)

    # Build test sequences using full train data as lookback history
    test_dates = set(df_test['Date'].apply(lambda d: pd.Timestamp(d).strftime('%Y-%m-%d')))
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    X_test, y_test, keys_test = _build_sequences_multi_with_lookback(
        df_combined, seq_len, feature_cols, test_dates
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, keys_train, keys_val, keys_test


# ────────────────────────────────────────────────────────────────────────────
# Model Architectures
# ────────────────────────────────────────────────────────────────────────────

class StockLSTMTunable(nn.Module):
    """
    Flexible LSTM architecture for hyperparameter tuning.
    All architecture parameters are configurable via constructor.
    Used for both LSTM-A (tuned) and LSTM-B (fixed architecture).
    Outputs logits for 2 classes.
    """
    def __init__(
        self,
        input_size: int,        # 4 for LSTM-A, 6 for LSTM-B
        hidden_size: int = 64,  # tuner-resolved for LSTM-A; fixed 64 for LSTM-B
        num_layers: int = 2,    # tuner-resolved for LSTM-A; fixed 2 for LSTM-B
        dropout: float = 0.2,   # tuner-resolved for LSTM-A; fixed 0.2 for LSTM-B
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,  # PyTorch ignores dropout
                                                          # for single-layer LSTM
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # take last timestep
        out = self.relu(self.fc1(out))
        return self.fc2(out)  # logits for 2 classes


class LSTMModelA(nn.Module):
    """
    LSTM-A: Bhandari-inspired technical indicator LSTM.
    4 features (MACD, RSI, ATR, Return_1d), architecture from hyperparameter tuning.
    Outputs logits for 2 classes.
    """
    def __init__(self, hidden_size=None, num_layers=None, dropout=None):
        super().__init__()
        n_feat = len(config.LSTM_A_FEATURES)
        # Use tuned values if provided, otherwise use first value from grid
        hidden = hidden_size if hidden_size else config.LSTM_A_ARCH_GRID["hidden_size"][0]
        layers = num_layers if num_layers else config.LSTM_A_ARCH_GRID["num_layers"][0]
        drop = dropout if dropout else config.LSTM_A_ARCH_GRID["dropout"][0]

        self.lstm = nn.LSTM(
            input_size=n_feat,
            hidden_size=hidden,
            num_layers=layers,
            dropout=drop if layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(drop)
        self.fc1 = nn.Linear(hidden, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc2(self.relu(self.fc1(out)))


class LSTMModelB(nn.Module):
    """
    Extended LSTM with 6 input features, 2 layers.
    Outputs logits for 2 classes.
    """
    def __init__(self):
        super().__init__()
        n_feat = len(config.LSTM_B_FEATURES)
        self.lstm = nn.LSTM(
            input_size=n_feat,
            hidden_size=config.LSTM_B_HIDDEN,
            num_layers=config.LSTM_B_LAYERS,
            dropout=config.LSTM_B_DROPOUT if config.LSTM_B_LAYERS > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.LSTM_B_DROPOUT)
        self.fc = nn.Linear(config.LSTM_B_HIDDEN, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


# ────────────────────────────────────────────────────────────────────────────
# Training Functions
# ────────────────────────────────────────────────────────────────────────────

def _clear_mps_cache():
    """Clear MPS memory cache if available."""
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _eval_loader_loss_auc(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, Optional[float]]:
    """Mean loss and ROC-AUC (positive class prob vs labels) on a loader."""
    model.eval()
    total_loss = 0.0
    n = 0
    probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * len(yb)
            n += len(yb)
            pr = torch.softmax(logits, dim=1)[:, 1]
            probs_list.append(pr.cpu().numpy())
            labels_list.append(yb.cpu().numpy())
    mean_loss = total_loss / max(n, 1)
    if not probs_list:
        return mean_loss, None
    y_score = np.concatenate(probs_list)
    y_true = np.concatenate(labels_list)
    auc = binary_auc_safe(y_true, y_score, log_on_fail=False)
    return mean_loss, auc


def _train_lstm_impl(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    max_epochs: int,
    patience: int,
    desc: str,
    batch_size: int | None = None,
    seed: int | None = None,
    lr_scheduler: Any | None = None,
    fold_idx: int | None = None,
) -> nn.Module:
    """
    Training loop with per-epoch train/val loss, AUC, LR logging, optional CSV,
    gradient norm audit, and heuristic warnings (flat AUC, train/val loss gap).
    """
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val),
    )

    if batch_size is None:
        batch_size = config.LSTM_A_BATCH if desc == "LSTM-A" else config.LSTM_B_BATCH

    if seed is None:
        seed = config.RANDOM_SEED
    train_gen = _make_torch_generator(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_gen,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

    max_grad_norm = getattr(config, "LSTM_MAX_GRAD_NORM", None)
    audit_grad = getattr(config, "LSTM_AUDIT_GRAD_NORM", False)
    log_every = getattr(config, "LSTM_LOG_EVERY_EPOCH", True)
    save_csv = getattr(config, "LSTM_SAVE_TRAINING_CSV", False)
    flat_n = getattr(config, "LSTM_FLAT_AUC_WARN_epochs", 8)
    flat_eps = getattr(config, "LSTM_FLAT_AUC_EPS", 0.02)
    of_ratio = getattr(config, "LSTM_OVERFIT_LOSS_RATIO", 3.0)
    of_n = getattr(config, "LSTM_OVERFIT_WARN_epochs", 6)

    epoch_rows: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_state: dict | None = None
    best_epoch = -1
    patience_ctr = 0
    flat_streak = 0
    flat_warned = False
    of_streak = 0
    of_warned = False
    stop_reason = "max_epochs"
    epoch = -1

    with tqdm(range(max_epochs), desc=desc, unit="epoch") as pbar:
        for epoch in pbar:
            model.train()
            train_loss_sum = 0.0
            train_count = 0
            last_gnorm: float | None = None

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()

                if max_grad_norm is not None:
                    gnorm_t = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(max_grad_norm)
                    )
                else:
                    gnorm_t = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float("inf")
                    )
                last_gnorm = float(gnorm_t)

                if audit_grad and last_gnorm is not None:
                    if last_gnorm < 1e-6 or last_gnorm > 1e3:
                        logger.warning(
                            "[%s] epoch %s gradient norm=%.4e (vanish/explode check)",
                            desc,
                            epoch + 1,
                            last_gnorm,
                        )

                optimizer.step()
                train_loss_sum += loss.item() * len(yb)
                train_count += len(yb)

            train_loss_batch = train_loss_sum / max(train_count, 1)

            tr_eval_loss, tr_auc = _eval_loader_loss_auc(
                model, train_loader, device, criterion
            )
            val_loss, val_auc = _eval_loader_loss_auc(
                model, val_loader, device, criterion
            )

            if lr_scheduler is not None:
                lr_scheduler.step(val_loss)
            lr = float(optimizer.param_groups[0]["lr"])

            row = {
                "epoch": epoch + 1,
                "train_loss_batch": round(train_loss_batch, 6),
                "train_eval_loss": round(tr_eval_loss, 6),
                "val_loss": round(val_loss, 6),
                "train_auc": tr_auc,
                "val_auc": val_auc,
                "lr": lr,
                "grad_norm_last": last_gnorm,
            }
            epoch_rows.append(row)

            if log_every:
                au = f" tr_auc={tr_auc:.4f} va_auc={val_auc:.4f}" if tr_auc is not None and val_auc is not None else ""
                logger.info(
                    "[%s] epoch %d/%d train_loss=%.5f tr_eval=%.5f val_loss=%.5f lr=%.2e%s",
                    desc,
                    epoch + 1,
                    max_epochs,
                    train_loss_batch,
                    tr_eval_loss,
                    val_loss,
                    lr,
                    au,
                )

            if tr_auc is not None and val_auc is not None:
                if (
                    abs(tr_auc - 0.5) < flat_eps
                    and abs(val_auc - 0.5) < flat_eps
                ):
                    flat_streak += 1
                else:
                    flat_streak = 0
                if flat_streak >= flat_n and not flat_warned:
                    logger.warning(
                        "[%s] Train and val AUC near 0.5 for %d consecutive epochs (no discrimination).",
                        desc,
                        flat_streak,
                    )
                    flat_warned = True

            if tr_eval_loss > 1e-12 and val_loss > of_ratio * tr_eval_loss:
                of_streak += 1
            else:
                of_streak = 0
            if of_streak >= of_n and not of_warned:
                logger.warning(
                    "[%s] Val loss >> train eval loss for %d epochs (possible overfitting).",
                    desc,
                    of_streak,
                )
                of_warned = True

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    stop_reason = "early_stop_patience"
                    pbar.set_postfix(
                        {"val_loss": f"{best_val_loss:.4f}", "status": "early stop"}
                    )
                    break

            pbar.set_postfix(
                {"val_loss": f"{val_loss:.4f}", "best": f"{best_val_loss:.4f}"}
            )

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epoch + 1

    model.load_state_dict(best_state)
    model.to(device)

    print(
        f"  [{desc}] stopped: {stop_reason} | best_epoch={best_epoch} "
        f"| best_val_loss={best_val_loss:.4f}"
    )

    if save_csv and epoch_rows:
        tag = desc.replace(" ", "_").lower()
        fd = fold_idx if fold_idx is not None else "na"
        out_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "training_logs")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"fold{fd}_{tag}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(epoch_rows[0].keys()))
            w.writeheader()
            w.writerows(epoch_rows)
        print(f"  [{desc}] training log: {path}")

    return model


def train_lstm_a(
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    optimizer_name=None,
    lr=None,
    hidden_size=None,
    num_layers=None,
    dropout=None,
    batch_size=None,
    seed=None,
    fold_idx: int | None = None,
):
    """
    Trains LSTM-A using specified or default hyperparameters.
    Returns trained model with best validation loss weights restored.
    Falls back to CPU if MPS runs out of memory.

    Parameters
    ----------
    optimizer_name : str, optional
        Optimizer name ('adam', 'adagrad', 'nadam'). Defaults to config value.
    lr : float, optional
        Learning rate. Defaults to config value.
    hidden_size, num_layers, dropout : int/float, optional
        Architecture params. Defaults to first value in arch grid.
    batch_size : int, optional
        Batch size. Defaults to config value.
    """
    _clear_mps_cache()

    # Use tuned or default values
    opt_name = optimizer_name or config.LSTM_A_OPTIMIZER
    learning_rate = lr or config.LSTM_A_LR
    bs = batch_size or config.LSTM_A_BATCH
    h_size = hidden_size or config.LSTM_A_ARCH_GRID["hidden_size"][0]
    n_layers = num_layers or config.LSTM_A_ARCH_GRID["num_layers"][0]
    drop = dropout or config.LSTM_A_ARCH_GRID["dropout"][0]

    n_feat = len(config.LSTM_A_FEATURES)
    train_seed = config.RANDOM_SEED if seed is None else seed
    _seed_everything(train_seed)

    # Try with the given device first
    try:
        model = StockLSTMTunable(
            input_size=n_feat,
            hidden_size=h_size,
            num_layers=n_layers,
            dropout=drop,
        ).to(device)

        optimizer = _build_optimizer(model, opt_name, learning_rate)
        criterion = nn.CrossEntropyLoss()

        return _train_lstm_impl(
            model, X_train, y_train, X_val, y_val, device,
            optimizer, criterion, config.LSTM_A_MAX_EPOCHS,
            config.LSTM_A_PATIENCE, "LSTM-A", bs, seed=train_seed,
            lr_scheduler=None, fold_idx=fold_idx,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "MPS" in str(e):
            print(f"  [LSTM-A] MPS out of memory, falling back to CPU...")
            _clear_mps_cache()
            cpu_device = torch.device('cpu')
            model = StockLSTMTunable(
                input_size=n_feat,
                hidden_size=h_size,
                num_layers=n_layers,
                dropout=drop,
            ).to(cpu_device)

            optimizer = _build_optimizer(model, opt_name, learning_rate)
            criterion = nn.CrossEntropyLoss()

            return _train_lstm_impl(
                model, X_train, y_train, X_val, y_val, cpu_device,
                optimizer, criterion, config.LSTM_A_MAX_EPOCHS,
                config.LSTM_A_PATIENCE, "LSTM-A", bs, seed=train_seed,
                lr_scheduler=None, fold_idx=fold_idx,
            )
        raise


def _train_lstm_b_impl(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    optimizer,
    scheduler,
    criterion,
    max_epochs,
    patience,
    seed=None,
    fold_idx: int | None = None,
):
    """Delegates to _train_lstm_impl with ReduceLROnPlateau."""
    return _train_lstm_impl(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        device,
        optimizer,
        criterion,
        max_epochs,
        patience,
        "LSTM-B",
        batch_size=config.LSTM_B_BATCH,
        seed=seed,
        lr_scheduler=scheduler,
        fold_idx=fold_idx,
    )


def train_lstm_b(
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    seed=None,
    fold_idx: int | None = None,
    learning_rate: float | None = None,
):
    """
    Trains LSTM-B using Adam with ReduceLROnPlateau scheduler.
    Returns trained model with best validation loss weights restored.
    Falls back to CPU if MPS runs out of memory.
    """
    _clear_mps_cache()
    train_seed = config.RANDOM_SEED if seed is None else seed
    _seed_everything(train_seed)
    lr_use = float(learning_rate) if learning_rate is not None else config.LSTM_B_LR

    def _create_model_and_optim(dev):
        model = LSTMModelB().to(dev)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr_use,
            weight_decay=config.LSTM_WD,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=config.LSTM_B_LR_PATIENCE,
            factor=config.LSTM_B_LR_FACTOR,
        )
        criterion = nn.CrossEntropyLoss()
        return model, optimizer, scheduler, criterion

    # Try with the given device first
    try:
        model, optimizer, scheduler, criterion = _create_model_and_optim(device)
        return _train_lstm_b_impl(
            model, X_train, y_train, X_val, y_val, device,
            optimizer, scheduler, criterion, config.LSTM_B_MAX_EPOCHS,
            config.LSTM_B_PATIENCE,
            seed=train_seed,
            fold_idx=fold_idx,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "MPS" in str(e):
            print(f"  [LSTM-B] MPS out of memory, falling back to CPU...")
            _clear_mps_cache()
            cpu_device = torch.device('cpu')
            model, optimizer, scheduler, criterion = _create_model_and_optim(cpu_device)
            return _train_lstm_b_impl(
                model, X_train, y_train, X_val, y_val, cpu_device,
                optimizer, scheduler, criterion, config.LSTM_B_MAX_EPOCHS,
                config.LSTM_B_PATIENCE,
                seed=train_seed,
                fold_idx=fold_idx,
            )
        raise


# ────────────────────────────────────────────────────────────────────────────
# Prediction Functions
# ────────────────────────────────────────────────────────────────────────────

def predict_lstm(model, X: np.ndarray, device=None) -> np.ndarray:
    """
    Run inference and return predicted probability of class 1.
    Model outputs logits — softmax is applied here.
    Uses batched inference for memory efficiency.

    Args:
        model: trained LSTMModelA or LSTMModelB
        X: np.array of sequences
        device: torch device (if None, uses model's current device)

    Returns:
        np.array of probabilities for class 1
    """
    # Detect model's actual device
    model_device = next(model.parameters()).device
    if device is None:
        device = model_device
    elif device != model_device:
        # Model might have been moved to CPU due to OOM, use model's device
        device = model_device

    model.eval()
    batch_size = 512  # Use reasonable batch size for inference
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i + batch_size]).to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs)


def align_predictions_to_df(probs: np.ndarray, keys: list, df: pd.DataFrame) -> np.ndarray:
    """
    Map LSTM output (aligned to dataset keys) back to all rows of df.

    Args:
        probs: predicted probabilities from predict_lstm
        keys: list of (date, ticker) from sequence building
        df: original DataFrame to align to

    Returns:
        np.array aligned to df rows (NaN where no prediction)
    """
    prob_map = {
        (pd.Timestamp(date).strftime('%Y-%m-%d'), ticker): float(prob)
        for (date, ticker), prob in zip(keys, probs)
    }

    result = np.full(len(df), np.nan)
    for i, (_, row) in enumerate(df.iterrows()):
        key = (pd.Timestamp(row['Date']).strftime('%Y-%m-%d'), row['Ticker'])
        if key in prob_map:
            result[i] = prob_map[key]
    return result
