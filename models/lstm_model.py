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
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


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


def _train_lstm_impl(model, X_train, y_train, X_val, y_val, device,
                     optimizer, criterion, max_epochs, patience, desc,
                     batch_size=None, seed=None):
    """
    Internal training loop with batched validation for memory efficiency.
    Returns trained model with best validation loss weights restored.
    """
    # Use DataLoaders for both train and validation (memory efficient)
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )

    # Determine batch size
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

    best_val_loss = float('inf')
    best_state = None
    patience_ctr = 0

    with tqdm(range(max_epochs), desc=desc, unit="epoch") as pbar:
        for epoch in pbar:
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

            # Batched validation for memory efficiency
            model.eval()
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    loss = criterion(model(xb), yb)
                    val_loss_sum += loss.item() * len(yb)
                    val_count += len(yb)
            val_loss = val_loss_sum / val_count

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Store state on CPU to save GPU memory
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    pbar.set_postfix({"val_loss": f"{best_val_loss:.4f}", "status": "early stop"})
                    break

            pbar.set_postfix({"val_loss": f"{val_loss:.4f}", "best": f"{best_val_loss:.4f}"})

    model.load_state_dict(best_state)
    model.to(device)
    print(f"  [{desc}] epoch {epoch+1}/{max_epochs} — early stop | val_loss={best_val_loss:.4f}")
    return model


def train_lstm_a(X_train, y_train, X_val, y_val, device,
                 optimizer_name=None, lr=None, hidden_size=None,
                 num_layers=None, dropout=None, batch_size=None, seed=None):
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
            config.LSTM_A_PATIENCE, "LSTM-A", bs, seed=train_seed
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
                config.LSTM_A_PATIENCE, "LSTM-A", bs, seed=train_seed
            )
        raise


def _train_lstm_b_impl(model, X_train, y_train, X_val, y_val, device,
                        optimizer, scheduler, criterion, max_epochs, patience,
                        seed=None):
    """
    Internal training loop for LSTM-B with batched validation and scheduler.
    """
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )

    batch_size = config.LSTM_B_BATCH
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

    best_val_loss = float('inf')
    best_state = None
    patience_ctr = 0

    with tqdm(range(max_epochs), desc="LSTM-B", unit="epoch") as pbar:
        for epoch in pbar:
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

            # Batched validation
            model.eval()
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    loss = criterion(model(xb), yb)
                    val_loss_sum += loss.item() * len(yb)
                    val_count += len(yb)
            val_loss = val_loss_sum / val_count

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    pbar.set_postfix({"val_loss": f"{best_val_loss:.4f}", "status": "early stop"})
                    break

            pbar.set_postfix({"val_loss": f"{val_loss:.4f}", "best": f"{best_val_loss:.4f}"})

    model.load_state_dict(best_state)
    model.to(device)
    print(f"  [LSTM-B] epoch {epoch+1}/{max_epochs} — early stop | val_loss={best_val_loss:.4f}")
    return model


def train_lstm_b(X_train, y_train, X_val, y_val, device, seed=None):
    """
    Trains LSTM-B using Adam with ReduceLROnPlateau scheduler.
    Returns trained model with best validation loss weights restored.
    Falls back to CPU if MPS runs out of memory.
    """
    _clear_mps_cache()
    train_seed = config.RANDOM_SEED if seed is None else seed
    _seed_everything(train_seed)

    def _create_model_and_optim(dev):
        model = LSTMModelB().to(dev)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LSTM_B_LR,
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
