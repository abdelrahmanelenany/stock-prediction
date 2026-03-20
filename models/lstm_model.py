"""
models/lstm_model.py
LSTM-A: Paper-faithful replication (Fischer & Krauss 2017) — 1 feature, seq_len=240
LSTM-B: Extended ablation — 6 curated features, seq_len=60

Both models output raw logits (2 classes) — use CrossEntropyLoss in training.
Inference applies softmax to get class probabilities.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


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
    Builds overlapping sequences of standardized 1-day returns for LSTM-A.
    Replicates Fischer & Krauss (2017) Section 3.2.1 exactly.

    Standardization is fit on training data ONLY, then applied to both splits.

    For test sequences, training data is used as lookback history so that
    predictions can be made even when test period < seq_len.

    Args:
        df_train: DataFrame with columns [Date, Ticker, Return_1d, Target], training fold only
        df_test:  DataFrame with same columns, test fold only

    Returns:
        X_train: np.array shape (N_train, seq_len, 1)
        y_train: np.array shape (N_train,)
        X_test:  np.array shape (N_test, seq_len, 1)
        y_test:  np.array shape (N_test,)
        keys_train: list of (date, ticker) for train alignment
        keys_test:  list of (date, ticker) for test alignment
    """
    seq_len = config.LSTM_A_SEQ_LEN
    feature_col = config.LSTM_A_FEATURES[0]  # 'Return_1d'

    # Fit scaler on training data ONLY
    mu = df_train[feature_col].mean()
    sigma = df_train[feature_col].std()

    # Standardize both splits using training statistics
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train['r_std'] = (df_train[feature_col] - mu) / sigma
    df_test['r_std'] = (df_test[feature_col] - mu) / sigma

    # Build training sequences from train data only
    X_train, y_train, keys_train = _build_sequences(df_train, seq_len, 'r_std')

    # Build test sequences using train data as lookback history
    # Combine last seq_len days of train with all of test for each ticker
    test_dates = set(df_test['Date'].apply(lambda d: pd.Timestamp(d).strftime('%Y-%m-%d')))
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    X_test, y_test, keys_test = _build_sequences_with_lookback(
        df_combined, seq_len, 'r_std', test_dates
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

class LSTMModelA(nn.Module):
    """
    Paper-faithful LSTM (Fischer & Krauss 2017).
    Single layer, h=25, input_size=1.
    Outputs logits for 2 classes.
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=config.LSTM_A_HIDDEN,
            num_layers=config.LSTM_A_LAYERS,
            dropout=0.0,  # no recurrent dropout for single layer
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.LSTM_A_DROPOUT)
        self.fc = nn.Linear(config.LSTM_A_HIDDEN, 2)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # take last timestep output
        return self.fc(out)  # logits for 2 classes


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
                     optimizer, criterion, max_epochs, patience, desc):
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

    batch_size = config.LSTM_A_BATCH if desc == "LSTM-A" else config.LSTM_B_BATCH
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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


def train_lstm_a(X_train, y_train, X_val, y_val, device):
    """
    Trains LSTM-A using RMSprop with early stopping.
    Returns trained model with best validation loss weights restored.
    Falls back to CPU if MPS runs out of memory.
    """
    _clear_mps_cache()

    # Try with the given device first
    try:
        model = LSTMModelA().to(device)
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=config.LSTM_A_LR,
            weight_decay=config.LSTM_WD,
        )
        criterion = nn.CrossEntropyLoss()

        return _train_lstm_impl(
            model, X_train, y_train, X_val, y_val, device,
            optimizer, criterion, config.LSTM_A_MAX_EPOCHS,
            config.LSTM_A_PATIENCE, "LSTM-A"
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "MPS" in str(e):
            print(f"  [LSTM-A] MPS out of memory, falling back to CPU...")
            _clear_mps_cache()
            cpu_device = torch.device('cpu')
            model = LSTMModelA().to(cpu_device)
            optimizer = torch.optim.RMSprop(
                model.parameters(),
                lr=config.LSTM_A_LR,
                weight_decay=config.LSTM_WD,
            )
            criterion = nn.CrossEntropyLoss()

            return _train_lstm_impl(
                model, X_train, y_train, X_val, y_val, cpu_device,
                optimizer, criterion, config.LSTM_A_MAX_EPOCHS,
                config.LSTM_A_PATIENCE, "LSTM-A"
            )
        raise


def _train_lstm_b_impl(model, X_train, y_train, X_val, y_val, device,
                        optimizer, scheduler, criterion, max_epochs, patience):
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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


def train_lstm_b(X_train, y_train, X_val, y_val, device):
    """
    Trains LSTM-B using Adam with ReduceLROnPlateau scheduler.
    Returns trained model with best validation loss weights restored.
    Falls back to CPU if MPS runs out of memory.
    """
    _clear_mps_cache()

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
            config.LSTM_B_PATIENCE
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
                config.LSTM_B_PATIENCE
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
