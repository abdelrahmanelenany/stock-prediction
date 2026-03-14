"""
models/lstm_model.py
Step 6d: PyTorch LSTM architecture, Dataset, and training loop.

Architecture:
  Input (batch, 60, 41)
  → LSTM(64 units, 2 layers, dropout=0.2)
  → last timestep (batch, 64)
  → Dropout(0.2)
  → Linear(64→32) → ReLU
  → Linear(32→1)  → Sigmoid
  → scalar probability of outperforming the cross-sectional median
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    SEQ_LEN, N_TOTAL_FEATURES,
    LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT,
    LSTM_LR, LSTM_BATCH, LSTM_MAX_EPOCHS, LSTM_PATIENCE,
)


class StockSequenceDataset(Dataset):
    """
    Builds (sequence, label) pairs per ticker using a sliding window.

    Each sequence is shape (SEQ_LEN, n_features).  The corresponding label is
    the Target value at the _last_ row of the window (position i in the ticker).

    Critically, rows 0..SEQ_LEN-1 per ticker cannot form a full window and are
    skipped — so predictions are only available for a subset of df_ts rows.
    The `keys` attribute stores (Date, Ticker) for every sequence so predictions
    can be aligned back to the original DataFrame in main.py.
    """

    def __init__(
        self,
        data_df,
        feature_cols: list[str],
        target_col: str,
        seq_len: int = SEQ_LEN,
        tickers=None,
    ):
        self.seq_len = seq_len
        sequences, labels, keys = [], [], []

        tickers = tickers if tickers is not None else data_df['Ticker'].unique()
        for ticker in tickers:
            stock = (
                data_df[data_df['Ticker'] == ticker]
                .sort_values('Date')
                .reset_index(drop=True)
            )
            X     = stock[feature_cols].values.astype(np.float32)
            y     = stock[target_col].values.astype(np.float32)
            dates = stock['Date'].values

            for i in range(seq_len, len(X)):
                sequences.append(X[i - seq_len : i])   # shape (seq_len, n_features)
                labels.append(y[i])
                keys.append((dates[i], ticker))         # for alignment in main.py

        self.sequences = torch.tensor(np.array(sequences), dtype=torch.float32)   # (N, seq_len, n_feat)
        self.labels    = torch.tensor(np.array(labels),    dtype=torch.float32)   # (N,)
        self.keys      = keys                                 # list of (date, ticker)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class StockLSTM(nn.Module):
    """
    Two-layer stacked LSTM followed by a small fully-connected head.
    Uses the last timestep's hidden state as the summary vector.
    """

    def __init__(
        self,
        input_size: int  = N_TOTAL_FEATURES,
        hidden_size: int = LSTM_HIDDEN,
        num_layers: int  = LSTM_LAYERS,
        dropout: float   = LSTM_DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(hidden_size, 32)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)               # (batch, seq_len, hidden)
        out = self.dropout(out[:, -1, :])   # last timestep → (batch, hidden)
        out = self.relu(self.fc1(out))      # (batch, 32)
        return self.sigmoid(self.fc2(out)).squeeze(1)   # (batch,)


def train_lstm(
    model: StockLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_epochs: int  = LSTM_MAX_EPOCHS,
    patience: int    = LSTM_PATIENCE,
    lr: float        = LSTM_LR,
) -> tuple[StockLSTM, list[float], list[float]]:
    """
    Trains the LSTM with early stopping on validation BCE loss.

    Returns the model loaded with best weights, plus train/val loss histories
    per epoch (used for thesis Figure F11 — loss curves).
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_loss  = float('inf')
    best_weights   = None
    patience_ctr   = 0
    train_losses   = []
    val_losses     = []

    with tqdm(range(max_epochs), desc="Training", unit="epoch") as epoch_bar:
        for epoch in epoch_bar:
            # ── Training pass ──────────────────────────────────────────────────
            model.train()
            running_loss = 0.0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running_loss += loss.item()
            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)

            # ── Validation pass ────────────────────────────────────────────────
            model.eval()
            running_val = 0.0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    running_val += criterion(
                        model(X_b.to(device)), y_b.to(device)
                    ).item()
            val_loss = running_val / len(val_loader)
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            # ── Early stopping ─────────────────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
                patience_ctr  = 0
            else:
                patience_ctr += 1

            epoch_bar.set_postfix({
                "train_loss": f"{train_loss:.4f}",
                "val_loss":   f"{val_loss:.4f}",
                "patience":   f"{patience_ctr}/{patience}",
            })

            if patience_ctr >= patience:
                tqdm.write(f"  LSTM early stop @ epoch {epoch + 1}  "
                           f"best val loss={best_val_loss:.4f}")
                break

    model.load_state_dict(best_weights)
    return model, train_losses, val_losses


def lstm_predict(
    model: StockLSTM,
    dataset: StockSequenceDataset,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, list]:
    """
    Run inference on a StockSequenceDataset and return:
      - probs : np.ndarray of shape (N,) — predicted probabilities
      - keys  : list of (date, ticker) tuples aligned to probs
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0)
    model.eval()
    all_probs = []
    with torch.no_grad():
        for X_b, _ in loader:
            all_probs.append(model(X_b.to(device)).cpu().numpy())
    return np.concatenate(all_probs), dataset.keys
