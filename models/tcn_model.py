"""
models/tcn_model.py
TCN (Temporal Convolutional Network, Bai et al. 2018) — second neural-network
model running alongside LSTM under identical fold/training infrastructure.

Architecture: stack of dilated causal Conv1d residual blocks. Consumes the same
(N, seq_len, n_features) tensor the LSTM consumes — transposed internally to the
(N, channels, seq_len) layout expected by nn.Conv1d. Last-timestep output feeds
a 2-class linear decoder and CrossEntropyLoss, so prediction is P(class=1).

Infrastructure reuses models/lstm_model.py helpers (seeding, optimizer builder,
sequence builders, prediction alignment) so behavior stays consistent with LSTM.
"""
from __future__ import annotations

import csv
import itertools
import logging
import os
import sys
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from evaluation.metrics_utils import binary_auc_safe
from models.lstm_model import (
    _build_optimizer,
    _build_sequences_multi,
    _build_sequences_multi_with_lookback,
    _clear_mps_cache,
    _eval_loader_loss_auc,
    _make_torch_generator,
    _seed_everything,
    align_predictions_to_df,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Architecture (Bai et al. 2018)
# ────────────────────────────────────────────────────────────────────────────

class Chomp1d(nn.Module):
    """Strip the right-side padding introduced to realise a causal convolution."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Two dilated causal conv layers + ReLU + dropout with a residual connection."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        def _conv(in_ch: int, out_ch: int) -> nn.Module:
            c = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
            # Initialize BEFORE weight_norm: after weight_norm is applied, `weight` becomes
            # a computed property (overwritten by a pre-hook each forward pass), so calling
            # kaiming_normal_ on it afterwards has no effect on the actual parameters.
            nn.init.kaiming_normal_(c.weight, nonlinearity='relu')
            if c.bias is not None:
                nn.init.zeros_(c.bias)
            return weight_norm(c) if use_weight_norm else c

        self.conv1 = _conv(n_inputs, n_outputs)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = _conv(n_outputs, n_outputs)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self) -> None:
        # conv1/conv2 are initialized in _conv before optional weight_norm application.
        # Only the downsample 1x1 conv (no weight_norm) needs explicit init here.
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='relu')
            if self.downsample.bias is not None:
                nn.init.zeros_(self.downsample.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    """
    TCN classifier. Input: (batch, seq_len, n_features). Internally transposed
    to (batch, n_features, seq_len) for Conv1d. Output: logits over 2 classes.
    Uses the final timestep of the top block as the classification vector.
    """

    def __init__(
        self,
        input_size: int,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = list(getattr(config, 'TCN_NUM_CHANNELS', [32, 32, 32, 32]))

        layers: list[nn.Module] = []
        n_levels = len(num_channels)
        for i in range(n_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    dilation=dilation_size, dropout=dropout,
                    use_weight_norm=use_weight_norm,
                )
            )
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, seq_len, n_features) -> (N, n_features, seq_len)
        x = x.transpose(1, 2)
        y = self.tcn(x)              # (N, C_out, seq_len)
        y = y[:, :, -1]              # last timestep (causal output)
        return self.fc(y)


def tcn_receptive_field(kernel_size: int, num_levels: int) -> int:
    """Theoretical receptive field for a stack with two convs per level and
    exponentially growing dilations (matches the TemporalBlock above)."""
    # Each level contributes 2 * (k-1) * dilation to the receptive field.
    rf = 1
    for i in range(num_levels):
        rf += 2 * (kernel_size - 1) * (2 ** i)
    return rf


# ────────────────────────────────────────────────────────────────────────────
# Sequence building — mirrors LSTM with a configurable feature_cols argument
# ────────────────────────────────────────────────────────────────────────────

def prepare_tcn_sequences_temporal_split(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    val_ratio: float = 0.2,
    seq_len: int | None = None,
):
    """
    Build TCN sequences with a TEMPORAL train/val split, scaler fit on TRUE
    training dates only. Returns same tuple layout as the LSTM helper:
        X_train, y_train, X_val, y_val, X_test, y_test,
        keys_train, keys_val, keys_test
    """
    if seq_len is None:
        seq_len = int(getattr(config, 'TCN_SEQ_LEN', config.SEQ_LEN))

    train_dates_sorted = sorted(df_train['Date'].unique())
    n_dates = len(train_dates_sorted)
    val_start_idx = int(n_dates * (1 - val_ratio))

    train_date_set = set(
        pd.Timestamp(d).strftime('%Y-%m-%d')
        for d in train_dates_sorted[:val_start_idx]
    )
    val_date_set = set(
        pd.Timestamp(d).strftime('%Y-%m-%d')
        for d in train_dates_sorted[val_start_idx:]
    )

    df_true_train = df_train[df_train['Date'].isin(train_dates_sorted[:val_start_idx])]
    scaler = StandardScaler()
    scaler.fit(df_true_train[feature_cols].values)

    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train[feature_cols] = scaler.transform(df_train[feature_cols].values)
    df_test[feature_cols] = scaler.transform(df_test[feature_cols].values)

    X_all, y_all, keys_all = _build_sequences_multi(df_train, seq_len, feature_cols)

    X_train_list: list[np.ndarray] = []
    y_train_list: list[int] = []
    keys_train: list[tuple] = []
    X_val_list: list[np.ndarray] = []
    y_val_list: list[int] = []
    keys_val: list[tuple] = []

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

    n_feat = len(feature_cols)
    X_train = (
        np.array(X_train_list, dtype=np.float32)
        if X_train_list else np.zeros((0, seq_len, n_feat), dtype=np.float32)
    )
    y_train = (
        np.array(y_train_list, dtype=np.int64)
        if y_train_list else np.zeros(0, dtype=np.int64)
    )
    X_val = (
        np.array(X_val_list, dtype=np.float32)
        if X_val_list else np.zeros((0, seq_len, n_feat), dtype=np.float32)
    )
    y_val = (
        np.array(y_val_list, dtype=np.int64)
        if y_val_list else np.zeros(0, dtype=np.int64)
    )

    test_dates = set(
        df_test['Date'].apply(lambda d: pd.Timestamp(d).strftime('%Y-%m-%d'))
    )
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    X_test, y_test, keys_test = _build_sequences_multi_with_lookback(
        df_combined, seq_len, feature_cols, test_dates
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, keys_train, keys_val, keys_test


# ────────────────────────────────────────────────────────────────────────────
# Training & prediction
# ────────────────────────────────────────────────────────────────────────────

def _build_tcn_optimizer(model: nn.Module, name: str, lr: float) -> torch.optim.Optimizer:
    """TCN-flavoured optimizer builder that routes through LSTM helper but
    uses TCN_WD for weight decay so both networks can be tuned independently."""
    wd = float(getattr(config, 'TCN_WD', getattr(config, 'LSTM_WD', 1e-4)))
    params = model.parameters()
    name = name.lower()
    if name == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == 'adagrad':
        return torch.optim.Adagrad(params, lr=lr, weight_decay=wd)
    if name == 'nadam':
        return torch.optim.NAdam(params, lr=lr, weight_decay=wd)
    if name == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
    # Fallback to shared builder for any other future optimizer choices.
    return _build_optimizer(model, name, lr)


def _train_tcn_impl(
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
    batch_size: int,
    seed: int | None = None,
    lr_scheduler: Any | None = None,
    fold_idx: int | None = None,
    desc: str = 'TCN',
) -> nn.Module:
    """Mirrors _train_lstm_impl: per-epoch loss/AUC/LR logging, gradient clipping,
    overfit + flat-AUC warnings, best-val-loss checkpoint, optional CSV log.

    Unlike LSTM (which checkpoints on val AUC), TCN checkpoints on val loss.
    Val loss is a proper scoring rule that rewards calibrated, well-spread
    probabilities — what the cross-sectional signal pipeline needs for consistent
    day-by-day rankings. Val AUC only rewards rank order and can be maximised by
    memorising lag patterns that don't generalise to trading returns.
    """
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

    if seed is None:
        seed = config.RANDOM_SEED
    train_gen = _make_torch_generator(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=train_gen)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    max_grad_norm = getattr(config, 'LSTM_MAX_GRAD_NORM', None)
    audit_grad = getattr(config, 'LSTM_AUDIT_GRAD_NORM', False)
    log_every = getattr(config, 'LSTM_LOG_EVERY_EPOCH', True)
    save_csv = getattr(config, 'LSTM_SAVE_TRAINING_CSV', False)
    flat_n = getattr(config, 'LSTM_FLAT_AUC_WARN_epochs', 8)
    flat_eps = getattr(config, 'LSTM_FLAT_AUC_EPS', 0.02)
    of_ratio = getattr(config, 'LSTM_OVERFIT_LOSS_RATIO', 3.0)
    of_n = getattr(config, 'LSTM_OVERFIT_WARN_epochs', 6)

    epoch_rows: list[dict[str, Any]] = []
    best_val_loss = float('inf')
    best_val_auc_at_ckpt = float('-inf')   # tracked for logging only
    best_state: dict | None = None
    best_epoch = -1
    patience_ctr = 0
    flat_streak = 0
    flat_warned = False
    of_streak = 0
    of_warned = False
    stop_reason = 'max_epochs'
    epoch = -1

    with tqdm(range(max_epochs), desc=desc, unit='epoch') as pbar:
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

                clip_norm = float(max_grad_norm) if max_grad_norm is not None else float('inf')
                gnorm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                last_gnorm = float(gnorm_t)

                if audit_grad and last_gnorm is not None:
                    if last_gnorm < 1e-6 or last_gnorm > 1e3:
                        logger.warning(
                            '[%s] epoch %s gradient norm=%.4e (vanish/explode check)',
                            desc, epoch + 1, last_gnorm,
                        )

                optimizer.step()
                train_loss_sum += loss.item() * len(yb)
                train_count += len(yb)

            train_loss_batch = train_loss_sum / max(train_count, 1)
            tr_eval_loss, tr_auc = _eval_loader_loss_auc(model, train_loader, device, criterion)
            val_loss, val_auc = _eval_loader_loss_auc(model, val_loader, device, criterion)

            if lr_scheduler is not None:
                lr_scheduler.step(val_loss)
            lr = float(optimizer.param_groups[0]['lr'])

            epoch_rows.append({
                'epoch': epoch + 1,
                'train_loss_batch': round(train_loss_batch, 6),
                'train_eval_loss': round(tr_eval_loss, 6),
                'val_loss': round(val_loss, 6),
                'train_auc': tr_auc,
                'val_auc': val_auc,
                'lr': lr,
                'grad_norm_last': last_gnorm,
            })

            if log_every:
                au = (
                    f' tr_auc={tr_auc:.4f} va_auc={val_auc:.4f}'
                    if tr_auc is not None and val_auc is not None else ''
                )
                logger.info(
                    '[%s] epoch %d/%d train_loss=%.5f tr_eval=%.5f val_loss=%.5f lr=%.2e%s',
                    desc, epoch + 1, max_epochs,
                    train_loss_batch, tr_eval_loss, val_loss, lr, au,
                )

            if tr_auc is not None and val_auc is not None:
                if abs(tr_auc - 0.5) < flat_eps and abs(val_auc - 0.5) < flat_eps:
                    flat_streak += 1
                else:
                    flat_streak = 0
                if flat_streak >= flat_n and not flat_warned:
                    logger.warning(
                        '[%s] Train and val AUC near 0.5 for %d consecutive epochs (no discrimination).',
                        desc, flat_streak,
                    )
                    flat_warned = True

            if tr_eval_loss > 1e-12 and val_loss > of_ratio * tr_eval_loss:
                of_streak += 1
            else:
                of_streak = 0
            if of_streak >= of_n and not of_warned:
                logger.warning(
                    '[%s] Val loss >> train eval loss for %d epochs (possible overfitting).',
                    desc, of_streak,
                )
                of_warned = True

            # Checkpoint on val loss (proper scoring rule) rather than val AUC.
            # Val AUC rewards rank order only; val loss rewards calibrated spreads.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_auc_at_ckpt = val_auc if val_auc is not None else float('-inf')
                best_epoch = epoch + 1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    stop_reason = 'early_stop_patience'
                    pbar.set_postfix({
                        'val_loss': f'{val_loss:.4f}',
                        'best_val_loss': f'{best_val_loss:.4f}',
                        'status': 'early stop',
                    })
                    break

            pbar.set_postfix({
                'val_loss': f'{val_loss:.4f}',
                'best_val_loss': f'{best_val_loss:.4f}',
            })

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epoch + 1

    model.load_state_dict(best_state)
    model.to(device)

    print(
        f'  [{desc}] stopped: {stop_reason} | best_epoch={best_epoch} '
        f'| best_val_loss={best_val_loss:.4f} | val_auc_at_ckpt={best_val_auc_at_ckpt:.4f}'
    )

    if save_csv and epoch_rows:
        tag = desc.replace(' ', '_').lower()
        fd = fold_idx if fold_idx is not None else 'na'
        out_dir = os.path.join(os.path.dirname(__file__), '..', 'reports', 'training_logs')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f'fold{fd}_{tag}.csv')
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(epoch_rows[0].keys()))
            w.writeheader()
            w.writerows(epoch_rows)
        print(f'  [{desc}] training log: {path}')

    return model


def train_tcn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    seed: int | None = None,
    fold_idx: int | None = None,
    learning_rate: float | None = None,
    optimizer_name: str | None = None,
    batch_size: int | None = None,
    num_channels: list[int] | None = None,
    kernel_size: int | None = None,
    dropout: float | None = None,
    use_weight_norm: bool | None = None,
    max_epochs: int | None = None,
    patience: int | None = None,
) -> nn.Module:
    """
    Train a TCNModel using ReduceLROnPlateau + early stopping on val AUC.
    Falls back to CPU if MPS runs out of memory. Returns the model with the
    best-validation-AUC epoch weights loaded.
    """
    _clear_mps_cache()
    train_seed = config.RANDOM_SEED if seed is None else int(seed)
    _seed_everything(train_seed)

    lr_use = float(learning_rate) if learning_rate is not None else float(config.TCN_LR)
    opt_name = optimizer_name if optimizer_name is not None else str(config.TCN_OPTIMIZER)
    bs = int(batch_size) if batch_size is not None else int(config.TCN_BATCH)
    nc = list(num_channels) if num_channels is not None else list(config.TCN_NUM_CHANNELS)
    ks = int(kernel_size) if kernel_size is not None else int(config.TCN_KERNEL_SIZE)
    dr = float(dropout) if dropout is not None else float(config.TCN_DROPOUT)
    wn = bool(use_weight_norm) if use_weight_norm is not None else bool(config.TCN_USE_WEIGHT_NORM)
    me = int(max_epochs) if max_epochs is not None else int(config.TCN_MAX_EPOCHS)
    pat = int(patience) if patience is not None else int(config.TCN_PATIENCE)
    n_feat = int(X_train.shape[-1])

    def _create(dev: torch.device):
        model = TCNModel(
            input_size=n_feat, num_channels=nc, kernel_size=ks,
            dropout=dr, use_weight_norm=wn,
        ).to(dev)
        optimizer = _build_tcn_optimizer(model, opt_name, lr_use)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=int(config.TCN_LR_PATIENCE),
            factor=float(config.TCN_LR_FACTOR),
        )
        ls = float(getattr(config, 'TCN_LABEL_SMOOTHING', 0.0))
        criterion = nn.CrossEntropyLoss(label_smoothing=ls)
        return model, optimizer, scheduler, criterion

    try:
        model, optimizer, scheduler, criterion = _create(device)
        return _train_tcn_impl(
            model, X_train, y_train, X_val, y_val, device,
            optimizer, criterion, me, pat, bs,
            seed=train_seed, lr_scheduler=scheduler, fold_idx=fold_idx,
        )
    except RuntimeError as e:
        msg = str(e).lower()
        if 'out of memory' in msg or 'mps' in msg:
            print('  [TCN] MPS out of memory, falling back to CPU...')
            _clear_mps_cache()
            cpu_device = torch.device('cpu')
            model, optimizer, scheduler, criterion = _create(cpu_device)
            return _train_tcn_impl(
                model, X_train, y_train, X_val, y_val, cpu_device,
                optimizer, criterion, me, pat, bs,
                seed=train_seed, lr_scheduler=scheduler, fold_idx=fold_idx,
            )
        raise


def predict_tcn(model: nn.Module, X: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    """Batched softmax inference returning P(class=1). Mirrors predict_lstm."""
    model_device = next(model.parameters()).device
    if device is None or device != model_device:
        device = model_device

    model.eval()
    batch_size = 512
    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i + batch_size]).to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())
    if not all_probs:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(all_probs)


# ────────────────────────────────────────────────────────────────────────────
# Two-phase hyperparameter tuning (Phase 1: training hp, Phase 2: arch+feature set)
# ────────────────────────────────────────────────────────────────────────────

def _run_tcn_replicates(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    device: torch.device,
    opt_name: str, lr: float, bs: int,
    num_channels: list[int], kernel_size: int, dropout: float,
    max_epochs: int, patience: int, seed: int,
    n_replicates: int,
) -> list[float]:
    """Run `n_replicates` independent training runs for one hyperparameter tuple;
    return a list of best-val-AUC scores. Mirrors _run_tuning_replicates for LSTM."""
    auc_scores: list[float] = []
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    n_feat = int(X_train.shape[-1])

    for rep in range(n_replicates):
        rep_seed = seed + rep
        _seed_everything(rep_seed)
        train_gen = _make_torch_generator(rep_seed)

        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, generator=train_gen)
        val_dl = DataLoader(val_ds, batch_size=bs * 2, shuffle=False)

        model = TCNModel(
            input_size=n_feat, num_channels=list(num_channels),
            kernel_size=int(kernel_size), dropout=float(dropout),
            use_weight_norm=bool(getattr(config, 'TCN_USE_WEIGHT_NORM', True)),
        ).to(device)
        optimizer = _build_tcn_optimizer(model, opt_name, lr)
        criterion = nn.CrossEntropyLoss()

        best_rep_auc = float('-inf')
        best_state = None
        patience_ctr = 0
        for _ in range(max_epochs):
            model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for xb, yb in val_dl:
                    logits = model(xb.to(device))
                    probs = torch.softmax(logits, dim=1)[:, 1]
                    preds.extend(probs.cpu().numpy())
                    labels.extend(yb.numpy())
            import numpy as _np
            preds_arr = _np.array(preds)
            labels_arr = _np.array(labels)
            valid_mask = _np.isfinite(preds_arr)
            if valid_mask.sum() > 0 and len(set(labels_arr[valid_mask])) > 1:
                ep_auc = roc_auc_score(labels_arr[valid_mask], preds_arr[valid_mask])
            else:
                ep_auc = 0.5

            if ep_auc > best_rep_auc:
                best_rep_auc = ep_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)
        auc_scores.append(best_rep_auc if best_rep_auc > float('-inf') else 0.5)

    return auc_scores


def tune_tcn_hyperparams(
    seqs_by_feature_set: dict[str, dict[str, np.ndarray]],
    device: torch.device,
    arch_grid: dict[str, Any],
    train_grid: dict[str, Any],
    tune_replicates: int,
    tune_patience: int,
    tune_max_epochs: int,
    seed_num_channels: list[int],
    seed_kernel_size: int,
    seed_dropout: float,
    seed_feature_set: str = 'full',
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Two-phase tuning for TCN, paralleling tune_lstm_hyperparams.

    Phase 1: (optimizer, learning_rate, batch_size) on `seed_feature_set`'s
             sequences with the seed architecture.
    Phase 2: (num_channels, kernel_size, dropout, feature_set) using Phase 1's
             best training hp. Feature-set selection rebuilds loaders from the
             corresponding pre-built sequence arrays in `seqs_by_feature_set`.

    `seqs_by_feature_set` has the shape:
        {
          'core': {'X_tr': ..., 'y_tr': ..., 'X_val': ..., 'y_val': ...},
          'full': {'X_tr': ..., 'y_tr': ..., 'X_val': ..., 'y_val': ...},
        }

    Returns a dict with keys:
        optimizer, lr, batch_size, num_channels, kernel_size, dropout, feature_set
    """
    if seed is None:
        seed = config.RANDOM_SEED

    if seed_feature_set not in seqs_by_feature_set:
        raise ValueError(
            f'seed_feature_set={seed_feature_set!r} not in {list(seqs_by_feature_set)}'
        )

    seed_block = seqs_by_feature_set[seed_feature_set]
    X_tr, y_tr = seed_block['X_tr'], seed_block['y_tr']
    X_val, y_val = seed_block['X_val'], seed_block['y_val']

    # ── Phase 1: training-hp sweep on the seed feature set ──────────────────
    combos = list(itertools.product(
        train_grid['optimizer'], train_grid['learning_rate'], train_grid['batch_size']
    ))
    print(f'[TCN Tuning - Phase 1] {len(combos)} training combos x {tune_replicates} replicates '
          f'on feature_set={seed_feature_set!r}')

    phase1_results = []
    for opt_name, lr, bs in combos:
        auc_scores = _run_tcn_replicates(
            X_tr, y_tr, X_val, y_val, device,
            opt_name=opt_name, lr=lr, bs=bs,
            num_channels=seed_num_channels,
            kernel_size=seed_kernel_size,
            dropout=seed_dropout,
            max_epochs=tune_max_epochs,
            patience=tune_patience,
            seed=seed, n_replicates=tune_replicates,
        )
        avg_auc = sum(auc_scores) / len(auc_scores)
        phase1_results.append({
            'optimizer': opt_name, 'lr': lr, 'batch_size': bs, 'avg_val_auc': avg_auc,
        })
        print(f'  opt={opt_name:7s}  lr={lr:.4f}  bs={bs:3d}  -> avg AUC={avg_auc:.4f}')

    best_p1 = max(phase1_results, key=lambda x: x['avg_val_auc'])
    print(f'[TCN Phase 1 best] {best_p1}')

    # ── Phase 2: architecture + feature-set sweep ──────────────────────────
    arch_combos = list(itertools.product(
        arch_grid['num_channels'],
        arch_grid['kernel_size'],
        arch_grid['dropout'],
        arch_grid.get('feature_set', [seed_feature_set]),
    ))
    print(f'\n[TCN Tuning - Phase 2] {len(arch_combos)} arch+feature combos x '
          f'{tune_replicates} replicates')

    phase2_results = []
    for num_channels, kernel_size, dropout, feature_set in arch_combos:
        if feature_set not in seqs_by_feature_set:
            print(f'  [skip] feature_set={feature_set!r} not provided')
            continue
        blk = seqs_by_feature_set[feature_set]
        auc_scores = _run_tcn_replicates(
            blk['X_tr'], blk['y_tr'], blk['X_val'], blk['y_val'], device,
            opt_name=best_p1['optimizer'], lr=best_p1['lr'], bs=best_p1['batch_size'],
            num_channels=list(num_channels),
            kernel_size=int(kernel_size),
            dropout=float(dropout),
            max_epochs=tune_max_epochs,
            patience=tune_patience,
            seed=seed + 10_000,
            n_replicates=tune_replicates,
        )
        avg_auc = sum(auc_scores) / len(auc_scores)
        phase2_results.append({
            'num_channels': list(num_channels),
            'kernel_size': int(kernel_size),
            'dropout': float(dropout),
            'feature_set': feature_set,
            'avg_val_auc': avg_auc,
        })
        print(
            f'  ch={num_channels}  k={kernel_size}  drop={dropout:.2f}  '
            f'feat={feature_set:<4}  -> avg AUC={avg_auc:.4f}'
        )

    best_p2 = max(phase2_results, key=lambda x: x['avg_val_auc'])
    print(f'[TCN Phase 2 best] {best_p2}')

    return {
        'optimizer': best_p1['optimizer'],
        'lr': best_p1['lr'],
        'batch_size': best_p1['batch_size'],
        'num_channels': best_p2['num_channels'],
        'kernel_size': best_p2['kernel_size'],
        'dropout': best_p2['dropout'],
        'feature_set': best_p2['feature_set'],
    }
