# LSTM Refactor Plan — Track A & Track B
**Guide for GitHub Copilot agent — implement step by step, pause for review after each step.**

---

## Context & Motivation

The current LSTM model feeds 52 features per timestep into the network. This is the root cause of poor performance. Fischer & Krauss (2017) — the paper this project replicates — use **exactly 1 feature** (the standardized 1-day return) over a **240-timestep sequence**. The model is designed to discover temporal patterns endogenously from raw return sequences, not to process a wide tabular feature matrix.

This plan implements two independent LSTM variants alongside the existing baseline models (LR, RF, XGBoost), removes all ensemble models, and produces clean before/after transaction cost results for every model.

---

## Scope of Changes

| What changes | What stays the same |
|---|---|
| LSTM split into LSTM-A and LSTM-B | LR, RF, XGBoost pipelines untouched |
| All ensemble models removed | Walk-forward fold structure unchanged |
| `config.py` gains two LSTM param blocks | Anti-leakage / standardization logic unchanged |
| New feature pipeline for LSTM-A (1 feature) | Binary cross-sectional median target unchanged |
| New feature pipeline for LSTM-B (6 features) | Portfolio construction (long/short top-k) unchanged |
| Results saved to `/reports` | Transaction cost grid (0 bps to 10 bps) unchanged |

---

## Models After Refactor

| Model ID | Description |
|---|---|
| `LR` | Logistic Regression (baseline, unchanged) |
| `RF` | Random Forest (unchanged) |
| `XGBoost` | XGBoost (unchanged) |
| `LSTM-A` | Paper-faithful replication — 1 feature, seq_len=240 |
| `LSTM-B` | Extended ablation — 6 curated features, seq_len=60 |

---

## Step 1 — Update `config.py`

**Replace** the single LSTM parameter block with two independent blocks. Do not remove existing LR, RF, or XGBoost parameters.

```python
# ── LSTM-A: Paper-faithful replication (Fischer & Krauss 2017) ─────────────
LSTM_A_FEATURES      = ['Return_1d']          # single feature, as per paper
LSTM_A_SEQ_LEN       = 240                    # ~1 trading year of history
LSTM_A_HIDDEN        = 25                     # matches paper's h=25
LSTM_A_LAYERS        = 1                      # single LSTM layer, as per paper
LSTM_A_DROPOUT       = 0.16                   # matches paper's dropout value
LSTM_A_OPTIMIZER     = 'rmsprop'              # paper explicitly uses RMSprop
LSTM_A_LR            = 0.001
LSTM_A_BATCH         = 512                    # large batch for stable gradients on 1 feature
LSTM_A_MAX_EPOCHS    = 1000                   # paper trains up to 1000 epochs
LSTM_A_PATIENCE      = 10                     # paper uses patience=10
LSTM_A_VAL_SPLIT     = 0.2                    # paper uses 80/20 train/val split

# ── LSTM-B: Extended ablation — curated 6-feature set ────────────────────
LSTM_B_FEATURES      = [
    'Return_1d',        # core price signal
    'RSI_14',           # bounded momentum [0, 100]
    'BB_PctB',          # position within Bollinger band
    'RealVol_20d',      # realized volatility regime
    'Volume_Ratio',     # relative volume anomaly
    'SectorRelReturn',  # cross-sectional sector context
]
LSTM_B_SEQ_LEN       = 60                     # shorter window; justified by wider feature set
LSTM_B_HIDDEN        = 64
LSTM_B_LAYERS        = 2
LSTM_B_DROPOUT       = 0.2
LSTM_B_OPTIMIZER     = 'adam'
LSTM_B_LR            = 0.001
LSTM_B_BATCH         = 128
LSTM_B_MAX_EPOCHS    = 300
LSTM_B_PATIENCE      = 15
LSTM_B_LR_PATIENCE   = 7
LSTM_B_LR_FACTOR     = 0.5
LSTM_B_VAL_SPLIT     = 0.2

# ── Shared LSTM settings ──────────────────────────────────────────────────
LSTM_WD              = 1e-5                   # weight decay (both models)

# ── Remove or comment out old single-LSTM block ───────────────────────────
# LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT, etc. — DELETE these
```

**Checkpoint:** Verify `config.py` has two clean LSTM blocks and no orphaned old keys before proceeding.

---

## Step 2 — Build Independent Feature Pipelines

The key constraint: **standardization must be fit on training data only.** This applies to both LSTM variants.

### 2a — LSTM-A feature preparation

LSTM-A uses only `Return_1d`. The standardization uses the mean and standard deviation computed from the training fold only, then applied to both train and test sequences.

```python
def prepare_lstm_a_sequences(df_train, df_test, config):
    """
    Builds overlapping sequences of standardized 1-day returns.
    Replicates Fischer & Krauss (2017) Section 3.2.1 exactly.

    Args:
        df_train: DataFrame with columns [Date, Ticker, Return_1d], training fold only
        df_test:  DataFrame with same columns, test fold only
        config:   config module

    Returns:
        X_train: np.array shape (N_train, seq_len, 1)
        y_train: np.array shape (N_train,)
        X_test:  np.array shape (N_test, seq_len, 1)
        y_test:  np.array shape (N_test,)
    """
    seq_len = config.LSTM_A_SEQ_LEN

    # Fit scaler on training data ONLY
    mu    = df_train['Return_1d'].mean()
    sigma = df_train['Return_1d'].std()

    # Standardize both splits using training statistics
    df_train = df_train.copy()
    df_test  = df_test.copy()
    df_train['r_std'] = (df_train['Return_1d'] - mu) / sigma
    df_test['r_std']  = (df_test['Return_1d']  - mu) / sigma

    # Build overlapping sequences per stock, sorted by Date ascending
    # For each stock s, and each date t >= seq_len,
    # sequence = [r_std_{t-seq_len+1}, ..., r_std_t]
    # target   = binary label at t+1
    # Implementation note: use a sliding window per ticker group.
    # Sequences should NOT cross stock boundaries.

    X_train, y_train = _build_sequences(df_train, seq_len, feature_col='r_std')
    X_test,  y_test  = _build_sequences(df_test,  seq_len, feature_col='r_std')

    return X_train, y_train, X_test, y_test


def _build_sequences(df, seq_len, feature_col):
    """
    Constructs overlapping sequences per ticker.
    df must have columns: [Date, Ticker, {feature_col}, Target]
    sorted by [Ticker, Date] ascending.
    """
    X_list, y_list = [], []
    for ticker, grp in df.sort_values(['Ticker', 'Date']).groupby('Ticker'):
        vals   = grp[feature_col].values        # shape (T,)
        labels = grp['Target'].values           # shape (T,)
        for i in range(seq_len, len(vals)):
            X_list.append(vals[i - seq_len:i])
            y_list.append(labels[i])
    X = np.array(X_list)[:, :, np.newaxis]     # shape (N, seq_len, 1)
    y = np.array(y_list)
    return X, y
```

### 2b — LSTM-B feature preparation

LSTM-B uses 6 features. A single `StandardScaler` is fit on the training fold for all 6 columns simultaneously.

```python
def prepare_lstm_b_sequences(df_train, df_test, config):
    """
    Builds overlapping multi-feature sequences.
    Scaler is fit on training fold only and applied to both splits.
    """
    from sklearn.preprocessing import StandardScaler

    seq_len      = config.LSTM_B_SEQ_LEN
    feature_cols = config.LSTM_B_FEATURES

    # Fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols].values)

    df_train = df_train.copy()
    df_test  = df_test.copy()
    df_train[feature_cols] = scaler.transform(df_train[feature_cols].values)
    df_test[feature_cols]  = scaler.transform(df_test[feature_cols].values)

    X_train, y_train = _build_sequences_multi(df_train, seq_len, feature_cols)
    X_test,  y_test  = _build_sequences_multi(df_test,  seq_len, feature_cols)

    return X_train, y_train, X_test, y_test


def _build_sequences_multi(df, seq_len, feature_cols):
    """
    Constructs overlapping multi-feature sequences per ticker.
    """
    X_list, y_list = [], []
    n_feat = len(feature_cols)
    for ticker, grp in df.sort_values(['Ticker', 'Date']).groupby('Ticker'):
        vals   = grp[feature_cols].values       # shape (T, n_feat)
        labels = grp['Target'].values
        for i in range(seq_len, len(vals)):
            X_list.append(vals[i - seq_len:i])  # shape (seq_len, n_feat)
            y_list.append(labels[i])
    X = np.array(X_list)                        # shape (N, seq_len, n_feat)
    y = np.array(y_list)
    return X, y
```

**Checkpoint:** Print shape of `X_train` and `X_test` for one fold before building models. Expected:
- LSTM-A: `(N, 240, 1)`
- LSTM-B: `(N, 60, 6)`

---

## Step 3 — Build LSTM Model Factories

Create two separate model-building functions. Do not share a single function with conditional branching — keep them fully independent to avoid config bleed.

### 3a — LSTM-A model

```python
import torch
import torch.nn as nn

class LSTMModelA(nn.Module):
    """
    Paper-faithful LSTM (Fischer & Krauss 2017).
    Single layer, h=25, input_size=1.
    """
    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = 1,
            hidden_size = config.LSTM_A_HIDDEN,
            num_layers  = config.LSTM_A_LAYERS,
            dropout     = 0.0,              # no recurrent dropout for single layer
            batch_first = True,
        )
        self.dropout = nn.Dropout(config.LSTM_A_DROPOUT)
        self.fc      = nn.Linear(config.LSTM_A_HIDDEN, 2)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        out     = self.dropout(out[:, -1, :])   # take last timestep output
        return self.fc(out)                     # logits for 2 classes


def train_lstm_a(X_train, y_train, X_val, y_val, config, device):
    """
    Trains LSTM-A using RMSprop with early stopping.
    Returns trained model with best validation loss weights restored.
    """
    model = LSTMModelA(config).to(device)
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr           = config.LSTM_A_LR,
        weight_decay = config.LSTM_WD,
    )
    criterion = nn.CrossEntropyLoss()

    # Convert to tensors
    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.LongTensor(y_train).to(device)
    X_v  = torch.FloatTensor(X_val).to(device)
    y_v  = torch.LongTensor(y_val).to(device)

    dataset    = torch.utils.data.TensorDataset(X_tr, y_tr)
    loader     = torch.utils.data.DataLoader(
        dataset, batch_size=config.LSTM_A_BATCH, shuffle=True
    )

    best_val_loss = float('inf')
    best_state    = None
    patience_ctr  = 0

    for epoch in range(config.LSTM_A_MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), y_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= config.LSTM_A_PATIENCE:
                break

    model.load_state_dict(best_state)
    return model
```

### 3b — LSTM-B model

```python
class LSTMModelB(nn.Module):
    """
    Extended LSTM with 6 input features, 2 layers, LR scheduler.
    """
    def __init__(self, config):
        super().__init__()
        n_feat = len(config.LSTM_B_FEATURES)
        self.lstm = nn.LSTM(
            input_size  = n_feat,
            hidden_size = config.LSTM_B_HIDDEN,
            num_layers  = config.LSTM_B_LAYERS,
            dropout     = config.LSTM_B_DROPOUT if config.LSTM_B_LAYERS > 1 else 0.0,
            batch_first = True,
        )
        self.dropout = nn.Dropout(config.LSTM_B_DROPOUT)
        self.fc      = nn.Linear(config.LSTM_B_HIDDEN, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out     = self.dropout(out[:, -1, :])
        return self.fc(out)


def train_lstm_b(X_train, y_train, X_val, y_val, config, device):
    """
    Trains LSTM-B using Adam with ReduceLROnPlateau scheduler.
    Returns trained model with best validation loss weights restored.
    """
    model = LSTMModelB(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = config.LSTM_B_LR,
        weight_decay = config.LSTM_WD,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience = config.LSTM_B_LR_PATIENCE,
        factor   = config.LSTM_B_LR_FACTOR,
    )
    criterion = nn.CrossEntropyLoss()

    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.LongTensor(y_train).to(device)
    X_v  = torch.FloatTensor(X_val).to(device)
    y_v  = torch.LongTensor(y_val).to(device)

    dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=config.LSTM_B_BATCH, shuffle=True
    )

    best_val_loss = float('inf')
    best_state    = None
    patience_ctr  = 0

    for epoch in range(config.LSTM_B_MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), y_v).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= config.LSTM_B_PATIENCE:
                break

    model.load_state_dict(best_state)
    return model
```

**Checkpoint:** Confirm both models compile and forward-pass on a dummy tensor before plugging into the walk-forward loop.

---

## Step 4 — Integrate Both LSTMs into the Walk-Forward Loop

The walk-forward loop must train and predict all 5 models per fold. The existing LR, RF, XGBoost logic remains untouched. Remove all ensemble signal generation.

```python
MODELS = ['LR', 'RF', 'XGBoost', 'LSTM-A', 'LSTM-B']

# Structure to collect per-fold predictions
all_preds = {m: [] for m in MODELS}   # list of (date, ticker, prob_class1)

for fold_idx, (train_idx, test_idx) in enumerate(walk_forward_folds):
    print(f"\n=== Fold {fold_idx + 1} ===")

    df_train_fold = full_df.iloc[train_idx]
    df_test_fold  = full_df.iloc[test_idx]

    # ── Baseline models (unchanged logic) ────────────────────────────────
    for model_name in ['LR', 'RF', 'XGBoost']:
        # existing feature prep, fit, predict_proba
        # append (date, ticker, prob) tuples to all_preds[model_name]
        ...

    # ── LSTM-A ───────────────────────────────────────────────────────────
    X_tr_a, y_tr_a, X_te_a, y_te_a = prepare_lstm_a_sequences(
        df_train_fold, df_test_fold, config
    )
    # 80/20 split of training data into train/val
    val_split = int(len(X_tr_a) * (1 - config.LSTM_A_VAL_SPLIT))
    model_a = train_lstm_a(
        X_tr_a[:val_split], y_tr_a[:val_split],
        X_tr_a[val_split:], y_tr_a[val_split:],
        config, device
    )
    model_a.eval()
    with torch.no_grad():
        logits_a = model_a(torch.FloatTensor(X_te_a).to(device))
        probs_a  = torch.softmax(logits_a, dim=1)[:, 1].cpu().numpy()
    # store probs_a with corresponding (date, ticker) from df_test_fold
    # NOTE: align carefully — sequences start at seq_len-th row per ticker
    all_preds['LSTM-A'].extend(zip(test_dates_a, test_tickers_a, probs_a))

    # ── LSTM-B ───────────────────────────────────────────────────────────
    X_tr_b, y_tr_b, X_te_b, y_te_b = prepare_lstm_b_sequences(
        df_train_fold, df_test_fold, config
    )
    val_split = int(len(X_tr_b) * (1 - config.LSTM_B_VAL_SPLIT))
    model_b = train_lstm_b(
        X_tr_b[:val_split], y_tr_b[:val_split],
        X_tr_b[val_split:], y_tr_b[val_split:],
        config, device
    )
    model_b.eval()
    with torch.no_grad():
        logits_b = model_b(torch.FloatTensor(X_te_b).to(device))
        probs_b  = torch.softmax(logits_b, dim=1)[:, 1].cpu().numpy()
    all_preds['LSTM-B'].extend(zip(test_dates_b, test_tickers_b, probs_b))
```

> **Important — sequence alignment:** When you build sequences from `df_test_fold`, the first `seq_len` rows per ticker have no valid sequence and are skipped. The `test_dates` and `test_tickers` arrays used for alignment must be drawn from the same rows that actually produce sequences — i.e., starting at index `seq_len` per ticker group, not from the beginning of the test fold. Misalignment here causes silent return attribution errors.

**Checkpoint:** After one fold, assert that `len(probs_a) == len(test_dates_a)` and that the date range of predictions matches the expected test fold window.

---

## Step 5 — Remove All Ensemble Models

Search the codebase for every reference to ensemble logic and delete it cleanly.

**Files/sections to remove:**

- Any function named `build_ensemble_signal`, `selective_ensemble`, `equal_weight_ensemble`, or similar
- Any loop or block that iterates over `['Ensemble (equal)', 'Ensemble (selective)']`
- Any `ensemble_weights`, `selective_threshold`, or ensemble-specific config keys
- Any plotting or reporting code that references ensemble model names

**After deletion, the only model names in the codebase must be:**
```python
MODELS = ['LR', 'RF', 'XGBoost', 'LSTM-A', 'LSTM-B']
```

**Checkpoint:** `grep -r "ensemble\|Ensemble\|selective" .` should return zero results (excluding this plan file and comments).

---

## Step 6 — Portfolio Construction (unchanged logic, apply to all 5 models)

The existing long/short portfolio construction is applied identically to each model. No changes needed to this logic — just confirm it loops over `MODELS` rather than a hardcoded list.

For each trading day `t`:
1. Rank all stocks by predicted probability of outperforming the median (descending)
2. Go long the top-`k` stocks, short the bottom-`k` stocks
3. Each leg is equal-weighted; portfolio is dollar-neutral
4. Hold for one day, compute next-day realized return

The `k` parameter comes from `config.TOP_K` (keep existing value).

---

## Step 7 — Compute Before and After Transaction Cost Returns

Both gross (before TC) and net (after TC) returns must be computed and stored for every model.

```python
def apply_transaction_costs(daily_returns_series, signals_df, tc_bps):
    """
    Deducts transaction costs on days where the portfolio changes.

    Args:
        daily_returns_series: pd.Series indexed by date, gross portfolio return
        signals_df: pd.DataFrame with columns [Date, Ticker, Signal]
                    where Signal ∈ {1 (long), -1 (short), 0 (hold)}
        tc_bps: transaction cost in basis points (e.g., 5 for 5 bps)

    Returns:
        pd.Series of net daily returns
    """
    tc_rate = tc_bps / 10_000

    # Compute daily turnover: fraction of portfolio that changes each day
    # A simple proxy: count positions that differ from previous day / total positions
    turnover = signals_df.groupby('Date').apply(_compute_turnover)

    net_returns = daily_returns_series - turnover * tc_rate
    return net_returns


def _compute_turnover(group):
    # Implement based on your existing signal structure
    # Returns a scalar fraction in [0, 1]
    ...
```

**TC values to report:**
- `0 bps` — gross return (before TC)
- `5 bps` — primary net return (after TC, main thesis result)
- `10 bps` — sensitivity check

Store all three for every model.

---

## Step 8 — Compute Performance Metrics

Apply the metrics function to both gross and net return series for all 5 models.

```python
def compute_metrics(returns_series, label):
    """
    Computes the full metrics suite for a daily returns series.

    Args:
        returns_series: pd.Series of daily portfolio returns
        label: string identifier for this model+TC combination

    Returns:
        dict with keys matching thesis Table T5 and T8 columns
    """
    ann_factor = 252

    sharpe     = (returns_series.mean() / returns_series.std()) * np.sqrt(ann_factor)
    sortino    = (returns_series.mean() / returns_series[returns_series < 0].std()) * np.sqrt(ann_factor)
    ann_ret    = (1 + returns_series).prod() ** (ann_factor / len(returns_series)) - 1
    ann_std    = returns_series.std() * np.sqrt(ann_factor)
    cum        = (1 + returns_series).cumprod()
    mdd        = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar     = ann_ret / abs(mdd) if mdd != 0 else np.nan
    win_rate   = (returns_series > 0).mean()
    var_1pct   = returns_series.quantile(0.01)

    return {
        'Model':                  label,
        'Sharpe Ratio':           round(sharpe, 3),
        'Sortino Ratio':          round(sortino, 3),
        'Annualized Return (%)':  round(ann_ret * 100, 2),
        'Annualized Std Dev (%)': round(ann_std * 100, 2),
        'Max Drawdown (%)':       round(mdd * 100, 2),
        'Calmar Ratio':           round(calmar, 3),
        'Win Rate (%)':           round(win_rate * 100, 2),
        'VaR 1% (%)':             round(var_1pct * 100, 4),
    }
```

---

## Step 9 — Save All Results to `/reports`

Create the `/reports` directory if it does not exist. Save all outputs at the end of the backtest run in a single reporting function.

### Directory structure after this step:
```
reports/
├── table_T5_gross_returns.csv          # before TC (0 bps)
├── table_T5_net_returns_5bps.csv       # after TC (5 bps)  ← primary result
├── table_T5_net_returns_10bps.csv      # after TC (10 bps)
├── table_T8_classification_metrics.csv
├── table_T6_subperiod_performance.csv
├── daily_returns_gross.csv             # raw daily returns per model, before TC
├── daily_returns_net_5bps.csv          # raw daily returns per model, after TC
├── signals_all_models.csv              # full signal log
└── backtest_summary.txt                # human-readable console-style summary
```

### Reporting function:

```python
import os
import pandas as pd

def save_all_results(results_dict, daily_returns_dict, signals_dict, reports_dir='reports'):
    """
    Saves all backtest results to the /reports folder.

    Args:
        results_dict: {
            'gross':   list of metric dicts (one per model),
            'net_5':   list of metric dicts,
            'net_10':  list of metric dicts,
            'classification': list of classification metric dicts,
            'subperiod': list of sub-period metric dicts,
        }
        daily_returns_dict: {
            'gross':  pd.DataFrame columns=[Date, LR, RF, XGBoost, LSTM-A, LSTM-B],
            'net_5':  pd.DataFrame same columns,
        }
        signals_dict: pd.DataFrame with all signals
        reports_dir: output directory path
    """
    os.makedirs(reports_dir, exist_ok=True)

    # ── Table T5: Risk-Return Metrics ─────────────────────────────────────
    pd.DataFrame(results_dict['gross']).to_csv(
        f'{reports_dir}/table_T5_gross_returns.csv', index=False
    )
    pd.DataFrame(results_dict['net_5']).to_csv(
        f'{reports_dir}/table_T5_net_returns_5bps.csv', index=False
    )
    pd.DataFrame(results_dict['net_10']).to_csv(
        f'{reports_dir}/table_T5_net_returns_10bps.csv', index=False
    )

    # ── Table T8: Classification Metrics ─────────────────────────────────
    pd.DataFrame(results_dict['classification']).to_csv(
        f'{reports_dir}/table_T8_classification_metrics.csv', index=False
    )

    # ── Table T6: Sub-Period Performance ─────────────────────────────────
    pd.DataFrame(results_dict['subperiod']).to_csv(
        f'{reports_dir}/table_T6_subperiod_performance.csv', index=False
    )

    # ── Raw daily returns ─────────────────────────────────────────────────
    daily_returns_dict['gross'].to_csv(
        f'{reports_dir}/daily_returns_gross.csv', index=False
    )
    daily_returns_dict['net_5'].to_csv(
        f'{reports_dir}/daily_returns_net_5bps.csv', index=False
    )

    # ── Signals ───────────────────────────────────────────────────────────
    signals_dict.to_csv(f'{reports_dir}/signals_all_models.csv', index=False)

    # ── Human-readable summary ────────────────────────────────────────────
    with open(f'{reports_dir}/backtest_summary.txt', 'w') as f:
        f.write(_format_summary(results_dict))

    print(f"\nAll results saved to /{reports_dir}/")


def _format_summary(results_dict):
    lines = []
    lines.append("=" * 60)
    lines.append("BACKTEST RESULTS SUMMARY")
    lines.append("=" * 60)

    for label, key in [
        ("GROSS RETURNS (0 bps TC)", 'gross'),
        ("NET RETURNS  (5 bps TC)",  'net_5'),
        ("NET RETURNS (10 bps TC)",  'net_10'),
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
```

**Checkpoint:** After the run, confirm all 9 files exist in `/reports` and that `backtest_summary.txt` contains results for all 5 models across all 3 TC levels.

---

## Step 10 — Console Output During Run

The main script should print progress in this format so the run is monitorable:

```
============================================================
BACKTEST — 5 MODELS × N FOLDS
============================================================

=== Fold 1/N ===
  [LR]      fit done
  [RF]      fit done
  [XGBoost] fit done
  [LSTM-A]  epoch 47/1000 — early stop | val_loss=0.6923
  [LSTM-B]  epoch 112/300 — early stop | val_loss=0.6918

=== Fold 2/N ===
  ...

============================================================
RESULTS — GROSS (0 bps)
============================================================
  LR          Sharpe= ...  Ann.Ret= ...%  MDD= ...%
  RF          Sharpe= ...  Ann.Ret= ...%  MDD= ...%
  XGBoost     Sharpe= ...  Ann.Ret= ...%  MDD= ...%
  LSTM-A      Sharpe= ...  Ann.Ret= ...%  MDD= ...%
  LSTM-B      Sharpe= ...  Ann.Ret= ...%  MDD= ...%

============================================================
RESULTS — NET (5 bps)
============================================================
  LR          Sharpe= ...  Ann.Ret= ...%  MDD= ...%
  RF          Sharpe= ...  Ann.Ret= ...%  MDD= ...%
  XGBoost     Sharpe= ...  Ann.Ret= ...%  MDD= ...%
  LSTM-A      Sharpe= ...  Ann.Ret= ...%  MDD= ...%
  LSTM-B      Sharpe= ...  Ann.Ret= ...%  MDD= ...%

All results saved to /reports/
```

---

## Anti-Leakage Checklist

Before running the full backtest, verify every item below. These are non-negotiable for academic validity.

- [ ] `StandardScaler` (LSTM-B) is fit only on `df_train_fold`, never on `df_test_fold`
- [ ] LSTM-A mean/std are computed from `df_train_fold['Return_1d']` only
- [ ] Validation split (80/20) is taken from the tail of the training sequences — do not shuffle time-series data
- [ ] Sequence construction never pulls rows from the test fold into a training sequence
- [ ] Target at date `t` uses only returns realized at `t+1`, not any future data
- [ ] `SectorRelReturn` in LSTM-B features is computed using cross-sectional data from `t` only (not future dates)
- [ ] No fitted objects (scaler, model weights) are carried from one fold to the next — fresh fit every fold
- [ ] The `walk_forward_folds` index generator uses non-overlapping test windows

---

## Parameter Reference Table

| Parameter | LSTM-A | LSTM-B | Rationale |
|---|---|---|---|
| Features | `Return_1d` (1) | 6 curated | Paper vs. extension |
| Sequence length | 240 | 60 | 1yr history vs. 3-month |
| Input size | 1 | 6 | Matches feature count |
| Hidden units | 25 | 64 | Paper spec vs. wider capacity |
| Layers | 1 | 2 | Paper spec vs. depth for 6 features |
| Dropout | 0.16 | 0.20 | Paper spec vs. standard |
| Optimizer | RMSprop | Adam | Paper spec vs. standard for multi-feature |
| Learning rate | 0.001 | 0.001 | Both |
| Batch size | 512 | 128 | Large for 1-feature stability |
| Max epochs | 1000 | 300 | Paper spec vs. practical budget |
| Early stop patience | 10 | 15 | Paper spec vs. LR scheduler lag |
| Val split | 20% | 20% | Paper spec — both |

---

## Notes for Copilot

1. Implement one step at a time. Do not skip ahead.
2. After each step marked with **Checkpoint**, stop and print/assert the stated condition before continuing.
3. Do not modify the LR, RF, or XGBoost training logic. Only integrate their predictions into the unified output format.
4. The sequence alignment note in Step 4 is the most error-prone part of this implementation — read it twice before coding.
5. Both LSTM models use `device = torch.device('mps')` on the MacBook Air M4. Add a fallback: `device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')`.
6. Save every model's raw daily return series (not just summary metrics) — the equity curve plots and sub-period analysis are generated from the raw series.
