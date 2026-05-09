# LSTM and TCN — Complete Implementation Details
> Accurate reference for thesis writing. All values sourced directly from `config.py`, `models/lstm_model.py`, `models/tcn_model.py`, `experiments/lstm_lr_sweep.py`, and `experiments/tcn_arch_sweep.py`.

---

## 1. Shared Infrastructure

Both LSTM and TCN share the same training infrastructure to ensure consistent, comparable behaviour.

### 1.1 Device Selection

Device priority: **CUDA → MPS → CPU**. The pipeline auto-detects at runtime. Both models fall back to CPU automatically if an MPS out-of-memory error occurs. TCN additionally exposes a `TCN_FORCE_CPU` flag (currently `False`) to force CPU training for full reproducibility, since MPS `Conv1d` is non-deterministic even with fixed seeds.

### 1.2 Reproducibility

All random number generators are seeded before each training run:
- Python `random`, `numpy.random`, `torch.manual_seed`
- `torch.cuda.manual_seed_all` (if CUDA available)
- `torch.mps.manual_seed` (if MPS available)
- A seeded `torch.Generator` is passed to each `DataLoader` for deterministic batch shuffling.

Per-fold seed: `fold_seed_base = RANDOM_SEED + fold * 1000` (base seed = 42).

### 1.3 Sequence Construction

Overlapping sliding-window sequences are built independently per ticker:

- For each ticker, the feature matrix is slid with a window of length `SEQ_LEN = 30` trading days.
- Each sequence `X[i]` is the feature window `features[i-30 : i]` (shape `(30, n_features)`), and the label `y[i]` is the binary target at day `i`.
- Keys `(date, ticker)` are stored alongside each sequence for alignment back to the original DataFrame.
- **Test sequences** use training data as lookback history (trailing window crosses the fold boundary) so that the first `seq_len` days of each test fold are not lost.

### 1.4 Temporal Train/Validation Split

Both models split the training fold temporally by **date**, not by row index or ticker:

1. Training dates are sorted chronologically.
2. The last 20% of unique dates (`val_ratio = 0.2`) become the validation set.
3. The `StandardScaler` is fitted on the **true training dates only** (first 80%), then applied to both train and val feature matrices.
4. This prevents any future information leaking into the scaler parameters.

---

## 2. LSTM Model

### 2.1 Architecture

| Parameter | Value |
|---|---|
| Model class | `LSTMModelB` (`nn.LSTM` + linear decoder) |
| Hidden size | 32 |
| Number of layers | 1 |
| Dropout | 0.0 (PyTorch ignores dropout for single-layer LSTM) |
| Output | 2-class logits (`nn.Linear(hidden_size, 2)`) |
| Sequence length | 30 trading days |
| Input features | Core + market + sector (23 features total) |

Forward pass: the LSTM processes the full sequence `(batch, seq_len, n_features)`; only the **last timestep** of the hidden state is extracted, passed through dropout (0.0 here), then through the linear decoder to produce 2-class logits.

**Feature set used (LSTM):**
- Core features (8): `Return_1d`, `Return_5d`, `Return_21d`, `RSI_14`, `BB_PctB`, `RealVol_20d`, `Volume_Ratio`, `SectorRelReturn`
- Market features (9): `Market_Return_1d/5d/21d`, `Market_Vol_20d/60d`, `RelToMarket_1d/5d/21d`, `Beta_60d`
- Sector features (6): `Sector_Return_1d/5d/21d`, `Sector_Vol_20d/60d`, `SectorRelZ_Return_1d`

### 2.2 Training Configuration

| Parameter | Value |
|---|---|
| Loss function | `nn.CrossEntropyLoss` |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 256 |
| Max epochs | 200 |
| Early stopping patience | 15 epochs |
| LR scheduler | `ReduceLROnPlateau(patience=7, factor=0.5)` |
| Scheduler trigger | Validation loss |
| Weight decay | 1e-4 |
| Gradient clipping | `clip_grad_norm_(max_norm=1.0)` |

### 2.3 Early Stopping and Checkpointing

The LSTM checkpoints on **validation AUC** (best val AUC epoch weights are restored). Training halts when validation AUC fails to improve for 15 consecutive epochs (`LSTM_B_PATIENCE = 15`).

### 2.4 Diagnostic Monitoring

Each epoch logs:
- Batch train loss and eval train loss
- Validation loss
- Train and validation AUC
- Current learning rate
- Last gradient norm

Two heuristic warnings are emitted during training:
- **Flat AUC warning**: if both train and val AUC remain within 0.02 of 0.5 for 8 consecutive epochs (no discrimination).
- **Overfitting warning**: if val loss > 3× train eval loss for 6 consecutive epochs.

Epoch-level logs are saved to `reports/training_logs/fold{N}_lstm.csv` when `LSTM_SAVE_TRAINING_CSV = True`.

### 2.5 Inference

Softmax is applied to the 2-class logits to produce `P(class=1)`, the probability that the stock outperforms the cross-sectional median over the next 21 days. Inference is batched at 512 sequences per forward pass.

---

## 3. LSTM Hyperparameter Tuning

### 3.1 Tuning Strategy

Tuning runs **on the first fold only** (`LSTM_B_TUNE_ON_FIRST_FOLD_ONLY = True`). The best hyperparameters found are reused across all subsequent folds. This is a deliberate cost/benefit trade-off: full per-fold tuning would multiply compute by the number of folds (~16) with minimal gain given the small data regime.

The tuning grid is a focused subset of the Bhandari §3.3 grid, scaled for the 50-stock universe:

| Dimension | Grid |
|---|---|
| Optimizer | `['adam', 'nadam']` |
| Learning rate | `[0.0003, 0.001, 0.003]` |
| Batch size | `[64, 128]` |

This produces 12 combinations (2 × 3 × 2).

Architecture is **fixed** during LSTM tuning (unlike TCN, which also tunes architecture in Phase 2). The fixed architecture during tuning uses the production defaults: hidden=32, layers=1, dropout=0.0.

### 3.2 Selection Criterion

Selection is by **validation net Sharpe ratio** (not AUC). The full signal pipeline (EMA smoothing → z-score ranking → confidence threshold → holding constraint → portfolio P&L) is run on the validation fold, and the Sharpe ratio of the resulting net returns is used as the selection metric.

**Return-aware guardrail**: the tuned candidate is accepted only if it outperforms the default configuration on validation Sharpe. If not, the defaults are kept.

Tuning parameters:
- Max epochs per candidate: 35 (`LSTM_B_TUNE_MAX_EPOCHS`)
- Early stopping patience during tuning: 4 epochs (`LSTM_B_TUNE_PATIENCE`)
- Replicates per configuration: 1 (`LSTM_B_TUNE_REPLICATES`)

### 3.3 LSTM LR Sweep Experiment (`experiments/lstm_lr_sweep.py`)

A separate diagnostic sweep explores the joint grid of learning rate × hidden size on fold 1:

| Dimension | Grid |
|---|---|
| Learning rate | `[0.0005, 0.001, 0.003, 0.005]` (from `LSTM_LR_GRID`) |
| Hidden size | `[32, 64]` (from `LSTM_B_ARCH_GRID["hidden_size"]`) |

This produces 8 combinations. All other hyperparameters are fixed at their production defaults.

**Budget**: capped at `LSTM_LR_SWEEP_MAX_EPOCHS = 40` epochs per candidate.

**Selection criterion**: validation net Sharpe (same full signal pipeline as main tuning).

**Outputs:**
- `reports/{universe}_lstm_tuning_results.csv` — one row per (lr, hidden_size) with `val_sharpe`, `val_auc`, `val_loss`, `n_epochs_trained`.
- `reports/training_logs/{universe}_lstm_sweep_best_training_log.csv` — epoch-level log for the best configuration.

**Key implementation detail**: sequences are built **once** before the sweep loop and reused across all (lr, hidden_size) combinations. Only the model weights and optimizer change per candidate. This avoids redundant data preprocessing inside the sweep.

---

## 4. TCN Model

### 4.1 Architecture

The TCN follows Bai et al. (2018): a stack of dilated causal residual blocks (`TemporalBlock`), each containing two `Conv1d` layers with exponentially growing dilations.

| Parameter | Value |
|---|---|
| Model class | `TCNModel` |
| Number of levels | 3 |
| Channels per level | `[16, 16, 16]` |
| Kernel size | 3 |
| Dilation at level `i` | `2^i` (dilations: 1, 2, 4) |
| Dropout | 0.2 |
| Weight normalization | `False` (disabled) |
| Output | 2-class logits (`nn.Linear(num_channels[-1], 2)`) |
| Sequence length | 30 trading days |
| Input features | Same as LSTM (core + market + sector, 23 features total) |

**Receptive field formula:**
```
RF = 1 + 2 * (kernel_size - 1) * (2^num_levels - 1)
   = 1 + 2 * (3-1) * (2^3 - 1)
   = 1 + 2 * 2 * 7 = 29
```
The default 3-level, kernel-3 TCN has a receptive field of 29, covering almost exactly the full 30-day sequence.

**Causal convolution**: `Chomp1d` strips the right-side padding introduced by `nn.Conv1d` to ensure each output position depends only on past inputs. This enforces strict causality.

**Residual connection**: each `TemporalBlock` adds a skip connection from input to output. If the input and output channel counts differ, a 1×1 `Conv1d` downsamples the skip connection.

**Weight normalization**: disabled (`TCN_USE_WEIGHT_NORM = False`) because `weight_norm` causes non-deterministic results on Apple MPS and also invalidates Kaiming initialization (the computed `weight` attribute is overwritten by the pre-hook each forward pass, so `kaiming_normal_` applied after `weight_norm` has no effect).

**Input layout**: the model receives `(batch, seq_len, n_features)` (same as LSTM) and transposes internally to `(batch, n_features, seq_len)` for `Conv1d`. The final timestep of the top block's output is extracted and fed to the linear decoder.

**Feature set used (TCN):** Identical to LSTM — core + market + sector (23 features).

### 4.2 Training Configuration

| Parameter | Value |
|---|---|
| Loss function | `nn.CrossEntropyLoss(label_smoothing=0.1)` |
| Label smoothing | 0.1 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 256 |
| Max epochs | 200 |
| Early stopping patience | 15 epochs |
| LR scheduler | `ReduceLROnPlateau(patience=7, factor=0.5)` |
| Scheduler trigger | Validation loss |
| Weight decay | 1e-4 |
| Gradient clipping | `clip_grad_norm_(max_norm=1.0)` |

**Label smoothing (0.1)**: prevents overconfident probability predictions. For a binary target with hard labels `{0, 1}`, label smoothing shifts targets to `{0.05, 0.95}`. This improves the spread of probabilities across the cross-section, which is important for ranking-based signal generation.

### 4.3 Early Stopping and Checkpointing

The TCN checkpoints on **validation loss** (not val AUC). This is a deliberate design difference from the LSTM:

> Val loss is a proper scoring rule that rewards calibrated, well-spread probabilities — what the cross-sectional signal pipeline needs for consistent day-by-day rankings. Val AUC only rewards rank order and can be maximised by memorising lag patterns that do not generalise to trading returns.

Training halts when validation loss fails to improve for 15 consecutive epochs (`TCN_PATIENCE = 15`).

### 4.4 Diagnostic Monitoring

TCN uses the same per-epoch diagnostic framework as LSTM (shared `_eval_loader_loss_auc`):
- Epoch-level train/val loss, train/val AUC, learning rate, gradient norm.
- Flat AUC and overfitting warnings (same thresholds as LSTM).
- Logs saved to `reports/training_logs/fold{N}_tcn.csv`.

### 4.5 Inference

Identical to LSTM: softmax on 2-class logits → `P(class=1)`. Batched at 512 per forward pass.

---

## 5. TCN Hyperparameter Tuning

### 5.1 Tuning Strategy

Two-phase tuning, also run on the **first fold only** (`TCN_TUNE_ON_FIRST_FOLD_ONLY = True`).

**Phase 1 — training hyperparameters:**

| Dimension | Grid |
|---|---|
| Optimizer | `['adam', 'nadam']` |
| Learning rate | `[3e-4, 1e-3, 3e-3]` |
| Batch size | `[64, 128]` |

12 combinations (2 × 3 × 2). Architecture is fixed at the seed defaults: `[16,16,16]` channels, kernel=3, dropout=0.2. Feature set fixed at `'full'`.

**Phase 2 — architecture + feature set:**

| Dimension | Grid |
|---|---|
| num_channels | `[[16,16,16], [32,32,32], [32,32,32,32]]` |
| kernel_size | `[3, 5]` |
| dropout | `[0.1, 0.2, 0.3]` |
| feature_set | `['core_market', 'full']` |

18 combinations (3 × 2 × 3 × 1 per feature_set, 2 feature_sets = 36 total). The best Phase-1 training hyperparameters are fixed during Phase 2.

**Feature sets in tuning:**
- `core_market`: 8 core + 9 market features (17 total)
- `full`: 8 core + 9 market + 6 sector features (23 total)

Sequences for both feature sets are pre-built before the tuning loop to avoid redundant preprocessing.

### 5.2 Selection Criterion

Phase 1 selects by average validation AUC across replicates. Phase 2 also selects by average validation AUC with the best Phase-1 training hyperparameters fixed. (Note: unlike LSTM tuning's Sharpe-based criterion, TCN Phase 1/2 tuning uses AUC. The Sharpe criterion is applied in the separate diagnostic sweep below, not in the main tuning loop.)

Tuning parameters:
- Max epochs per candidate: 35 (`TCN_TUNE_MAX_EPOCHS`)
- Early stopping patience during tuning: 4 epochs (`TCN_TUNE_PATIENCE`)
- Replicates per configuration: 1 (`TCN_TUNE_REPLICATES`)

---

## 6. TCN Architecture Sweep Experiment (`experiments/tcn_arch_sweep.py`)

A diagnostic sweep over kernel size × number of levels × channel width on fold 1.

| Dimension | Grid |
|---|---|
| Kernel size | `[2, 3, 5]` |
| Number of levels | `[3, 4, 5]` |
| Channels per level | `[32, 64]` |

This produces up to 18 combinations (3 × 3 × 2).

**Receptive field constraint**: configurations where `RF < SEQ_LEN (30)` are **skipped** and logged without training. The RF formula is:
```
RF = 1 + 2 * (kernel_size - 1) * (2^num_levels - 1)
```
This ensures the model's receptive field covers at least the full input sequence. For example:
- kernel=2, levels=3: RF = 1 + 2×1×7 = 15 → **skipped** (15 < 30)
- kernel=3, levels=3: RF = 1 + 2×2×7 = 29 → **skipped** (29 < 30)
- kernel=3, levels=4: RF = 1 + 2×2×15 = 61 → **trained**
- kernel=5, levels=3: RF = 1 + 2×4×7 = 57 → **trained**

**Budget**: capped at `TCN_SWEEP_MAX_EPOCHS = 40` epochs per candidate.

**Fixed settings during sweep**: dropout=0.2 (`TCN_DROPOUT`), optimizer=Adam, lr=0.001, batch=256, all from production defaults.

**Selection criterion**: validation net Sharpe ratio (full signal pipeline, same as LSTM sweep).

**Output**: `reports/{universe}_tcn_tuning_results.csv` — one row per valid combination with columns: `kernel_size`, `num_levels`, `num_channels`, `receptive_field`, `val_sharpe`, `val_auc`, `n_params`.

**Parameter count** is reported for each configuration (`n_params = sum of trainable parameters`), providing a model complexity reference.

---

## 7. Key Design Differences: LSTM vs TCN

| Aspect | LSTM | TCN |
|---|---|---|
| Inductive bias | Sequential, gated memory | Dilated causal convolutions |
| Checkpointing criterion | Best validation **AUC** | Best validation **loss** |
| Tuning phases | Phase 1 only (training hp) | Phase 1 (training hp) + Phase 2 (arch + feature set) |
| Weight normalization | N/A | Disabled (`TCN_USE_WEIGHT_NORM = False`) |
| Label smoothing | None (0.0) | 0.1 |
| Receptive field | Full sequence via hidden state | Explicit: 29 days (default 3-level, kernel-3) |
| Parallelism | Sequential (recurrence dependency) | Fully parallel (convolutions) |
| Param count (default) | ~7k (32h × 1L) | ~10k ([16,16,16] × kernel-3) |
| MPS determinism | Deterministic | Non-deterministic (use CPU flag if needed) |

---

## 8. Signal Generation from Model Probabilities

Both models output `P(class=1)` — the probability that the stock outperforms the cross-sectional median 21-day forward return. These probabilities feed an identical signal pipeline:

1. **EMA smoothing** (`SIGNAL_SMOOTH_ALPHA = 0.0`): disabled — raw probabilities are used.
2. **Cross-sectional z-scoring** (`SIGNAL_USE_ZSCORE = True`): each day, probabilities are standardised within the cross-section before ranking.
3. **Ranking**: top-K stocks → Long, bottom-K → Short. `K_STOCKS = 5`.
4. **Confidence threshold** (`SIGNAL_CONFIDENCE_THRESHOLD = 0.0`): currently disabled.
5. **Holding period constraint** (`MIN_HOLDING_DAYS = 5`): once a position is entered, it is held for at least 5 days before exit or flip.
6. **Signal inversion** (`invert_signals = True` for large-cap): the portfolio direction is inverted at the `compute_portfolio_returns` level — model "winners" are shorted and "losers" are longed, exploiting mean reversion in liquid large-cap stocks.

Transaction costs: 5 bps per half-turn (`TC_BPS = 5`). No slippage (`SLIPPAGE_BPS = 0.0`).

---

## 9. Ensemble Construction

The final Ensemble model is the **mean probability** of LR + LSTM + TCN. RF and XGBoost are excluded from the ensemble because they produced negative Sharpe ratios in the large-cap universe. Inclusion is controlled by `include_lstm_b_in_ensemble = True` and `include_tcn_in_ensemble = True` in `LARGE_CAP_CONFIG`.

---

## 10. Permutation Feature Importance

Both LSTM and TCN compute permutation importance on each fold's test set:
- Each feature column is shuffled independently.
- The drop in AUC after shuffling is recorded as the feature's importance score.
- Results are saved per-fold and averaged across folds to `reports/{universe}_feature_importances_*.csv`.

---

## 11. Probability Calibration (Optional)

Both models support isotonic regression calibration fitted on validation-set predictions per fold (`LSTM_CALIBRATE_PROBS = False`, `TCN_CALIBRATE_PROBS = False` — currently disabled). When enabled, the calibration corrects overconfident logit-saturation (probability std ~0.34 for LSTM) to a well-calibrated range comparable to the baseline models.

---

## 12. Output Files

| File | Content |
|---|---|
| `reports/{universe}_lstm_tuning_results.csv` | LSTM lr × hidden_size sweep results |
| `reports/{universe}_tcn_tuning_results.csv` | TCN kernel × levels × channels sweep results |
| `reports/training_logs/fold{N}_lstm.csv` | Epoch-level train/val loss, AUC, LR per LSTM fold |
| `reports/training_logs/fold{N}_tcn.csv` | Epoch-level train/val loss, AUC, LR per TCN fold |
| `reports/training_logs/{universe}_lstm_sweep_best_training_log.csv` | Best-config epoch log from LR sweep |
| `reports/{universe}_full_predictions.csv` | Raw per-fold model probabilities |
| `reports/{universe}_feature_importances_per_fold.csv` | Per-fold permutation importance |
| `reports/{universe}_feature_importances_avg.csv` | Averaged feature importances |
