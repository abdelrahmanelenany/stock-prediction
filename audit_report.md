# Codebase Audit Report — Neural Networks for Stock Behaviour Prediction
**Date:** 2026-04-10  
**Auditor:** Claude Code (automated audit)  
**Branch:** main  
**Status at audit:** `DEV_MODE=True`, LSTM-A skipped, Ensemble missing, results for LR/RF/XGB/LSTM only

---

## 1. Executive Summary — Top 5 Critical Issues

The following five issues **must be resolved before any thesis-quality results are credible**:

| Priority | Issue | Impact |
|---|---|---|
| **C1** | `DEV_MODE=True` is hardcoded — LSTM-A is permanently skipped and never appears in reports | All reported results are incomplete; no LSTM-A, no Ensemble |
| **C2** | `LSTM_B_HIDDEN_SIZE=32 / NUM_LAYERS=1 / DROPOUT=0.0` — architecture is 2× smaller than spec (64/2/0.2) | LSTM underperforms spec; comparison invalid |
| **C3** | **Ensemble model is not implemented** in `run_walk_forward_pipeline` — absent from all reports | Core contribution (Ensemble) missing entirely |
| **C4** | `TARGET_HORIZON_DAYS=21` with **daily portfolio rebalancing** — 21-day prediction target but 1-day P&L | Signal-to-portfolio horizon mismatch; thesis rationale required |
| **C5** | **Hyperparameter tuning replicates** use `val_loss` for early stopping and measure AUC on the **last epoch**, not best-AUC epoch — tuning selects suboptimal configs | All tuned models may use inferior hyperparameters |

---

## 2. Complete Findings Table

### Category A — Known Confirmed Bugs

| ID | File | Line(s) | Severity | Description | Recommended Fix |
|---|---|---|---|---|---|
| A1 | `config.py` | 88–107 | **FIXED** | `SMALL_CAP_SECTOR_MAP` — all tickers now have correct sector assignments | No action required |
| A2 | `models/lstm_model.py` | 879, 993–998 | **FIXED** | LSTM early stopping now uses `val_auc` for checkpointing via `_train_lstm_impl` | No action required |
| A3 | `config.py` | 286; `backtest/signals.py` | Medium | `SIGNAL_CONFIDENCE_THRESHOLD=0.55` applied to z-scored signals. The code is now CORRECT in applying this as a z-score threshold (not raw probability), but 0.55 z-score still silently filters positions when cross-sectional dispersion is low. | Set `SIGNAL_CONFIDENCE_THRESHOLD = 0.0` for pure-ranking behaviour per spec. Document how many positions were filtered in the diagnostics printout. |
| A4 | `config.py` | 316–318 | **Critical** | `LSTM_B_HIDDEN_SIZE=32`, `LSTM_B_NUM_LAYERS=1`, `LSTM_B_DROPOUT=0.0` — architecture is half the spec (64/2/0.2). With market+sector features enabled, LSTM has 23 input features but only 32 hidden units — aggressive compression. | Set `LSTM_B_HIDDEN_SIZE=64`, `LSTM_B_NUM_LAYERS=2`, `LSTM_B_DROPOUT=0.2` to match spec. Update arch tuning grid min values. |
| A5 | `config.py` | 112, 402; `main.py` | **Critical** | `DEV_MODE=True` defined **twice** (lines 112 and 402). The second definition overrides the first. With `DEV_MODE=True`, main.py line 535 (`if not DEV_MODE:`) permanently skips LSTM-A, and line 828–829 excludes LSTM-A from `model_cols`. All existing reports lack LSTM-A. | Set `DEV_MODE=False` for final thesis run. Remove the duplicate definition at line 112. |

---

### Category B — Data Integrity & Leakage Risks

| ID | File | Line(s) | Severity | Description | Recommended Fix |
|---|---|---|---|---|---|
| B1 | `main.py` | 426–428 | **OK** | Walk-forward boundary integrity — `.isin()` on non-overlapping integer index slices is correct. No date appears in two splits. | No action required |
| B2 | `main.py` | 439–479 | **OK** | Wavelet denoising — thresholds computed from `df_tr` only; applied per-split independently with `apply_wavelet_denoising_causal`. | No action required |
| B3 | `pipeline/standardizer.py` | 72–75 | **OK** | Scaler fits only on `X_train`; `.transform()` on val/test. No contamination. | No action required |
| B4 | All feature lists | n/a | **OK** | `Return_NextDay` is absent from `LSTM_A_FEATURES`, `LSTM_B_FEATURES`, `BASELINE_FEATURE_COLS`. No target leakage. | No action required |
| B5 | `pipeline/features.py` | 424–445 | **OK** | `SectorRelReturn` uses leave-one-out sector mean — self-contribution correctly removed via `(sector_sums - df['Return_1d']) / (sector_counts - 1)`. | No action required |
| B6 | `models/lstm_model.py` | 529–679 | Medium | **LSTM sequence lookback correctness**: `prepare_lstm_b_sequences_temporal_split` receives `df_train_fold = pd.concat([df_tr, df_v])` (main.py line 531). The LSTM trains on walk-forward val data (`df_v`) that baseline models only use for hyperparameter selection. This creates a **training data asymmetry** — LSTMs see more data than baselines. | Document this design decision explicitly. Consider whether the LSTM's internal 20% temporal val split covers an appropriate date range and whether baselines should also concatenate train+val. |
| B7 | `pipeline/features.py` | 471–506 | Low | `compute_market_context_features` computes `Market_Return_1d` as the cross-sectional **mean of all tickers** (not leave-one-out). `RelToMarket_1d = Return_1d - Market_Return_1d` contains a self-contribution bias of 1/N per ticker (2% for N=50, 3.3% for N=30). | Use leave-one-out market mean: `(sum - self) / (N - 1)`. Impact is small for N=50 but should be documented for small-cap universe. |

---

### Category C — Model Architecture & Training Correctness

| ID | File | Line(s) | Severity | Description | Recommended Fix |
|---|---|---|---|---|---|
| C1 | `models/lstm_model.py` | 879, 993–998 | **FIXED** | LSTM-A and LSTM both checkpoint on best `val_auc` via `_train_lstm_impl`. | No action required |
| C2 | `models/lstm_model.py` | 151–290 | **OK** | LSTM-A two-phase tuning: Phase 1 (optimizer/lr/batch_size) → Phase 2 (hidden/layers/dropout). Phases are correctly separated; Phase 2 receives best Phase 1 config. | No action required |
| C3 | `models/lstm_model.py` | 869, 903–910 | **OK** | Gradient clipping at `max_norm=1.0` applied in `_train_lstm_impl` via `LSTM_MAX_GRAD_NORM`. | No action required |
| C4 | `main.py` | 822–829 | **Critical** | **Ensemble model absent** from `run_walk_forward_pipeline`. The `model_cols` dict only contains LR, RF, XGBoost, LSTM (and LSTM-A when not DEV_MODE). CLAUDE.md specifies Ensemble = mean of all 5. | Implement Ensemble after the fold loop: average `Prob_LR`, `Prob_RF`, `Prob_XGB`, `Prob_LSTM_A`, `Prob_LSTM_B` per (Date, Ticker) and run it through the full backtest pipeline. |
| C5 | `models/lstm_model.py` | 107–148 | High | **Tuning replicate early stopping uses `val_loss`** (not `val_auc`). The AUC for hyperparameter selection is measured on the **final model state** (at training endpoint), NOT at the best-AUC epoch. If AUC peaks before val_loss stops improving, tuning selects inferior configs. | In `_run_tuning_replicates`: track `best_val_auc` and `best_state`, restore before measuring final AUC. This aligns tuning selection with production training behaviour. |
| C6 | `models/calibration.py` | all | Low | Probability calibration module exists (per-fold, fit on val, apply to test — architecturally correct) but is **not integrated into main.py**. | Optionally wire into the pipeline after LSTM prediction, or document as an unused extension. |
| C7 | `models/lstm_model.py` | 753–777 | Medium | `LSTMModelB` uses a **single FC layer** (`self.fc = nn.Linear(hidden, 2)`) while `StockLSTMTunable` (used by LSTM-A) uses a **two-layer MLP** (fc1 → ReLU → fc2). This architectural inconsistency makes the comparison between LSTM-A and LSTM confounded. | Add a second FC layer to `LSTMModelB` to match `StockLSTMTunable`'s decoder, or document the intentional difference. |
| C8 | `models/lstm_model.py` | 693–694 | Low | Stale comments: "4 features" for LSTM-A, "6 features" for LSTM — LSTM-A has 6 features (after adding momentum), LSTM has 23 features when market/sector features enabled. | Update comments to reflect actual feature counts. |
| C9 | `models/lstm_model.py` | 1196–1200; 936 | Low | `ReduceLROnPlateau` for LSTM is stepped on `val_loss` (line 936), but model checkpointing uses `val_auc`. LR reduction fires when loss plateaus but the saved model is the best-AUC epoch. This is a minor inconsistency but unlikely to cause major issues. | Document this or align both to use `val_auc` as the scheduler metric. |

---

### Category D — Backtesting & Portfolio Construction

| ID | File | Line(s) | Severity | Description | Recommended Fix |
|---|---|---|---|---|---|
| D1 | `backtest/portfolio.py` | 87–101 | **OK** | TC charged only on position CHANGES (delta turnover tracking via `prev_signals`). No over-charging. | No action required |
| D2 | `backtest/portfolio.py` | 68–83 | High | **`invert_signals=True` by default** for large-cap. The portfolio FLIPS all Long↔Short signals. This means the large-cap models are **anti-predictive** — they predict "will outperform" but the portfolio bets on "will underperform". The raw model AUC is ~0.48–0.49 (below random) and the inverted AUC is ~0.51–0.52. **This finding is empirically interesting but must be justified and explained.** | Document as a mean-reversion finding. Report BOTH raw model AUC (before inversion) AND inverted AUC in Table T8. Verify whether small-cap universe shows same pattern. |
| D3 | `config.py` | 273 | Medium | `K_STOCKS=5` (CLAUDE.md spec says 10). With N=50 stocks and k=5, long and short legs each represent 10% of the universe. | Spec mismatch — either update CLAUDE.md or explain the rationale for k=5 in the thesis. |
| D4 | `backtest/metrics.py` | 58–60 | **OK** | Sharpe uses `sqrt(252)` annualization on excess returns. Computed on net-of-TC daily returns. | No action required |
| D5 | `backtest/metrics.py` | 75–77 | **OK** | MDD computed on cumulative wealth curve `(1+r).cumprod()`, peak-to-trough definition. | No action required |
| D6 | `main.py` | 966–974 | High | **Sub-period analysis is never computed** — it only runs when `'LSTM-A' in port_returns_net_5`, which is always False because `DEV_MODE=True`. `subperiod_metrics` silently stays `None` and Table T6 is never saved. | After fixing DEV_MODE, also wire sub-period analysis to run on the BEST-performing model (not hardcoded LSTM-A), or run it on all models and save all. |
| D7 | `config.py` | 283 | **Critical** | `TARGET_HORIZON_DAYS=21` but portfolio rebalances DAILY. The models predict 21-day forward return rank, but the portfolio earns 1-day P&L using `Return_NextDay`. `MIN_HOLDING_DAYS=5` enforces a 5-day hold constraint but does not match the 21-day prediction horizon. **This is a coherent design for momentum but needs explicit justification.** | Choose one of: (a) use `TARGET_HORIZON_DAYS=1` (original paper approach), (b) use daily rebalancing with a 5-day look-ahead target (`TARGET_HORIZON_DAYS=5`), or (c) keep 21-day target and explain in thesis that the model learns medium-term momentum rank but portfolio executes daily. |

---

### Category E — Feature Engineering Quality

| ID | File | Line(s) | Severity | Description | Recommended Fix |
|---|---|---|---|---|---|
| E1 | `config.py` | 285 | **OK** | `SIGNAL_SMOOTH_ALPHA=0.0` — smoothing is disabled (handled correctly in `smooth_probabilities` via `alpha=None` path). | No action required |
| E2 | `pipeline/features.py` | 327–329 | **OK** | `Return_5d` and `Return_21d` use `pct_change(5)` and `pct_change(21)` — trailing (backward-looking). No look-ahead bias. | No action required |
| E3 | `pipeline/features.py` | 378 | Low | `Volume_Ratio` denominator is `rolling(20).mean()` which includes the current day's volume in the average. This is a mild within-day circularity (denominator partially based on today's value), but since all features are known at market close, this is acceptable. | No action required, but document as deliberate design. |
| E4 | `pipeline/features.py` | 355 | **OK** | `BB_PctB` denominator has `+ 1e-10` guard against zero-volatility periods. | No action required |
| E5 | `pipeline/features.py` | 343–349 | **OK** | `ATR_14` introduces NaN for first 14 rows per ticker. `build_feature_matrix` drops these via `dropna(subset=FEATURE_COLS)`. Per-fold wavelet recomputation also does `dropna(subset=FEATURE_COLS)`. | No action required |
| E6 | `config.py` | 183–221 | Low | `ALL_FEATURE_COLS` is built dynamically via a loop + conditional extends. Reversal features (`NegReturn_1d`, `NegReturn_5d`, `RSI_Reversal`, `NegMACD`, `BB_Reversal`) are also computed unconditionally in `compute_technical_features` regardless of `invert_features` flag. These features are in the cache but not in any model's feature set — wasted compute and storage. | Remove unconditional computation of reversal features from `compute_technical_features` and gate it on `invert_features`. |

---

### Category F — Code Quality & Reproducibility

| ID | File | Line(s) | Severity | Description | Recommended Fix |
|---|---|---|---|---|---|
| F1 | `main.py` | 111–123 | Medium | `set_global_seed` sets `torch.manual_seed` (CPU) and `torch.cuda.manual_seed_all` (GPU) but **does NOT call `torch.mps.manual_seed(seed)`** for Apple Silicon MPS backend. MPS operations may be non-deterministic across runs. | Add `if torch.backends.mps.is_available(): torch.mps.manual_seed(seed)` to `set_global_seed`. |
| F2 | `config.py` | 112, 402 | High | `DEV_MODE` is **defined twice** — first at line 112 (`DEV_MODE = True`) and again at line 402 (`DEV_MODE = True`). Python uses the last definition but having two definitions is a maintenance hazard: changing one and not the other will cause silent bugs. | Remove the first definition (line 112). Keep only the authoritative definition at line 402 with a clear comment. |
| F3 | `main.py` | 289–338 | **OK** | When `load_cached=True`, feature cache is loaded but the **full walk-forward training loop always runs**. Cache only skips download + feature computation, not model training. | No action required |
| F4 | `main.py` | 426–428 | **OK** | Missing tickers handled gracefully — `groupby('Ticker')` in sequence builders skips absent tickers; baseline predictions are NaN-filled and filtered by `dropna(subset=[prob_col])`. | No action required |
| F5 | `config.py` | 376 | Low | `FEATURE_COLS_AFTER_SELECTION` contains a manually-edited list from `analysis/feature_correlation.py` but is **never referenced in main.py or any model**. The pipeline always uses `ALL_FEATURE_COLS` / `BASELINE_FEATURE_COLS` etc. | Either wire `FEATURE_COLS_AFTER_SELECTION` into the pipeline as an option or remove it from config. |
| F6 | `config.py` | 300–301 | Medium | DEV mode increases batch sizes to 256 (`LSTM_A_BATCH = 256 if DEV_MODE else 128`). Results obtained in DEV mode (larger batches) may differ from thesis-spec results (batch=128). | Ensure final thesis runs use `DEV_MODE=False` for reproducible batch sizes. |

---

## 3. Improvement Roadmap (Phase 3 Items)

### Impact vs. Effort Matrix

```
                    LOW EFFORT          HIGH EFFORT
                ─────────────────────────────────────
HIGH IMPACT  |  G. LayerNorm (G35)   G. TCN (G37)
             |  H. Wavelet ablation  J. Diebold-Mariano
             |  J. Sub-period regime    (J45)
             |  J. Fold-level distrib.
             |─────────────────────────────────────
LOW IMPACT   |  H. Interaction feats I. PBO metric
             |  F. Regime indicator  I. Market neutrality
             |                          check
```

### Detailed Assessment

| ID | Item | Thesis Impact | Effort | M4 Feasibility | Recommendation |
|---|---|---|---|---|---|
| G34 | Additive attention over LSTM timesteps | **High** — thesis-differentiating, interpretability | Low (10–15 lines) | Full MPS support | **Do it** — add after LSTM output, before dropout+FC |
| G35 | `LayerNorm(hidden_size)` after LSTM | Medium — training stability across folds | Very low (1 line) | Full MPS support | **Do it** — trivial improvement |
| G36 | Bidirectional LSTM | **NOT VIABLE** — uses future context | — | — | **Do not implement** in causal setting |
| G37 | TCN (Temporal Convolutional Network) | High — architectural contribution | High (200+ lines) | Good MPS support | Consider as LSTM-C after fixing existing models |
| H38 | Wavelet ablation (`USE_WAVELET_DENOISING=False`) | **High** — required thesis contribution | Very low (1 config change) | n/a | **Do it** — run both and report Table T5 comparison |
| H39 | Interaction features for tree models | Medium — minor boost for LR/RF/XGB | Low | n/a | Nice-to-have if time permits |
| H40 | `HighVol_Regime` indicator feature | Medium — supports AMH thesis framing | Low (20 lines) | n/a | Worth adding; computable from universe vol |
| I41 | Simplified PBO (IS vs OOS Sharpe ratio) | Medium — strengthens anti-overfitting claims | Medium (50 lines) | n/a | Implement simple IS/OOS Sharpe degradation ratio |
| I42 | Extended TC sensitivity (15, 20 bps) | **Already done** — `compute_tc_sensitivity` defaults `[0,2,5,10,15,20,25,30]` | Zero | n/a | Already implemented; just call and report |
| I43 | Market-neutrality beta check | High — examiner will ask about factor exposure | Medium (30 lines) | n/a | Add to `compute_metrics`: average portfolio beta to market |
| I44 | Turnover reporting | **Already done** — `compute_turnover_and_holding_stats` exists | Zero | n/a | Already implemented; wire to T5 output table |
| J45 | Diebold-Mariano test for model comparison | High — statistical significance of model differences | Medium (20 lines + scipy) | n/a | **Do it** — simple paired t-test on daily return differences |
| J46 | Sub-period regime analysis (all 5 regimes) | **High** — core AMH evidence | Zero (code exists) | n/a | **Do it** once DEV_MODE and LSTM-A fixed |
| J47 | Feature importance stability across folds | Medium — overfitting evidence | Medium | n/a | Use existing RF `feature_importances_` per fold |
| J48 | Fold-level Sharpe distribution plot | **High** — visualises reliability | Low | n/a | `fold_sharpe_per_model.csv` already saved; just plot it |

---

## 4. Implementation Order

### Tier 1 — Must-Fix Before Any Results Are Credible

1. **Fix `DEV_MODE`** (`config.py` lines 112 + 402): Set `DEV_MODE=False`, remove duplicate definition.
2. **Fix LSTM architecture** (`config.py` lines 316–318): `LSTM_B_HIDDEN_SIZE=64`, `LSTM_B_NUM_LAYERS=2`, `LSTM_B_DROPOUT=0.2`.
3. **Implement Ensemble model** (`main.py` after fold loop): Average Prob_LR + Prob_RF + Prob_XGB + Prob_LSTM_A + Prob_LSTM_B; run through full backtest.
4. **Fix tuning replicate AUC measurement** (`lstm_model.py` `_run_tuning_replicates`): Track best-state by val_AUC within the replicate, restore before final AUC measurement.
5. **Fix `set_global_seed` MPS seed** (`main.py` line 117): Add `torch.mps.manual_seed(seed)`.
6. **Set `SIGNAL_CONFIDENCE_THRESHOLD = 0.0`** (`config.py` line 286) for pure-ranking behaviour.

### Tier 2 — Should-Fix to Improve Performance and Correctness

7. **Decide on `TARGET_HORIZON_DAYS`** (`config.py` line 283): Choose 1 (original paper), 5 (weekly momentum), or 21 (monthly) and document the choice. Regenerate targets cache.
8. **Fix sub-period analysis** (`main.py` lines 966–974): Run on best-performing model or all models (not hardcoded LSTM-A only).
9. **Add `invert_signals` documentation** in thesis: explain raw model is anti-predictive for large-cap (mean-reversion effect); report both inverted and raw AUC.
10. **Fix `LSTMModelB` decoder**: Add two-layer MLP (fc1 → ReLU → fc2) to match `StockLSTMTunable`.
11. **Fix `compute_market_context_features` leave-one-out**: Correct `RelToMarket_1d` to exclude self-contribution.
12. **Wire `compute_turnover_and_holding_stats` output** to T5 reporting (it's already computed per-model at line 860 but not saved).

### Tier 3 — Nice-to-Have Thesis Contributions

13. **Add attention mechanism** to both LSTM variants (additive attention over timesteps).
14. **Add `LayerNorm`** after LSTM output in both variants.
15. **Run wavelet ablation** (`USE_WAVELET_DENOISING=False` baseline run).
16. **Implement Diebold-Mariano / paired t-test** for model comparison significance.
17. **Add `HighVol_Regime` binary feature** for regime-conditional behavior.
18. **Add market-neutrality beta diagnostic** to `compute_metrics`.
19. **Add fold-level Sharpe box plots** (data already in `fold_sharpe_per_model.csv`).
20. **Run extended TC sensitivity** using existing `compute_tc_sensitivity` (already supports 15/20 bps).

---

## 5. CLAUDE.md Delta — Sections Requiring Updates After Fixes

The following CLAUDE.md sections contain outdated or inaccurate information that must be updated after fixes are applied.

### Section: `config.py — Single Source of Truth`

| Field | Current CLAUDE.md | Actual code | Update needed |
|---|---|---|---|
| `TICKERS` | 70-stock S&P 500 universe | 50 large-cap / 30 small-cap via `UNIVERSE_MODE` | Update to describe both universes |
| `START_DATE` | `2015-01-01` | `2019-01-01` | Update |
| `END_DATE` | `2024-12-31` | `2024-12-31` | OK |
| `TRAIN_DAYS` | 500 | 252 | Update |
| `VAL_DAYS` | 125 | 63 | Update |
| `TEST_DAYS` | 125 | 63 | Update |
| `SEQ_LEN` | 60 | 30 | Update |
| `K_STOCKS` | 10 | 5 | Update |
| `LSTM_B_HIDDEN_SIZE` | 64 | 32 (bug — fix to 64) | After fix: spec will match |
| `LSTM_B_NUM_LAYERS` | 2 | 1 (bug — fix to 2) | After fix: spec will match |
| `LSTM_B_DROPOUT` | 0.2 | 0.0 (bug — fix to 0.2) | After fix: spec will match |
| `SIGNAL_CONFIDENCE_THRESHOLD` | 0.0 (pure ranking) | 0.55 (should be 0.0) | After fix: spec will match |
| `TARGET_HORIZON_DAYS` | Not mentioned | 21 | Add this critical parameter |
| `UNIVERSE_MODE` | Not mentioned | `"large_cap"` / `"small_cap"` | Document the mode flag |

### Section: `Step 6 — Models` (Summary Table)

- Update LSTM feature count from "8" to "8 base + 9 market + 6 sector = 23 total (when market/sector features enabled)"
- Update Ensemble description to note it is missing from current implementation (see Tier 1 fix)
- Add note: `DEV_MODE=True` currently disables LSTM-A and Ensemble

### Section: `Step 7 — Trading Signals`

- Update `SIGNAL_CONFIDENCE_THRESHOLD` from 0.55 to 0.0 after fix
- Document `invert_signals=True` for large-cap: the portfolio bets AGAINST raw model predictions

### Section: `Step 10 — main.py Pipeline Overview`

- Add `TARGET_HORIZON_DAYS` to data loading description
- Note that sub-period analysis is currently non-functional (fixed after Tier 1 fixes)
- Reflect actual K_STOCKS=5 (not 10)

### Section: `Implementation Status Summary — ✅ Completed`

- Move Ensemble from ✅ to ❌ (not implemented in current main.py)
- Add a note under LSTM-A: currently disabled by DEV_MODE

### Section: `Required Thesis Outputs — Tables`

- T6 (Sub-period): mark as **broken** until DEV_MODE fix and LSTM-A run
- Add T7 note: `compute_tc_sensitivity` already includes 15/20/25/30 bps — just call it

### Section: `LSTM architecture row in Tables`

- Fix: `LSTM_B_HIDDEN_SIZE=32, LSTM_B_NUM_LAYERS=1, LSTM_B_DROPOUT=0.0` until fix is applied

---

## 6. Current Results Context

From `reports/large_cap_backtest_summary.txt` (run at time of audit, DEV_MODE=True, large_cap):

| Model | Net Sharpe (5bps) | Ann. Return | MDD |
|---|---|---|---|
| LR | 0.195 | 9.13% | -40.88% |
| RF | 0.227 | 9.62% | -41.49% |
| XGBoost | -0.044 | 2.80% | -49.41% |
| LSTM | **0.434** | 15.58% | -42.26% |
| LSTM-A | *(skipped)* | — | — |
| Ensemble | *(not implemented)* | — | — |

**Key observations:**
- Classification AUC ≈ 0.51–0.53 after signal inversion (`invert_signals=True`). Raw model AUC ≈ 0.47–0.49 — models are anti-predictive on large-cap. This is an empirical mean-reversion finding requiring justification.
- LSTM net Sharpe of 0.434 with the current suboptimal architecture (32/1/0.0) may improve further after architecture fix. Results should be re-run after Tier 1 fixes.
- Transaction costs erode Sharpe significantly: LR drops from 0.361 (gross) to 0.195 (net), LSTM from 0.569 to 0.434.

---

*End of Audit Report*
