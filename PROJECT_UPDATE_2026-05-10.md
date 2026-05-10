# Project Update — 2026-05-10

> **Purpose:** Comprehensive record of recent implementation changes, feature layer design,
> ablation results, and final backtest numbers for both universes. Intended for Claude Chat
> to update project state and assist with thesis section rewrites.

---

## 1. Summary of Recent Changes

### 1a. Feature Layer Refactoring (`config.py`, commits `fca2522` + `f665b9e`)

The per-model feature flag system was overhauled from boolean flags
(`BASELINE_MARKET_FEATURES_ENABLED`, etc.) to **explicit, universe-aware feature layer
assignments**. The new design is documented inline in `config.py`:

```
L1 = core technical features      (_CORE_FEATURE_COLS)   — 8 features
L2 = market context features      (_MARKET_FEATURE_COLS)  — 9 features
L3 = sector context features      (_SECTOR_FEATURE_COLS)  — 6 features
```

**Final feature layer assignments (as shipped):**

| Model         | Large Cap | Small Cap | N Features (large) | N Features (small) |
|---------------|-----------|-----------|--------------------|--------------------|
| LR/RF/XGBoost | L1 + L2   | L1        | 17                 | 8                  |
| LSTM          | L1+L2+L3  | L1+L2+L3  | 23                 | 23                 |
| TCN           | L1 + L2   | L1 + L2   | 17                 | 17                 |

**Rationale (confirmed by ablation, see §3):**
- Sector features (L3) add noise for tree/linear models at this universe size.
- TCN performs best on `core_market` (L1+L2); adding L3 hurts large-cap TCN.
- LSTM benefits from L3 in large-cap (confirmed by main run results: Sharpe 0.487 net).
- Small-cap baselines use L1 only; market context features hurt RF and XGBoost there.

### 1b. New Config Parameters Added

| Parameter | Value | Purpose |
|---|---|---|
| `TUNE_USE_FROZEN_HPS` | `False` | Load `{universe}_tuned_hyperparams.json` instead of re-tuning |
| `COMPUTE_PERMUTATION_IMPORTANCE` | `True` | Skip ~(n_features × 2 × n_folds) inference passes when False |
| `TCN_FEATURE_SET_DEFAULT` | `"core_market"` | Changed from `"full"` — aligns with ablation finding |

### 1c. TCN Architecture: `feature_set` Grid Restricted

In `TCN_ARCH_GRID`, the `feature_set` option was narrowed from `["core_market", "full"]` to
`["core_market"]` only, reflecting ablation evidence that sector features hurt TCN performance.

### 1d. Feature-Layer Ablation Experiment Added (`experiments/feature_layer_ablation.py`)

New experiment script runs the full walk-forward pipeline under three feature conditions:
- **L1 only** — all models use only core technical features (8 features)
- **L1+L2** — all models receive core + market context (17 features)
- **L1+L2+L3** — LSTM and TCN additionally receive sector context (23 features)

Outputs saved to `reports/ablation/`.

---

## 2. Feature Layer Definitions (Reference)

### L1 — Core Technical Features (8 features)
`Return_1d`, `Return_5d`, `Return_21d`, `RSI_14`, `BB_PctB`, `RealVol_20d`, `Volume_Ratio`, `SectorRelReturn`

And: `MACD`, `ATR_14` (used by baselines only, part of `BASELINE_FEATURE_COLS` via `_CORE_FEATURE_COLS`)

> Note: `_CORE_FEATURE_COLS` used in the layer assignments contains 8 features excluding MACD/ATR/reversal aliases.
> The full `BASELINE_FEATURE_COLS` for large cap is L1+L2 (17 features).

### L2 — Market Context Features (9 features)
`Market_Return_1d`, `Market_Return_5d`, `Market_Return_21d`, `Market_Vol_20d`, `Market_Vol_60d`,
`RelToMarket_1d`, `RelToMarket_5d`, `RelToMarket_21d`, `Beta_60d`

### L3 — Sector Context Features (6 features)
`Sector_Return_1d`, `Sector_Return_5d`, `Sector_Return_21d`, `Sector_Vol_20d`, `Sector_Vol_60d`,
`SectorRelZ_Return_1d`

---

## 3. Feature-Layer Ablation Results (Table T10)

Experiment run: 2026-05-09, 17 folds each condition, no tuning (frozen hyperparams).

### 3a. Large-Cap Universe

| Model    | Condition | Val AUC | Mean Net Sharpe (per fold) |
|----------|-----------|---------|---------------------------|
| LR       | L1 only   | 0.4999  | -0.610                    |
| LR       | L1+L2     | 0.4926  | +0.059 ✓                  |
| RF       | L1 only   | 0.5115  | -0.990                    |
| RF       | L1+L2     | 0.5112  | +0.160 ✓                  |
| XGBoost  | L1 only   | 0.5092  | -1.498                    |
| XGBoost  | L1+L2     | 0.5161  | -0.226 ✓                  |
| LSTM     | L1 only   | 0.5500  | -0.729                    |
| LSTM     | L1+L2     | 0.5624  | +0.048                    |
| LSTM     | L1+L2+L3  | 0.5557  | +0.516 ✓ **best**         |
| TCN      | L1 only   | 0.5119  | -0.640                    |
| TCN      | L1+L2     | 0.5210  | -0.155 ✓ **best**         |
| TCN      | L1+L2+L3  | 0.5200  | -0.516 ✗                  |

**Key large-cap finding:** Market context (L2) is essential for all models. Sector context (L3)
helps LSTM but hurts TCN. LR without market context is the worst performer (-0.610 Sharpe/fold).

### 3b. Small-Cap Universe

| Model    | Condition | Val AUC | Mean Net Sharpe (per fold) |
|----------|-----------|---------|---------------------------|
| LR       | L1 only   | 0.5200  | +0.105                    |
| LR       | L1+L2     | 0.5133  | +0.190 ✓                  |
| RF       | L1 only   | 0.5150  | +0.190 ✓ **best**         |
| RF       | L1+L2     | 0.5248  | -0.386 ✗                  |
| XGBoost  | L1 only   | 0.5146  | -0.121                    |
| XGBoost  | L1+L2     | 0.5232  | -0.328 ✗                  |
| LSTM     | L1 only   | 0.5519  | -0.073                    |
| LSTM     | L1+L2     | 0.5494  | +0.694 ✓ **best**         |
| LSTM     | L1+L2+L3  | 0.5665  | -0.129 ✗                  |
| TCN      | L1 only   | 0.5260  | -0.262                    |
| TCN      | L1+L2     | 0.5132  | -0.021 ✓ **best**         |
| TCN      | L1+L2+L3  | 0.5192  | -0.102 ✗                  |

**Key small-cap finding:** Small-cap RF is optimal with L1 only (market context hurts it).
LSTM peaks at L1+L2 for small-cap (sector context hurts). TCN consistently improves
with L1+L2 vs L1 but L3 adds noise.

> **Discrepancy note:** The final main-run small-cap assignment gives LSTM L1+L2+L3
> (following the large-cap precedent). The ablation suggests L1+L2 would be better for
> small-cap LSTM. This is a design decision worth discussing in the thesis limitations section.

---

## 4. Tuned Hyperparameters (Saved in `{universe}_tuned_hyperparams.json`)

### Large Cap

**LSTM:** Default fixed architecture used (null — no LSTM tuning was accepted for large-cap).
Fixed: hidden=32, layers=1, dropout=0.0, lr=0.001, batch=256, optimizer=adam.

**TCN tuned:**
```json
{
  "optimizer": "adam",
  "lr": 0.003,
  "batch_size": 128,
  "num_channels": [16, 16, 16],
  "kernel_size": 5,
  "dropout": 0.3,
  "feature_set": "core_market"
}
```

### Small Cap

**LSTM tuned:**
```json
{
  "optimizer": "adam",
  "lr": 0.001,
  "batch_size": 128,
  "hidden_size": 64,
  "num_layers": 1,
  "dropout": 0.2
}
```

**TCN tuned:**
```json
{
  "optimizer": "nadam",
  "lr": 0.0003,
  "batch_size": 64,
  "num_channels": [32, 32, 32, 32],
  "kernel_size": 3,
  "dropout": 0.1,
  "feature_set": "core_market"
}
```

---

## 5. Final Backtest Results

All results: 2019–2024, walk-forward, rolling windows (252/63/63 days), 17 folds, K=5,
TC=5 bps per half-turn, invert_signals=True for large-cap.

### 5a. Large-Cap Universe — Table T5

#### Gross Returns (0 bps TC)

| Model    | N Days | Ann. Return (%) | Ann. Std (%) | Sharpe | Sortino | MDD (%) | Calmar | Win Rate (%) | VaR 1% (%) |
|----------|--------|----------------|--------------|--------|---------|---------|--------|-------------|-----------|
| LR       | 1071   | 13.32          | 25.91        | 0.337  | 0.518   | -38.21  | 0.349  | 51.54       | -3.96     |
| RF       | 1071   | 15.59          | 21.17        | 0.506  | 0.807   | -30.17  | 0.517  | 51.17       | -3.27     |
| XGBoost  | 1071   | 11.18          | 21.09        | 0.324  | 0.518   | -30.96  | 0.361  | 50.79       | -3.34     |
| LSTM     | 1071   | 21.48          | 24.84        | 0.631  | 0.940   | -36.63  | 0.587  | 52.47       | -3.94     |
| TCN      | 1071   | 12.70          | 20.40        | 0.401  | 0.652   | -25.80  | 0.492  | 51.73       | -2.83     |
| Ensemble | 1071   | 25.36          | 25.41        | 0.741  | 1.189   | -36.00  | 0.704  | 52.29       | -3.96     |

#### Net Returns (5 bps TC)

| Model    | N Days | Ann. Return (%) | Ann. Std (%) | Sharpe | Sortino | MDD (%) | Calmar | Win Rate (%) | VaR 1% (%) |
|----------|--------|----------------|--------------|--------|---------|---------|--------|-------------|-----------|
| LR       | 1071   | 8.78           | 25.91        | 0.179  | 0.275   | -39.96  | 0.220  | 50.98       | -3.98     |
| RF       | 1071   | 9.20           | 21.17        | 0.237  | 0.378   | -32.57  | 0.282  | 50.33       | -3.30     |
| XGBoost  | 1071   | 5.07           | 21.09        | 0.055  | 0.088   | -35.03  | 0.145  | 49.77       | -3.36     |
| LSTM     | 1071   | 17.19          | 24.84        | 0.487  | 0.724   | -38.02  | 0.452  | 52.10       | -3.96     |
| TCN      | 1071   | 4.39           | 20.41        | 0.026  | 0.041   | -32.15  | 0.137  | 50.51       | -2.87     |
| Ensemble | 1071   | 20.31          | 25.41        | 0.579  | 0.929   | -38.79  | 0.524  | 51.73       | -3.97     |

### 5b. Small-Cap Universe — Table T5

#### Gross Returns (0 bps TC)

| Model    | N Days | Ann. Return (%) | Ann. Std (%) | Sharpe | Sortino | MDD (%) | Calmar | Win Rate (%) | VaR 1% (%) |
|----------|--------|----------------|--------------|--------|---------|---------|--------|-------------|-----------|
| LR       | 1071   | 11.93          | 47.20        | 0.159  | 0.219   | -57.93  | 0.206  | 51.73       | -8.17     |
| RF       | 1071   | 24.77          | 37.03        | 0.496  | 0.746   | -43.74  | 0.566  | 51.63       | -6.10     |
| XGBoost  | 1071   | 13.24          | 38.26        | 0.226  | 0.319   | -45.39  | 0.292  | 50.98       | -6.63     |
| LSTM     | 1071   | 50.70          | 45.88        | 0.812  | 1.258   | -50.39  | 1.006  | 50.05       | -7.24     |
| TCN      | 1071   | 27.67          | 38.31        | 0.539  | 0.778   | -55.71  | 0.497  | 52.75       | -6.92     |
| Ensemble | 1071   | 43.59          | 44.12        | 0.735  | 1.004   | -57.49  | 0.758  | 53.41       | -7.50     |

#### Net Returns (5 bps TC)

| Model    | N Days | Ann. Return (%) | Ann. Std (%) | Sharpe | Sortino | MDD (%) | Calmar | Win Rate (%) | VaR 1% (%) |
|----------|--------|----------------|--------------|--------|---------|---------|--------|-------------|-----------|
| LR       | 1071   | 4.69           | 47.19        | 0.017  | 0.023   | -64.75  | 0.072  | 51.35       | -8.19     |
| RF       | 1071   | 14.68          | 37.04        | 0.268  | 0.402   | -50.77  | 0.289  | 50.70       | -6.12     |
| XGBoost  | 1071   | 4.28           | 38.26        | 0.011  | 0.015   | -50.80  | 0.084  | 50.61       | -6.67     |
| LSTM     | 1071   | 45.40          | 45.88        | 0.734  | 1.137   | -50.89  | 0.892  | 49.67       | -7.25     |
| TCN      | 1071   | 16.37          | 38.32        | 0.297  | 0.429   | -61.59  | 0.266  | 52.01       | -6.95     |
| Ensemble | 1071   | 35.18          | 44.11        | 0.598  | 0.817   | -58.01  | 0.606  | 53.03       | -7.51     |

---

## 6. Classification Metrics — Table T8

### Large Cap (invert_signals=True; metrics evaluated on 1−prob)

| Model    | Accuracy (%) | AUC-ROC | F1 Score | Daily AUC (mean) | Daily AUC (std) |
|----------|-------------|---------|---------|-----------------|----------------|
| LR       | 51.71       | 0.5313  | 0.5188  | 0.5238          | 0.1554         |
| RF       | 51.14       | 0.5194  | 0.5267  | 0.5173          | 0.1348         |
| XGBoost  | 51.09       | 0.5171  | 0.5212  | 0.5152          | 0.1332         |
| LSTM     | 51.57       | 0.5292  | 0.5388  | 0.5246          | 0.1327         |
| TCN      | 51.81       | 0.5233  | 0.5056  | 0.5235          | 0.1404         |
| Ensemble | 52.07       | 0.5330  | 0.5320  | 0.5287          | 0.1433         |

### Small Cap (invert_signals=False)

| Model    | Accuracy (%) | AUC-ROC | F1 Score | Daily AUC (mean) | Daily AUC (std) |
|----------|-------------|---------|---------|-----------------|----------------|
| LR       | 51.08       | 0.5143  | 0.5423  | 0.5225          | 0.1237         |
| RF       | 51.02       | 0.5114  | 0.5289  | 0.5106          | 0.1152         |
| XGBoost  | 51.00       | 0.5093  | 0.5418  | 0.5066          | 0.1134         |
| LSTM     | 51.03       | 0.5182  | 0.5117  | 0.5160          | 0.1109         |
| TCN      | 49.76       | 0.4973  | 0.5116  | 0.4982          | 0.1108         |
| Ensemble | 50.98       | 0.5179  | 0.5127  | 0.5175          | 0.1122         |

---

## 7. Sub-Period Performance — Table T6

> Pre-COVID and COVID crash periods have no data (walk-forward training window begins
> late 2019; these folds have insufficient training days). Only three periods are populated.

### Large Cap (net returns, 5 bps TC)

| Model    | Period           | N Days | Ann. Return (%) | Sharpe | MDD (%)  |
|----------|-----------------|--------|----------------|--------|---------|
| LR       | Recovery/bull   | 382    | 8.15           | 0.207  | -16.58  |
| LR       | 2022 bear       | 251    | 10.49          | 0.189  | -26.28  |
| LR       | 2023–24 AI rally| 438    | 8.36           | 0.161  | -39.96  |
| RF       | Recovery/bull   | 382    | 3.76           | -0.006 | -15.31  |
| RF       | 2022 bear       | 251    | 12.04          | 0.281  | -24.31  |
| RF       | 2023–24 AI rally| 438    | 12.51          | 0.376  | -32.57  |
| XGBoost  | Recovery/bull   | 382    | 2.34           | -0.089 | -20.68  |
| XGBoost  | 2022 bear       | 251    | 21.34          | 0.602  | -21.11  |
| XGBoost  | 2023–24 AI rally| 438    | -1.01          | -0.222 | -35.03  |
| LSTM     | Recovery/bull   | 382    | 24.19          | 0.838  | -22.31  |
| LSTM     | 2022 bear       | 251    | 2.28           | -0.058 | -21.92  |
| LSTM     | 2023–24 AI rally| 438    | 20.45          | 0.554  | -38.02  |
| TCN      | Recovery/bull   | 382    | 3.30           | -0.033 | -25.10  |
| TCN      | 2022 bear       | 251    | 8.84           | 0.175  | -26.95  |
| TCN      | 2023–24 AI rally| 438    | 2.87           | -0.049 | -32.15  |
| Ensemble | Recovery/bull   | 382    | 15.90          | 0.551  | -23.17  |
| Ensemble | 2022 bear       | 251    | **50.36**      | **1.206**| -16.96 |
| Ensemble | 2023–24 AI rally| 438    | 9.38           | 0.197  | -38.79  |

### Small Cap (net returns, 5 bps TC)

| Model    | Period           | N Days | Ann. Return (%) | Sharpe | MDD (%)  |
|----------|-----------------|--------|----------------|--------|---------|
| LR       | Recovery/bull   | 382    | 36.65          | 0.570  | -42.03  |
| LR       | 2022 bear       | 251    | 3.34           | -0.011 | -39.63  |
| LR       | 2023–24 AI rally| 438    | -16.42         | -0.466 | -51.45  |
| RF       | Recovery/bull   | 382    | 27.19          | 0.536  | -29.51  |
| RF       | 2022 bear       | 251    | 6.81           | 0.066  | -34.55  |
| RF       | 2023–24 AI rally| 438    | 9.14           | 0.152  | -32.39  |
| XGBoost  | Recovery/bull   | 382    | 3.05           | -0.021 | -34.29  |
| XGBoost  | 2022 bear       | 251    | 8.39           | 0.100  | -40.43  |
| XGBoost  | 2023–24 AI rally| 438    | 3.06           | -0.021 | -37.67  |
| LSTM     | Recovery/bull   | 382    | **113.26**     | **1.531**| -30.66 |
| LSTM     | 2022 bear       | 251    | 49.32          | 0.748  | -29.90  |
| LSTM     | 2023–24 AI rally| 438    | 2.48           | -0.031 | -50.89  |
| TCN      | Recovery/bull   | 382    | 72.74          | 1.258  | -26.25  |
| TCN      | 2022 bear       | 251    | 42.86          | 0.882  | -20.19  |
| TCN      | 2023–24 AI rally| 438    | -26.74         | -0.931 | -61.17  |
| Ensemble | Recovery/bull   | 382    | 113.79         | 1.686  | -32.31  |
| Ensemble | 2022 bear       | 251    | 34.66          | 0.587  | -32.61  |
| Ensemble | 2023–24 AI rally| 438    | -9.24          | -0.299 | -58.01  |

---

## 8. Top Feature Importances (Large Cap, Averaged Across Folds)

All importances normalised within model. Top features by cross-model consensus:

| Feature        | LR coef | RF MDI  | XGB gain | LSTM perm | TCN perm |
|---------------|---------|---------|---------|----------|---------|
| Beta_60d      | 0.212   | 0.215   | 0.140   | 0.082    | 0.116   |
| RealVol_20d   | 0.167   | 0.115   | 0.086   | 0.073    | 0.062   |
| RelToMarket_21d| 0.093  | 0.086   | 0.079   | 0.067    | 0.059   |
| Market_Vol_60d| 0.015   | 0.107   | 0.094   | 0.012    | 0.053   |
| Market_Vol_20d| 0.057   | 0.075   | 0.071   | 0.036    | 0.035   |
| Return_21d    | 0.065   | 0.066   | 0.066   | 0.061    | **0.174** |
| RSI_14        | 0.075   | 0.053   | 0.057   | 0.038    | 0.104   |
| RelToMarket_5d| 0.056   | 0.040   | 0.049   | 0.054    | 0.055   |
| BB_PctB       | 0.080   | 0.036   | 0.044   | 0.075    | 0.059   |

Market features (`Beta_60d`, `RealVol_20d`, `RelToMarket_*`, `Market_Vol_*`) dominate
importance across all models — confirming the value of L2 in large-cap.
TCN places unusual weight on `Return_21d` (monthly momentum) and `RSI_14`.

---

## 9. Interpretation Notes for Thesis

### 9a. Large-Cap vs Small-Cap Comparison

- Large-cap net Sharpe: **Ensemble 0.579**, LSTM 0.487 — moderate but consistent.
- Small-cap net Sharpe: **LSTM 0.734**, Ensemble 0.598 — substantially higher but with
  much higher volatility (~45% ann. std vs ~25% for large-cap).
- Large-cap uses `invert_signals=True` (mean-reversion strategy); small-cap uses
  `invert_signals=False` (momentum strategy).
- Transaction costs hit small-cap harder: gross→net degradation is larger (e.g., LSTM:
  50.7%→45.4% small-cap vs 21.5%→17.2% large-cap in absolute return terms).

### 9b. TCN in Context

- TCN (gross) large-cap: Sharpe 0.401 — competitive with baselines but below LSTM (0.631).
- TCN (net) large-cap: Sharpe 0.026 — near-zero; high turnover under 5 bps TC erodes returns.
- TCN is sensitive to feature set: `core_market` tuned achieved 0.585 net Sharpe in the
  standalone ablation vs -0.134 untuned. The main run (Sharpe 0.026 net) was run without
  using frozen hyperparams from the ablation — worth noting.
- Small-cap TCN net: Sharpe 0.297 — more viable than large-cap TCN.

### 9c. LSTM Dominance in Small-Cap

- Small-cap LSTM recovers 45.4% annualised net return (Sharpe 0.734) — the strongest
  single-model result. Recovery/bull sub-period: 113.26% annualised.
- Large-cap LSTM is strong in Recovery/bull (24.2% ann.) and 2023–24 AI rally (20.5%),
  but nearly flat in 2022 bear (+2.28%).

### 9d. 2022 Bear Market

- Ensemble performs best across all models in this period for large-cap (50.36% ann., Sharpe 1.206).
- For small-cap, LSTM (49.32%) and TCN (42.86%) perform strongly in 2022.
- 2022 regime characterised by rate hikes and equity drawdowns — cross-sectional dispersion
  creates exploitable relative-value signals.

### 9e. 2023–24 AI Rally

- Large-cap models struggle: LSTM 20.45% Sharpe 0.554, but TCN (-0.049) and XGBoost (-0.222) negative.
- Small-cap models deteriorate sharply: TCN -26.74%, LSTM +2.48%, Ensemble -9.24%.
- AI/large-cap concentration narrows dispersion in small-cap, reducing predictability.

### 9f. Classification vs Trading Performance Gap

- AUC-ROC values (0.51–0.53) appear modest but are consistent with the efficient-market
  literature — any persistent edge above 50% is economically meaningful.
- Large-cap: TCN has highest accuracy (51.81%) but near-zero net Sharpe — accuracy does
  not translate to profitability due to transaction costs.
- The gap between classification accuracy and Sharpe ratio motivates the use of val Sharpe
  (not val AUC) as the tuning selection criterion.

### 9g. Feature-Layer Ablation (New Contribution, Table T10)

- The ablation confirms that feature layer design is a material modelling decision.
- Adding L2 (market context) universally benefits large-cap baselines.
- Adding L3 (sector context) benefits large-cap LSTM but not TCN or small-cap LSTM.
- This supports differentiated feature assignment per model and universe rather than
  a one-size-fits-all feature set.

---

## 10. Files Changed / Added (Since Last Thesis Snapshot)

| File | Change |
|------|--------|
| `config.py` | Feature layer refactoring; `TUNE_USE_FROZEN_HPS`; `COMPUTE_PERMUTATION_IMPORTANCE`; `TCN_FEATURE_SET_DEFAULT` changed to `"core_market"`; `TCN_ARCH_GRID` feature_set restricted |
| `main.py` | Frozen hyperparameter loading; conditional permutation importance |
| `experiments/feature_layer_ablation.py` | New — L1/L1+L2/L1+L2+L3 ablation experiment |
| `figures/generate_F12_ablation_feature_layer.py` | New — grouped bar chart for feature layer ablation |
| `reports/ablation/*.csv` | New — ablation results (6 CSVs: per-fold + summary + T10 for each universe) |
| `reports/tcn_feature_set_ablation.csv` | TCN-only feature set sweep (no tuning) |
| `reports/tcn_feature_set_ablation_tuned.csv` | TCN-only feature set sweep (with tuning) |
| `reports/large_cap_tuned_hyperparams.json` | Saved TCN hyperparams (LSTM null for large-cap) |
| `reports/small_cap_tuned_hyperparams.json` | Saved LSTM + TCN hyperparams for small-cap |
| All `reports/large_cap_*.csv` and `small_cap_*.csv` | Updated with current run results |

---

## 11. CLAUDE.md Sections That Need Updating

The following CLAUDE.md sections are now stale and should be rewritten:

1. **Feature Sets section** — Replace the per-model feature flag description with the new
   L1/L2/L3 layer table (§2 above). Update `BASELINE_FEATURE_COLS` description to note
   universe-conditional assignment.

2. **`TCN_FEATURE_SET_DEFAULT`** — Update from `"full"` to `"core_market"` in the TCN
   Architecture table.

3. **`TCN_ARCH_GRID`** — `feature_set` now only `["core_market"]`, not `["core_market", "full"]`.

4. **New config parameters** — Add `TUNE_USE_FROZEN_HPS` and `COMPUTE_PERMUTATION_IMPORTANCE`
   to the config reference table.

5. **Required Thesis Outputs** — Add Table T10 (Feature-Layer Ablation) and Figure F12
   (Feature-Layer Ablation Bar Chart) as completed items.

6. **Implementation Status** — Mark F12 as ✅ completed; add feature-layer ablation experiment
   to completed experiments list.

7. **`TCN_FEATURE_COLS_FULL`** note — LSTM still uses `full` (L1+L2+L3) for large-cap, but
   TCN default is `core_market`. Update the model summary table accordingly.
