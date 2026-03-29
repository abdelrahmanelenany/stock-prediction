---
name: backtest-result-auditor
description: audits backtest and experiment outputs for weak out-of-sample behavior, suspicious metric gaps, turnover problems, drawdown issues, and signs of overfitting or leakage. use when asked to analyze backtest results, interpret model performance, diagnose failure modes, or compare runs.
---

# Backtest Result Auditor

Use this skill to turn raw experiment artifacts into a concise diagnosis.

## Objectives
- Identify whether the strategy is failing because of signal quality, execution assumptions, risk concentration, instability, or likely leakage.
- Separate "good prediction metrics" from "good trading outcomes."
- Produce a short, ranked diagnosis instead of a vague commentary dump.

## Audit checklist
Review as many of the following as are available:
- train / validation / test metrics
- walk-forward segment metrics
- Sharpe, Sortino, CAGR, max drawdown
- hit rate, precision, recall, F1, AUC, MCC, log loss
- turnover, holding period, exposure, long-short balance
- transaction cost assumptions
- benchmark-relative performance
- monthly or regime-level breakdowns
- confusion matrix or class balance
- calibration behavior, if available

## Red flags to explicitly check
- very large train-to-test performance gap
- high AUC or accuracy with weak Sharpe or poor PnL
- strong gross returns but weak after-cost returns
- excessive turnover
- one short regime carrying the whole result
- highly unstable fold-by-fold or window-by-window performance
- implausibly smooth or high returns
- improvements that appear only after a code or preprocessing change that may have caused leakage

## Required output format
Return:
1. One-paragraph diagnosis
2. Top 3 likely causes
3. What evidence supports each cause
4. Best next action

## Preferred reasoning style
- Be concrete.
- Name the likely bottleneck.
- Do not recommend architecture changes if the real issue is methodology or costs.
- Do not recommend hyperparameter tuning if the result is structurally broken.

## Example diagnoses
- "Signal quality appears modest, but turnover and transaction costs are destroying economic performance."
- "The main issue is instability across walk-forward windows, suggesting regime dependence or overfitting."
- "The metric jump is suspicious and should be treated as potential leakage until the pipeline is audited."