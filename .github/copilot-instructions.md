# Repository-wide Copilot instructions

This repository is for machine learning and deep learning research on stock behavior prediction with proper backtesting.

## Core goals
- Improve out-of-sample performance, not just in-sample metrics.
- Preserve methodological rigor.
- Prefer reproducible, config-driven experimentation.
- Optimize for economically meaningful results, not vanity ML metrics.

## Non-negotiable constraints
- Never introduce look-ahead bias, target leakage, or data leakage.
- Never fit scalers, imputers, winsorization thresholds, PCA, feature selection, or any preprocessing on validation/test/future data.
- Never use future timestamps, future prices, future cross-sectional information, or future corporate-event knowledge in features for time t.
- Respect chronological ordering everywhere.
- Treat all performance improvements as suspicious until leakage checks pass.

## Validation and backtesting rules
- Prefer walk-forward, rolling, or expanding-window evaluation over random splits.
- Keep train, validation, and test periods strictly separated in time.
- If the code uses cross-sectional targets, preserve the intended target definition unless explicitly asked to change it.
- Report both ML metrics and trading metrics.
- Always mention which split or period each metric belongs to.

## Model-change rules
- Prefer small, auditable changes over large rewrites.
- Change one major factor at a time unless explicitly asked to run a broader redesign.
- Prefer editing config files and modular code over hardcoding values.
- Preserve backward compatibility where practical.
- Do not silently change the target, label definition, transaction-cost assumptions, or portfolio construction logic.

## Feature-engineering rules
- Favor features that are plausibly available at prediction time.
- Check whether new features overlap with existing ones before adding them.
- Prefer robust scaling/normalization that can be applied in a time-safe way.
- Flag features that may encode future information, survivorship bias, or post-close data unavailable at decision time.

## Trading and risk rules
- Do not accept an improvement that raises AUC or F1 but weakens Sharpe, drawdown, turnover, or after-cost performance unless explicitly justified.
- Always consider transaction costs, turnover, concentration, and regime dependence.
- Distinguish between signal quality improvement and portfolio construction improvement.

## Experiment workflow
- Start by reading the relevant pipeline, config, and latest experiment artifacts.
- Summarize the current failure mode before changing code.
- State the hypothesis for each change.
- After edits, summarize exactly what changed.
- Whenever possible, add or update experiment notes in markdown.

## Expected answer style
When analyzing or changing this repository:
1. Briefly diagnose the issue.
2. Explain the hypothesis.
3. Make the smallest useful change.
4. Summarize expected impact and risks.
5. Report before/after results if available.
6. Explicitly mention any uncertainty.

## Preferred priorities
1. Methodological correctness
2. Out-of-sample robustness
3. Trading performance after costs
4. Code clarity and reproducibility
5. Training speed