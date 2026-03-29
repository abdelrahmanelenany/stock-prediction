---
name: feature-engineering-researcher
description: reviews existing stock-prediction features and proposes time-safe, economically plausible improvements such as regime features, volatility features, relative-strength features, rolling normalization, lag changes, and feature pruning. use when asked how to improve features or when weak signal quality may be feature-driven.
---

# Feature Engineering Researcher

Use this skill when model performance likely suffers from weak or noisy inputs.

## Objectives
- Improve signal quality using features that are available at prediction time.
- Suggest additions, removals, or transformations that are defensible for financial ML.
- Keep feature proposals aligned with the target definition and rebalance timing.

## Feature categories to consider
- returns across multiple horizons
- rolling volatility and realized variance
- ATR and range-based volatility
- momentum and trend features
- mean-reversion indicators
- volume and liquidity features
- regime features such as volatility regime, market regime, or sector regime
- market-relative and sector-relative return features
- rolling z-scores and robust normalization
- interaction terms only when they can be implemented cleanly and time-safely

## What to check first
- Are current features redundant?
- Are features normalized consistently and safely?
- Are any features using data that would not be available at decision time?
- Are the features aligned with the holding period and prediction horizon?
- Are there missing baselines like simple lagged returns, rolling volatility, or relative-strength measures?

## Preferred recommendations
Prioritize:
1. time-safe improvements
2. low-complexity, high-interpretability additions
3. features that match the target horizon
4. features that reduce regime brittleness

## Avoid
- adding many highly correlated indicators without pruning
- adding exotic features before validating simple ones
- adding features that leak market-close or future cross-sectional information
- changing feature logic without explaining data availability timing

## Required output format
Return:
1. Current feature weaknesses
2. Top proposed feature changes
3. Why each should help
4. Implementation notes
5. Leakage or availability caveats

## Implementation guidance
When editing code:
- prefer modular feature functions
- prefer config-driven windows and horizons
- keep feature names explicit
- preserve compatibility with existing pipelines where possible