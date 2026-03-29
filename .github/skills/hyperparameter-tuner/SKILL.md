---
name: hyperparameter-tuner
description: proposes safe, bounded hyperparameter improvements for stock-prediction experiments and edits config-driven search spaces for learning rate, dropout, hidden size, sequence length, batch size, regularization, schedulers, and early stopping. use when the pipeline is sound and performance may improve through better tuning rather than major redesign.
---

# Hyperparameter Tuner

Use this skill when the model and methodology are broadly reasonable but performance may be limited by suboptimal settings.

## Objectives
- Tune efficiently.
- Avoid search explosions.
- Prefer config edits over hardcoded values.
- Keep searches reproducible and reviewable.

## Tune these first
- learning rate
- weight decay / L2 regularization
- dropout
- hidden size
- number of layers
- batch size
- sequence length
- early stopping patience
- scheduler settings
- thresholding rules if probability-to-position logic exists

## Search policy
- Start with narrow, high-probability ranges.
- Favor logarithmic intuition for learning rate and regularization.
- Do not expand the search space unless earlier runs justify it.
- Keep the number of simultaneous free variables manageable.

## Red flags
- tuning after a likely leakage issue
- tuning many parameters on a tiny validation set
- interpreting one lucky run as a robust improvement
- changing parameters in code instead of in config without a good reason

## Required output format
Return:
1. Most likely mis-tuned parameters
2. Proposed new values or search ranges
3. Why these ranges make sense
4. Exact config/code edits needed
5. How to compare runs fairly

## Implementation guidance
Prefer:
- YAML or config-file edits
- a fixed random seed strategy when appropriate
- explicit experiment naming
- compact summaries of old vs new settings