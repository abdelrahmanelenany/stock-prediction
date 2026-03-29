---
name: architecture-experimenter
description: proposes and implements controlled model-architecture changes for stock-prediction pipelines, including lstm, gru, tcn, transformer, mlp, dropout, residual, attention, and head-design changes. use when the validation methodology is sound and the next bottleneck is likely model capacity or architecture choice.
---

# Architecture Experimenter

Use this skill only after basic methodology and leakage risks are under control.

## Objectives
- Improve model architecture without turning the experiment into an unbounded rewrite.
- Recommend small, benchmarkable changes.
- Preserve comparability across runs.

## Default policy
- Change one major architectural factor at a time.
- Keep a strong baseline.
- Prefer a clean ablation path.
- Do not stack several architecture innovations at once unless explicitly asked.

## Examples of acceptable changes
- hidden size changes
- number of layers
- dropout placement and strength
- bidirectional vs unidirectional sequence encoders when appropriate
- replacing or adding a simple MLP head
- adding residual connections
- switching between LSTM, GRU, TCN, or a simple Transformer baseline
- revising pooling or final-step extraction logic
- changing loss function if it better matches the target

## What to check before suggesting a change
- Is the model underfitting or overfitting?
- Is sequence length appropriate?
- Is the current architecture too complex for the data volume?
- Is the head too weak or too strong relative to the encoder?
- Are simpler baselines under-tested?

## Avoid
- architecture churn without evidence
- very complex models on small or noisy datasets
- changing architecture before auditing preprocessing and validation
- claiming architecture superiority from a single run

## Required output format
Return:
1. Architectural weakness identified
2. Proposed architecture change
3. Why it fits this repository
4. Minimal implementation plan
5. Validation requirements