---
name: walk-forward-validator
description: checks time-series and cross-sectional evaluation pipelines for look-ahead bias, leakage, unsafe preprocessing, bad split logic, and improper walk-forward validation. use when validating methodology, auditing suspicious gains, or before trusting model improvements.
---

# Walk-Forward Validator

Use this skill as the methodological gatekeeper.

## Objectives
- Verify that the reported performance could realistically have been achieved without future information.
- Audit every step where leakage commonly enters the pipeline.
- Reject performance gains that are not methodologically trustworthy.

## Mandatory checks
Check whether any of the following are fit or computed using future data:
- scalers
- imputers
- winsorization thresholds
- PCA or feature selection
- target encoding
- rolling statistics with improper alignment
- cross-sectional statistics from the wrong date
- label construction using information beyond the intended horizon

## Split integrity checklist
- train, validation, and test are chronologically ordered
- no random shuffle where time order matters
- no overlap that contaminates the future prediction period
- re-training / re-fitting happens only with historically available data
- feature windows are aligned correctly with targets
- prediction timestamp matches data availability timestamp

## Common failure patterns
- scaler fit once on the whole dataset
- rolling windows centered or misaligned
- label creation shifted incorrectly
- sector or market aggregates computed with future constituents or future returns
- post-close data used for same-day decisions without justification
- threshold tuning on test data

## Required output format
Return:
1. Leakage risk summary
2. Confirmed safe elements
3. Suspected unsafe elements
4. Exact code locations or logic patterns to inspect
5. Whether current performance should be trusted

## Default stance
If a gain looks unusually strong, be skeptical first.