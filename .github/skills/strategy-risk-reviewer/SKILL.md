---
name: strategy-risk-reviewer
description: reviews whether prediction improvements actually translate into better trading outcomes after costs, turnover, concentration, and regime effects. use when comparing models, evaluating portfolio performance, or deciding whether a statistically better model is economically better.
---

# Strategy Risk Reviewer

Use this skill to judge economic usefulness, not just model quality.

## Objectives
- Determine whether the strategy improvement is real after costs and risk.
- Separate prediction quality from portfolio construction quality.
- Catch cases where better probabilities do not create better trades.

## Evaluate these dimensions
- Sharpe and Sortino
- max drawdown
- CAGR or total return
- turnover and average holding period
- gross vs net performance
- concentration and exposure balance
- long-only vs long-short behavior
- benchmark-relative behavior
- regime dependence

## Key questions
- Does the improvement survive costs?
- Is the gain concentrated in one short subperiod?
- Is turnover too high?
- Is the strategy effectively just a market beta bet?
- Is drawdown now worse even if headline return improved?
- Did the signal improve, or did only sizing / selection logic change?

## Required output format
Return:
1. Economic verdict
2. Main risk concerns
3. Whether the change should be accepted
4. If rejected, what kind of change is more promising next

## Decision rule
Do not endorse a model change purely because classification metrics improved.