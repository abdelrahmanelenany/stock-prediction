---
name: experiment-summarizer
description: summarizes research iterations into compact markdown notes with hypothesis, changes, configs, metrics, risks, and next actions. use when an experiment finishes, code is modified, or results need to be recorded in a clean research log.
---

# Experiment Summarizer

Use this skill to leave a clear paper trail after each meaningful analysis or code change.

## Objectives
- Produce concise, decision-ready experiment notes.
- Make it easy to compare runs later.
- Record not just what changed, but why.

## Required sections
Use this structure:

### Experiment title
A short descriptive name.

### Hypothesis
What was expected to improve and why.

### Changes made
- code changes
- config changes
- feature changes
- architecture changes
- backtest or validation changes

### Evaluation setup
- dataset/split or walk-forward setup
- target definition if relevant
- cost assumptions if relevant

### Results
- ML metrics
- trading metrics
- before vs after comparison

### Verdict
One of:
- keep
- discard
- investigate further

### Risks / caveats
Anything that weakens confidence in the conclusion.

### Next best action
The single most useful follow-up.

## Style rules
- Be short.
- Be specific.
- Prefer bullets over long prose.
- Do not hide uncertainty.