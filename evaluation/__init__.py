"""Evaluation utilities: metrics, audits, and diagnostic helpers."""

from evaluation.metrics_utils import (
    binary_auc_safe,
    classification_sanity_checks,
    log_split_balance,
)

__all__ = [
    "binary_auc_safe",
    "classification_sanity_checks",
    "log_split_balance",
]
