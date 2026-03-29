"""
Reusable classification metrics and audit helpers.

Ensures AUC is computed from continuous scores (probabilities), not hard labels.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def binary_auc_safe(
    y_true: np.ndarray,
    y_score: np.ndarray,
    log_on_fail: bool = True,
) -> float | None:
    """
    ROC-AUC for binary labels using predicted scores (positive-class probability).

    Returns None if AUC is undefined (single class in y_true or invalid inputs).
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    if len(y_true) == 0 or len(y_score) == 0:
        if log_on_fail:
            logger.warning("binary_auc_safe: empty inputs")
        return None
    if len(np.unique(y_true)) < 2:
        if log_on_fail:
            logger.debug("binary_auc_safe: single class in y_true — AUC undefined")
        return None
    if np.any(np.isnan(y_score)) or np.any(np.isinf(y_score)):
        if log_on_fail:
            logger.warning("binary_auc_safe: NaN/Inf in y_score")
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError as e:
        if log_on_fail:
            logger.warning("binary_auc_safe: %s", e)
        return None


def classification_sanity_checks(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    name: str = "split",
    prob_std_floor: float = 1e-8,
) -> dict[str, Any]:
    """
    Detect pathological prediction distributions (constant scores, NaNs).

    Returns a dict suitable for logging or fold reports.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    out: dict[str, Any] = {"name": name}
    out["n"] = int(len(y_true))
    out["n_nan_prob"] = int(np.isnan(y_prob).sum())
    out["n_inf_prob"] = int(np.isinf(y_prob).sum())
    if len(y_true) > 0:
        uniq, cnt = np.unique(y_true, return_counts=True)
        out["class_counts"] = {int(u): int(c) for u, c in zip(uniq, cnt)}
    else:
        out["class_counts"] = {}
    std_p = float(np.nanstd(y_prob)) if len(y_prob) else 0.0
    out["prob_std"] = std_p
    out["near_constant_prob"] = std_p < prob_std_floor
    if out["near_constant_prob"] and len(y_prob) > 1:
        logger.warning(
            "[%s] Near-constant predicted probabilities (std=%.2e)",
            name,
            std_p,
        )
    if out["n_nan_prob"] or out["n_inf_prob"]:
        logger.warning(
            "[%s] Invalid probs: nan=%s inf=%s",
            name,
            out["n_nan_prob"],
            out["n_inf_prob"],
        )
    return out


def log_split_balance(y: np.ndarray, split_name: str, log: logging.Logger | None = None) -> None:
    """Log class balance for a binary (or multi-class) label vector."""
    log = log or logger
    y = np.asarray(y).ravel()
    if len(y) == 0:
        log.info("[%s] class balance: empty", split_name)
        return
    uniq, cnt = np.unique(y, return_counts=True)
    parts = [f"{int(u)}:{int(c)}" for u, c in zip(uniq, cnt)]
    log.info("[%s] class balance %s (n=%d)", split_name, ", ".join(parts), len(y))
