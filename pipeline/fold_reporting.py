"""Per-fold audit artifacts (dates, sample counts, class balance, quick metrics)."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np


def save_fold_report(
    fold: dict,
    df_tr: pd.DataFrame,
    df_v: pd.DataFrame,
    df_ts: pd.DataFrame,
    target_col: str,
    extra: dict[str, Any] | None = None,
    reports_dir: str = "reports/fold_reports",
) -> str:
    """
    Write a JSON summary for one walk-forward fold.

    Parameters
    ----------
    fold : dict
        Fold descriptor from generate_walk_forward_folds.
    df_tr, df_v, df_ts : DataFrame
        Train / val / test splits for this fold.
    target_col : str
        Binary label column name.
    extra : dict, optional
        Additional keys (e.g. val AUC, test AUC).
    """
    os.makedirs(reports_dir, exist_ok=True)
    n_tr = len(df_tr)
    n_v = len(df_v)
    n_ts = len(df_ts)

    def _balance(df: pd.DataFrame) -> dict[str, int]:
        if target_col not in df.columns or len(df) == 0:
            return {}
        vc = df[target_col].value_counts()
        return {str(k): int(v) for k, v in vc.items()}

    payload: dict[str, Any] = {
        "fold": int(fold["fold"]),
        "train_window_mode": fold.get("train_window_mode", "rolling"),
        "stride_days": fold.get("stride_days"),
        "train": {
            "start": str(fold["train_start_date"])[:10],
            "end": str(fold["train_end_date"])[:10],
            "n_rows": n_tr,
            "target_balance": _balance(df_tr),
        },
        "val": {
            "start": str(fold["val_start_date"])[:10],
            "end": str(fold["val_end_date"])[:10],
            "n_rows": n_v,
            "target_balance": _balance(df_v),
        },
        "test": {
            "start": str(fold["test_start_date"])[:10],
            "end": str(fold["test_end_date"])[:10],
            "n_rows": n_ts,
            "target_balance": _balance(df_ts),
        },
    }
    if extra:
        # JSON-serialize safe
        safe_extra = {}
        for k, v in extra.items():
            if isinstance(v, (np.floating, np.integer)):
                safe_extra[k] = float(v) if isinstance(v, np.floating) else int(v)
            elif isinstance(v, (float, int, str, bool)) or v is None:
                safe_extra[k] = v
            else:
                safe_extra[k] = str(v)
        payload["metrics"] = safe_extra

    path = os.path.join(reports_dir, f"fold_{fold['fold']:03d}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
