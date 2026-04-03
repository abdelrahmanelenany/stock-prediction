"""
pipeline/walk_forward.py
Step 4: Walk-forward fold generator for time-series cross-validation.

Fold structure (rolls forward by one test period each time):
    |--- Train 252 ---|--- Val 63 ---|--- Test 63 ---|
                                        |--- Train 252 ---|--- Val 63 ---|--- Test 63 ---|
                                     ...

Modes:
  rolling    — fixed train length; window slides by stride_days
  expanding  — train always dates[0:val_start); val/test blocks slide by stride_days
"""
from __future__ import annotations

import sys
import os
from typing import Literal
import pandas as pd
import config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import TRAIN_DAYS, VAL_DAYS, TEST_DAYS, WALK_FORWARD_STRIDE, TRAIN_WINDOW_MODE


def generate_walk_forward_folds(
    dates_sorted: list,
    train_days: int = TRAIN_DAYS,
    val_days: int = VAL_DAYS,
    test_days: int = TEST_DAYS,
    stride_days: int | None = None,
    train_window_mode: Literal["rolling", "expanding"] | str | None = None,
) -> list[dict]:
    """
    Generates walk-forward folds with strict chronological ordering.

    Parameters
    ----------
    dates_sorted : list
        Sorted list of unique trading dates (strings or Timestamps).
    train_days, val_days, test_days : int
        Sizes of each segment (rolling mode) or minimum train span (expanding).
    stride_days : int, optional
        Index advance between folds. Default: config WALK_FORWARD_STRIDE or test_days.
    train_window_mode : str
        'rolling' (default) or 'expanding'.

    Returns
    -------
    list of dict with index slices into dates_sorted and boundary dates.
    """
    if stride_days is None:
        stride_days = WALK_FORWARD_STRIDE if WALK_FORWARD_STRIDE is not None else test_days
    mode = (train_window_mode or TRAIN_WINDOW_MODE or "rolling").lower()
    if mode not in ("rolling", "expanding"):
        raise ValueError(f"train_window_mode must be 'rolling' or 'expanding', got {mode!r}")

    total = len(dates_sorted)
    folds: list[dict] = []

    if mode == "rolling":
        window = train_days + val_days + test_days
        start = 0
        while start + window <= total:
            train_end = start + train_days
            val_end = train_end + val_days
            test_end = val_end + test_days

            folds.append({
                'fold': len(folds) + 1,
                'train': (start, train_end),
                'val': (train_end, val_end),
                'test': (val_end, test_end),
                'train_start_date': dates_sorted[start],
                'train_end_date': dates_sorted[train_end - 1],
                'val_start_date': dates_sorted[train_end],
                'val_end_date': dates_sorted[val_end - 1],
                'test_start_date': dates_sorted[val_end],
                'test_end_date': dates_sorted[test_end - 1],
                'train_window_mode': mode,
                'stride_days': stride_days,
            })
            start += stride_days
    else:
        # Expanding train: train indices [0, val_start), val [val_start, val_end), test [...]
        val_start = train_days
        while True:
            val_end = val_start + val_days
            test_end = val_end + test_days
            if test_end > total:
                break
            folds.append({
                'fold': len(folds) + 1,
                'train': (0, val_start),
                'val': (val_start, val_end),
                'test': (val_end, test_end),
                'train_start_date': dates_sorted[0],
                'train_end_date': dates_sorted[val_start - 1],
                'val_start_date': dates_sorted[val_start],
                'val_end_date': dates_sorted[val_end - 1],
                'test_start_date': dates_sorted[val_end],
                'test_end_date': dates_sorted[test_end - 1],
                'train_window_mode': mode,
                'stride_days': stride_days,
            })
            val_start += stride_days

    print(
        f"Generated {len(folds)} folds mode={mode} stride={stride_days} "
        f"(train={'expanding from 0' if mode == 'expanding' else train_days}, "
        f"val={val_days}d, test={test_days}d, total dates={total})"
    )
    assert len(folds) >= 8, (
        f"Only {len(folds)} folds generated. Check TRAIN/VAL/TEST_DAYS vs date range. "
        f"Need at least 8 folds for statistically meaningful walk-forward evaluation."
    )
    return folds


def print_fold_summary(folds: list[dict]) -> None:
    """Pretty-print a summary table of all folds (for thesis Table T2)."""
    header = (f"{'Fold':>4} | {'Train start':>12} | {'Train end':>12} | "
              f"{'Val start':>12} | {'Val end':>12} | "
              f"{'Test start':>12} | {'Test end':>12}")
    print(header)
    print('-' * len(header))
    for f in folds:
        print(
            f"{f['fold']:>4} | "
            f"{str(f['train_start_date'])[:10]:>12} | "
            f"{str(f['train_end_date'])[:10]:>12} | "
            f"{str(f['val_start_date'])[:10]:>12} | "
            f"{str(f['val_end_date'])[:10]:>12} | "
            f"{str(f['test_start_date'])[:10]:>12} | "
            f"{str(f['test_end_date'])[:10]:>12}"
        )


if __name__ == '__main__':
    features_cache = f'data/processed/features_{config.UNIVERSE_MODE}.csv'
    data = pd.read_csv(features_cache, parse_dates=['Date'])
    dates = sorted(data['Date'].unique())
    folds = generate_walk_forward_folds(dates)
    print()
    print_fold_summary(folds)
