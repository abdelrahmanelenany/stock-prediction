"""
pipeline/walk_forward.py
Step 4: Walk-forward fold generator for time-series cross-validation.

Fold structure (rolls forward by one test period each time):
  |--- Train 500 ---|--- Val 125 ---|--- Test 125 ---|
                    |--- Train 500 ---|--- Val 125 ---|--- Test 125 ---|
                                     ...
"""
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import TRAIN_DAYS, VAL_DAYS, TEST_DAYS


def generate_walk_forward_folds(
    dates_sorted: list,
    train_days: int = TRAIN_DAYS,
    val_days: int = VAL_DAYS,
    test_days: int = TEST_DAYS,
) -> list[dict]:
    """
    Generates non-overlapping test-period walk-forward folds.

    Parameters
    ----------
    dates_sorted : list
        Sorted list of unique trading dates (strings or Timestamps).
    train_days, val_days, test_days : int
        Number of trading days in each split (from config.py).

    Returns
    -------
    list of dict, each containing:
        fold               : fold number (1-indexed)
        train              : (start_idx, end_idx) index slice into dates_sorted
        val                : (start_idx, end_idx)
        test               : (start_idx, end_idx)
        train_start_date   : first date of training window
        train_end_date     : last  date of training window
        val_start_date     : first date of validation window
        val_end_date       : last  date of validation window
        test_start_date    : first date of test window
        test_end_date      : last  date of test window
    """
    total = len(dates_sorted)
    window = train_days + val_days + test_days
    folds = []
    start = 0

    while start + window <= total:
        train_end = start + train_days
        val_end   = train_end + val_days
        test_end  = val_end + test_days

        folds.append({
            'fold':            len(folds) + 1,
            'train':           (start, train_end),
            'val':             (train_end, val_end),
            'test':            (val_end, test_end),
            'train_start_date': dates_sorted[start],
            'train_end_date':   dates_sorted[train_end - 1],
            'val_start_date':   dates_sorted[train_end],
            'val_end_date':     dates_sorted[val_end - 1],
            'test_start_date':  dates_sorted[val_end],
            'test_end_date':    dates_sorted[test_end - 1],
        })
        start += test_days   # roll forward by exactly one test period

    print(f"Generated {len(folds)} folds "
          f"(train={train_days}d, val={val_days}d, test={test_days}d, "
          f"total dates={total})")
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
    data = pd.read_csv('data/processed/features.csv', parse_dates=['Date'])
    dates = sorted(data['Date'].unique())
    folds = generate_walk_forward_folds(dates)
    print()
    print_fold_summary(folds)
