"""
pipeline/standardizer.py
Step 5: Feature standardization — fit on train only, transform all splits.

Supports both StandardScaler (default) and MinMaxScaler (Bhandari §4.5).
The scaler type is configured via config.SCALER_TYPE.

ANTI-LEAKAGE: Scaler is ALWAYS fit on training data only, then applied to
validation and test sets.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Union

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


def get_scaler(scaler_type: str = None) -> Union[StandardScaler, MinMaxScaler]:
    """
    Factory function for scaler instantiation.

    Parameters
    ----------
    scaler_type : str, optional
        'standard' (default) or 'minmax'. If None, uses config.SCALER_TYPE.

    Returns
    -------
    sklearn scaler instance (unfitted)
    """
    if scaler_type is None:
        scaler_type = getattr(config, 'SCALER_TYPE', 'standard')

    if scaler_type.lower() == 'minmax':
        return MinMaxScaler()
    return StandardScaler()


def standardize_fold(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    scaler_type: str = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Union[StandardScaler, MinMaxScaler]]:
    """
    Fit a scaler on training data, then apply to all three splits.

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray, shape (n_samples, n_features)
        Raw (unscaled) feature arrays for each split.
    scaler_type : str, optional
        'standard' or 'minmax'. Defaults to config.SCALER_TYPE.

    Returns
    -------
    X_train_s, X_val_s, X_test_s : np.ndarray
        Scaled arrays.
    scaler : StandardScaler or MinMaxScaler
        Fitted scaler (saved per fold so LSTM sequences can be scaled
        consistently via scaled_df helper in main.py).

    Notes
    -----
    StandardScaler: zero mean, unit variance (robust to outliers in returns)
    MinMaxScaler: scales to [0, 1] (Bhandari §4.5 approach)
    """
    scaler = get_scaler(scaler_type)
    X_train_s = scaler.fit_transform(X_train)  # fit + transform on train only
    X_val_s   = scaler.transform(X_val)        # transform only
    X_test_s  = scaler.transform(X_test)       # transform only
    return X_train_s, X_val_s, X_test_s, scaler


def winsorize_fold(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    lower_q: float = 0.005,
    upper_q: float = 0.995,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Clip features to train-derived quantiles (column-wise), then apply same
    bounds to val and test. PIT-safe: quantiles use training rows only.
    """
    X_train = np.asarray(X_train, dtype=np.float64).copy()
    X_val = np.asarray(X_val, dtype=np.float64).copy()
    X_test = np.asarray(X_test, dtype=np.float64).copy()
    n_feat = X_train.shape[1]
    for j in range(n_feat):
        col = X_train[:, j]
        lo = float(np.quantile(col, lower_q))
        hi = float(np.quantile(col, upper_q))
        if lo > hi:
            lo, hi = hi, lo
        X_train[:, j] = np.clip(X_train[:, j], lo, hi)
        X_val[:, j] = np.clip(X_val[:, j], lo, hi)
        X_test[:, j] = np.clip(X_test[:, j], lo, hi)
    return X_train, X_val, X_test


def standardize_train_val(
    X_train: np.ndarray,
    X_val: np.ndarray,
    scaler_type: str = None,
) -> tuple[np.ndarray, np.ndarray, Union[StandardScaler, MinMaxScaler]]:
    """
    Fit a scaler on training data, then apply to train and validation splits.
    Useful for hyperparameter tuning where test set is not yet needed.

    Parameters
    ----------
    X_train, X_val : np.ndarray, shape (n_samples, n_features)
        Raw (unscaled) feature arrays.
    scaler_type : str, optional
        'standard' or 'minmax'. Defaults to config.SCALER_TYPE.

    Returns
    -------
    X_train_s, X_val_s : np.ndarray
        Scaled arrays.
    scaler : StandardScaler or MinMaxScaler
        Fitted scaler.
    """
    scaler = get_scaler(scaler_type)
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    return X_train_s, X_val_s, scaler
