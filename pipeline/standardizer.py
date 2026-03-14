"""
pipeline/standardizer.py
Step 5: Fit-on-train-only StandardScaler helper.

CRITICAL anti-leakage rule: the scaler is always fitted exclusively on
training data. Validation and test sets are only transformed, never used
to fit. This prevents any future information from contaminating the model.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler


def standardize_fold(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler on training data, then apply to all three splits.

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray, shape (n_samples, n_features)
        Raw (unscaled) feature arrays for each split.

    Returns
    -------
    X_train_s, X_val_s, X_test_s : np.ndarray
        Scaled arrays (zero mean, unit variance relative to training set).
    scaler : StandardScaler
        Fitted scaler (saved per fold so LSTM sequences can be scaled
        consistently via scaled_df helper in main.py).
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)  # fit + transform on train only
    X_val_s   = scaler.transform(X_val)        # transform only
    X_test_s  = scaler.transform(X_test)       # transform only
    return X_train_s, X_val_s, X_test_s, scaler
