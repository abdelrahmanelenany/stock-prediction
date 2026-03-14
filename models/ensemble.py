"""
models/ensemble.py
Step 6e: Equal-weighted probability ensemble of all four models.
"""
import numpy as np
import xgboost as xgb


def ensemble_predict(
    lr_model,
    rf_model,
    xgb_model: xgb.Booster,
    X_np: np.ndarray,
) -> np.ndarray:
    """
    Equal-weighted average of class-1 probabilities from the three
    baseline models (LR, RF, XGBoost). Used in backtest/signals.py
    as Prob_ENS alongside the LSTM probability.

    Parameters
    ----------
    lr_model  : fitted LogisticRegression
    rf_model  : fitted RandomForestClassifier
    xgb_model : fitted xgb.Booster
    X_np      : np.ndarray, shape (n_samples, n_features) — scaled features

    Returns
    -------
    np.ndarray of shape (n_samples,) — averaged probability of class 1
    """
    p_lr  = lr_model.predict_proba(X_np)[:, 1]
    p_rf  = rf_model.predict_proba(X_np)[:, 1]
    p_xgb = xgb_model.predict(xgb.DMatrix(X_np))
    return np.mean([p_lr, p_rf, p_xgb], axis=0)
