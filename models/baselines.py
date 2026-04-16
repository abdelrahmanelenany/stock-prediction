"""
models/baselines.py
Step 6a/b/c: Logistic Regression, Random Forest, XGBoost classifiers.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    RF_PARAM_GRID,
    XGB_PARAM_GRID, XGB_COLSAMPLE,
    XGB_ROUNDS, XGB_EARLY_STOP, XGB_REG_ALPHA, XGB_REG_LAMBDA,
    RANDOM_SEED,
)


def train_logistic(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """
    L2-regularised Logistic Regression with time-series-aware CV over C.
    Uses TimeSeriesSplit to respect temporal ordering within the training window.
    """
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    lr = LogisticRegression(
        solver='lbfgs', max_iter=1000, random_state=RANDOM_SEED
    )
    tscv = TimeSeriesSplit(n_splits=5)
    cv = GridSearchCV(lr, param_grid, cv=tscv, scoring='roc_auc', n_jobs=-1)
    cv.fit(X_train, y_train)
    print(f"  LR  best C={cv.best_params_['C']:.4f}  CV AUC={cv.best_score_:.4f}")
    return cv.best_estimator_


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> RandomForestClassifier:
    """
    Random Forest with validation-based hyperparameter selection.
    Evaluates all combinations in RF_PARAM_GRID on the held-out
    validation set (AUC-ROC) to select the best configuration.
    """
    best_auc = -1.0
    best_model = None
    best_params = {}

    from itertools import product
    keys = list(RF_PARAM_GRID.keys())
    values = list(RF_PARAM_GRID.values())

    for combo in product(*values):
        params = dict(zip(keys, combo))
        rf = RandomForestClassifier(
            max_features='sqrt',
            n_jobs=-1,
            random_state=RANDOM_SEED,
            **params,
        )
        rf.fit(X_train, y_train)
        val_probs = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_probs)

        if auc > best_auc:
            best_auc = auc
            best_model = rf
            best_params = params

    print(f"  RF  best params={best_params}  val AUC={best_auc:.4f}")
    return best_model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> xgb.Booster:
    """
    XGBoost with grid search over XGB_PARAM_GRID.
    Each combo uses early stopping on validation AUC.
    Returns the booster with the best validation AUC.
    """
    from itertools import product

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)

    best_auc   = -1.0
    best_model = None
    best_p     = {}

    keys   = list(XGB_PARAM_GRID.keys())
    values = list(XGB_PARAM_GRID.values())

    for combo in product(*values):
        p = dict(zip(keys, combo))
        params = {
            'max_depth':        p['max_depth'],
            'eta':              p['eta'],
            'subsample':        p['subsample'],
            'colsample_bytree': XGB_COLSAMPLE,
            'reg_alpha':        XGB_REG_ALPHA,
            'reg_lambda':       XGB_REG_LAMBDA,
            'objective':        'binary:logistic',
            'eval_metric':      'auc',
            'seed':             RANDOM_SEED,
        }
        model = xgb.train(
            params, dtrain,
            num_boost_round=XGB_ROUNDS,
            evals=[(dval, 'val')],
            early_stopping_rounds=XGB_EARLY_STOP,
            verbose_eval=0,
        )
        if model.best_score > best_auc:
            best_auc   = model.best_score
            best_model = model
            best_p     = p

    print(f"  XGB best params={best_p}  val AUC={best_auc:.4f}")
    return best_model


def extract_feature_importances(
    lr_model: LogisticRegression,
    rf_model: RandomForestClassifier,
    xgb_model: xgb.Booster,
    feature_cols: list,
) -> dict:
    """
    Extract per-feature importance scores from the three baseline models.

    Returns a dict keyed by feature name, each value a dict with keys
    'LR_coef', 'RF_importance', 'XGB_gain'.

    LR  : absolute value of the coefficient (magnitude of linear weight).
    RF  : mean decrease in impurity (sklearn feature_importances_).
    XGB : average gain across all splits on that feature.

    All three are normalised to sum to 1 so they are comparable across models.
    Missing XGB features (zero gain) are filled with 0 before normalisation.
    """
    n = len(feature_cols)
    result = {f: {'LR_coef': 0.0, 'RF_importance': 0.0, 'XGB_gain': 0.0}
              for f in feature_cols}

    # ── LR coefficients ───────────────────────────────────────────────────
    coef = np.abs(lr_model.coef_).flatten()          # shape (n_features,)
    coef_norm = coef / (coef.sum() + 1e-12)
    for i, feat in enumerate(feature_cols):
        result[feat]['LR_coef'] = float(coef_norm[i])

    # ── RF mean-decrease impurity ──────────────────────────────────────────
    rf_imp = rf_model.feature_importances_           # shape (n_features,)
    rf_norm = rf_imp / (rf_imp.sum() + 1e-12)
    for i, feat in enumerate(feature_cols):
        result[feat]['RF_importance'] = float(rf_norm[i])

    # ── XGBoost gain ──────────────────────────────────────────────────────
    # DMatrix without feature names uses 'f0', 'f1', ... internally.
    raw_gain = xgb_model.get_score(importance_type='gain')   # {f0: val, ...}
    xgb_arr = np.zeros(n)
    for fname, val in raw_gain.items():
        # fname is 'f0', 'f1', etc.  Extract the index.
        try:
            idx = int(fname[1:])
            if idx < n:
                xgb_arr[idx] = val
        except ValueError:
            pass
    xgb_norm = xgb_arr / (xgb_arr.sum() + 1e-12)
    for i, feat in enumerate(feature_cols):
        result[feat]['XGB_gain'] = float(xgb_norm[i])

    return result
