"""
models/baselines.py
Step 6a/b/c: Logistic Regression, Random Forest, XGBoost classifiers.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    RF_N_ESTIMATORS, RF_MAX_DEPTH,
    XGB_MAX_DEPTH, XGB_ETA, XGB_SUBSAMPLE, XGB_COLSAMPLE,
    XGB_ROUNDS, XGB_EARLY_STOP, RANDOM_SEED,
)


def train_logistic(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """
    L2-regularised Logistic Regression with 5-fold CV over C.
    Solver lbfgs handles large n_features reliably.
    """
    param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    lr = LogisticRegression(
        penalty='l2', solver='lbfgs', max_iter=1000, random_state=RANDOM_SEED
    )
    cv = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    cv.fit(X_train, y_train)
    print(f"  LR  best C={cv.best_params_['C']:.4f}  CV AUC={cv.best_score_:.4f}")
    return cv.best_estimator_


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Random Forest with 500 trees and sqrt feature sampling (Krauss et al. 2017).
    """
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        max_features='sqrt',
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    rf.fit(X_train, y_train)
    print(f"  RF  n_estimators={RF_N_ESTIMATORS}  max_depth={RF_MAX_DEPTH}")
    return rf


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> xgb.Booster:
    """
    XGBoost with early stopping on AUC evaluated on the validation set.
    Returns the best-iteration booster (no overfitting).
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    params = {
        'max_depth':        XGB_MAX_DEPTH,
        'eta':              XGB_ETA,
        'subsample':        XGB_SUBSAMPLE,
        'colsample_bytree': XGB_COLSAMPLE,
        'objective':        'binary:logistic',
        'eval_metric':      'auc',
        'seed':             RANDOM_SEED,
    }
    model = xgb.train(
        params, dtrain,
        num_boost_round=XGB_ROUNDS,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=XGB_EARLY_STOP,
        verbose_eval=100,
    )
    print(f"  XGB best_iteration={model.best_iteration}  "
          f"best_val_AUC={model.best_score:.4f}")
    return model
