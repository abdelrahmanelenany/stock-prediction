"""
models/calibration.py
Probability calibration for LSTM outputs to fix signal imbalance.

The LSTM model outputs probabilities that are often biased above 0.5,
resulting in far more Long signals than Short signals. This module
provides calibration methods to center probabilities around 0.5,
restoring the intended long-short hedge.

Calibration is fit on validation set per fold, then applied to test
predictions. This ensures no look-ahead bias.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    """
    Calibrates raw model probabilities using isotonic regression or Platt scaling.

    Isotonic regression is non-parametric and makes no distributional assumptions.
    It learns a monotonic mapping from raw probabilities to calibrated ones,
    preserving rank order while centering the distribution.

    Usage:
        calibrator = ProbabilityCalibrator(method='isotonic')
        calibrator.fit(val_probs, val_labels)
        calibrated_test_probs = calibrator.transform(test_probs)
    """

    def __init__(self, method: str = 'isotonic'):
        """
        Initialize calibrator.

        Parameters
        ----------
        method : str
            'isotonic' - Isotonic regression (non-parametric, preserves rank)
            'platt' - Platt scaling (sigmoid/logistic fit)
        """
        self.method = method
        self.calibrator = None
        self._fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> 'ProbabilityCalibrator':
        """
        Fit calibrator on validation set probabilities and labels.

        Parameters
        ----------
        probs : np.ndarray
            Raw model probabilities, shape (n_samples,)
        labels : np.ndarray
            True binary labels, shape (n_samples,)

        Returns
        -------
        self
        """
        probs = np.asarray(probs).ravel()
        labels = np.asarray(labels).ravel()

        # Remove NaN values
        valid_mask = ~(np.isnan(probs) | np.isnan(labels))
        probs = probs[valid_mask]
        labels = labels[valid_mask]

        if len(probs) < 10:
            raise ValueError(f"Need at least 10 samples for calibration, got {len(probs)}")

        if self.method == 'isotonic':
            self.calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds='clip'
            )
            self.calibrator.fit(probs, labels)
        elif self.method == 'platt':
            # Platt scaling: fit logistic regression on log-odds
            # Transform probs to log-odds, fit LR, then transform back
            eps = 1e-7
            log_odds = np.log((probs + eps) / (1 - probs + eps))
            self.calibrator = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
            self.calibrator.fit(log_odds.reshape(-1, 1), labels)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self._fitted = True
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply calibration to new probabilities.

        Parameters
        ----------
        probs : np.ndarray
            Raw model probabilities, shape (n_samples,)

        Returns
        -------
        np.ndarray
            Calibrated probabilities, shape (n_samples,)
        """
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        probs = np.asarray(probs).ravel()
        original_shape = probs.shape

        # Handle NaN values
        nan_mask = np.isnan(probs)
        result = np.full_like(probs, np.nan, dtype=np.float64)

        if nan_mask.all():
            return result

        valid_probs = probs[~nan_mask]

        if self.method == 'isotonic':
            calibrated = self.calibrator.predict(valid_probs)
        elif self.method == 'platt':
            eps = 1e-7
            log_odds = np.log((valid_probs + eps) / (1 - valid_probs + eps))
            calibrated = self.calibrator.predict_proba(log_odds.reshape(-1, 1))[:, 1]

        result[~nan_mask] = calibrated
        return result

    def fit_transform(self, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(probs, labels)
        return self.transform(probs)


def calibrate_probabilities_per_fold(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    test_probs: np.ndarray,
    method: str = 'isotonic',
) -> tuple:
    """
    Calibrate probabilities within a single walk-forward fold.

    This is the main entry point for integrating calibration into the pipeline.
    Fits the calibrator on validation data, then transforms both validation
    and test probabilities.

    Parameters
    ----------
    val_probs : np.ndarray
        Validation set raw probabilities
    val_labels : np.ndarray
        Validation set true labels
    test_probs : np.ndarray
        Test set raw probabilities to calibrate
    method : str
        Calibration method ('isotonic' or 'platt')

    Returns
    -------
    tuple
        (calibrated_val_probs, calibrated_test_probs, calibrator)
    """
    calibrator = ProbabilityCalibrator(method=method)
    calibrator.fit(val_probs, val_labels)

    cal_val = calibrator.transform(val_probs)
    cal_test = calibrator.transform(test_probs)

    return cal_val, cal_test, calibrator


def compute_calibration_diagnostics(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute calibration diagnostics for a set of probabilities.

    Returns
    -------
    dict with keys:
        - 'mean_prob': Mean predicted probability
        - 'mean_label': Mean actual label (should be ~0.5 for balanced data)
        - 'bias': mean_prob - mean_label (positive = overconfident on class 1)
        - 'ece': Expected Calibration Error
        - 'bin_stats': Per-bin statistics
    """
    probs = np.asarray(probs).ravel()
    labels = np.asarray(labels).ravel()

    # Remove NaN
    valid_mask = ~(np.isnan(probs) | np.isnan(labels))
    probs = probs[valid_mask]
    labels = labels[valid_mask]

    mean_prob = probs.mean()
    mean_label = labels.mean()
    bias = mean_prob - mean_label

    # ECE: Expected Calibration Error
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = []

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_prob = probs[mask].mean()
            bin_label = labels[mask].mean()
            bin_size = mask.sum()
            ece += (bin_size / len(probs)) * abs(bin_prob - bin_label)
            bin_stats.append({
                'bin': i,
                'range': (bin_edges[i], bin_edges[i + 1]),
                'mean_prob': bin_prob,
                'mean_label': bin_label,
                'count': bin_size,
            })

    return {
        'mean_prob': mean_prob,
        'mean_label': mean_label,
        'bias': bias,
        'ece': ece,
        'bin_stats': bin_stats,
    }
