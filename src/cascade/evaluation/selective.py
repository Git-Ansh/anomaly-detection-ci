"""
Selective prediction framework: coverage-accuracy tradeoff analysis.
Implements abstention logic and coverage-accuracy curves.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def coverage_accuracy_curve(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Compute coverage-accuracy tradeoff at multiple confidence thresholds.

    Coverage = fraction of samples where model makes a prediction.
    Accuracy = accuracy on the predicted (confident) subset only.

    Args:
        y_true: True labels
        proba: Calibrated probabilities (n_samples, n_classes)
        thresholds: Confidence thresholds to evaluate (default: 0.50 to 0.99)

    Returns:
        DataFrame with columns: threshold, coverage, accuracy, n_predicted, n_correct
    """
    if thresholds is None:
        thresholds = np.arange(0.50, 1.00, 0.01)

    confidence = np.max(proba, axis=1) if proba.ndim > 1 else np.maximum(proba, 1 - proba)
    predicted = np.argmax(proba, axis=1) if proba.ndim > 1 else (proba >= 0.5).astype(int)

    results = []
    n_total = len(y_true)

    for t in thresholds:
        mask = confidence >= t
        n_predicted = mask.sum()
        coverage = n_predicted / n_total

        if n_predicted > 0:
            accuracy = (y_true[mask] == predicted[mask]).mean()
            n_correct = (y_true[mask] == predicted[mask]).sum()
        else:
            accuracy = np.nan
            n_correct = 0

        results.append({
            'threshold': t,
            'coverage': coverage,
            'accuracy': accuracy,
            'n_predicted': int(n_predicted),
            'n_correct': int(n_correct),
            'n_deferred': int(n_total - n_predicted)
        })

    return pd.DataFrame(results)


def selective_risk(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """
    Compute selective prediction risk at a given threshold.

    Args:
        y_true: True labels
        proba: Calibrated probabilities
        threshold: Confidence threshold

    Returns:
        Dict with coverage, accuracy, risk (1-accuracy), and n_deferred
    """
    confidence = np.max(proba, axis=1) if proba.ndim > 1 else np.maximum(proba, 1 - proba)
    predicted = np.argmax(proba, axis=1) if proba.ndim > 1 else (proba >= 0.5).astype(int)

    mask = confidence >= threshold
    n_total = len(y_true)
    n_predicted = mask.sum()
    coverage = n_predicted / n_total

    if n_predicted > 0:
        accuracy = (y_true[mask] == predicted[mask]).mean()
    else:
        accuracy = np.nan

    return {
        'coverage': coverage,
        'accuracy': accuracy,
        'risk': 1 - accuracy if not np.isnan(accuracy) else np.nan,
        'n_predicted': int(n_predicted),
        'n_deferred': int(n_total - n_predicted),
        'threshold': threshold
    }


def find_threshold_for_coverage(
    proba: np.ndarray,
    target_coverage: float
) -> float:
    """
    Find the confidence threshold that achieves approximately the target coverage.

    Args:
        proba: Calibrated probabilities
        target_coverage: Desired fraction of samples to predict on

    Returns:
        Confidence threshold
    """
    confidence = np.max(proba, axis=1) if proba.ndim > 1 else np.maximum(proba, 1 - proba)
    # Threshold = quantile such that (1 - target_coverage) fraction is below it
    threshold = np.quantile(confidence, 1 - target_coverage)
    return float(threshold)


def find_threshold_for_accuracy(
    y_true: np.ndarray,
    proba: np.ndarray,
    target_accuracy: float = 0.90,
    min_coverage: float = 0.10
) -> float:
    """
    Find the lowest confidence threshold that achieves target accuracy.

    Args:
        y_true: True labels
        proba: Calibrated probabilities
        target_accuracy: Desired accuracy on predicted subset
        min_coverage: Minimum acceptable coverage

    Returns:
        Confidence threshold
    """
    curve = coverage_accuracy_curve(y_true, proba)
    valid = curve[(curve['accuracy'] >= target_accuracy) & (curve['coverage'] >= min_coverage)]

    if len(valid) == 0:
        # Can't achieve target, return highest accuracy threshold
        return float(curve.loc[curve['accuracy'].idxmax(), 'threshold'])

    # Return threshold with highest coverage that meets accuracy target
    return float(valid.loc[valid['coverage'].idxmax(), 'threshold'])


def workload_reduction(
    n_total: int,
    n_deferred: int
) -> Dict[str, float]:
    """
    Compute workload reduction metrics.

    Args:
        n_total: Total number of items
        n_deferred: Items deferred to human review

    Returns:
        Dict with automated count, deferred count, and reduction percentage
    """
    n_automated = n_total - n_deferred
    reduction_pct = n_automated / n_total * 100 if n_total > 0 else 0

    return {
        'n_total': n_total,
        'n_automated': n_automated,
        'n_deferred': n_deferred,
        'reduction_pct': reduction_pct
    }
