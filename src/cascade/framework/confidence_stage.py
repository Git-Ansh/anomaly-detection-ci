"""
Generalizable Confidence-Gated Classification Stage.

A single reusable building block for the cascade. Each stage:
1. Trains a classifier on labeled data
2. Calibrates probabilities (isotonic / Platt scaling)
3. Tunes confidence thresholds via out-of-fold predictions
4. At inference: predicts class + confidence, gates uncertain cases

This is dataset-agnostic. To adapt to a new CI system:
- Provide features, labels, and class names
- The stage handles calibration, threshold tuning, and routing automatically
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold


class ConfidenceStage:
    """
    A single confidence-gated classification stage.

    Encapsulates: model training, probability calibration, OOF threshold tuning,
    per-class confidence gates, and prediction with abstention.

    Usage:
        stage = ConfidenceStage(
            name='disposition',
            classes={4: 'Actionable', 6: 'Wontfix', 7: 'Fixed', 1: 'Downstream'},
            target_accuracy=0.85,
        )
        stage.fit(X_train, y_train, feature_names=feature_cols)
        predictions = stage.predict(X_test)
        # predictions['class'] = predicted class or -1 (deferred)
        # predictions['confidence'] = calibrated probability
        # predictions['is_confident'] = True/False
    """

    def __init__(
        self,
        name: str,
        classes: Dict[int, str],
        target_accuracy: float = 0.85,
        calibration_method: str = 'isotonic',
        n_cv_folds: int = 5,
        random_state: int = 42,
        model: Optional[BaseEstimator] = None,
        defer_label: int = -1,
    ):
        """
        Args:
            name: Stage name (for logging)
            classes: Mapping of class_code -> class_name
            target_accuracy: Target accuracy per class for threshold tuning
            calibration_method: 'isotonic' or 'sigmoid'
            n_cv_folds: Number of CV folds for calibration and OOF
            random_state: Random seed
            model: Custom model (default: RandomForestClassifier)
            defer_label: Label to use for deferred/uncertain predictions
        """
        self.name = name
        self.classes = classes
        self.target_accuracy = target_accuracy
        self.calibration_method = calibration_method
        self.n_cv_folds = n_cv_folds
        self.random_state = random_state
        self.defer_label = defer_label

        self._model = model
        self._calibrated_model = None
        self._label_encoder = None
        self._scaler = None
        self._per_class_thresholds = None
        self._feature_names = None
        self._oof_proba = None
        self._is_fitted = False

    def _default_model(self) -> BaseEstimator:
        """Default model if none provided."""
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        scale: bool = True,
    ) -> 'ConfidenceStage':
        """
        Train the stage: fit model, calibrate, tune thresholds.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (original class codes, not encoded)
            feature_names: Optional feature column names
            scale: Whether to apply StandardScaler

        Returns:
            self
        """
        self._feature_names = feature_names

        # Encode labels to 0..n-1
        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(y)

        # Scale features
        if scale:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
        else:
            self._scaler = None
            X_scaled = X

        n_classes = len(self._label_encoder.classes_)
        class_counts = np.bincount(y_enc, minlength=n_classes)
        print(f"[{self.name}] Training: {len(y)} samples, {n_classes} classes")
        for i, code in enumerate(self._label_encoder.classes_):
            label = self.classes.get(int(code), str(code))
            print(f"  {label} ({code}): {class_counts[i]}")

        # Get base model
        base_model = self._model if self._model is not None else self._default_model()

        # Step 1: Generate OOF predictions for threshold tuning
        print(f"[{self.name}] Generating OOF predictions ({self.n_cv_folds}-fold)...")
        self._oof_proba = self._get_oof_predictions(base_model, X_scaled, y_enc)

        # Step 2: Find per-class thresholds
        self._per_class_thresholds = self._find_per_class_thresholds(
            y_enc, self._oof_proba
        )
        print(f"[{self.name}] Per-class thresholds:")
        for i, code in enumerate(self._label_encoder.classes_):
            label = self.classes.get(int(code), str(code))
            print(f"  {label}: {self._per_class_thresholds[i]:.2f}")

        # Step 3: Train calibrated model on full data
        print(f"[{self.name}] Training calibrated model...")
        self._calibrated_model = CalibratedClassifierCV(
            clone(base_model),
            method=self.calibration_method,
            cv=self.n_cv_folds,
        )
        self._calibrated_model.fit(X_scaled, y_enc)

        # Report OOF performance
        self._report_oof_performance(y_enc)

        self._is_fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
        return_proba: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Predict with confidence gating.

        Args:
            X: Feature matrix
            return_proba: Whether to include full probability matrix

        Returns:
            Dict with keys:
                'class': predicted class codes (-1 for deferred)
                'confidence': calibrated max probability
                'is_confident': boolean mask
                'predicted_raw': predicted class codes (ignoring threshold)
                'proba': full probability matrix (if return_proba=True)
        """
        if not self._is_fitted:
            raise RuntimeError(f"Stage '{self.name}' is not fitted. Call fit() first.")

        if self._scaler is not None:
            X_scaled = self._scaler.transform(X)
        else:
            X_scaled = X

        proba = self._calibrated_model.predict_proba(X_scaled)
        confidence = np.max(proba, axis=1)
        predicted_idx = np.argmax(proba, axis=1)
        predicted_codes = self._label_encoder.inverse_transform(predicted_idx)

        # Apply per-class thresholds
        per_sample_threshold = self._per_class_thresholds[predicted_idx]
        is_confident = confidence >= per_sample_threshold

        gated_codes = np.where(is_confident, predicted_codes, self.defer_label)

        result = {
            'class': gated_codes,
            'confidence': confidence,
            'is_confident': is_confident,
            'predicted_raw': predicted_codes,
        }

        if return_proba:
            result['proba'] = proba
            result['class_codes'] = self._label_encoder.classes_

        return result

    def get_oof_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return OOF predictions for downstream stage training.

        Returns:
            (oof_proba, label_encoder_classes)
        """
        if self._oof_proba is None:
            raise RuntimeError("No OOF predictions available. Call fit() first.")
        return self._oof_proba, self._label_encoder.classes_

    def coverage_accuracy_curve(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Compute coverage-accuracy tradeoff on a test set.

        Args:
            X: Test features
            y_true: True class codes
            thresholds: Confidence thresholds to evaluate

        Returns:
            DataFrame with threshold, coverage, accuracy columns
        """
        if thresholds is None:
            thresholds = np.arange(0.40, 0.96, 0.05)

        preds = self.predict(X)
        confidence = preds['confidence']
        predicted_raw = preds['predicted_raw']

        results = []
        for t in thresholds:
            mask = confidence >= t
            n = mask.sum()
            if n > 0:
                acc = (y_true[mask] == predicted_raw[mask]).mean()
                results.append({
                    'threshold': t,
                    'coverage': n / len(y_true),
                    'accuracy': acc,
                    'n_predicted': int(n),
                    'n_deferred': int(len(y_true) - n),
                })

        return pd.DataFrame(results)

    def _get_oof_predictions(
        self,
        base_model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Generate out-of-fold probability predictions."""
        n_classes = len(np.unique(y))
        oof_proba = np.zeros((len(y), n_classes))

        skf = StratifiedKFold(
            n_splits=self.n_cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        for train_idx, val_idx in skf.split(X, y):
            fold_model = clone(base_model)
            fold_model.fit(X[train_idx], y[train_idx])
            oof_proba[val_idx] = fold_model.predict_proba(X[val_idx])

        return oof_proba

    def _find_per_class_thresholds(
        self,
        y_true: np.ndarray,
        proba: np.ndarray,
        min_samples: int = 5,
    ) -> np.ndarray:
        """Find per-class confidence thresholds targeting self.target_accuracy."""
        n_classes = proba.shape[1]
        thresholds = np.full(n_classes, 0.50)
        predicted = np.argmax(proba, axis=1)

        for c in range(n_classes):
            class_mask = predicted == c
            if class_mask.sum() < min_samples:
                continue

            class_true = y_true[class_mask]
            class_conf = np.max(proba[class_mask], axis=1)

            best_t = 0.50
            best_cov = 0.0
            for t in np.arange(0.40, 0.96, 0.01):
                t_mask = class_conf >= t
                if t_mask.sum() < min_samples:
                    continue
                acc = (class_true[t_mask] == c).mean()
                cov = t_mask.mean()
                if acc >= self.target_accuracy and cov > best_cov:
                    best_t = t
                    best_cov = cov

            thresholds[c] = best_t

        return thresholds

    def _report_oof_performance(self, y_enc: np.ndarray):
        """Print OOF performance summary."""
        oof_conf = np.max(self._oof_proba, axis=1)
        oof_pred = np.argmax(self._oof_proba, axis=1)

        # At per-class thresholds
        per_sample_t = self._per_class_thresholds[oof_pred]
        is_conf = oof_conf >= per_sample_t
        n_conf = is_conf.sum()

        if n_conf > 0:
            acc = (y_enc[is_conf] == oof_pred[is_conf]).mean()
            cov = n_conf / len(y_enc)
            print(f"[{self.name}] OOF: {acc:.1%} accuracy, {cov:.1%} coverage "
                  f"({n_conf}/{len(y_enc)})")

    @property
    def class_names(self) -> Dict[int, str]:
        """Return the class code -> name mapping."""
        return self.classes.copy()

    @property
    def feature_names(self) -> Optional[List[str]]:
        """Return feature names if provided during fit."""
        return self._feature_names
