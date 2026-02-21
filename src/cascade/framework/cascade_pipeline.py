"""
Generalizable Confidence-Gated Cascade Pipeline.

Chains multiple ConfidenceStages with configurable routing.
Each stage processes a subset of inputs, and its confident outputs
are routed to the next stage or to a terminal decision.

To apply to a new CI system:
1. Define stages (classes, features, target accuracy)
2. Define routing rules (which output goes where)
3. Provide labeled training data
4. Call pipeline.fit() then pipeline.predict()

The framework handles calibration, threshold tuning, and routing automatically.
No code changes needed -- only configuration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass, field

from cascade.framework.confidence_stage import ConfidenceStage


@dataclass
class StageConfig:
    """Configuration for a single cascade stage."""
    name: str
    classes: Dict[int, str]
    target_accuracy: float = 0.85
    # Columns to use as features (extracted from DataFrame)
    feature_columns: Optional[List[str]] = None
    # Column containing the target label
    target_column: str = 'status'
    # Label merge map (e.g., {8: 4} to merge Backedout into Ack)
    label_merge: Optional[Dict[int, int]] = None
    # Filter function: given DataFrame, return filtered DataFrame
    # Used to select which rows this stage trains/predicts on
    input_filter: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
    # Routing: class_code -> action
    # 'terminal' = final decision, 'defer' = human review, 'next' = pass to next stage
    routing: Dict[int, str] = field(default_factory=dict)
    # Custom model (optional, default: RandomForest)
    model: Optional[Any] = None
    # Whether to scale features
    scale: bool = True
    # Output column name prefix
    output_prefix: Optional[str] = None


class GeneralCascade:
    """
    A configurable cascade of confidence-gated classification stages.

    Architecture:
        Input -> Stage 0 -> {confident: route, uncertain: defer}
                             |
                Stage 1 -> {confident: route, uncertain: defer}
                             |
                Stage N -> {confident: terminal, uncertain: defer}

    Each stage:
    - Trains a calibrated classifier
    - Tunes per-class confidence thresholds via OOF
    - At inference: routes confident predictions, defers uncertain ones
    """

    def __init__(
        self,
        stages: List[StageConfig],
        random_state: int = 42,
        defer_label: int = -1,
    ):
        """
        Args:
            stages: Ordered list of stage configurations
            random_state: Random seed for reproducibility
            defer_label: Label used for deferred predictions
        """
        self.stage_configs = stages
        self.random_state = random_state
        self.defer_label = defer_label
        self._stages: Dict[str, ConfidenceStage] = {}
        self._is_fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        calibration_method: str = 'isotonic',
        n_cv_folds: int = 5,
    ) -> 'GeneralCascade':
        """
        Train all cascade stages sequentially.

        Args:
            df: Training DataFrame with features and labels
            calibration_method: 'isotonic' or 'sigmoid'
            n_cv_folds: Number of CV folds

        Returns:
            self
        """
        print("=" * 70)
        print("TRAINING GENERAL CASCADE PIPELINE")
        print("=" * 70)

        for i, config in enumerate(self.stage_configs):
            print(f"\n[{i+1}/{len(self.stage_configs)}] Training: {config.name}")
            print("-" * 50)

            # Filter training data for this stage
            stage_df = df.copy()
            if config.input_filter is not None:
                stage_df = config.input_filter(stage_df)

            if len(stage_df) == 0:
                print(f"  WARNING: No training data for stage '{config.name}'")
                continue

            # Apply label merge
            target_col = config.target_column
            if config.label_merge:
                stage_df = stage_df.copy()
                stage_df[target_col] = stage_df[target_col].replace(config.label_merge)

            # Filter to valid classes
            valid = stage_df[target_col].isin(config.classes.keys())
            stage_df = stage_df[valid].copy()

            if len(stage_df) == 0:
                print(f"  WARNING: No valid samples for stage '{config.name}'")
                continue

            # Extract features and target
            feature_cols = config.feature_columns
            if feature_cols is None:
                # Auto-detect: all numeric columns except target
                feature_cols = [
                    c for c in stage_df.select_dtypes(include=[np.number]).columns
                    if c != target_col
                ]

            available_cols = [c for c in feature_cols if c in stage_df.columns]
            X = stage_df[available_cols].fillna(0).values
            y = stage_df[target_col].values

            # Create and fit stage
            stage = ConfidenceStage(
                name=config.name,
                classes=config.classes,
                target_accuracy=config.target_accuracy,
                calibration_method=calibration_method,
                n_cv_folds=n_cv_folds,
                random_state=self.random_state,
                model=config.model,
                defer_label=self.defer_label,
            )
            stage.fit(X, y, feature_names=available_cols, scale=config.scale)
            self._stages[config.name] = stage

        self._is_fitted = True
        print("\n" + "=" * 70)
        print("CASCADE TRAINING COMPLETE")
        print("=" * 70)
        return self

    def predict(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Run the full cascade on input data.

        For each row:
        1. Stage 0 processes it
        2. If confident -> route per config (terminal or next stage)
        3. If uncertain -> defer (label = defer_label)
        4. Repeat for subsequent stages on routed rows

        Args:
            df: Input DataFrame with feature columns

        Returns:
            DataFrame with added prediction columns per stage
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        result_df = df.copy()
        # Track which rows are still active (not yet terminally decided or deferred)
        active_mask = np.ones(len(df), dtype=bool)

        for i, config in enumerate(self.stage_configs):
            stage = self._stages.get(config.name)
            if stage is None:
                continue

            prefix = config.output_prefix or config.name

            # Initialize columns
            result_df[f'{prefix}_pred'] = self.defer_label
            result_df[f'{prefix}_confidence'] = 0.0
            result_df[f'{prefix}_is_confident'] = False

            # Apply input filter to determine which active rows this stage sees
            if config.input_filter is not None:
                active_df = result_df[active_mask].copy()
                try:
                    filtered = config.input_filter(active_df)
                    stage_indices = filtered.index
                except Exception:
                    stage_indices = active_df.index
            else:
                stage_indices = result_df[active_mask].index

            if len(stage_indices) == 0:
                continue

            # Extract features
            feature_cols = config.feature_columns
            if feature_cols is None:
                feature_cols = stage.feature_names or []

            available_cols = [c for c in feature_cols if c in result_df.columns]
            X = result_df.loc[stage_indices, available_cols].fillna(0).values

            # Predict
            preds = stage.predict(X)

            # Store results
            result_df.loc[stage_indices, f'{prefix}_pred'] = preds['class']
            result_df.loc[stage_indices, f'{prefix}_confidence'] = preds['confidence']
            result_df.loc[stage_indices, f'{prefix}_is_confident'] = preds['is_confident']

            # Route: update active mask based on routing rules
            for idx, (pred_class, is_conf) in zip(
                stage_indices,
                zip(preds['class'], preds['is_confident'])
            ):
                if not is_conf:
                    # Deferred -> stays active for next stage or becomes deferred
                    route = config.routing.get(self.defer_label, 'defer')
                    if route == 'defer':
                        active_mask[result_df.index.get_loc(idx)] = False
                else:
                    route = config.routing.get(int(pred_class), 'terminal')
                    if route == 'terminal':
                        active_mask[result_df.index.get_loc(idx)] = False
                    # 'next' -> stays active for next stage

        # Add summary columns
        result_df['cascade_is_automated'] = ~active_mask | self._has_terminal_prediction(result_df)

        return result_df

    def _has_terminal_prediction(self, df: pd.DataFrame) -> pd.Series:
        """Check which rows received a terminal (confident) prediction from any stage."""
        has_pred = pd.Series(False, index=df.index)
        for config in self.stage_configs:
            prefix = config.output_prefix or config.name
            conf_col = f'{prefix}_is_confident'
            if conf_col in df.columns:
                has_pred = has_pred | df[conf_col].astype(bool)
        return has_pred

    def evaluate(
        self,
        df: pd.DataFrame,
        predictions: pd.DataFrame,
        true_label_column: str = 'status',
        label_merge: Optional[Dict[int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate cascade performance against ground truth.

        Args:
            df: Original data with true labels
            predictions: Output from predict()
            true_label_column: Column with true labels
            label_merge: Optional merge map for true labels

        Returns:
            Dict of evaluation metrics
        """
        true = df[true_label_column].values.copy()
        if label_merge:
            true = pd.Series(true).replace(label_merge).values

        results = {}

        # Per-stage evaluation
        for config in self.stage_configs:
            prefix = config.output_prefix or config.name
            pred_col = f'{prefix}_pred'
            conf_col = f'{prefix}_is_confident'

            if pred_col not in predictions.columns:
                continue

            pred = predictions[pred_col].values
            is_conf = predictions[conf_col].values.astype(bool)

            # Apply stage's label merge to true labels for comparison
            stage_true = true.copy()
            if config.label_merge:
                stage_true = pd.Series(stage_true).replace(config.label_merge).values

            n_conf = is_conf.sum()
            stage_result = {
                'n_total': len(true),
                'n_confident': int(n_conf),
                'coverage': n_conf / len(true) if len(true) > 0 else 0,
            }

            if n_conf > 0:
                stage_result['accuracy'] = (
                    stage_true[is_conf] == pred[is_conf]
                ).mean()

            results[config.name] = stage_result

        # Coverage-accuracy curve (end-to-end)
        # Find the last stage's confidence for each row
        final_conf = np.zeros(len(predictions))
        final_pred = np.full(len(predictions), self.defer_label)

        for config in self.stage_configs:
            prefix = config.output_prefix or config.name
            pred_col = f'{prefix}_pred'
            conf_col = f'{prefix}_confidence'
            is_conf_col = f'{prefix}_is_confident'

            if pred_col not in predictions.columns:
                continue

            mask = predictions[is_conf_col].values.astype(bool)
            final_conf[mask] = np.maximum(final_conf[mask],
                                           predictions.loc[mask, conf_col].values)
            # Use the prediction from the last confident stage
            final_pred[mask] = predictions.loc[mask, pred_col].values

        # Apply the most inclusive merge map for e2e
        e2e_true = true.copy()
        for config in self.stage_configs:
            if config.label_merge:
                e2e_true = pd.Series(e2e_true).replace(config.label_merge).values

        results['end_to_end'] = {}
        for t in [0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
            mask = final_conf >= t
            n = mask.sum()
            if n > 0:
                acc = (e2e_true[mask] == final_pred[mask]).mean()
                results['end_to_end'][f't_{t:.2f}'] = {
                    'accuracy': float(acc),
                    'coverage': float(n / len(true)),
                    'n_automated': int(n),
                    'n_deferred': int(len(true) - n),
                }

        return results

    def get_stage(self, name: str) -> Optional[ConfidenceStage]:
        """Get a fitted stage by name."""
        return self._stages.get(name)

    def print_evaluation(self, eval_results: Dict[str, Any]):
        """Pretty-print evaluation results."""
        print("\n" + "=" * 70)
        print("CASCADE EVALUATION")
        print("=" * 70)

        for config in self.stage_configs:
            if config.name in eval_results:
                r = eval_results[config.name]
                print(f"\n  {config.name}:")
                print(f"    Confident: {r['n_confident']}/{r['n_total']} "
                      f"({r['coverage']:.1%})")
                if 'accuracy' in r:
                    print(f"    Accuracy:  {r['accuracy']:.1%}")

        if 'end_to_end' in eval_results:
            print(f"\n  End-to-end coverage-accuracy curve:")
            print(f"  {'Threshold':>10} {'Coverage':>10} {'Accuracy':>10} {'Automated':>10}")
            print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
            for key, val in sorted(eval_results['end_to_end'].items()):
                t = key.replace('t_', '')
                print(f"  {t:>10} {val['coverage']:>10.1%} "
                      f"{val['accuracy']:>10.1%} {val['n_automated']:>10}")
