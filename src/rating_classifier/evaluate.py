"""
evaluate.py – Evaluation metrics and cross-validation helpers.

Responsibilities
----------------
- Compute accuracy, macro/weighted F1, and full classification report.
- Run stratified k-fold cross-validation by refitting an estimator per fold.
- Return results as plain Python dicts / DataFrames (no side-effects).
"""

from __future__ import annotations

import logging
from typing import Dict, List, TypedDict

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score

from rating_classifier.config import CLASS_ORDER, TRAIN_CFG

logger = logging.getLogger(__name__)


class CrossValidationResult(TypedDict):
    """Summary statistics and individual fold scores."""

    mean: float
    std: float
    scores: list[float]


def compute_metrics(
    y_true: pd.Series,
    y_pred,
    class_order: List[str] = CLASS_ORDER,
) -> Dict[str, float]:
    """Compute accuracy, balance-aware scores, and the classification report.

    Parameters
    ----------
    y_true:
        Ground-truth integer labels.
    y_pred:
        Predicted integer labels.
    class_order:
        Human-readable class names in label-index order.

    Returns
    -------
    dict
        Accuracy, balanced accuracy, macro precision and recall,
        and macro and weighted F1 values.
    """
    metrics = {
        "Accuracy":    float(accuracy_score(y_true, y_pred)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Macro Precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Macro Recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "Macro F1":    float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Weighted F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    logger.info(
        "Metrics → Accuracy: %.4f | Balanced Accuracy: %.4f | Macro F1: %.4f | "
        "Weighted F1: %.4f",
        metrics["Accuracy"], metrics["Balanced Accuracy"],
        metrics["Macro F1"], metrics["Weighted F1"],
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_order))),
        target_names=class_order,
        zero_division=0,
    )
    logger.info("Classification report:\n%s", report)
    return metrics


def cross_validate(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = TRAIN_CFG.cv_folds,
    scoring: str = "accuracy",
) -> CrossValidationResult:
    """Run stratified k-fold cross-validation and return mean ± std.

    Parameters
    ----------
    estimator:
        A fitted (or unfitted) scikit-learn estimator.
    X:
        Full feature matrix.
    y:
        Full label vector.
    cv:
        Number of cross-validation folds.
    scoring:
        Scoring metric string accepted by scikit-learn.

    Returns
    -------
    dict
        ``{"mean": float, "std": float, "scores": list[float]}``
    """
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    result: CrossValidationResult = {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "scores": scores.tolist(),
    }
    logger.info(
        "%d-fold CV %s: %.4f ± %.4f",
        cv, scoring, result["mean"], result["std"],
    )
    return result


def build_final_comparison(
    baseline_df: pd.DataFrame,
    tuned_metrics: Dict[str, float],
    model_label: str = "Random Forest (Tuned)",
) -> pd.DataFrame:
    """Append tuned-model metrics to the baseline comparison table.

    Parameters
    ----------
    baseline_df:
        DataFrame returned by :func:`~rating_classifier.models.train_baselines`.
    tuned_metrics:
        Metrics dict returned by :func:`compute_metrics`.
    model_label:
        Display name for the tuned model row.

    Returns
    -------
    pd.DataFrame
        Combined comparison table sorted by Macro F1 (descending).
    """
    tuned_row = pd.DataFrame([{"Model": model_label, **tuned_metrics}])
    final = pd.concat([baseline_df, tuned_row]).sort_values("Macro F1", ascending=False)
    return final.reset_index(drop=True)
