"""
visualize.py – Plotting utilities for the Rating Classifier.

Produces four publication-ready figures and saves them to the output
directory.  All functions are pure (they receive data, save a file, and
return the saved path).

Figures
-------
1. Class distribution bar chart.
2. Model comparison bar chart (Macro F1).
3. Normalised confusion matrix heatmap.
4. Top-15 feature importances horizontal bar chart.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from rating_classifier.config import CLASS_ORDER

logger = logging.getLogger(__name__)


def _save(fig_path: Path) -> Path:
    """Save the current matplotlib figure to *fig_path* and close it."""
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.info("Figure saved → %s", fig_path)
    return fig_path


def plot_class_distribution(
    audience_series: pd.Series,
    out_dir: Path,
    filename: str = "01_class_distribution.png",
) -> Path:
    """Bar chart of audience category counts.

    Parameters
    ----------
    audience_series:
        ``audience`` column of the processed dataframe.
    out_dir:
        Directory where the figure is saved.
    filename:
        Output filename.

    Returns
    -------
    Path
        Absolute path of the saved figure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    audience_series.value_counts().plot(kind="bar", color="#4C72B0")
    plt.title("Audience Category Distribution")
    plt.ylabel("Count")
    return _save(out_dir / filename)


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    out_dir: Path,
    filename: str = "02_model_comparison.png",
) -> Path:
    """Horizontal bar chart comparing models by Macro F1.

    Parameters
    ----------
    comparison_df:
        DataFrame with columns ``Model`` and ``Macro F1``.
    out_dir:
        Directory where the figure is saved.
    filename:
        Output filename.

    Returns
    -------
    Path
        Absolute path of the saved figure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    sns.barplot(data=comparison_df, y="Model", x="Macro F1", color="#55A868")
    plt.title("Model Comparison (Macro F1)")
    return _save(out_dir / filename)


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred,
    out_dir: Path,
    class_order: List[str] = CLASS_ORDER,
    filename: str = "03_confusion_matrix.png",
) -> Path:
    """Normalised confusion matrix heatmap for the tuned Random Forest.

    Parameters
    ----------
    y_true:
        Ground-truth integer labels.
    y_pred:
        Predicted integer labels.
    out_dir:
        Directory where the figure is saved.
    class_order:
        Human-readable class names in label-index order.
    filename:
        Output filename.

    Returns
    -------
    Path
        Absolute path of the saved figure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(len(class_order))), normalize="true"
    )
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_order,
        yticklabels=class_order,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Tuned RF, normalized)")
    return _save(out_dir / filename)


def plot_feature_importance(
    feature_importances,
    feature_names: List[str],
    out_dir: Path,
    top_n: int = 15,
    filename: str = "04_feature_importance.png",
) -> Path:
    """Horizontal bar chart of the top-N most important features.

    Parameters
    ----------
    feature_importances:
        Array of importance values from ``estimator.feature_importances_``.
    feature_names:
        List of feature column names (same order as *feature_importances*).
    out_dir:
        Directory where the figure is saved.
    top_n:
        Number of top features to display.
    filename:
        Output filename.

    Returns
    -------
    Path
        Absolute path of the saved figure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    imp = (
        pd.Series(feature_importances, index=feature_names)
        .sort_values(ascending=False)
        .head(top_n)
    )
    logger.info("Top 10 features:\n%s", imp.head(10).round(4).to_string())

    plt.figure(figsize=(7, 5))
    imp[::-1].plot(kind="barh", color="#C44E52")
    plt.title(f"Top {top_n} Feature Importances")
    return _save(out_dir / filename)
