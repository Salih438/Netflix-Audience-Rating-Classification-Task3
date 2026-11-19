"""
pipeline.py – End-to-end orchestration of the Rating Classifier workflow.

The :func:`run` function is the single entry point that:
1. Loads and maps data.
2. Splits the raw rows into train / test sets.
3. Fits the feature transformer on training rows and transforms both splits.
4. Fits a StandardScaler on the **training set only** (no data leakage).
5. Trains baseline models.
6. Tunes the Random Forest with GridSearchCV.
7. Evaluates and cross-validates the best model.
8. Saves plots and the model-comparison CSV.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rating_classifier.config import CLASS_ORDER, PATHS, TRAIN_CFG
from rating_classifier.data import load_data, map_audience
from rating_classifier.evaluate import (
    build_final_comparison,
    compute_metrics,
    cross_validate,
)
from rating_classifier.features import FeatureTransformer
from rating_classifier.models import get_model_specs, train_baselines, tune_random_forest
from rating_classifier.visualize import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
)

logger = logging.getLogger(__name__)


def run(
    data_path: Path | str | None = None,
    out_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Execute the full classification pipeline end-to-end.

    Parameters
    ----------
    data_path:
        Path to ``Dataset.csv``.  Defaults to
        :data:`~rating_classifier.config.PATHS.raw_data`.
    out_dir:
        Directory for output plots and CSV.  Defaults to
        :data:`~rating_classifier.config.PATHS.outputs`.

    Returns
    -------
    pd.DataFrame
        Final model comparison table (all models + tuned RF).
    """
    # ------------------------------------------------------------------
    # 0. Resolve paths
    # ------------------------------------------------------------------
    csv_path = Path(data_path) if data_path else PATHS.raw_data
    out_path = Path(out_dir) if out_dir else PATHS.outputs
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Step 1 – Load and label data")
    logger.info("=" * 60)
    df_raw = load_data(csv_path)
    df = map_audience(df_raw)

    # ------------------------------------------------------------------
    # 1. Visualise class distribution
    # ------------------------------------------------------------------
    plot_class_distribution(df["audience"], out_path)

    # ------------------------------------------------------------------
    # 2. Split before learning data-dependent feature parameters
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 2 – Train/test split")
    logger.info("=" * 60)
    label_index = {label: index for index, label in enumerate(CLASS_ORDER)}
    y = df["audience"].map(label_index)
    if y.isna().any():
        raise ValueError("Audience mapping produced an unknown or missing target.")
    if y.nunique() < 2:
        raise ValueError("At least two audience classes are required for training.")
    if y.value_counts().min() < TRAIN_CFG.grid_search_folds:
        raise ValueError(
            f"Each audience class needs at least {TRAIN_CFG.grid_search_folds} "
            "samples for stratified cross-validation."
        )

    # ------------------------------------------------------------------
    # 3. Train-only feature engineering
    # ------------------------------------------------------------------
    df_train, df_test, y_train, y_test = train_test_split(
        df, y.astype(int),
        test_size=TRAIN_CFG.test_size,
        stratify=y,
        random_state=TRAIN_CFG.random_state,
    )
    logger.info(
        "Split: %d train / %d test samples", len(df_train), len(df_test)
    )

    feature_transformer = FeatureTransformer()
    X_train = feature_transformer.fit_transform(df_train)
    X_test = feature_transformer.transform(df_test)
    scaler = StandardScaler().fit(X_train)

    # ------------------------------------------------------------------
    # 4. Baseline model training
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 4 – Train baseline models")
    logger.info("=" * 60)
    model_specs = get_model_specs(random_state=TRAIN_CFG.random_state)
    baseline_df = train_baselines(X_train, X_test, y_train, y_test, scaler, model_specs)

    # ------------------------------------------------------------------
    # 5. Hyperparameter tuning
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 5 – Tune Random Forest")
    logger.info("=" * 60)
    best_rf = tune_random_forest(
        df_train, y_train,
        random_state=TRAIN_CFG.random_state,
        cv_folds=TRAIN_CFG.grid_search_folds,
        preprocessor=FeatureTransformer(),
    )

    # ------------------------------------------------------------------
    # 6. Evaluate tuned model
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 6 – Evaluate tuned model")
    logger.info("=" * 60)
    y_pred = best_rf.predict(df_test)
    tuned_metrics = compute_metrics(y_test, y_pred, class_order=CLASS_ORDER)
    train_pred = best_rf.predict(df_train)
    train_metrics = compute_metrics(y_train, train_pred, class_order=CLASS_ORDER)
    tuned_metrics.update({
        "Train Accuracy": train_metrics["Accuracy"],
        "Train Macro F1": train_metrics["Macro F1"],
    })

    cross_validate(
        best_rf, df, y.astype(int),
        cv=TRAIN_CFG.cv_folds,
        scoring="f1_macro",
    )
    # NOTE: cross_validate() already logs the CV Macro F1 mean ± std
    # internally; no duplicate log call here.

    # ------------------------------------------------------------------
    # 8. Build final comparison table and save CSV
    # ------------------------------------------------------------------
    final_df = build_final_comparison(baseline_df, tuned_metrics)
    csv_out = out_path / "model_comparison.csv"
    final_df.round(4).to_csv(csv_out, index=False)
    logger.info("Model comparison saved → %s", csv_out)

    # ------------------------------------------------------------------
    # 9. Visualisations
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 7 – Save visualisations")
    logger.info("=" * 60)
    plot_model_comparison(final_df, out_path)
    plot_confusion_matrix(y_test, y_pred, out_path, class_order=CLASS_ORDER)
    if not isinstance(best_rf, Pipeline):
        raise TypeError("Tuned Random Forest must be a preprocessing Pipeline.")
    plot_feature_importance(
        best_rf.named_steps["model"].feature_importances_,
        best_rf.named_steps["features"].get_feature_names_out().tolist(),
        out_path,
    )

    logger.info("Pipeline complete. All outputs saved to '%s'.", out_path)
    return final_df
