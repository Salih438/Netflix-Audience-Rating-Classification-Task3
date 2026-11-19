"""
models.py – Model factory, baseline training and hyperparameter tuning.

Responsibilities
----------------
- Provide a factory function that returns all classifier instances used in
  the project (keeps model definitions in one place).
- Train all baseline models on a training split.
- Run GridSearchCV to tune the RandomForestClassifier.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Protocol, Tuple

import numpy as np

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from rating_classifier.config import RF_PARAM_GRID, TRAIN_CFG
from rating_classifier.features import FeatureTransformer

logger = logging.getLogger(__name__)

FeatureMatrix = pd.DataFrame | np.ndarray


class Classifier(Protocol):
    """Minimal interface shared by the baseline sklearn classifiers."""

    def fit(self, X: FeatureMatrix, y: pd.Series) -> "Classifier":
        ...

    def predict(self, X: FeatureMatrix) -> np.ndarray:
        ...


ModelSpec = Dict[str, Tuple[Classifier, bool]]  # name → (estimator, needs_scaling)


def get_model_specs(random_state: int = TRAIN_CFG.random_state) -> ModelSpec:
    """Return the dictionary of baseline model specifications.

    Each value is a ``(estimator, needs_scaling)`` tuple.  ``needs_scaling``
    is ``True`` for models that benefit from standardised features (e.g.
    Logistic Regression) and ``False`` for tree-based models.

    Parameters
    ----------
    random_state:
        Seed used for all stochastic models.

    Returns
    -------
    dict
        ``{model_name: (estimator_instance, needs_scaling)}``
    """
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            True,
        ),
        "Decision Tree": (
            DecisionTreeClassifier(
                max_depth=8, class_weight="balanced", random_state=random_state
            ),
            False,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
            ),
            False,
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=random_state),
            False,
        ),
    }


def train_baselines(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scaler: StandardScaler,
    model_specs: ModelSpec | None = None,
) -> pd.DataFrame:
    """Train all baseline classifiers and return a comparison dataframe.

    The scaler is applied only to models that require it
    (``needs_scaling=True``).  The scaler must already be **fitted on
    training data only** to prevent data leakage.

    Parameters
    ----------
    X_train, X_test:
        Training and test feature matrices.
    y_train, y_test:
        Training and test label vectors.
    scaler:
        A *pre-fitted* :class:`~sklearn.preprocessing.StandardScaler`.
    model_specs:
        Model specification dict as returned by :func:`get_model_specs`.
        Defaults to the standard set of four classifiers.

    Returns
    -------
    pd.DataFrame
        Rows indexed by model name with test metrics, plus train Accuracy
        and train Macro F1 for overfitting comparison.
    """
    if model_specs is None:
        model_specs = get_model_specs()

    rows: List[dict] = []
    for name, (model, needs_scaling) in model_specs.items():
        if needs_scaling:
            Xa = scaler.transform(X_train)
            Xb = scaler.transform(X_test)
        else:
            Xa, Xb = X_train, X_test

        model.fit(Xa, y_train)
        preds = model.predict(Xb)

        train_preds = model.predict(Xa)
        row = {
            "Model":       name,
            "Accuracy":    accuracy_score(y_test, preds),
            "Balanced Accuracy": balanced_accuracy_score(y_test, preds),
            "Macro Precision": precision_score(y_test, preds, average="macro", zero_division=0),
            "Macro Recall": recall_score(y_test, preds, average="macro", zero_division=0),
            "Macro F1":    f1_score(y_test, preds, average="macro", zero_division=0),
            "Weighted F1": f1_score(y_test, preds, average="weighted", zero_division=0),
            "Train Accuracy": accuracy_score(y_train, train_preds),
            "Train Macro F1": f1_score(y_train, train_preds, average="macro", zero_division=0),
        }
        rows.append(row)
        logger.info(
            "%-22s  Accuracy=%.4f  Macro F1=%.4f",
            name, row["Accuracy"], row["Macro F1"],
        )

    baseline_df = pd.DataFrame(rows).sort_values("Macro F1", ascending=False)
    logger.info("Baseline model comparison:\n%s", baseline_df.round(4).to_string(index=False))
    return baseline_df


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict | None = None,
    random_state: int = TRAIN_CFG.random_state,
    cv_folds: int = TRAIN_CFG.grid_search_folds,
    preprocessor: FeatureTransformer | None = None,
) -> RandomForestClassifier | Pipeline:
    """Tune a RandomForestClassifier with GridSearchCV (macro F1 scoring).

    Parameters
    ----------
    X_train:
        Training feature matrix.
    y_train:
        Training label vector.
    param_grid:
        Parameter grid for :class:`~sklearn.model_selection.GridSearchCV`.
        Defaults to :data:`~rating_classifier.config.RF_PARAM_GRID`.
    random_state:
        Seed for reproducibility.
    cv_folds:
        Number of stratified folds used by GridSearchCV.

    Returns
    -------
    RandomForestClassifier or Pipeline
        Best estimator found by GridSearchCV (already fitted on *X_train*).
        When *preprocessor* is supplied, the returned Pipeline contains it.
    """
    if param_grid is None:
        param_grid = RF_PARAM_GRID

    estimator: RandomForestClassifier | Pipeline
    if preprocessor is None:
        estimator = RandomForestClassifier(
            class_weight="balanced", n_jobs=-1, random_state=random_state
        )
        search_grid = param_grid
    else:
        estimator = Pipeline(
            steps=[
                ("features", preprocessor),
                ("model", RandomForestClassifier(
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=random_state,
                )),
            ]
        )
        search_grid = {f"model__{key}": values for key, values in param_grid.items()}

    cv = StratifiedKFold(cv_folds, shuffle=True, random_state=random_state)
    gs = GridSearchCV(
        estimator,
        search_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )

    logger.info("Starting GridSearchCV for RandomForest (%d-fold) …", cv_folds)
    gs.fit(X_train, y_train)

    logger.info(
        "Best RF parameters: %s | CV Macro F1: %.4f",
        gs.best_params_,
        gs.best_score_,
    )
    return gs.best_estimator_
