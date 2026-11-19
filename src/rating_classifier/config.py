"""
config.py – Central configuration for the Rating Classifier project.

All constants, paths, and hyperparameters live here so that every other
module stays free of magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


# ---------------------------------------------------------------------------
# Project root (two levels up from this file: src/rating_classifier/config.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Filesystem paths used throughout the project."""

    raw_data: Path = PROJECT_ROOT / "data" / "raw" / "Dataset.csv"
    outputs: Path = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyper-parameters and split settings."""

    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    grid_search_folds: int = 3


# ---------------------------------------------------------------------------
# Rating → Audience mapping  (NR / UR intentionally absent → dropped)
# ---------------------------------------------------------------------------
RATING_MAP: Dict[str, str] = {
    "TV-Y":    "Kids",
    "TV-Y7":   "Kids",
    "TV-Y7-FV":"Kids",
    "TV-G":    "Kids",
    "G":       "Kids",
    "TV-PG":   "Older Kids",
    "PG":      "Older Kids",
    "TV-14":   "Teens",
    "PG-13":   "Teens",
    "TV-MA":   "Adults",
    "R":       "Adults",
    "NC-17":   "Adults",
}

# Ordered class list – preserves label encoding used during training
CLASS_ORDER: List[str] = ["Kids", "Older Kids", "Teens", "Adults"]

# Top-N countries to keep as one-hot columns (rest → "Other")
TOP_COUNTRIES: int = 10

# Minimum genre occurrence to include as a feature column
MIN_GENRE_COUNT: int = 50

# Numerical feature columns (order matters for reproducibility)
NUMERIC_FEATURES: List[str] = [
    "is_movie",
    "minutes",
    "seasons",
    "release_year",
    "year_added",
    "month_added",
    "release_gap",
    "has_director",
    "n_countries",
    "n_genres",
]

# ---------------------------------------------------------------------------
# GridSearch parameter grid for RandomForestClassifier
# ---------------------------------------------------------------------------
RF_PARAM_GRID: Dict[str, list] = {
    "n_estimators":    [200, 400],
    "max_depth":       [None, 15, 25],
    "min_samples_leaf":[1, 3],
    "max_features":    ["sqrt", 0.5],
}


# ---------------------------------------------------------------------------
# Convenience singleton instances
# ---------------------------------------------------------------------------
PATHS = Paths()
TRAIN_CFG = TrainingConfig()
