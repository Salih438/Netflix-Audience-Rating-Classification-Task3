"""
data.py – Data loading and label mapping for the Rating Classifier.

Responsibilities
----------------
- Load the raw CSV from disk with helpful error messages.
- Map 14 raw rating values to 4 audience-category labels.
- Drop rows whose rating cannot be mapped (NR / UR).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from rating_classifier.config import RATING_MAP

logger = logging.getLogger(__name__)

# Required columns – validated on load so failures are caught early
_REQUIRED_COLUMNS = {
    "rating", "type", "duration", "date_added",
    "release_year", "director", "country", "listed_in",
}


def load_data(csv_path: Path | str) -> pd.DataFrame:
    """Load the Netflix CSV dataset from *csv_path*.

    Parameters
    ----------
    csv_path:
        Path to ``Dataset.csv`` (or any compatible CSV file).

    Returns
    -------
    pd.DataFrame
        Raw dataframe with all original columns intact.

    Raises
    ------
    FileNotFoundError
        If *csv_path* does not exist.
    ValueError
        If required columns are missing from the file.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. "
            "Make sure the file is placed at data/raw/Dataset.csv "
            "or pass the correct path via --data."
        )

    logger.info("Loading dataset from %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Raw dataset shape: %s", df.shape)

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Required column(s) missing from dataset: {sorted(missing)}. "
            "Check that you are using the correct Dataset.csv file."
        )

    logger.info("Original rating distribution:\n%s", df["rating"].value_counts().to_string())
    return df


def map_audience(
    df: pd.DataFrame,
    rating_map: Dict[str, str] | None = None,
) -> pd.DataFrame:
    """Add an ``audience`` column by mapping raw ratings to 4 categories.

    Rows whose rating is not present in *rating_map* (e.g. NR, UR) are
    dropped because those labels are ambiguous for supervised learning.

    Parameters
    ----------
    df:
        Raw dataframe as returned by :func:`load_data`.
    rating_map:
        Mapping from raw rating string to audience category.
        Defaults to :data:`~rating_classifier.config.RATING_MAP`.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an additional ``audience`` column; ambiguous rows
        removed.
    """
    if rating_map is None:
        rating_map = RATING_MAP

    if "rating" not in df.columns:
        raise ValueError("Missing target column: rating")
    if df.empty:
        raise ValueError("Cannot map audience labels from an empty dataframe.")

    if df["rating"].isna().any():
        raise ValueError("rating contains missing values.")

    df = df.copy()
    df["audience"] = df["rating"].map(rating_map)

    unknown = sorted(
        df.loc[df["audience"].isna(), "rating"].dropna().unique().tolist()
    )
    unexpected = [value for value in unknown if value not in {"NR", "UR"}]
    if unexpected:
        raise ValueError(f"Unknown rating value(s): {unexpected}")

    before = len(df)
    df = df.dropna(subset=["audience"]).copy()
    dropped = before - len(df)

    logger.info(
        "Audience mapping complete. Dropped %d rows with unmapped ratings (NR/UR). "
        "Remaining: %d rows. Unmapped ratings: %s.",
        dropped,
        len(df),
        unknown,
    )
    logger.info("Audience category distribution:\n%s", df["audience"].value_counts().to_string())
    return df
