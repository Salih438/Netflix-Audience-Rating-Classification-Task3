"""
features.py – Feature engineering for the Rating Classifier.

Responsibilities
----------------
- Parse ``duration`` into numeric ``minutes`` / ``seasons`` columns.
- Extract date-related features from ``date_added``.
- One-hot encode the primary country (top-N + "Other").
- Multi-hot encode genres (columns appearing ≥ MIN_GENRE_COUNT times).
- Assemble and return the final feature matrix ``X`` and label vector ``y``.

Design note
-----------
All functions are pure (no side-effects) and operate on copies of the input
dataframe so that they can safely be reused by Task 2 and Task 4 without
unexpected mutation.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from rating_classifier.config import (
    CLASS_ORDER,
    MIN_GENRE_COUNT,
    NUMERIC_FEATURES,
    TOP_COUNTRIES,
)

logger = logging.getLogger(__name__)

_REQUIRED_FEATURE_COLUMNS = {
    "type", "duration", "date_added", "release_year",
    "director", "country", "listed_in",
}


# ---------------------------------------------------------------------------
# Individual feature-engineering steps
# ---------------------------------------------------------------------------


def parse_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Extract numeric duration value and split into ``minutes`` / ``seasons``.

    For Movies  : ``minutes = duration_val``, ``seasons = 0``
    For TV Shows: ``seasons = duration_val``, ``minutes = 0``

    Parameters
    ----------
    df:
        Dataframe that must contain ``type`` and ``duration`` columns.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with new columns: ``is_movie``, ``duration_val``,
        ``minutes``, ``seasons``.
    """
    missing = {"type", "duration"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature column(s): {sorted(missing)}")

    df = df.copy()
    if df.empty:
        raise ValueError("Cannot parse duration from an empty dataframe.")
    if df["duration"].isna().any():
        raise ValueError("duration contains missing values.")
    duration_text = df["duration"].astype("string").str.strip()
    is_movie = df["type"].eq("Movie").fillna(False)
    is_show = df["type"].eq("TV Show").fillna(False)
    valid_movie = is_movie & duration_text.str.fullmatch(r"[1-9]\d* min")
    valid_show = is_show & duration_text.str.fullmatch(r"1 Season|(?:[2-9]|[1-9]\d+) Seasons")
    invalid = ~(valid_movie | valid_show)
    if invalid.any():
        bad_rows = df.index[invalid].tolist()[:3]
        raise ValueError(
            f"duration contains invalid values at rows {bad_rows}; "
            "expected minutes for a Movie or seasons for a TV Show."
        )
    df["is_movie"] = is_movie.astype(int)
    df["duration_val"] = duration_text.str.split().str[0].astype(float)
    df["minutes"] = np.where(df["is_movie"] == 1, df["duration_val"], 0)
    df["seasons"] = np.where(df["is_movie"] == 0, df["duration_val"], 0)
    logger.debug("Duration parsed. is_movie counts: %s", df["is_movie"].value_counts().to_dict())
    return df


def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract year, month and release-gap features from ``date_added``.

    ``release_gap`` = ``year_added`` − ``release_year`` captures how long
    after production a title was added to Netflix.

    Parameters
    ----------
    df:
        Dataframe containing ``date_added`` and ``release_year`` columns.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with new columns: ``year_added``, ``month_added``,
        ``release_gap``.
    """
    missing = {"date_added", "release_year"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature column(s): {sorted(missing)}")

    df = df.copy()
    if df.empty:
        raise ValueError("Cannot extract date features from an empty dataframe.")
    dt = pd.to_datetime(df["date_added"], errors="coerce")
    years = dt.dt.year
    median_year = years.median()
    if pd.isna(median_year):
        raise ValueError("date_added must contain at least one valid date to learn the median year.")
    df["year_added"] = years.fillna(median_year)
    df["month_added"] = dt.dt.month.fillna(1)
    df["release_gap"] = df["year_added"] - df["release_year"]
    logger.debug("Date features extracted. year_added range: %s–%s",
                 df["year_added"].min(), df["year_added"].max())
    return df


def extract_director_country_genre_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary / count flags for director, country, and genre columns.

    Parameters
    ----------
    df:
        Dataframe containing ``director``, ``country``, and ``listed_in``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with new columns: ``has_director``, ``n_countries``,
        ``n_genres``.
    """
    missing = {"director", "country", "listed_in"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature column(s): {sorted(missing)}")

    df = df.copy()
    for column in ("director", "country", "listed_in"):
        df[column] = df[column].fillna("Not Given").astype("string")
    df["has_director"] = (
        df["director"].str.strip().str.lower() != "not given"
    ).astype(int)
    df["n_countries"] = df["country"].apply(
        lambda s: 0 if str(s).lower() == "not given" else len(str(s).split(","))
    )
    df["n_genres"] = df["listed_in"].str.count(",") + 1
    return df


def encode_country(
    df: pd.DataFrame,
    top_n: int = TOP_COUNTRIES,
    known_top: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """One-hot encode the primary country, grouping rare countries as "Other".

    Parameters
    ----------
    df:
        Dataframe with a ``country`` column.
    top_n:
        Number of most-frequent countries to keep as separate columns.
    known_top:
        Pre-computed list of top countries (use this at inference time to
        ensure the same columns as during training).

    Returns
    -------
    country_ohe : pd.DataFrame
        One-hot encoded country columns (prefix ``c_``).
    top_countries : list[str]
        The list of top countries used for encoding (pass back as
        ``known_top`` at inference time).
    """
    df = df.copy()
    df["main_country"] = df["country"].str.split(",").str[0].str.strip()

    if known_top is None:
        top_countries = df["main_country"].value_counts().head(top_n).index.tolist()
    else:
        top_countries = known_top

    df["main_country"] = df["main_country"].where(
        df["main_country"].isin(top_countries), "Other"
    )
    country_ohe = pd.get_dummies(df["main_country"], prefix="c")
    logger.debug("Country OHE shape: %s", country_ohe.shape)
    return country_ohe, top_countries


def encode_genres(
    df: pd.DataFrame,
    min_count: int = MIN_GENRE_COUNT,
    known_genres: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Multi-hot encode genres, keeping only those with ≥ *min_count* rows.

    Parameters
    ----------
    df:
        Dataframe with a ``listed_in`` column.
    min_count:
        Minimum number of occurrences for a genre to be included.
    known_genres:
        Pre-computed genre column names (use at inference time).

    Returns
    -------
    genre_mhe : pd.DataFrame
        Multi-hot encoded genre columns (prefix ``g_``).
    genre_cols : list[str]
        The list of genre column names used (pass back as ``known_genres``
        at inference time).
    """
    g = df["listed_in"].str.get_dummies(sep=", ")

    if known_genres is None:
        g = g.loc[:, g.sum() >= min_count]
        genre_cols = g.columns.tolist()
    else:
        # At inference time, align to training columns (fill missing with 0)
        g = g.reindex(columns=[c.replace("g_", "") for c in known_genres], fill_value=0)
        genre_cols = known_genres

    g.columns = ["g_" + c.strip() for c in g.columns]
    logger.debug("Genre MHE shape: %s", g.shape)
    return g, genre_cols


# ---------------------------------------------------------------------------
# Main feature-building entry point
# ---------------------------------------------------------------------------


def build_features(
    df: pd.DataFrame,
    class_order: List[str] = CLASS_ORDER,
    top_countries: int = TOP_COUNTRIES,
    min_genre_count: int = MIN_GENRE_COUNT,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Run the full feature-engineering pipeline and return ``(X, y)``.

    This convenience function fits preprocessing on every supplied row.
    For evaluation, split the raw rows first and use a `FeatureTransformer`
    fitted on training rows to transform both partitions.

    Parameters
    ----------
    df:
        Cleaned dataframe with an ``audience`` column (output of
        :func:`~rating_classifier.data.map_audience`).
    class_order:
        Ordered list of audience class labels used to build the integer
        label vector ``y``.
    top_countries:
        Number of top countries for one-hot encoding.
    min_genre_count:
        Minimum genre occurrences to include in the feature matrix.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix, all columns cast to ``float``.
    y : pd.Series
        Integer label vector (0 = Kids, 1 = Older Kids, 2 = Teens,
        3 = Adults).
    """
    if df.empty:
        raise ValueError("Cannot build features from an empty dataframe.")
    missing = _REQUIRED_FEATURE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature column(s): {sorted(missing)}")

    transformer = FeatureTransformer(
        top_countries=top_countries,
        min_genre_count=min_genre_count,
    )
    X = transformer.fit_transform(df)

    y = _encode_target(df, class_order)

    logger.info("Feature matrix shape: %s | Classes: %s", X.shape, class_order)

    if X.isnull().any().any():
        null_cols = X.columns[X.isnull().any()].tolist()
        logger.warning("NaN values found in feature columns: %s", null_cols)

    return X, y


def _encode_target(df: pd.DataFrame, class_order: List[str]) -> pd.Series:
    """Encode the audience target and reject unknown labels explicitly."""
    if "audience" not in df.columns:
        raise ValueError("Missing target column: audience")
    label_index = {c: i for i, c in enumerate(class_order)}
    y = df["audience"].map(label_index)
    if y.isna().any():
        unknown = sorted(df.loc[y.isna(), "audience"].dropna().unique().tolist())
        missing_count = df["audience"].isna().sum()
        raise ValueError(
            f"Unknown or missing audience target value(s): {unknown}; "
            f"missing={missing_count}."
        )
    return y.astype(int)


class FeatureTransformer(BaseEstimator):
    """Fit metadata-derived feature vocabularies on training data only."""

    def __init__(
        self,
        top_countries: int = TOP_COUNTRIES,
        min_genre_count: int = MIN_GENRE_COUNT,
    ) -> None:
        self.top_countries = top_countries
        self.min_genre_count = min_genre_count

    def fit(self, X: pd.DataFrame, _y: pd.Series | None = None) -> "FeatureTransformer":
        del _y
        self._validate_input(X)
        if X.empty:
            raise ValueError("Cannot fit feature transformer on empty data.")

        prepared = self._prepare(X, fit=True)
        _, self.top_countries_ = encode_country(
            prepared, top_n=self.top_countries
        )
        _, self.genre_columns_ = encode_genres(
            prepared, min_count=self.min_genre_count
        )
        self.year_added_median_ = pd.to_datetime(
            prepared["date_added"], errors="coerce"
        ).dt.year.median()
        self.feature_names_out_ = self._transform_prepared(prepared).columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(X)
        if not hasattr(self, "top_countries_"):
            raise ValueError("FeatureTransformer must be fitted before transform.")
        prepared = self._prepare(X, fit=False)
        result = self._transform_prepared(prepared)
        return result.reindex(columns=self.feature_names_out_, fill_value=0.0).astype(float)

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        **fit_params: object,
    ) -> pd.DataFrame:
        del fit_params
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self) -> np.ndarray:
        if not hasattr(self, "feature_names_out_"):
            raise ValueError("FeatureTransformer must be fitted first.")
        return np.asarray(self.feature_names_out_, dtype=object)

    def _validate_input(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Features must be provided as a pandas DataFrame.")
        missing = _REQUIRED_FEATURE_COLUMNS - set(X.columns)
        if missing:
            raise ValueError(f"Missing feature column(s): {sorted(missing)}")

    def _prepare(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        prepared = X.copy()
        prepared = parse_duration(prepared)
        if fit:
            return extract_date_features(prepared)

        dt = pd.to_datetime(prepared["date_added"], errors="coerce")
        prepared["year_added"] = dt.dt.year.fillna(self.year_added_median_)
        prepared["month_added"] = dt.dt.month.fillna(1)
        prepared["release_gap"] = prepared["year_added"] - prepared["release_year"]
        return prepared

    def _transform_prepared(self, prepared: pd.DataFrame) -> pd.DataFrame:
        prepared = extract_director_country_genre_flags(prepared)
        country_ohe, _ = encode_country(
            prepared, known_top=self.top_countries_
        )
        genre_mhe, _ = encode_genres(
            prepared, known_genres=self.genre_columns_
        )
        return pd.concat(
            [prepared[NUMERIC_FEATURES], country_ohe, genre_mhe],
            axis=1,
        )
