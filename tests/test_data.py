"""
tests/test_data.py – Unit tests for data loading and audience mapping.

Tests
-----
- RATING_MAP maps every known raw rating to the correct audience category.
- Rows with NR / UR ratings are dropped after mapping.
- Rows with valid ratings are retained.
- An informative FileNotFoundError is raised for a missing CSV.
- A ValueError is raised when required columns are missing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rating_classifier.config import RATING_MAP
from rating_classifier.data import load_data, map_audience


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(ratings: list[str]) -> pd.DataFrame:
    """Create a minimal dataframe with the columns required by map_audience."""
    return pd.DataFrame(
        {
            "rating":      ratings,
            "type":        ["Movie"] * len(ratings),
            "duration":    ["90 min"] * len(ratings),
            "date_added":  ["January 1, 2020"] * len(ratings),
            "release_year":[2019] * len(ratings),
            "director":    ["Some Director"] * len(ratings),
            "country":     ["United States"] * len(ratings),
            "listed_in":   ["Dramas"] * len(ratings),
        }
    )


# ---------------------------------------------------------------------------
# RATING_MAP correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_rating, expected_audience",
    [
        ("TV-Y",    "Kids"),
        ("TV-Y7",   "Kids"),
        ("TV-Y7-FV","Kids"),
        ("TV-G",    "Kids"),
        ("G",       "Kids"),
        ("TV-PG",   "Older Kids"),
        ("PG",      "Older Kids"),
        ("TV-14",   "Teens"),
        ("PG-13",   "Teens"),
        ("TV-MA",   "Adults"),
        ("R",       "Adults"),
        ("NC-17",   "Adults"),
    ],
)
def test_rating_map_values(raw_rating: str, expected_audience: str) -> None:
    """Every known raw rating maps to the expected audience category."""
    assert RATING_MAP[raw_rating] == expected_audience


def test_rating_map_covers_all_12_ratings() -> None:
    """RATING_MAP must have exactly 12 entries (no accidental additions)."""
    assert len(RATING_MAP) == 12


# ---------------------------------------------------------------------------
# NR / UR handling
# ---------------------------------------------------------------------------


def test_nr_ur_rows_are_dropped() -> None:
    """Rows with NR or UR ratings must be dropped after mapping."""
    df = _make_df(["TV-MA", "NR", "UR", "TV-Y"])
    result = map_audience(df)
    assert len(result) == 2, "Only TV-MA and TV-Y rows should survive."
    assert "NR" not in result["rating"].values
    assert "UR" not in result["rating"].values


def test_valid_rows_retained() -> None:
    """All rows with mappable ratings must be retained."""
    ratings = list(RATING_MAP.keys())
    df = _make_df(ratings)
    result = map_audience(df)
    assert len(result) == len(ratings)


def test_audience_column_created() -> None:
    """The ``audience`` column must exist after mapping."""
    df = _make_df(["TV-MA", "TV-Y"])
    result = map_audience(df)
    assert "audience" in result.columns


def test_audience_values_correct() -> None:
    """The audience column values must match expectations."""
    df = _make_df(["TV-MA", "TV-Y", "TV-PG"])
    result = map_audience(df).reset_index(drop=True)
    assert result.loc[0, "audience"] == "Adults"
    assert result.loc[1, "audience"] == "Kids"
    assert result.loc[2, "audience"] == "Older Kids"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_load_data_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    """load_data must raise FileNotFoundError for a non-existent path."""
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_data(tmp_path / "nonexistent.csv")


def test_load_data_missing_columns_raises_value_error(tmp_path: Path) -> None:
    """load_data must raise ValueError when required columns are absent."""
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"only_col": [1, 2, 3]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="Required column"):
        load_data(bad_csv)


def test_map_audience_missing_rating_raises_value_error() -> None:
    """A missing target column must produce an actionable validation error."""
    df = _make_df(["TV-MA"]).drop(columns=["rating"])
    with pytest.raises(ValueError, match="Missing target column"):
        map_audience(df)


def test_map_audience_unknown_rating_raises_value_error() -> None:
    """Unexpected rating values must not silently become missing labels."""
    df = _make_df(["NOT_A_RATING"])
    with pytest.raises(ValueError, match="Unknown rating"):
        map_audience(df)
