"""
tests/test_features.py – Unit tests for the feature engineering module.

Tests
-----
- ``parse_duration`` correctly extracts minutes (Movie) and seasons (TV Show).
- ``build_features`` produces no NaN values.
- The feature matrix column count matches expectations.
- Genre and country encoding produce correctly prefixed columns.
"""

from __future__ import annotations

import pandas as pd
import pytest

from rating_classifier.config import CLASS_ORDER, NUMERIC_FEATURES
from rating_classifier.features import (
    FeatureTransformer,
    build_features,
    encode_country,
    encode_genres,
    extract_date_features,
    parse_duration,
)


# ---------------------------------------------------------------------------
# Minimal sample dataframe factory
# ---------------------------------------------------------------------------


def _make_sample(n: int = 10) -> pd.DataFrame:
    """Return a minimal dataframe with all required columns."""
    movies = n // 2
    shows = n - movies
    return pd.DataFrame(
        {
            "type":         ["Movie"] * movies + ["TV Show"] * shows,
            "duration":     ["90 min"] * movies + ["2 Seasons"] * shows,
            "date_added":   ["January 1, 2020"] * n,
            "release_year": [2018] * n,
            "director":     ["John Doe"] * n,
            "country":      ["United States"] * n,
            "listed_in":    ["Dramas, Action & Adventure"] * n,
            "audience":     ["Adults"] * n,
            "rating":       ["TV-MA"] * n,
        }
    )


# ---------------------------------------------------------------------------
# parse_duration tests
# ---------------------------------------------------------------------------


def test_parse_duration_movie_minutes() -> None:
    """'90 min' should yield minutes=90, seasons=0 for a Movie row."""
    df = _make_sample(2)
    df["type"] = "Movie"
    df["duration"] = "90 min"
    result = parse_duration(df)
    assert (result["minutes"] == 90).all(), "Movie minutes should be 90."
    assert (result["seasons"] == 0).all(), "Movie seasons should be 0."


def test_parse_duration_tv_seasons() -> None:
    """'2 Seasons' should yield seasons=2, minutes=0 for a TV Show row."""
    df = _make_sample(2)
    df["type"] = "TV Show"
    df["duration"] = "2 Seasons"
    result = parse_duration(df)
    assert (result["seasons"] == 2).all(), "TV Show seasons should be 2."
    assert (result["minutes"] == 0).all(), "TV Show minutes should be 0."


def test_parse_duration_is_movie_flag() -> None:
    """is_movie should be 1 for Movies and 0 for TV Shows."""
    df = _make_sample(4)  # 2 movies, 2 shows
    result = parse_duration(df)
    assert result.loc[result["type"] == "Movie", "is_movie"].eq(1).all()
    assert result.loc[result["type"] == "TV Show", "is_movie"].eq(0).all()


def test_parse_duration_invalid_value_raises_value_error() -> None:
    """Malformed duration metadata must fail with an actionable error."""
    df = _make_sample(1)
    df["duration"] = "not a duration"
    with pytest.raises(ValueError, match="duration contains invalid"):
        parse_duration(df)


@pytest.mark.parametrize(
    ("content_type", "duration"),
    [("Movie", "2 Seasons"), ("Movie", "-90 min"), ("TV Show", "1.5 Seasons"),
     ("Unknown", "90 min")],
)
def test_parse_duration_rejects_invalid_units_and_values(
    content_type: str, duration: str
) -> None:
    df = _make_sample().assign(type=content_type, duration=duration)
    with pytest.raises(ValueError, match="duration contains invalid"):
        parse_duration(df)


def test_build_features_empty_dataframe_raises_value_error() -> None:
    """Empty training data must not produce a success-shaped empty matrix."""
    with pytest.raises(ValueError, match="empty dataframe"):
        build_features(_make_sample(0), class_order=CLASS_ORDER)


# ---------------------------------------------------------------------------
# extract_date_features tests
# ---------------------------------------------------------------------------


def test_extract_date_features_year_month() -> None:
    """January 1, 2020 should yield year_added=2020, month_added=1."""
    df = _make_sample(2)
    result = extract_date_features(df)
    assert (result["year_added"] == 2020).all()
    assert (result["month_added"] == 1).all()


def test_release_gap_calculation() -> None:
    """release_gap = year_added - release_year."""
    df = _make_sample(2)
    result = extract_date_features(df)
    expected_gap = 2020 - 2018
    assert (result["release_gap"] == expected_gap).all()


def test_feature_transformer_requires_a_valid_training_date() -> None:
    df = _make_sample().assign(date_added=None)
    with pytest.raises(ValueError, match="at least one valid date"):
        FeatureTransformer().fit(df)


# ---------------------------------------------------------------------------
# encode_country tests
# ---------------------------------------------------------------------------


def test_encode_country_prefix() -> None:
    """Country OHE columns must start with 'c_'."""
    df = _make_sample(4)
    country_ohe, _ = encode_country(df)
    assert all(c.startswith("c_") for c in country_ohe.columns)


def test_encode_country_known_top() -> None:
    """When known_top is supplied, same columns are produced at inference."""
    df = _make_sample(6)
    _, top = encode_country(df)
    ohe_train, _ = encode_country(df, known_top=top)
    ohe_infer, _ = encode_country(df, known_top=top)
    assert list(ohe_train.columns) == list(ohe_infer.columns)


# ---------------------------------------------------------------------------
# encode_genres tests
# ---------------------------------------------------------------------------


def test_encode_genres_prefix() -> None:
    """Genre MHE columns must start with 'g_'."""
    df = _make_sample(100)  # need enough rows for min_count threshold
    genre_mhe, _ = encode_genres(df, min_count=1)
    assert all(c.startswith("g_") for c in genre_mhe.columns)


# ---------------------------------------------------------------------------
# build_features integration tests
# ---------------------------------------------------------------------------


def test_build_features_no_nan() -> None:
    """The feature matrix produced by build_features must have no NaN values."""
    df = _make_sample(20)
    X, _ = build_features(df, class_order=CLASS_ORDER)
    assert not X.isnull().any().any(), "Feature matrix must not contain NaN."


def test_build_features_expected_numeric_columns() -> None:
    """All NUMERIC_FEATURES must appear as columns in X."""
    df = _make_sample(20)
    X, _ = build_features(df, class_order=CLASS_ORDER)
    for col in NUMERIC_FEATURES:
        assert col in X.columns, f"Expected column '{col}' in feature matrix."


def test_build_features_label_vector_range() -> None:
    """Label values must be within [0, len(CLASS_ORDER) - 1]."""
    df = _make_sample(20)
    _, y = build_features(df, class_order=CLASS_ORDER)
    assert y.between(0, len(CLASS_ORDER) - 1).all(), "Label out of valid range."


def test_build_features_x_all_float() -> None:
    """Every column of X must be of float dtype."""
    df = _make_sample(20)
    X, _ = build_features(df, class_order=CLASS_ORDER)
    non_float = [c for c in X.columns if not pd.api.types.is_float_dtype(X[c])]
    assert not non_float, f"Non-float columns found: {non_float}"


def test_build_features_column_count_stable() -> None:
    """Running build_features twice on the same data must yield identical shapes."""
    df = _make_sample(20)
    X1, _ = build_features(df, class_order=CLASS_ORDER)
    X2, _ = build_features(df, class_order=CLASS_ORDER)
    assert X1.shape == X2.shape, "Feature matrix shape must be deterministic."


def test_feature_transformer_learns_categories_from_training_only() -> None:
    """Test-only countries and genres must map to the training long tail."""
    train = _make_sample(4)
    test = _make_sample(2)
    train["country"] = "United States"
    test["country"] = "Test Country"
    train["listed_in"] = "Dramas"
    test["listed_in"] = "Test Genre"

    transformer = FeatureTransformer(top_countries=1, min_genre_count=1)
    train_features = transformer.fit_transform(train)
    test_features = transformer.transform(test)

    assert "c_Test Country" not in train_features.columns
    assert "c_Test Country" not in test_features.columns
    assert "g_Test Genre" not in test_features.columns
    assert list(train_features.columns) == list(test_features.columns)


def test_feature_transformer_rejects_unknown_target() -> None:
    """Unknown labels must be rejected instead of encoded as NaN."""
    df = _make_sample(2)
    df["audience"] = "Unrecognised"
    with pytest.raises(ValueError, match="Unknown or missing audience"):
        build_features(df, class_order=CLASS_ORDER)


def test_build_features_rejects_nan_target() -> None:
    """Missing target labels must not enter the model as NaN."""
    df = _make_sample(2)
    df["audience"] = pd.Series(["Adults", pd.NA], dtype="string")
    with pytest.raises(ValueError, match="Unknown or missing audience"):
        build_features(df, class_order=CLASS_ORDER)
