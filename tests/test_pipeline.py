"""
tests/test_pipeline.py – End-to-end pipeline smoke test on a 500-row sample.

The test verifies that the complete pipeline runs without errors on a small
slice of the real dataset and that the output dataframe has the expected
structure.  Actual numeric values are NOT asserted here because GridSearchCV
results on a 500-row subsample will differ from the full-dataset run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure the src package is importable when tests are run from the project root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rating_classifier.config import PATHS
from rating_classifier.pipeline import run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_sample_csv(tmp_path: Path, n: int = 500) -> Path:
    """Read the first *n* rows of the real dataset and write to *tmp_path*."""
    real_csv = PATHS.raw_data
    if not real_csv.exists():
        pytest.skip(f"Real dataset not found at {real_csv}; skipping pipeline test.")
    df = pd.read_csv(real_csv, nrows=n)
    sample_csv = tmp_path / "sample.csv"
    df.to_csv(sample_csv, index=False)
    return sample_csv


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pipeline_runs_on_sample(tmp_path: Path) -> None:
    """Pipeline must complete without raising on a 500-row sample."""
    sample_csv = _create_sample_csv(tmp_path)
    out_dir = tmp_path / "outputs"

    result_df = run(data_path=sample_csv, out_dir=out_dir)

    # Structural assertions
    assert isinstance(result_df, pd.DataFrame), "run() must return a DataFrame."
    assert not result_df.empty, "Result DataFrame must not be empty."

    required_cols = {"Model", "Accuracy", "Macro F1", "Weighted F1"}
    assert required_cols.issubset(result_df.columns), (
        f"Missing columns in result: {required_cols - set(result_df.columns)}"
    )


def test_pipeline_produces_output_files(tmp_path: Path) -> None:
    """Pipeline must save all four figures and the CSV."""
    sample_csv = _create_sample_csv(tmp_path)
    out_dir = tmp_path / "outputs"

    run(data_path=sample_csv, out_dir=out_dir)

    expected_files = [
        "01_class_distribution.png",
        "02_model_comparison.png",
        "03_confusion_matrix.png",
        "04_feature_importance.png",
        "model_comparison.csv",
    ]
    for fname in expected_files:
        fpath = out_dir / fname
        assert fpath.exists(), f"Expected output file not found: {fpath}"
        assert fpath.stat().st_size > 0, f"Output file is empty: {fpath}"


def test_pipeline_comparison_df_model_count(tmp_path: Path) -> None:
    """Result DataFrame must contain exactly 5 models (4 baselines + tuned RF)."""
    sample_csv = _create_sample_csv(tmp_path)
    out_dir = tmp_path / "outputs"

    result_df = run(data_path=sample_csv, out_dir=out_dir)

    assert len(result_df) == 5, (
        f"Expected 5 models in comparison, got {len(result_df)}."
    )


def test_pipeline_metrics_in_valid_range(tmp_path: Path) -> None:
    """All Accuracy and Macro F1 values must be between 0 and 1."""
    sample_csv = _create_sample_csv(tmp_path)
    out_dir = tmp_path / "outputs"

    result_df = run(data_path=sample_csv, out_dir=out_dir)

    assert result_df["Accuracy"].between(0, 1).all(), "Accuracy out of [0, 1]."
    assert result_df["Macro F1"].between(0, 1).all(), "Macro F1 out of [0, 1]."


def test_pipeline_rejects_single_class_training_data(tmp_path: Path) -> None:
    """Training must fail clearly when stratified multi-class learning is impossible."""
    df = pd.read_csv(_create_sample_csv(tmp_path, n=10))
    df["rating"] = "TV-MA"
    sample_csv = tmp_path / "single_class.csv"
    df.to_csv(sample_csv, index=False)

    with pytest.raises(ValueError, match="At least two audience classes"):
        run(data_path=sample_csv, out_dir=tmp_path / "outputs")


def test_pipeline_evaluates_two_supported_audience_classes(tmp_path: Path) -> None:
    sample = pd.read_csv(_create_sample_csv(tmp_path))
    two_classes = sample.loc[sample["rating"].isin(["TV-Y", "TV-14"])]
    sample_csv = tmp_path / "two_classes.csv"
    two_classes.to_csv(sample_csv, index=False)
    out_dir = tmp_path / "outputs"

    comparison = run(data_path=sample_csv, out_dir=out_dir)

    assert len(comparison) == 5
    assert (out_dir / "03_confusion_matrix.png").is_file()
