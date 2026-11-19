"""
main.py – Command-line entry point for the Rating Classifier.

Usage
-----
    python main.py
    python main.py --data data/raw/Dataset.csv --out outputs

Arguments
---------
--data  : Path to the input CSV file (default: data/raw/Dataset.csv).
--out   : Directory for output plots and CSV   (default: outputs/).
--log   : Logging level: DEBUG | INFO | WARNING (default: INFO).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Suppress sklearn/joblib parallel UserWarnings BEFORE any other import so
# that the setting is inherited by joblib worker sub-processes as well.
# ---------------------------------------------------------------------------
import os
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning:sklearn.utils.parallel")

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the src/ package is importable when running from the project root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rating_classifier.pipeline import run  # noqa: E402


def _configure_logging(level: str) -> None:
    """Set up root logger with a readable format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Netflix Audience Rating Classifier – Task 3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/Dataset.csv"),
        help="Path to the input CSV dataset.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs"),
        help="Directory to save output plots and model_comparison.csv.",
    )
    parser.add_argument(
        "--log",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run the classification pipeline."""
    args = _parse_args(argv)
    _configure_logging(args.log)
    logger = logging.getLogger(__name__)
    logger.info("Starting Rating Classifier pipeline.")
    logger.info("  data → %s", args.data)
    logger.info("  out  → %s", args.out)

    final_df = run(data_path=args.data, out_dir=args.out)

    logger.info("\n%s", final_df.round(4).to_string(index=False))
    logger.info("Done.")


if __name__ == "__main__":
    main()
