"""
robustness_check.py – Multi-seed robustness and baseline checks.

Evaluates:
1. Multi-seed Tuned Random Forest (seeds 42, 123, 456, 789, 999) using
   standard project GridSearchCV settings.
2. Naive DummyClassifier (most_frequent) baseline on seed 42 split.
3. 95% Confidence Interval for seed 42 test Accuracy (normal approximation).
4. Negative release_gap count and minimum value in the dataset.
5. Code audit of class_weight configuration across all models.

Saves results to outputs/robustness_summary.csv and prints to terminal.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Suppress joblib/sklearn UserWarnings
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning:sklearn.utils.parallel")

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from rating_classifier.config import CLASS_ORDER, PATHS, RF_PARAM_GRID, TRAIN_CFG
from rating_classifier.data import load_data, map_audience
from rating_classifier.features import FeatureTransformer, extract_date_features, parse_duration
from rating_classifier.models import tune_random_forest


def run_robustness_check(out_dir: Path = PATHS.outputs) -> pd.DataFrame:
    start_time = time.time()
    print("=" * 70)
    print("TASK 3 – ROBUSTNESS, BASELINE & STATISTICAL AUDIT CHECK")
    print("=" * 70)

    # 1. Load and label untransformed rows
    df_raw = load_data(PATHS.raw_data)
    df_clean = map_audience(df_raw)
    label_index = {label: index for index, label in enumerate(CLASS_ORDER)}
    y = df_clean["audience"].map(label_index).astype(int)

    # -----------------------------------------------------------------------
    # (a) Multi-seed evaluation of Tuned Random Forest
    # -----------------------------------------------------------------------
    seeds = [42, 123, 456, 789, 999]
    seed_results = []
    print("\n--- (a) Multi-Seed Tuned Random Forest Evaluation ---")
    for s in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(
            df_clean, y,
            test_size=TRAIN_CFG.test_size,
            stratify=y,
            random_state=s,
        )
        tuned_rf = tune_random_forest(
            X_tr, y_tr,
            param_grid=RF_PARAM_GRID,
            random_state=s,
            cv_folds=TRAIN_CFG.grid_search_folds,
            preprocessor=FeatureTransformer(),
        )
        preds = tuned_rf.predict(X_te)
        acc = accuracy_score(y_te, preds)
        mf1 = f1_score(y_te, preds, average="macro")
        seed_results.append({"seed": s, "accuracy": acc, "macro_f1": mf1})
        print(f"Seed {s:4d} -> Accuracy: {acc:.4f} | Macro F1: {mf1:.4f}")

    seed_df = pd.DataFrame(seed_results)
    mean_acc = seed_df["accuracy"].mean()
    std_acc = seed_df["accuracy"].std()
    mean_f1 = seed_df["macro_f1"].mean()
    std_f1 = seed_df["macro_f1"].std()

    print(f"\nMulti-Seed Summary ({len(seeds)} seeds: {seeds}):")
    print(f"  Test Accuracy: {mean_acc:.4f} ± {std_acc:.4f} (min={seed_df['accuracy'].min():.4f}, max={seed_df['accuracy'].max():.4f})")
    print(f"  Test Macro F1: {mean_f1:.4f} ± {std_f1:.4f} (min={seed_df['macro_f1'].min():.4f}, max={seed_df['macro_f1'].max():.4f})")

    # -----------------------------------------------------------------------
    # (b) DummyClassifier (most_frequent) baseline on seed 42 split
    # -----------------------------------------------------------------------
    print("\n--- (b) Naive Baseline: DummyClassifier(strategy='most_frequent') ---")
    X_tr42, X_te42, y_tr42, y_te42 = train_test_split(
        df_clean, y,
        test_size=TRAIN_CFG.test_size,
        stratify=y,
        random_state=42,
    )
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_tr42, y_tr42)
    dummy_preds = dummy.predict(X_te42)
    dummy_acc = accuracy_score(y_te42, dummy_preds)
    dummy_mf1 = f1_score(y_te42, dummy_preds, average="macro")
    print(f"DummyClassifier (Seed 42) -> Accuracy: {dummy_acc:.4f} | Macro F1: {dummy_mf1:.4f}")
    majority_class = y_tr42.value_counts().idxmax()
    majority_class_name = CLASS_ORDER[majority_class]
    majority_support = int((y_te42 == majority_class).sum())
    print(f"  (Always predicts majority class: '{majority_class_name}', support: {majority_support} / {len(y_te42)})")

    # -----------------------------------------------------------------------
    # (c) 95% Confidence Interval for Seed 42 Test Accuracy (Normal approx)
    # -----------------------------------------------------------------------
    print("\n--- (c) 95% Confidence Interval for Seed 42 Test Accuracy ---")
    seed42_acc = seed_df.loc[seed_df["seed"] == 42, "accuracy"].values[0]
    n_test = len(y_te42)
    # Wald interval: p +/- 1.96 * sqrt(p * (1 - p) / n)
    se = np.sqrt(seed42_acc * (1.0 - seed42_acc) / n_test)
    margin_95 = 1.959964 * se
    ci_lower = seed42_acc - margin_95
    ci_upper = seed42_acc + margin_95
    print(f"Test size n: {n_test}")
    print(f"Seed 42 Accuracy: {seed42_acc:.4f}")
    print(f"Standard Error (SE): {se:.4f}")
    print(f"95% CI (Normal approx): [{ci_lower:.4f}, {ci_upper:.4f}] (± {margin_95:.4f})")

    # -----------------------------------------------------------------------
    # (d) Negative release_gap check in dataset
    # -----------------------------------------------------------------------
    print("\n--- (d) Negative release_gap Check in Dataset ---")
    d_dates = extract_date_features(parse_duration(df_clean))
    neg_mask = d_dates["release_gap"] < 0
    neg_count = neg_mask.sum()
    min_gap = float(d_dates["release_gap"].min())
    print(f"Total titles with release_gap < 0: {neg_count} (out of {len(d_dates)} clean titles)")
    print(f"Minimum release_gap value: {min_gap}")
    if neg_count > 0:
        sample_neg = d_dates.loc[neg_mask, ["show_id", "title", "release_year", "year_added", "release_gap"]].head(3)
        print("Sample records with negative gap:")
        print(sample_neg.to_string(index=False))

    # -----------------------------------------------------------------------
    # (e) Code Audit of class_weight in models.py
    # -----------------------------------------------------------------------
    print("\n--- (e) Model class_weight Configuration Audit ---")
    models_py_path = PROJECT_ROOT / "src" / "rating_classifier" / "models.py"
    models_content = models_py_path.read_text(encoding="utf-8").splitlines()
    print(f"Source: {models_py_path.name}")
    for idx, line in enumerate(models_content, 1):
        if any(m in line for m in ["LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier", "GradientBoostingClassifier"]):
            print(f"  Line {idx:3d}: {line.strip()}")
        elif "class_weight" in line:
            print(f"  Line {idx:3d}: {line.strip()}")

    print("\nFinding on class_weight:")
    print("  - Logistic Regression:  class_weight='balanced' (models.py:52)")
    print("  - Decision Tree:        class_weight='balanced' (models.py:57)")
    print("  - Random Forest:        class_weight='balanced' (models.py:64)")
    print("  - Gradient Boosting:    NO class_weight parameter (models.py:69) -> scikit-learn does not support it natively")

    # -----------------------------------------------------------------------
    # Save results to outputs/robustness_summary.csv
    # -----------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "robustness_summary.csv"

    summary_rows = [
        {"metric_category": "Dummy Baseline", "parameter": "Strategy", "value": "most_frequent"},
        {"metric_category": "Dummy Baseline", "parameter": "Accuracy", "value": f"{dummy_acc:.4f}"},
        {"metric_category": "Dummy Baseline", "parameter": "Macro F1", "value": f"{dummy_mf1:.4f}"},
        {"metric_category": "Tuned RF Seed 42", "parameter": "Accuracy", "value": f"{seed42_acc:.4f}"},
        {"metric_category": "Tuned RF Seed 42", "parameter": "Macro F1", "value": f"{seed_df.loc[seed_df['seed']==42, 'macro_f1'].values[0]:.4f}"},
        {"metric_category": "Tuned RF Seed 42 95% CI", "parameter": "CI Lower", "value": f"{ci_lower:.4f}"},
        {"metric_category": "Tuned RF Seed 42 95% CI", "parameter": "CI Upper", "value": f"{ci_upper:.4f}"},
        {"metric_category": "Tuned RF Seed 42 95% CI", "parameter": "Margin (±)", "value": f"{margin_95:.4f}"},
        {"metric_category": "Tuned RF 5-Seed Mean ± Std", "parameter": "Accuracy", "value": f"{mean_acc:.4f} ± {std_acc:.4f}"},
        {"metric_category": "Tuned RF 5-Seed Mean ± Std", "parameter": "Macro F1", "value": f"{mean_f1:.4f} ± {std_f1:.4f}"},
        {"metric_category": "Tuned RF 5-Seed Min-Max", "parameter": "Accuracy Range", "value": f"[{seed_df['accuracy'].min():.4f}, {seed_df['accuracy'].max():.4f}]"},
        {"metric_category": "Tuned RF 5-Seed Min-Max", "parameter": "Macro F1 Range", "value": f"[{seed_df['macro_f1'].min():.4f}, {seed_df['macro_f1'].max():.4f}]"},
        {"metric_category": "Negative release_gap", "parameter": "Row Count", "value": str(neg_count)},
        {"metric_category": "Negative release_gap", "parameter": "Min Value", "value": str(min_gap)},
    ]

    # Add individual seed rows
    for _, row in seed_df.iterrows():
        summary_rows.append({
            "metric_category": f"Seed {int(row['seed'])}",
            "parameter": "Accuracy / Macro F1",
            "value": f"{row['accuracy']:.4f} / {row['macro_f1']:.4f}",
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved robustness summary to: {csv_path}")

    elapsed = time.time() - start_time
    print(f"\nExecution time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PATHS.outputs)
    run_robustness_check(out_dir=parser.parse_args().out)
