"""Training and evaluation script for breast density classification.

This script loads the provided Excel dataset, builds a binary target for high
breast density, performs a randomized cross-validated grid search with
reproducible seeds, and exports the full set of metrics in CSV and JSON.

Usage
-----
python experiments/breast_density_experiment.py --data breast_data_250812.xlsx \
    --output-dir results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load the Excel dataset with pandas."""

    df = pd.read_excel(data_path)
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, List[str]]:
    """Return model features, target, and feature names.

    The binary target flags cases with breast density grade 3 or higher.
    """

    work = df.copy()
    work["Breast_Density_Grading"] = pd.to_numeric(
        work["Breast_Density_Grading"], errors="coerce"
    )
    work["high_density"] = (work["Breast_Density_Grading"] >= 3).astype(int)

    drop_columns = {"File Analyzed", "진료년월일", "Breast_Density_Grading", "high_density"}
    feature_cols = [col for col in work.columns if col not in drop_columns]
    X = work[feature_cols]
    y = work["high_density"]
    return X, y, feature_cols


def build_pipeline(feature_names: Iterable[str]) -> GridSearchCV:
    """Create the preprocessing + model pipeline with a grid search."""

    numeric_features = list(feature_names)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    clf = Pipeline(steps=[("preprocess", preprocessor), ("classifier", model)])

    param_grid = {
        "classifier__n_estimators": [150, 250],
        "classifier__max_depth": [None, 8, 14],
        "classifier__min_samples_leaf": [1, 3],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    return GridSearchCV(
        clf,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        return_train_score=True,
    )


def compute_threshold_metrics(
    y_true: pd.Series, positive_probas: np.ndarray, thresholds: Iterable[float]
) -> pd.DataFrame:
    """Compute threshold-dependent classification metrics."""

    rows = []
    for threshold in thresholds:
        preds = (positive_probas >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(y_true, preds),
                "precision": precision_score(y_true, preds, zero_division=0),
                "recall": recall_score(y_true, preds, zero_division=0),
                "f1": f1_score(y_true, preds, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def save_table(table: pd.DataFrame, output_path: Path) -> None:
    """Save a table to CSV and JSON next to each other."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path.with_suffix(".csv"), index=False)
    table.to_json(output_path.with_suffix(".json"), orient="records", indent=2)


def main(args: argparse.Namespace) -> None:
    np.random.seed(RANDOM_STATE)

    df = load_dataset(Path(args.data))
    X, y, feature_names = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    grid = build_pipeline(feature_names)
    grid.fit(X_train, y_train)

    cv_results = pd.DataFrame(grid.cv_results_)
    save_table(cv_results, Path(args.output_dir) / "cv_results")

    best_model = grid.best_estimator_
    test_probas = best_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probas >= 0.5).astype(int)

    test_metrics = pd.DataFrame(
        [
            {
                "best_params": json.dumps(grid.best_params_),
                "accuracy": accuracy_score(y_test, test_preds),
                "precision": precision_score(y_test, test_preds, zero_division=0),
                "recall": recall_score(y_test, test_preds, zero_division=0),
                "f1": f1_score(y_test, test_preds, zero_division=0),
                "roc_auc": roc_auc_score(y_test, test_probas),
            }
        ]
    )
    save_table(test_metrics, Path(args.output_dir) / "test_metrics")

    thresholds = np.linspace(0.1, 0.9, num=9)
    threshold_metrics = compute_threshold_metrics(y_test, test_probas, thresholds)
    save_table(threshold_metrics, Path(args.output_dir) / "threshold_metrics")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=str,
        default="breast_data_250812.xlsx",
        help="Path to the Excel dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Where to write CSV/JSON results.",
    )
    main(parser.parse_args())
