"""Run logistic regression hyperparameter search with cross-validation.

This script loads a dataset, splits it into train and test sets, and searches
for the best LogisticRegression configuration using GridSearchCV. It reports the
best parameters, cross-validated ROC AUC, the top-performing parameter
combinations, and basic test-set metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to a CSV or Excel file containing features and a target column.",
    )
    parser.add_argument(
        "--target-column",
        default="target",
        help="Name of the target column (default: %(default)s).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split fraction passed to train_test_split (default: %(default)s).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top hyperparameter combinations to display (default: %(default)s).",
    )
    return parser.parse_args()


def load_dataframe(path: Path, target_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. Available columns: {df.columns.tolist()}"
        )

    return df


def split_features_target(df: pd.DataFrame, target_column: str):
    y = df[target_column]
    X = df.drop(columns=[target_column])

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric feature columns found after dropping the target column.")
    if len(numeric_cols) != X.shape[1]:
        raise ValueError(
            "All feature columns must be numeric. Please encode categorical variables before running."
        )

    X_numeric = X[numeric_cols]
    return X_numeric, y


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=5000,
                    random_state=42,
                    multi_class="auto",
                ),
            ),
        ]
    )


def build_param_grid() -> List[dict]:
    c_values = [0.01, 0.1, 1, 10, 100]
    l1_ratios: Iterable[float] = [0.1, 0.5, 0.9]

    common = {
        "logreg__C": c_values,
        "logreg__class_weight": [None, "balanced"],
    }

    l2_params = {
        **common,
        "logreg__penalty": ["l2"],
        "logreg__solver": ["lbfgs", "saga"],
    }

    elasticnet_params = {
        **common,
        "logreg__penalty": ["elasticnet"],
        "logreg__solver": ["saga"],
        "logreg__l1_ratio": list(l1_ratios),
    }

    return [l2_params, elasticnet_params]


def main() -> None:
    args = parse_args()
    df = load_dataframe(args.dataset, args.target_column)
    X, y = split_features_target(df, args.target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    param_grid = build_param_grid()

    search = GridSearchCV(
        pipeline,
        param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    print("Best parameters:")
    print(search.best_params_)
    print(f"Best CV ROC AUC: {search.best_score_:.4f}\n")

    results = pd.DataFrame(search.cv_results_)
    top_results = results.sort_values(by="mean_test_score", ascending=False).head(args.top_n)
    display_cols = [
        "mean_test_score",
        "std_test_score",
        "mean_fit_time",
        "mean_score_time",
        "params",
    ]
    print(f"Top {args.top_n} parameter combinations:")
    print(top_results[display_cols].to_string(index=False))
    print()

    best_model = search.best_estimator_
    proba = best_model.predict_proba(X_test)[:, 1]

    if y.nunique() > 2:
        test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test), multi_class="ovr")
    else:
        test_auc = roc_auc_score(y_test, proba)

    print(f"Test ROC AUC: {test_auc:.4f}")
    print("Test classification report:")
    print(classification_report(y_test, best_model.predict(X_test)))


if __name__ == "__main__":
    main()
