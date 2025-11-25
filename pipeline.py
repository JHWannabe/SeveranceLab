from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


def load_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Unsupported file format. Use .csv or .xlsx files.")


def split_features_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame.")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def build_preprocessor(
    numeric_columns: Iterable[str],
    categorical_columns: Iterable[str],
    use_robust_scaler: bool = False,
) -> ColumnTransformer:
    numeric_scaler = RobustScaler() if use_robust_scaler else StandardScaler()

    numeric_transformer = Pipeline([
        ("scaler", numeric_scaler),
    ])

    categorical_transformer = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, list(numeric_columns)),
            ("categorical", categorical_transformer, list(categorical_columns)),
        ],
        remainder="drop",
    )


def build_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    use_robust_scaler: bool = False,
) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    numeric_columns = X.select_dtypes(include=["number"]).columns
    categorical_columns = X.select_dtypes(exclude=["number"]).columns

    preprocessor = build_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        use_robust_scaler=use_robust_scaler,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    return model, X_train, X_test, y_train, y_test


def train_and_evaluate(
    data_path: Path,
    target_column: str,
    use_robust_scaler: bool = False,
) -> float:
    df = load_dataframe(data_path)
    X, y = split_features_target(df, target_column)
    model, X_train, X_test, y_train, y_test = build_pipeline(
        X, y, use_robust_scaler=use_robust_scaler
    )
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train logistic regression pipeline")
    parser.add_argument("data_path", type=Path, help="Path to CSV or Excel data file")
    parser.add_argument("target", help="Target column name")
    parser.add_argument(
        "--use-robust-scaler",
        action="store_true",
        help="Use RobustScaler instead of StandardScaler for numeric features",
    )

    args = parser.parse_args()

    accuracy = train_and_evaluate(
        data_path=args.data_path,
        target_column=args.target,
        use_robust_scaler=args.use_robust_scaler,
    )

    print(f"Hold-out accuracy: {accuracy:.3f}")
