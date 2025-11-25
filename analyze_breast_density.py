"""Utilities for loading and exploring the breast density dataset.

The script performs the following operations:
- Load the xlsx file with a predefined subset of columns.
- Print dataframe info, descriptive statistics, and the class distribution of the
  Osteoporosis column.
- Handle missing values via simple drop-na or sklearn SimpleImputer.
- Generate histogram and boxplot visualizations for numeric columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer

COLUMNS: List[str] = [
    "age",
    "bmi",
    "breastDensity(%)AVG",
    "breast_Density_Grading",
    "DenseArea(sqcm)AVG",
    "Osteoporosis",
]


def load_data(xlsx_path: Path, columns: Iterable[str] = COLUMNS) -> pd.DataFrame:
    """Load the Excel dataset restricted to the selected columns."""
    return pd.read_excel(xlsx_path, usecols=list(columns))


def report_overview(df: pd.DataFrame) -> None:
    """Print dataframe info, summary statistics, and target distribution."""
    print("DataFrame Info:\n----------------")
    df.info()
    print("\nDescriptive Statistics:\n----------------------")
    print(df.describe(include="all"))
    print("\nOsteoporosis Distribution (normalized):\n---------------------------------------")
    print(df["Osteoporosis"].value_counts(normalize=True))


def handle_missing(
    df: pd.DataFrame,
    strategy: str = "drop",
    categorical_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Handle missing values either by dropping rows or imputing.

    Parameters
    ----------
    df:
        Input dataframe.
    strategy:
        "drop" to remove rows with missing values or "impute" to apply
        SimpleImputer (mean for numeric columns, most_frequent for categorical).
    categorical_columns:
        Optional explicit list of categorical column names. Defaults to the
        object dtype columns in the dataframe.
    """

    if strategy == "drop":
        return df.dropna()

    if strategy != "impute":
        raise ValueError("strategy must be either 'drop' or 'impute'")

    cleaned_df = df.copy()
    detected_categoricals = cleaned_df.select_dtypes(include=["object", "category"]).columns
    categorical_cols = list(categorical_columns) if categorical_columns is not None else list(detected_categoricals)

    numeric_cols = cleaned_df.columns.difference(categorical_cols)

    if len(numeric_cols) > 0:
        numeric_imputer = SimpleImputer(strategy="mean")
        cleaned_df[numeric_cols] = numeric_imputer.fit_transform(cleaned_df[numeric_cols])

    if len(categorical_cols) > 0:
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        cleaned_df[categorical_cols] = categorical_imputer.fit_transform(cleaned_df[categorical_cols])

    return cleaned_df


def plot_numeric_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    """Save histogram and boxplot for each numeric column."""
    numeric_cols = df.select_dtypes(include=["number"]).columns
    output_dir.mkdir(parents=True, exist_ok=True)

    for col in numeric_cols:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(df[col], kde=True, ax=axes[0])
        axes[0].set_title(f"Histogram of {col}")
        sns.boxplot(x=df[col], ax=axes[1])
        axes[1].set_title(f"Boxplot of {col}")
        plt.tight_layout()
        file_path = output_dir / f"{col}_distribution.png"
        plt.savefig(file_path)
        plt.close(fig)


def plot_boxplots_by_target(df: pd.DataFrame, target: str, output_dir: Path) -> None:
    """Save boxplots of numeric features grouped by the target column."""
    numeric_cols = df.select_dtypes(include=["number"]).columns
    output_dir.mkdir(parents=True, exist_ok=True)

    for col in numeric_cols:
        fig = plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x=target, y=col)
        plt.title(f"{col} by {target}")
        plt.tight_layout()
        file_path = output_dir / f"{col}_by_{target}.png"
        plt.savefig(file_path)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore breast density dataset")
    parser.add_argument(
        "xlsx_path",
        type=Path,
        help="Path to the xlsx file containing the dataset.",
    )
    parser.add_argument(
        "--impute",
        action="store_true",
        help="Use SimpleImputer for missing values instead of dropping rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory where plots will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_data(args.xlsx_path)
    report_overview(df)

    missing_strategy = "impute" if args.impute else "drop"
    cleaned_df = handle_missing(df, strategy=missing_strategy)

    print(f"\nAfter missing value handling ({missing_strategy}): {cleaned_df.shape[0]} rows remain")

    plot_numeric_distributions(cleaned_df, args.output_dir)
    plot_boxplots_by_target(cleaned_df, target="Osteoporosis", output_dir=args.output_dir)


if __name__ == "__main__":
    main()
