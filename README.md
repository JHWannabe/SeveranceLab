# SeveranceLab

This repository trains a logistic regression classifier with a preprocessing pipeline:

1. Numeric and categorical columns are separated automatically from the feature matrix.
2. Numeric features pass through `StandardScaler` (or `RobustScaler`), and categorical features are encoded with `OneHotEncoder(handle_unknown="ignore")` via a `ColumnTransformer`.
3. The preprocessing is chained with `LogisticRegression(max_iter=1000)` inside a scikit-learn `Pipeline`.
4. Data is split with `train_test_split(test_size=0.2, stratify=y, random_state=42)`.

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train and evaluate:

```bash
python pipeline.py breast_data_250812.xlsx <target_column_name>
```

Pass `--use-robust-scaler` to swap in `RobustScaler` for numeric features.
