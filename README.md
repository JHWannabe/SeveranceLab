# SeveranceLab

Utility script for exploring the provided breast density dataset.

## Requirements
- Python 3.10+
- pandas
- seaborn
- matplotlib
- scikit-learn
- openpyxl

If network access for `pip install` is restricted, install the dependencies in an
offline environment or via a pre-provisioned wheelhouse.

## Usage
Run the analysis script against the Excel file:

```bash
python analyze_breast_density.py breast_data_250812.xlsx
```

Use SimpleImputer instead of dropping rows during missing-value handling:

```bash
python analyze_breast_density.py breast_data_250812.xlsx --impute
```

Plots will be saved to the `plots/` directory by default. Use `--output-dir` to
choose another location. The script prints dataframe info, descriptive
statistics, and the normalized distribution of the `Osteoporosis` column.
