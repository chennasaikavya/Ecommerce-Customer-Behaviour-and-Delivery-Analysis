
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

# Load our previously generated reports
column_report = pd.read_csv(
    RESULTS_DIR / "column_quality_report.csv"
)

dataset_report = pd.read_csv(
    RESULTS_DIR / "dataset_quality_summary.csv"
)

print("=" * 65)
print("COLUMNS CONTAINING MISSING VALUES")
print("=" * 65)

# Display only columns with missing values
missing_columns = column_report[
    column_report["Missing_Count"] > 0
].sort_values(
    ["Dataset", "Missing_Count"],
    ascending=[True, False]
)

if missing_columns.empty:
    print("No missing values found!")
else:
    print(missing_columns.to_string(index=False))

print("\n" + "=" * 65)
print("DATASETS CONTAINING DUPLICATE ROWS")
print("=" * 65)

duplicates = dataset_report[
    dataset_report["Duplicate_Rows"] > 0
]

print(duplicates.to_string(index=False))

# Save focused reports
missing_columns.to_csv(
    RESULTS_DIR / "missing_columns_only.csv",
    index=False
)

duplicates.to_csv(
    RESULTS_DIR / "duplicate_datasets_only.csv",
    index=False
)

print("\nDetailed inspection completed!")
print("Reports saved inside results folder.")
