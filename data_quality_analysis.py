
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

column_reports = []
dataset_reports = []

print("=" * 70)
print("DETAILED DATA QUALITY ANALYSIS")
print("=" * 70)

csv_files = sorted(DATA_DIR.glob("*.csv"))

for file in csv_files:

    print("\n" + "-" * 70)
    print("Dataset:", file.name)

    try:
        df = pd.read_csv(file, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(
            file,
            encoding="latin-1",
            low_memory=False
        )

    # Basic information
    rows, columns = df.shape

    # Missing values in each column
    missing = df.isnull().sum()

    # Rows containing at least one missing value
    rows_with_missing = df.isnull().any(axis=1).sum()

    # Duplicate rows
    duplicates = df.duplicated().sum()

    print("Total rows:", rows)
    print("Total columns:", columns)
    print("Rows with missing values:", rows_with_missing)
    print("Duplicate rows:", duplicates)

    print("\nColumn-wise missing value details:")

    for column in df.columns:

        missing_count = int(missing[column])

        missing_percentage = (
            (missing_count / rows) * 100
            if rows > 0 else 0
        )

        print(
            f"{column}: "
            f"{missing_count} missing "
            f"({missing_percentage:.2f}%)"
        )

        column_reports.append({
            "Dataset": file.name,
            "Column": column,
            "Data_Type": str(df[column].dtype),
            "Missing_Count": missing_count,
            "Missing_Percentage": round(
                missing_percentage, 2
            )
        })

    dataset_reports.append({
        "Dataset": file.name,
        "Total_Rows": rows,
        "Total_Columns": columns,
        "Rows_With_Missing": int(rows_with_missing),
        "Duplicate_Rows": int(duplicates)
    })

    del df

# Save detailed reports
column_report = pd.DataFrame(column_reports)
dataset_report = pd.DataFrame(dataset_reports)

column_report.to_csv(
    RESULTS_DIR / "column_quality_report.csv",
    index=False
)

dataset_report.to_csv(
    RESULTS_DIR / "dataset_quality_summary.csv",
    index=False
)

print("\n" + "=" * 70)
print("DATA QUALITY ANALYSIS COMPLETED!")
print("=" * 70)

print("\nReports saved successfully inside results folder.")
print("- column_quality_report.csv")
print("- dataset_quality_summary.csv")
