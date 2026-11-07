
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

print("=" * 65)
print("E-COMMERCE DATASET INSPECTION")
print("=" * 65)

# Find all CSV files
csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    print("No CSV files found!")
else:
    print(f"\nTotal datasets found: {len(csv_files)}")

    summary = []

    # Inspect each dataset
    for file in csv_files:

        print("\n" + "-" * 65)
        print("DATASET:", file.name)

        try:
            df = pd.read_csv(file, low_memory=False)

        except UnicodeDecodeError:
            df = pd.read_csv(
                file,
                encoding="latin-1",
                low_memory=False
            )

        rows, columns = df.shape

        # Missing values
        missing = df.isnull().sum()
        missing_columns = missing[missing > 0]

        # Duplicate records
        duplicates = df.duplicated().sum()

        print("Rows:", rows)
        print("Columns:", columns)
        print("Duplicate rows:", duplicates)
        print("Total missing values:", missing.sum())

        print("\nColumn names:")
        print(list(df.columns))

        if len(missing_columns) > 0:
            print("\nColumns containing missing values:")
            print(missing_columns)
        else:
            print("\nNo missing values found!")

        # Store summary
        summary.append({
            "Dataset": file.name,
            "Rows": rows,
            "Columns": columns,
            "Missing_Values": int(missing.sum()),
            "Duplicate_Rows": int(duplicates)
        })

        del df

    # Create summary report
    report = pd.DataFrame(summary)

    report_path = RESULTS_DIR / "dataset_overview.csv"
    report.to_csv(report_path, index=False)

    print("\n" + "=" * 65)
    print("FINAL DATASET SUMMARY")
    print("=" * 65)

    print(report.to_string(index=False))

    print("\nInspection completed successfully!")
    print("Report saved to:", report_path)
