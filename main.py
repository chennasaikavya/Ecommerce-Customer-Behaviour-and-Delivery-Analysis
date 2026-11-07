
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

print("=" * 60)
print("E-COMMERCE CUSTOMER BEHAVIOUR AND DELIVERY ANALYSIS")
print("=" * 60)

if DATA_DIR.exists():

    print("\nDataset folder found successfully!")

    csv_files = sorted(DATA_DIR.glob("*.csv"))

    print(f"\nTotal CSV files found: {len(csv_files)}")

    for file in csv_files:
        print(" -", file.name)

else:
    print("Dataset folder not found!")

print("\nProject setup check completed!")
