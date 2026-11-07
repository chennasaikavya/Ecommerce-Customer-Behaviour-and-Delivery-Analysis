
from pathlib import Path
import subprocess
import sys

# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

scripts = [
    "data_cleaning.py",
    "eda_analysis.py",
    "customer_segmentation.py",
    "delivery_analysis.py",
    "customer_satisfaction_analysis.py",
    "poor_review_prediction.py"
]

print("=" * 65)
print("E-COMMERCE CUSTOMER BEHAVIOUR AND DELIVERY ANALYSIS")
print("COMPLETE PROJECT EXECUTION")
print("=" * 65)

# ==========================================
# RUN ALL PROJECT FEATURES
# ==========================================

for index, script in enumerate(scripts, start=1):

    script_path = BASE_DIR / script

    print("\n" + "=" * 65)
    print(f"STEP {index}/{len(scripts)}: {script}")
    print("=" * 65)

    if not script_path.exists():
        print(f"ERROR: {script} was not found!")
        sys.exit(1)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR
    )

    if result.returncode != 0:
        print(f"\nERROR: {script} failed!")
        print("Project execution stopped.")
        sys.exit(result.returncode)

    print(f"\nSUCCESS: {script} completed!")

# ==========================================
# FINAL COMPLETION
# ==========================================

print("\n" + "=" * 65)
print("ALL PROJECT FEATURES COMPLETED SUCCESSFULLY!")
print("=" * 65)

print("\nImplemented Features:")
print("1. Customer Behaviour Analysis")
print("2. Delivery Performance Analysis")
print("3. Customer Satisfaction Analysis")
print("4. Poor Customer Review Prediction")

print("\nAll reports and visualizations are available")
print("inside the results folder.")

print("\nPROJECT EXECUTION FINISHED!")
