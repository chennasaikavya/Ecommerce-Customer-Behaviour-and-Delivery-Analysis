
from pathlib import Path
import pandas as pd

# ==========================================
# E-COMMERCE DATA CLEANING AND PREPARATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
CLEANED_DIR = RESULTS_DIR / "cleaned_data"

CLEANED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("E-COMMERCE CUSTOMER BEHAVIOUR AND DELIVERY ANALYSIS")
print("DATA CLEANING AND PREPARATION")
print("=" * 70)

csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    print("ERROR: No CSV datasets found!")
    raise SystemExit

cleaning_summary = []

# Date columns in the orders dataset
order_date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

# Date columns in the reviews dataset
review_date_columns = [
    "review_creation_date",
    "review_answer_timestamp"
]

# Numeric product columns
product_numeric_columns = [
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm"
]

for file in csv_files:

    print("\n" + "-" * 70)
    print("Processing:", file.name)

    # Load original dataset
    try:
        df = pd.read_csv(file, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(
            file,
            encoding="latin-1",
            low_memory=False
        )

    original_rows = len(df)
    missing_before = int(df.isna().sum().sum())
    duplicates_removed = 0

    # --------------------------------------
    # 1. GEOLOCATION DATASET
    # --------------------------------------

    if file.name == "olist_geolocation_dataset.csv":

        duplicates_removed = int(df.duplicated().sum())

        # Remove only exact duplicate rows
        df = df.drop_duplicates().copy()

        print("Exact duplicate rows removed:",
              duplicates_removed)

    # --------------------------------------
    # 2. ORDER DATASET
    # --------------------------------------

    elif file.name == "olist_orders_dataset.csv":

        # Standardize order status
        if "order_status" in df.columns:
            df["order_status"] = (
                df["order_status"]
                .astype("string")
                .str.strip()
                .str.lower()
            )

        # Convert date columns
        for column in order_date_columns:
            if column in df.columns:
                df[column] = pd.to_datetime(
                    df[column],
                    errors="coerce"
                )

        print("Order statuses standardized.")
        print("Order dates converted to datetime.")

    # --------------------------------------
    # 3. CUSTOMER REVIEW DATASET
    # --------------------------------------

    elif file.name == "olist_order_reviews_dataset.csv":

        # Preserve missing-value information
        df["review_title_was_missing"] = (
            df["review_comment_title"].isna()
        )

        df["review_message_was_missing"] = (
            df["review_comment_message"].isna()
        )

        # Replace missing text with empty strings
        df["review_comment_title"] = (
            df["review_comment_title"]
            .fillna("")
            .astype(str)
        )

        df["review_comment_message"] = (
            df["review_comment_message"]
            .fillna("")
            .astype(str)
        )

        # Combine title and message for NLP
        df["review_text"] = (
            df["review_comment_title"] + " " +
            df["review_comment_message"]
        ).str.strip()

        # Convert review dates
        for column in review_date_columns:
            if column in df.columns:
                df[column] = pd.to_datetime(
                    df[column],
                    errors="coerce"
                )

        print("Missing review text handled.")
        print("Combined review_text column created.")

    # --------------------------------------
    # 4. PRODUCT DATASET
    # --------------------------------------

    elif file.name == "olist_products_dataset.csv":

        # Record missing category information
        df["product_category_was_missing"] = (
            df["product_category_name"].isna()
        )

        # Preserve missing categories as Unknown
        df["product_category_name"] = (
            df["product_category_name"]
            .fillna("unknown")
        )

        # Preserve missing numeric values and
        # create indicators for later analysis
        for column in product_numeric_columns:

            if column in df.columns:

                df[column + "_was_missing"] = (
                    df[column].isna()
                )

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

        print("Unknown product categories handled.")
        print("Numeric missing-value indicators created.")

    # --------------------------------------
    # 5. OTHER DATASETS
    # --------------------------------------

    # Keep other datasets unchanged because
    # our initial inspection found no missing
    # values or exact duplicate rows in them.

    # --------------------------------------
    # SAVE CLEANED DATASET
    # --------------------------------------

    output_path = CLEANED_DIR / file.name

    df.to_csv(output_path, index=False)

    missing_after = int(df.isna().sum().sum())

    # Record cleaning statistics
    cleaning_summary.append({
        "Dataset": file.name,
        "Original_Rows": original_rows,
        "Final_Rows": len(df),
        "Duplicates_Removed": duplicates_removed,
        "Missing_Before": missing_before,
        "Missing_After": missing_after
    })

    print("Original rows:", original_rows)
    print("Final rows:", len(df))
    print("Missing values before:", missing_before)
    print("Missing values after:", missing_after)
    print("Saved:", output_path.name)

    del df

# ==========================================
# SAVE CLEANING SUMMARY
# ==========================================

summary = pd.DataFrame(cleaning_summary)

summary_path = RESULTS_DIR / "data_cleaning_summary.csv"

summary.to_csv(summary_path, index=False)

print("\n" + "=" * 70)
print("DATA CLEANING COMPLETED SUCCESSFULLY!")
print("=" * 70)

print(summary.to_string(index=False))

print("\nCleaned datasets saved in:")
print(CLEANED_DIR)

print("\nCleaning summary saved in:")
print(summary_path)

print("\nAll original datasets remain unchanged.")
