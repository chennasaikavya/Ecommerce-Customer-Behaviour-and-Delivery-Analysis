
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

# Load required datasets
orders = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv")
reviews = pd.read_csv(DATA_DIR / "olist_order_reviews_dataset.csv")
products = pd.read_csv(DATA_DIR / "olist_products_dataset.csv")

print("=" * 65)
print("MISSING DATA INVESTIGATION")
print("=" * 65)

# ------------------------------------------------
# 1. INVESTIGATE MISSING DELIVERY DATES
# ------------------------------------------------

print("\n1. ORDER STATUS AND MISSING DELIVERY DATES")

order_status_summary = orders.groupby("order_status").agg(
    total_orders=("order_id", "count"),
    missing_approval=("order_approved_at",
                      lambda x: x.isna().sum()),
    missing_carrier=("order_delivered_carrier_date",
                     lambda x: x.isna().sum()),
    missing_customer_delivery=("order_delivered_customer_date",
                               lambda x: x.isna().sum())
).reset_index()

print(order_status_summary.to_string(index=False))

order_status_summary.to_csv(
    RESULTS_DIR / "order_status_missing_analysis.csv",
    index=False
)

# ------------------------------------------------
# 2. INVESTIGATE CUSTOMER REVIEWS
# ------------------------------------------------

print("\n2. CUSTOMER REVIEW INVESTIGATION")

reviews["title_missing"] = reviews["review_comment_title"].isna()
reviews["message_missing"] = reviews["review_comment_message"].isna()

review_patterns = reviews.groupby(
    ["title_missing", "message_missing"]
).size().reset_index(name="review_count")

print("\nMissing review text patterns:")
print(review_patterns.to_string(index=False))

print("\nCustomer rating distribution:")
print(reviews["review_score"].value_counts().sort_index())

review_rating_summary = reviews.groupby("review_score").agg(
    total_reviews=("review_id", "count"),
    missing_titles=("title_missing", "sum"),
    missing_messages=("message_missing", "sum")
).reset_index()

print("\nMissing review text by rating:")
print(review_rating_summary.to_string(index=False))

review_rating_summary.to_csv(
    RESULTS_DIR / "review_missing_analysis.csv",
    index=False
)

# ------------------------------------------------
# 3. INVESTIGATE MISSING PRODUCT INFORMATION
# ------------------------------------------------

print("\n3. PRODUCT MISSING DATA INVESTIGATION")

product_columns = [
    "product_category_name",
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm"
]

product_missing_patterns = (
    products[product_columns]
    .isna()
    .value_counts()
    .reset_index(name="product_count")
)

print("\nProduct missing-value patterns:")
print(product_missing_patterns.to_string(index=False))

product_missing_patterns.to_csv(
    RESULTS_DIR / "product_missing_patterns.csv",
    index=False
)

# ------------------------------------------------
# COMPLETION
# ------------------------------------------------

print("\n" + "=" * 65)
print("MISSING DATA INVESTIGATION COMPLETED!")
print("=" * 65)

print("\nThree investigation reports saved in results folder.")
