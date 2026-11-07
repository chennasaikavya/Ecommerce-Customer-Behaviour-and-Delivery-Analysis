
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# E-COMMERCE EXPLORATORY DATA ANALYSIS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent
CLEANED_DIR = PROJECT_DIR / "results" / "cleaned_data"
EDA_DIR = PROJECT_DIR / "results" / "eda"
PLOTS_DIR = EDA_DIR / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

print("=" * 70)
print("E-COMMERCE CUSTOMER BEHAVIOUR AND DELIVERY ANALYSIS")
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)


def load_data(filename):
    path = CLEANED_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path, low_memory=False)


# ==========================================
# LOAD REQUIRED DATASETS
# ==========================================

print("\nLoading cleaned datasets...")

orders = load_data("olist_orders_dataset.csv")
customers = load_data("olist_customers_dataset.csv")
items = load_data("olist_order_items_dataset.csv")
products = load_data("olist_products_dataset.csv")
reviews = load_data("olist_order_reviews_dataset.csv")
sellers = load_data("olist_sellers_dataset.csv")
translations = load_data("product_category_name_translation.csv")

# Convert dates into datetime format
date_columns = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for column in date_columns:
    orders[column] = pd.to_datetime(
        orders[column], errors="coerce"
    )

print("All required datasets loaded successfully!")


# ==========================================
# FEATURE 1: CUSTOMER BEHAVIOUR ANALYSIS
# ==========================================

print("\n1. CUSTOMER BEHAVIOUR ANALYSIS")

# Only completed delivered orders
delivered_orders = orders[
    orders["order_status"] == "delivered"
].copy()

# Calculate total order spending
order_spending = items.groupby(
    "order_id", as_index=False
).agg(
    item_price=("price", "sum"),
    freight_value=("freight_value", "sum"),
    item_count=("order_item_id", "count")
)

order_spending["order_spending"] = (
    order_spending["item_price"] +
    order_spending["freight_value"]
)

# Connect orders with customer information
customer_data = delivered_orders[
    ["order_id", "customer_id"]
].merge(
    customers[
        ["customer_id", "customer_unique_id"]
    ],
    on="customer_id",
    how="left"
).merge(
    order_spending,
    on="order_id",
    how="left"
)

customer_data = customer_data.dropna(
    subset=["customer_unique_id", "order_spending"]
)

# Aggregate purchases by actual unique customer
customer_summary = customer_data.groupby(
    "customer_unique_id", as_index=False
).agg(
    total_orders=("order_id", "nunique"),
    total_spending=("order_spending", "sum"),
    average_order_spending=("order_spending", "mean"),
    total_items=("item_count", "sum")
)

# Classify customers
customer_summary["customer_type"] = np.where(
    customer_summary["total_orders"] >= 2,
    "Frequent",
    "Occasional"
)

customer_summary.to_csv(
    EDA_DIR / "customer_behaviour_summary.csv",
    index=False
)

print("\nCustomer classification:")
print(customer_summary["customer_type"].value_counts())

print("\nAverage customer spending:")
print(customer_summary["total_spending"].describe())

# Graph 1: Frequent vs occasional customers
frequency_counts = (
    customer_summary["customer_type"]
    .value_counts()
    .reindex(["Frequent", "Occasional"], fill_value=0)
)

plt.figure(figsize=(8, 5))
plt.bar(frequency_counts.index, frequency_counts.values)
plt.title("Frequent vs Occasional Customers")
plt.xlabel("Customer Type")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "customer_frequency.png")
plt.close()

# Graph 2: Customer spending distribution
upper_limit = customer_summary[
    "total_spending"
].quantile(0.99)

spending_for_plot = customer_summary[
    "total_spending"
].clip(upper=upper_limit)

plt.figure(figsize=(9, 5))
plt.hist(spending_for_plot, bins=40)
plt.title("Customer Spending Distribution")
plt.xlabel("Total Spending (Price + Freight)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "customer_spending.png")
plt.close()


# ==========================================
# FEATURE 1: POPULAR PRODUCT CATEGORIES
# ==========================================

print("\n2. POPULAR PRODUCT CATEGORY ANALYSIS")

category_products = products[
    ["product_id", "product_category_name"]
].merge(
    translations,
    on="product_category_name",
    how="left"
)

category_products["category"] = (
    category_products["product_category_name_english"]
    .fillna(category_products["product_category_name"])
    .fillna("unknown")
)

delivered_items = items.merge(
    delivered_orders[["order_id"]],
    on="order_id",
    how="inner"
)

delivered_items = delivered_items.merge(
    category_products[["product_id", "category"]],
    on="product_id",
    how="left"
)

delivered_items["category"] = (
    delivered_items["category"].fillna("unknown")
)

category_sales = delivered_items.groupby(
    "category", as_index=False
).agg(
    units_sold=("order_item_id", "count"),
    unique_orders=("order_id", "nunique")
).sort_values(
    "units_sold", ascending=False
)

category_sales.to_csv(
    EDA_DIR / "product_category_analysis.csv",
    index=False
)

print("\nTop 10 popular product categories:")
print(category_sales.head(10).to_string(index=False))

# Graph 3: Top 10 product categories
top_categories = category_sales.head(10).sort_values(
    "units_sold"
)

plt.figure(figsize=(10, 6))
plt.barh(
    top_categories["category"],
    top_categories["units_sold"]
)
plt.title("Top 10 Popular Product Categories")
plt.xlabel("Units Sold")
plt.ylabel("Product Category")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "popular_categories.png")
plt.close()


# ==========================================
# FEATURE 2: DELIVERY PERFORMANCE ANALYSIS
# ==========================================

print("\n3. DELIVERY PERFORMANCE ANALYSIS")

delivery_metrics = delivered_orders[
    [
        "order_id",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]
].dropna(
    subset=[
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]
).copy()

# Actual delivery duration
delivery_metrics["actual_delivery_days"] = (
    delivery_metrics["order_delivered_customer_date"] -
    delivery_metrics["order_purchase_timestamp"]
).dt.total_seconds() / 86400

# Difference between actual and estimated delivery
delivery_metrics["delay_days"] = (
    delivery_metrics["order_delivered_customer_date"] -
    delivery_metrics["order_estimated_delivery_date"]
).dt.total_seconds() / 86400

delivery_metrics["late_delivery"] = (
    delivery_metrics["delay_days"] > 0
)

delivery_metrics.to_csv(
    EDA_DIR / "delivery_metrics.csv",
    index=False
)

total_deliveries = len(delivery_metrics)
late_deliveries = int(
    delivery_metrics["late_delivery"].sum()
)

late_percentage = (
    late_deliveries / total_deliveries * 100
    if total_deliveries else 0
)

print("\nTotal valid delivered orders:", total_deliveries)
print("Late deliveries:", late_deliveries)
print(f"Late delivery percentage: {late_percentage:.2f}%")

print(
    "Average actual delivery days:",
    round(delivery_metrics["actual_delivery_days"].mean(), 2)
)

print(
    "Average delay days:",
    round(delivery_metrics["delay_days"].mean(), 2)
)

# Graph 4: Late vs on-time or early deliveries
delivery_counts = pd.Series({
    "Late": late_deliveries,
    "On time or early": total_deliveries - late_deliveries
})

plt.figure(figsize=(8, 5))
plt.bar(delivery_counts.index, delivery_counts.values)
plt.title("Late vs On-Time or Early Deliveries")
plt.xlabel("Delivery Status")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "delivery_status.png")
plt.close()

# Graph 5: Delivery delay distribution
lower = delivery_metrics["delay_days"].quantile(0.01)
upper = delivery_metrics["delay_days"].quantile(0.99)

delay_for_plot = delivery_metrics[
    "delay_days"
].clip(lower=lower, upper=upper)

plt.figure(figsize=(9, 5))
plt.hist(delay_for_plot, bins=40)
plt.axvline(0, linestyle="--", label="Estimated delivery date")
plt.title("Actual Delivery Delay Distribution")
plt.xlabel("Delay in Days (Positive = Late)")
plt.ylabel("Number of Orders")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "delivery_delay.png")
plt.close()


# ==========================================
# SELLER DELIVERY PERFORMANCE
# ==========================================

print("\n4. SELLER DELIVERY ANALYSIS")

order_seller = items[
    ["order_id", "seller_id"]
].drop_duplicates()

order_seller = order_seller.merge(
    delivery_metrics[
        ["order_id", "delay_days", "late_delivery"]
    ],
    on="order_id",
    how="inner"
)

seller_summary = order_seller.groupby(
    "seller_id", as_index=False
).agg(
    delivered_orders=("order_id", "nunique"),
    average_delay_days=("delay_days", "mean"),
    late_rate=("late_delivery", "mean")
)

seller_summary["late_percentage"] = (
    seller_summary["late_rate"] * 100
)

seller_summary = seller_summary.merge(
    sellers[["seller_id", "seller_city", "seller_state"]],
    on="seller_id",
    how="left"
)

seller_summary.to_csv(
    EDA_DIR / "seller_delivery_analysis.csv",
    index=False
)

# Consider sellers with at least 20 delivered orders
eligible_sellers = seller_summary[
    seller_summary["delivered_orders"] >= 20
]

worst_sellers = eligible_sellers.nlargest(
    10, "average_delay_days"
).sort_values("average_delay_days")

print("\nSellers with highest average delivery delays")
print(worst_sellers.to_string(index=False))

if not worst_sellers.empty:
    plt.figure(figsize=(10, 6))
    plt.barh(
        worst_sellers["seller_id"],
        worst_sellers["average_delay_days"]
    )
    plt.title("10 Sellers with Highest Average Delivery Delays")
    plt.xlabel("Average Delay (Days)")
    plt.ylabel("Seller ID")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "seller_delivery_delays.png")
    plt.close()


# ==========================================
# FEATURE 3: CUSTOMER SATISFACTION ANALYSIS
# ==========================================

print("\n5. CUSTOMER SATISFACTION ANALYSIS")

reviews["review_score"] = pd.to_numeric(
    reviews["review_score"],
    errors="coerce"
)

valid_reviews = reviews.dropna(
    subset=["review_score"]
).copy()

rating_counts = (
    valid_reviews["review_score"]
    .value_counts()
    .sort_index()
    .rename_axis("review_score")
    .reset_index(name="total_reviews")
)

rating_counts.to_csv(
    EDA_DIR / "review_rating_analysis.csv",
    index=False
)

print("\nCustomer rating distribution:")
print(rating_counts.to_string(index=False))

# Graph 6: Review rating distribution
plt.figure(figsize=(8, 5))
plt.bar(
    rating_counts["review_score"].astype(str),
    rating_counts["total_reviews"]
)
plt.title("Customer Review Rating Distribution")
plt.xlabel("Review Score")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "review_ratings.png")
plt.close()


# ==========================================
# DELIVERY DELAY VS CUSTOMER RATINGS
# ==========================================

print("\n6. DELIVERY DELAY VS CUSTOMER RATINGS")

review_by_order = valid_reviews.groupby(
    "order_id", as_index=False
).agg(
    review_score=("review_score", "mean")
)

review_delay = delivery_metrics[
    ["order_id", "delay_days", "late_delivery"]
].merge(
    review_by_order,
    on="order_id",
    how="inner"
)

review_delay["rating_group"] = (
    review_delay["review_score"]
    .round()
    .astype(int)
    .astype(str)
)

review_delay_summary = review_delay.groupby(
    "rating_group", as_index=False
).agg(
    total_orders=("order_id", "nunique"),
    average_delay_days=("delay_days", "mean"),
    late_rate=("late_delivery", "mean")
)

review_delay_summary["late_percentage"] = (
    review_delay_summary["late_rate"] * 100
)

review_delay_summary.to_csv(
    EDA_DIR / "delay_vs_rating_analysis.csv",
    index=False
)

print("\nDelivery delay by rating:")
print(review_delay_summary.to_string(index=False))

# Graph 7: Delivery delay vs rating
plot_lower = review_delay["delay_days"].quantile(0.01)
plot_upper = review_delay["delay_days"].quantile(0.99)

review_delay_plot = review_delay[
    review_delay["delay_days"].between(
        plot_lower, plot_upper
    )
]

plt.figure(figsize=(9, 5))
sns.boxplot(
    data=review_delay_plot,
    x="rating_group",
    y="delay_days",
    order=["1", "2", "3", "4", "5"]
)
plt.axhline(0, linestyle="--")
plt.title("Delivery Delay vs Customer Review Rating")
plt.xlabel("Customer Review Rating")
plt.ylabel("Delivery Delay (Days)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "delay_vs_rating.png")
plt.close()


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n" + "=" * 70)
print("EXPLORATORY DATA ANALYSIS COMPLETED!")
print("=" * 70)

print("\nCustomer records analysed:", len(customer_summary))
print("Product categories analysed:", len(category_sales))
print("Delivery records analysed:", total_deliveries)
print("Customer reviews analysed:", len(valid_reviews))

print("\nEDA reports saved in:")
print(EDA_DIR)

print("\nVisualizations saved in:")
print(PLOTS_DIR)

print("\nAll EDA analyses completed successfully!")
