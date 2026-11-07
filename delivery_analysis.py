
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DELIVERY_FILE = BASE_DIR / "results" / "eda" / "delivery_metrics.csv"
ITEMS_FILE = BASE_DIR / "data" / "olist_order_items_dataset.csv"
SELLERS_FILE = BASE_DIR / "data" / "olist_sellers_dataset.csv"

OUTPUT_DIR = BASE_DIR / "results" / "delivery_analysis"
PLOTS_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

print("=" * 60)
print("E-COMMERCE DELIVERY PERFORMANCE ANALYSIS")
print("=" * 60)

# ==========================================
# 1. LOAD DATASETS
# ==========================================

delivery = pd.read_csv(DELIVERY_FILE)
items = pd.read_csv(ITEMS_FILE)
sellers = pd.read_csv(SELLERS_FILE)

delivery["delay_days"] = pd.to_numeric(
    delivery["delay_days"], errors="coerce"
)

delivery = delivery.dropna(
    subset=["order_id", "delay_days"]
).drop_duplicates(subset=["order_id"])

if "late_delivery" in delivery.columns:
    delivery["late_delivery"] = (
        delivery["late_delivery"]
        .astype(str).str.lower().isin(["true", "1", "yes"])
    )
else:
    delivery["late_delivery"] = delivery["delay_days"] > 0

print("\nDatasets loaded successfully!")

# ==========================================
# 2. OVERALL DELIVERY PERFORMANCE
# ==========================================

total_orders = len(delivery)
late_orders = int(delivery["late_delivery"].sum())
early_or_ontime = total_orders - late_orders

late_rate = late_orders / total_orders * 100

summary = {
    "Total_Delivered_Orders": total_orders,
    "Late_Orders": late_orders,
    "On_Time_or_Early_Orders": early_or_ontime,
    "Late_Delivery_Rate_Percent": round(late_rate, 2),
    "On_Time_Rate_Percent": round(
        early_or_ontime / total_orders * 100, 2
    ),
    "Average_Actual_Delivery_Days": round(
        delivery["actual_delivery_days"].mean(), 2
    ),
    "Average_Delay_Days": round(
        delivery["delay_days"].mean(), 2
    ),
    "Average_Lateness_Days": round(
        delivery.loc[
            delivery["late_delivery"], "delay_days"
        ].mean(), 2
    )
}

pd.DataFrame([summary]).to_csv(
    OUTPUT_DIR / "delivery_performance_summary.csv",
    index=False
)

print("\nOVERALL DELIVERY PERFORMANCE")
for key, value in summary.items():
    print(f"{key}: {value}")

# ==========================================
# 3. DELIVERY STATUS VISUALIZATION
# ==========================================

status_data = pd.DataFrame({
    "Delivery Status": ["On Time / Early", "Late"],
    "Orders": [early_or_ontime, late_orders]
})

plt.figure(figsize=(8, 5))
sns.barplot(
    data=status_data,
    x="Delivery Status",
    y="Orders"
)

plt.title("On-Time vs Late Delivery Performance")
plt.xlabel("Delivery Status")
plt.ylabel("Number of Orders")
plt.tight_layout()

plt.savefig(PLOTS_DIR / "delivery_status_comparison.png")
plt.close()

# ==========================================
# 4. DELIVERY DELAY DISTRIBUTION
# ==========================================

plt.figure(figsize=(9, 5))

sns.histplot(
    data=delivery,
    x="delay_days",
    bins=50
)

plt.axvline(
    0,
    linestyle="--",
    color="red",
    label="Estimated delivery date"
)

plt.title("Distribution of Delivery Delays")
plt.xlabel("Delay Days (Negative = Early, Positive = Late)")
plt.ylabel("Number of Orders")
plt.legend()
plt.tight_layout()

plt.savefig(PLOTS_DIR / "delivery_delay_distribution.png")
plt.close()

# ==========================================
# 5. SELLER DELIVERY PERFORMANCE
# ==========================================

order_sellers = items[
    ["order_id", "seller_id"]
].drop_duplicates()

seller_orders = order_sellers.merge(
    delivery[
        ["order_id", "delay_days", "late_delivery"]
    ],
    on="order_id",
    how="inner"
)

seller_stats = seller_orders.groupby("seller_id").agg(
    delivered_orders=("order_id", "nunique"),
    average_delay_days=("delay_days", "mean"),
    late_orders=("late_delivery", "sum"),
    late_rate=("late_delivery", "mean")
).reset_index()

late_only = (
    seller_orders[seller_orders["late_delivery"]]
    .groupby("seller_id")["delay_days"]
    .mean()
    .rename("average_late_days")
)

seller_stats = seller_stats.merge(
    late_only,
    on="seller_id",
    how="left"
)

seller_stats = seller_stats.merge(
    sellers[["seller_id", "seller_city", "seller_state"]],
    on="seller_id",
    how="left"
)

seller_stats["late_rate_percent"] = (
    seller_stats["late_rate"] * 100
)

seller_stats.to_csv(
    OUTPUT_DIR / "seller_delivery_performance.csv",
    index=False
)

# Consider sellers with at least 20 delivered orders

eligible_sellers = seller_stats[
    seller_stats["delivered_orders"] >= 20
]

top_sellers = eligible_sellers.sort_values(
    "late_rate",
    ascending=False
).head(10).copy()

top_sellers.to_csv(
    OUTPUT_DIR / "top_10_late_sellers.csv",
    index=False
)

print("\nTOP 10 SELLERS WITH HIGHEST LATE RATES")
print(
    top_sellers[
        [
            "seller_id",
            "seller_city",
            "seller_state",
            "delivered_orders",
            "late_rate_percent",
            "average_late_days"
        ]
    ].to_string(index=False)
)

# ==========================================
# 6. SELLER VISUALIZATION
# ==========================================

if not top_sellers.empty:

    top_sellers["seller_label"] = (
        top_sellers["seller_id"].str[:8]
        + " - "
        + top_sellers["seller_city"].fillna("Unknown")
    )

    plt.figure(figsize=(11, 6))

    sns.barplot(
        data=top_sellers,
        x="late_rate_percent",
        y="seller_label"
    )

    plt.title("Top 10 Sellers by Late Delivery Rate")
    plt.xlabel("Late Delivery Rate (%)")
    plt.ylabel("Seller")
    plt.tight_layout()

    plt.savefig(PLOTS_DIR / "top_late_sellers.png")
    plt.close()

# ==========================================
# 7. SELLER STATE ANALYSIS
# ==========================================

state_data = seller_orders.merge(
    sellers[["seller_id", "seller_state"]],
    on="seller_id",
    how="left"
)

state_stats = state_data.groupby("seller_state").agg(
    seller_order_records=("order_id", "size"),
    late_rate=("late_delivery", "mean"),
    average_delay_days=("delay_days", "mean")
).reset_index()

state_stats["late_rate_percent"] = (
    state_stats["late_rate"] * 100
)

state_stats.to_csv(
    OUTPUT_DIR / "seller_state_delivery_performance.csv",
    index=False
)

top_states = state_stats[
    state_stats["seller_order_records"] >= 100
].sort_values(
    "late_rate",
    ascending=False
).head(10)

if not top_states.empty:

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=top_states,
        x="late_rate_percent",
        y="seller_state"
    )

    plt.title("States with Highest Seller-Order Late Rates")
    plt.xlabel("Late Delivery Rate (%)")
    plt.ylabel("Seller State")
    plt.tight_layout()

    plt.savefig(PLOTS_DIR / "state_delivery_performance.png")
    plt.close()

# ==========================================
# 8. COMPLETION
# ==========================================

print("\n" + "=" * 60)
print("DELIVERY PERFORMANCE ANALYSIS COMPLETED!")
print("=" * 60)

print("\nReports saved in:")
print(OUTPUT_DIR)

print("\nVisualizations saved in:")
print(PLOTS_DIR)
