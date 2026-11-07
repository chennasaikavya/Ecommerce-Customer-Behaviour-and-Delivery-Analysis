
from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

REVIEWS_FILE = DATA_DIR / "olist_order_reviews_dataset.csv"
ORDERS_FILE = DATA_DIR / "olist_orders_dataset.csv"

OUTPUT_DIR = BASE_DIR / "results" / "customer_satisfaction"
PLOTS_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

print("=" * 65)
print("CUSTOMER SATISFACTION AND SENTIMENT ANALYSIS")
print("=" * 65)

# ==========================================
# 2. LOAD DATASETS
# ==========================================

reviews = pd.read_csv(REVIEWS_FILE)
orders = pd.read_csv(ORDERS_FILE)

print("\nDatasets loaded successfully!")

# ==========================================
# 3. PREPARE DELIVERY DATA
# ==========================================

date_columns = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for col in date_columns:
    orders[col] = pd.to_datetime(
        orders[col], errors="coerce"
    )

delivered_orders = orders[
    orders["order_status"] == "delivered"
].copy()

delivered_orders = delivered_orders.dropna(
    subset=[
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]
)

# Actual delivery duration
delivered_orders["actual_delivery_days"] = (
    delivered_orders["order_delivered_customer_date"]
    - delivered_orders["order_purchase_timestamp"]
).dt.total_seconds() / 86400

# Difference between actual and estimated date
# Positive = late, negative = early

delivered_orders["delay_days"] = (
    delivered_orders["order_delivered_customer_date"]
    - delivered_orders["order_estimated_delivery_date"]
).dt.total_seconds() / 86400

delivered_orders["late_delivery"] = (
    delivered_orders["delay_days"] > 0
)

delivery_columns = delivered_orders[
    [
        "order_id",
        "actual_delivery_days",
        "delay_days",
        "late_delivery"
    ]
]

# ==========================================
# 4. MERGE REVIEWS AND DELIVERY DATA
# ==========================================

data = reviews.merge(
    delivery_columns,
    on="order_id",
    how="inner"
)

data["review_score"] = pd.to_numeric(
    data["review_score"], errors="coerce"
)

data = data.dropna(
    subset=["review_score", "delay_days"]
)

data = data[
    data["review_score"].between(1, 5)
].copy()

data["delivery_status"] = np.where(
    data["late_delivery"],
    "Late",
    "On time / early"
)

print("\nReviews matched with delivered orders:", len(data))

# ==========================================
# 5. CUSTOMER RATING ANALYSIS
# ==========================================

rating_distribution = (
    data["review_score"]
    .value_counts()
    .sort_index()
    .reset_index()
)

rating_distribution.columns = [
    "Review_Score", "Number_of_Reviews"
]

rating_distribution.to_csv(
    OUTPUT_DIR / "rating_distribution.csv",
    index=False
)

average_rating = data["review_score"].mean()

print("\nCUSTOMER RATING ANALYSIS")
print("Total reviews:", len(data))
print("Average rating:", round(average_rating, 2))
print("\nRating distribution:")
print(rating_distribution.to_string(index=False))

# ==========================================
# 6. DELIVERY DELAY VS CUSTOMER RATINGS
# ==========================================

delivery_rating = data.groupby(
    "delivery_status"
).agg(
    total_reviews=("review_score", "count"),
    average_rating=("review_score", "mean"),
    poor_rating_count=(
        "review_score",
        lambda x: (x <= 2).sum()
    )
).reset_index()

delivery_rating["poor_rating_percentage"] = (
    delivery_rating["poor_rating_count"]
    / delivery_rating["total_reviews"] * 100
)

delivery_rating["average_rating"] = (
    delivery_rating["average_rating"].round(2)
)

delivery_rating.to_csv(
    OUTPUT_DIR / "rating_by_delivery_status.csv",
    index=False
)

# Correlation between delivery delay and rating

pearson_corr = data[
    ["delay_days", "review_score"]
].corr(method="pearson").iloc[0, 1]

spearman_corr = data[
    ["delay_days", "review_score"]
].corr(method="spearman").iloc[0, 1]

correlation_report = pd.DataFrame({
    "Method": ["Pearson", "Spearman"],
    "Correlation": [pearson_corr, spearman_corr]
})

correlation_report.to_csv(
    OUTPUT_DIR / "delay_rating_correlation.csv",
    index=False
)

print("\nDELIVERY DELAY VS CUSTOMER RATINGS")
print(delivery_rating.to_string(index=False))

print("\nCorrelation results:")
print(correlation_report.to_string(index=False))

# ==========================================
# 7. RATING ANALYSIS BY DELAY GROUP
# ==========================================

data["delay_category"] = pd.cut(
    data["delay_days"],
    bins=[-np.inf, 0, 3, 7, np.inf],
    labels=[
        "Early / On time",
        "1-3 Days Late",
        "4-7 Days Late",
        "More than 7 Days Late"
    ]
)

delay_rating = data.groupby(
    "delay_category",
    observed=False
).agg(
    total_reviews=("review_score", "count"),
    average_rating=("review_score", "mean")
).reset_index()

delay_rating["average_rating"] = (
    delay_rating["average_rating"].round(2)
)

delay_rating.to_csv(
    OUTPUT_DIR / "rating_by_delay_category.csv",
    index=False
)

# ==========================================
# 8. PORTUGUESE NLP SENTIMENT ANALYSIS
# ==========================================

positive_words = {
    "bom", "boa", "otimo", "otima", "excelente",
    "maravilhoso", "maravilhosa", "perfeito", "perfeita",
    "adorei", "amei", "gostei", "recomendo",
    "satisfeito", "satisfeita", "rapido", "rapida",
    "pontual", "qualidade", "lindo", "linda",
    "feliz", "eficiente", "atencioso", "atenciosa",
    "funciona", "funcionando", "conforme", "correto",
    "bem", "obrigado", "obrigada", "otimo"
}

negative_words = {
    "ruim", "pessimo", "pessima", "horrivel",
    "defeito", "quebrado", "quebrada",
    "danificado", "danificada", "atrasado",
    "atrasada", "atraso", "demora", "demorado",
    "demorada", "problema", "errado", "errada",
    "faltou", "faltando", "decepcionado",
    "decepcionada", "insatisfeito", "insatisfeita",
    "cancelado", "cancelada", "incompleto",
    "incompleta", "infelizmente", "fraco", "fraca",
    "reclamei", "devolver", "devolucao",
    "lento", "lenta", "descaso", "terrivel"
}

negation_words = {"nao", "nunca", "jamais", "sem"}

intensifiers = {"muito", "super", "extremamente", "bastante"}

def normalize_text(text):
    text = str(text).lower()

    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    tokens = re.findall(r"[a-z]+", text)

    return tokens


def analyze_sentiment(text):

    if pd.isna(text) or not str(text).strip():
        return pd.Series({
            "sentiment": "No comment",
            "sentiment_score": 0
        })

    tokens = normalize_text(text)

    score = 0

    for i, word in enumerate(tokens):

        if word in positive_words:
            polarity = 1

        elif word in negative_words:
            polarity = -1

        else:
            continue

        previous_words = tokens[max(0, i - 3):i]

        # Reverse sentiment when preceded by negation
        if any(w in negation_words for w in previous_words):
            polarity *= -1

        # Increase sentiment strength
        if any(w in intensifiers for w in previous_words):
            polarity *= 1.5

        score += polarity

    if score > 0:
        sentiment = "Positive"

    elif score < 0:
        sentiment = "Negative"

    else:
        sentiment = "Neutral"

    return pd.Series({
        "sentiment": sentiment,
        "sentiment_score": score
    })


# Combine review title and review message

data["review_text"] = (
    data["review_comment_title"].fillna("").astype(str)
    + " "
    + data["review_comment_message"].fillna("").astype(str)
).str.strip()

sentiment_results = data["review_text"].apply(
    analyze_sentiment
)

data["sentiment"] = sentiment_results["sentiment"]
data["sentiment_score"] = sentiment_results["sentiment_score"]

# ==========================================
# 9. SENTIMENT SUMMARY
# ==========================================

commented_reviews = data[
    data["sentiment"] != "No comment"
].copy()

sentiment_summary = (
    commented_reviews["sentiment"]
    .value_counts()
    .rename_axis("Sentiment")
    .reset_index(name="Number_of_Reviews")
)

sentiment_summary["Percentage"] = (
    sentiment_summary["Number_of_Reviews"]
    / len(commented_reviews) * 100
).round(2)

sentiment_summary.to_csv(
    OUTPUT_DIR / "sentiment_analysis_summary.csv",
    index=False
)

# Sentiment and delivery relationship

sentiment_delivery = pd.crosstab(
    commented_reviews["delivery_status"],
    commented_reviews["sentiment"]
)

sentiment_delivery.to_csv(
    OUTPUT_DIR / "sentiment_by_delivery_status.csv"
)

# Save detailed review analysis

data.to_csv(
    OUTPUT_DIR / "customer_review_sentiment_results.csv",
    index=False
)

print("\nSENTIMENT ANALYSIS")
print("Reviews containing written comments:", len(commented_reviews))
print(sentiment_summary.to_string(index=False))

# ==========================================
# 10. VISUALIZATION - RATING DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=rating_distribution,
    x="Review_Score",
    y="Number_of_Reviews"
)

plt.title("Customer Review Rating Distribution")
plt.xlabel("Review Score")
plt.ylabel("Number of Reviews")
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "customer_rating_distribution.png"
)
plt.close()

# ==========================================
# 11. VISUALIZATION - DELIVERY VS RATING
# ==========================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=delivery_rating,
    x="delivery_status",
    y="average_rating"
)

plt.title("Average Customer Rating by Delivery Status")
plt.xlabel("Delivery Status")
plt.ylabel("Average Rating")
plt.ylim(0, 5)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "delivery_vs_customer_rating.png"
)
plt.close()

# ==========================================
# 12. VISUALIZATION - DELAY GROUPS
# ==========================================

plt.figure(figsize=(10, 5))

sns.barplot(
    data=delay_rating,
    x="delay_category",
    y="average_rating"
)

plt.title("Customer Ratings Across Delivery Delay Groups")
plt.xlabel("Delivery Delay Category")
plt.ylabel("Average Rating")
plt.ylim(0, 5)
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "rating_by_delay_category.png"
)
plt.close()

# ==========================================
# 13. VISUALIZATION - SENTIMENT
# ==========================================

if not sentiment_summary.empty:

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=sentiment_summary,
        x="Sentiment",
        y="Number_of_Reviews"
    )

    plt.title("Customer Review Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "customer_sentiment_distribution.png"
    )
    plt.close()

# ==========================================
# 14. FINAL OUTPUT
# ==========================================

print("\n" + "=" * 65)
print("CUSTOMER SATISFACTION ANALYSIS COMPLETED!")
print("=" * 65)

print("\nTotal reviews analysed:", len(data))
print("Reviews with written comments:", len(commented_reviews))
print("Average customer rating:", round(average_rating, 2))
print("Delay-rating Pearson correlation:", round(pearson_corr, 4))

print("\nReports saved in:")
print(OUTPUT_DIR)

print("\nVisualizations saved in:")
print(PLOTS_DIR)
