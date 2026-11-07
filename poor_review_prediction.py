
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve
)

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

OUTPUT_DIR = BASE_DIR / "results" / "poor_review_prediction"
PLOTS_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

print("=" * 65)
print("POOR CUSTOMER REVIEW PREDICTION")
print("=" * 65)

# ==========================================
# 2. LOAD DATASETS
# ==========================================

orders = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv")
reviews = pd.read_csv(DATA_DIR / "olist_order_reviews_dataset.csv")
items = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")
payments = pd.read_csv(DATA_DIR / "olist_order_payments_dataset.csv")
customers = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")
sellers = pd.read_csv(DATA_DIR / "olist_sellers_dataset.csv")

print("\nAll datasets loaded successfully!")

# ==========================================
# 3. PREPARE REVIEW TARGET
# ==========================================

reviews["review_score"] = pd.to_numeric(
    reviews["review_score"], errors="coerce"
)

review_data = reviews.groupby("order_id").agg(
    review_score=("review_score", "mean")
).reset_index()

# ==========================================
# 4. PREPARE DELIVERY INFORMATION
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

data = orders[
    orders["order_status"] == "delivered"
].copy()

data = data.dropna(subset=date_columns)

data = data.merge(
    review_data,
    on="order_id",
    how="inner"
)

data = data.dropna(subset=["review_score"])

data["actual_delivery_days"] = (
    data["order_delivered_customer_date"]
    - data["order_purchase_timestamp"]
).dt.total_seconds() / 86400

data["delay_days"] = (
    data["order_delivered_customer_date"]
    - data["order_estimated_delivery_date"]
).dt.total_seconds() / 86400

data["late_delivery"] = (
    data["delay_days"] > 0
).astype(int)

data["estimated_delivery_days"] = (
    data["order_estimated_delivery_date"]
    - data["order_purchase_timestamp"]
).dt.total_seconds() / 86400

data["purchase_hour"] = (
    data["order_purchase_timestamp"].dt.hour
)

data["purchase_day"] = (
    data["order_purchase_timestamp"].dt.dayofweek
)

data["purchase_month"] = (
    data["order_purchase_timestamp"].dt.month
)

# ==========================================
# 5. CREATE TARGET VARIABLE
# ==========================================

# 1 = Poor review (1 or 2 stars)
# 0 = Non-poor review (3, 4 or 5 stars)

data["poor_review"] = (
    data["review_score"] <= 2
).astype(int)

print("\nTARGET DISTRIBUTION:")
print(data["poor_review"].value_counts())

print("\nPoor review percentage:",
      round(data["poor_review"].mean() * 100, 2), "%")

# ==========================================
# 6. CUSTOMER LOCATION FEATURES
# ==========================================

data = data.merge(
    customers[["customer_id", "customer_state"]],
    on="customer_id",
    how="left"
)

# ==========================================
# 7. ORDER ITEM FEATURES
# ==========================================

item_features = items.groupby("order_id").agg(
    item_count=("order_item_id", "count"),
    total_price=("price", "sum"),
    total_freight=("freight_value", "sum"),
    avg_item_price=("price", "mean"),
    unique_sellers=("seller_id", "nunique"),
    unique_products=("product_id", "nunique")
).reset_index()

data = data.merge(
    item_features,
    on="order_id",
    how="left"
)

# ==========================================
# 8. PAYMENT FEATURES
# ==========================================

def get_mode(series):
    modes = series.mode(dropna=True)

    if len(modes) > 0:
        return modes.iloc[0]

    return "Unknown"


payment_features = payments.groupby("order_id").agg(
    payment_value=("payment_value", "sum"),
    max_installments=("payment_installments", "max"),
    payment_methods=("payment_type", "nunique"),
    payment_records=("payment_sequential", "count"),
    payment_type=("payment_type", get_mode)
).reset_index()

data = data.merge(
    payment_features,
    on="order_id",
    how="left"
)

# ==========================================
# 9. SELLER LOCATION FEATURES
# ==========================================

seller_order_data = items[
    ["order_id", "seller_id"]
].merge(
    sellers[["seller_id", "seller_state"]],
    on="seller_id",
    how="left"
)

seller_features = seller_order_data.groupby(
    "order_id"
).agg(
    seller_state=("seller_state", get_mode)
).reset_index()

data = data.merge(
    seller_features,
    on="order_id",
    how="left"
)

# ==========================================
# 10. SELECT MACHINE LEARNING FEATURES
# ==========================================

numeric_features = [
    "actual_delivery_days",
    "delay_days",
    "estimated_delivery_days",
    "late_delivery",
    "purchase_hour",
    "purchase_day",
    "purchase_month",
    "item_count",
    "total_price",
    "total_freight",
    "avg_item_price",
    "unique_sellers",
    "unique_products",
    "payment_value",
    "max_installments",
    "payment_methods",
    "payment_records"
]

categorical_features = [
    "payment_type",
    "customer_state",
    "seller_state"
]

feature_columns = numeric_features + categorical_features

# Remove infinite values
data = data.replace(
    [np.inf, -np.inf], np.nan
)

X = data[feature_columns].copy()
y = data["poor_review"].copy()
order_ids = data["order_id"].copy()

print("\nTotal records for machine learning:", len(X))
print("Total input features:", len(feature_columns))

# ==========================================
# 11. TRAIN AND TEST SPLIT
# ==========================================

if y.nunique() < 2 or y.value_counts().min() < 2:
    raise ValueError(
        "Insufficient examples in one target class for stratified training."
    )

X_train, X_test, y_train, y_test, id_train, id_test = (
    train_test_split(
        X,
        y,
        order_ids,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ==========================================
# 12. DATA PREPROCESSING
# ==========================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ==========================================
# 13. DEFINE MACHINE LEARNING MODELS
# ==========================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
}

# ==========================================
# 14. TRAIN AND EVALUATE MODELS
# ==========================================

results = []
trained_models = {}
model_probabilities = {}
model_predictions = {}

print("\nTRAINING MACHINE LEARNING MODELS...")

for name, classifier in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test, probabilities
    )

    pr_auc = average_precision_score(
        y_test, probabilities
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc
    })

    trained_models[name] = pipeline
    model_probabilities[name] = probabilities
    model_predictions[name] = predictions

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc, 4))
    print("PR-AUC:", round(pr_auc, 4))

# ==========================================
# 15. COMPARE MODEL PERFORMANCE
# ==========================================

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by=["F1_Score", "Recall"],
    ascending=False
)

comparison.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

print("\nMODEL PERFORMANCE COMPARISON")
print(comparison.to_string(index=False))

best_model_name = comparison.iloc[0]["Model"]

print("\nBEST MODEL:", best_model_name)

# ==========================================
# 16. SAVE PREDICTIONS
# ==========================================

best_predictions = model_predictions[best_model_name]
best_probabilities = model_probabilities[best_model_name]

prediction_results = pd.DataFrame({
    "order_id": id_test.to_numpy(),
    "actual_poor_review": y_test.to_numpy(),
    "predicted_poor_review": best_predictions,
    "poor_review_probability": best_probabilities
})

prediction_results.to_csv(
    OUTPUT_DIR / "poor_review_predictions.csv",
    index=False
)

# ==========================================
# 17. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=[0, 1]
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Poor", "Poor"],
    yticklabels=["Not Poor", "Poor"]
)

plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "best_model_confusion_matrix.png"
)

plt.close()

# ==========================================
# 18. MODEL COMPARISON VISUALIZATION
# ==========================================

plt.figure(figsize=(9, 5))

sns.barplot(
    data=comparison,
    x="Model",
    y="F1_Score"
)

plt.title("F1-Score Comparison of Machine Learning Models")
plt.xlabel("Machine Learning Algorithm")
plt.ylabel("F1 Score")
plt.xticks(rotation=10)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "model_f1_comparison.png"
)

plt.close()

# ==========================================
# 19. ROC CURVE COMPARISON
# ==========================================

plt.figure(figsize=(8, 6))

for name, probabilities in model_probabilities.items():

    fpr, tpr, thresholds = roc_curve(
        y_test, probabilities
    )

    auc_value = roc_auc_score(
        y_test, probabilities
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc_value:.3f})"
    )

plt.plot(
    [0, 1], [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "roc_curve_comparison.png"
)

plt.close()

# ==========================================
# 20. FEATURE IMPORTANCE
# ==========================================

rf_pipeline = trained_models["Random Forest"]

feature_names = (
    rf_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

importances = (
    rf_pipeline
    .named_steps["model"]
    .feature_importances_
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

importance_df.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

top_features = importance_df.head(20).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 8))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title("Top 20 Factors Influencing Poor Review Prediction")
plt.xlabel("Random Forest Feature Importance")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "feature_importance.png"
)

plt.close()

# ==========================================
# 21. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 65)
print("POOR REVIEW PREDICTION COMPLETED SUCCESSFULLY!")
print("=" * 65)

print("\nTotal records analysed:", len(data))
print("Poor review records:", int(y.sum()))
print("Non-poor review records:", int((y == 0).sum()))

print("\nBest performing model:", best_model_name)

print("\nTOP 10 IMPORTANT FEATURES:")
print(
    importance_df.head(10).to_string(index=False)
)

print("\nResults saved in:")
print(OUTPUT_DIR)

print("\nVisualizations saved in:")
print(PLOTS_DIR)

print("\nALL THREE MODELS TRAINED SUCCESSFULLY!")
