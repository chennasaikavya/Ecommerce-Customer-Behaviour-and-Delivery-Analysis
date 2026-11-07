
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ==========================================
# CUSTOMER SEGMENTATION USING K-MEANS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_DIR / "results" / "eda" /
    "customer_behaviour_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR / "results" / "customer_segmentation"
)

PLOTS_DIR = OUTPUT_DIR / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

print("=" * 65)
print("CUSTOMER SEGMENTATION USING K-MEANS")
print("=" * 65)

# ==========================================
# 1. LOAD CUSTOMER DATA
# ==========================================

df = pd.read_csv(INPUT_FILE)

print("\nCustomer dataset loaded successfully!")
print("Total customers:", len(df))

# Features used for clustering
features = [
    "total_orders",
    "total_spending",
    "average_order_spending",
    "total_items"
]

numeric_data = df[features].apply(
    pd.to_numeric, errors="coerce"
)

valid_mask = (
    numeric_data.notna().all(axis=1) &
    (numeric_data >= 0).all(axis=1)
)

customer_data = df.loc[valid_mask].copy()
X = numeric_data.loc[valid_mask].copy()

print("Valid customer records:", len(X))

if len(X) < 7:
    raise ValueError("Not enough valid customers for clustering.")

# ==========================================
# 2. DATA TRANSFORMATION
# ==========================================

# Reduce the influence of highly skewed spending values
X_log = np.log1p(X)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

print("\nData transformation completed!")

# ==========================================
# 3. FIND SUITABLE NUMBER OF CLUSTERS
# ==========================================

inertias = []
silhouette_scores = []
k_values = range(2, min(7, len(X)))

for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    inertias.append(model.inertia_)

    score = silhouette_score(
        X_scaled,
        labels,
        sample_size=min(10000, len(X_scaled)),
        random_state=42
    )

    silhouette_scores.append(score)

    print(
        f"K = {k} | "
        f"Inertia = {model.inertia_:.2f} | "
        f"Silhouette = {score:.4f}"
    )

# Select the K with the highest silhouette score
best_index = int(np.argmax(silhouette_scores))
best_k = list(k_values)[best_index]

print("\nSelected number of clusters:", best_k)

# ==========================================
# 4. ELBOW METHOD GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    inertias,
    marker="o"
)

plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "elbow_method.png"
)

plt.close()

# ==========================================
# 5. SILHOUETTE SCORE GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    silhouette_scores,
    marker="o"
)

plt.title("Silhouette Score Comparison")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "silhouette_scores.png"
)

plt.close()

# ==========================================
# 6. FINAL K-MEANS MODEL
# ==========================================

final_model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

customer_data["cluster"] = (
    final_model.fit_predict(X_scaled)
)

# Save individual customer classifications
customer_data.to_csv(
    OUTPUT_DIR / "customer_segments.csv",
    index=False
)

print("\nK-Means clustering completed!")

# ==========================================
# 7. CUSTOMER CLUSTER PROFILES
# ==========================================

cluster_profiles = customer_data.groupby(
    "cluster"
).agg(
    customer_count=("customer_unique_id", "count"),
    average_orders=("total_orders", "mean"),
    average_spending=("total_spending", "mean"),
    average_order_value=("average_order_spending", "mean"),
    average_items=("total_items", "mean")
).reset_index()

cluster_profiles.to_csv(
    OUTPUT_DIR / "cluster_profiles.csv",
    index=False
)

print("\nCUSTOMER CLUSTER PROFILES")
print(cluster_profiles.round(2).to_string(index=False))

# ==========================================
# 8. PCA VISUALIZATION
# ==========================================

pca = PCA(n_components=2)

pca_data = pca.fit_transform(X_scaled)

plot_data = pd.DataFrame({
    "PC1": pca_data[:, 0],
    "PC2": pca_data[:, 1],
    "Cluster": customer_data["cluster"].values
})

# Sample for a clearer and faster graph
plot_data = plot_data.sample(
    n=min(10000, len(plot_data)),
    random_state=42
)

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=plot_data,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="viridis",
    alpha=0.6
)

plt.title("Customer Segmentation Using K-Means")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "customer_clusters.png"
)

plt.close()

# ==========================================
# 9. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 65)
print("CUSTOMER SEGMENTATION COMPLETED SUCCESSFULLY!")
print("=" * 65)

print("Total customers segmented:", len(customer_data))
print("Selected clusters:", best_k)

print("\nResults saved in:")
print(OUTPUT_DIR)

print("\nGraphs saved in:")
print(PLOTS_DIR)
