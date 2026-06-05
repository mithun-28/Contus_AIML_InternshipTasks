import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("customer_segmentation_processed.csv")

print("=" * 60)
print("Dataset Shape:", df.shape)
print("=" * 60)

X = df.copy()

# =====================================================
# SCALING
# =====================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =====================================================
# PCA
# Retain 95% variance
# =====================================================

pca = PCA(n_components=0.95, random_state=42)

X_pca = pca.fit_transform(X_scaled)

print(f"\nOriginal Features : {X.shape[1]}")
print(f"PCA Features      : {X_pca.shape[1]}")
print(
    f"Explained Variance: "
    f"{pca.explained_variance_ratio_.sum():.4f}"
)

# =====================================================
# FIND BEST K FOR KMEANS
# =====================================================

print("\n" + "=" * 60)
print("SEARCHING BEST K")
print("=" * 60)

best_k = 2
best_score = -1

for k in range(2, 11):

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = km.fit_predict(X_pca)

    score = silhouette_score(X_pca, labels)

    print(
        f"K={k} | Silhouette={score:.4f}"
    )

    if score > best_score:
        best_score = score
        best_k = k

print("\nBest K =", best_k)
print("Best Silhouette =", round(best_score, 4))

# =====================================================
# FINAL KMEANS
# =====================================================

print("\n" + "=" * 60)
print("FINAL KMEANS")
print("=" * 60)

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_pca)

print(
    "Silhouette Score:",
    silhouette_score(X_pca, kmeans_labels)
)

print(
    "Davies Bouldin:",
    davies_bouldin_score(X_pca, kmeans_labels)
)

print(
    "Calinski Harabasz:",
    calinski_harabasz_score(X_pca, kmeans_labels)
)

# =====================================================
# GAUSSIAN MIXTURE MODEL
# =====================================================

print("\n" + "=" * 60)
print("GAUSSIAN MIXTURE")
print("=" * 60)

gmm = GaussianMixture(
    n_components=best_k,
    random_state=42
)

gmm_labels = gmm.fit_predict(X_pca)

print(
    "Silhouette Score:",
    silhouette_score(X_pca, gmm_labels)
)

print(
    "Davies Bouldin:",
    davies_bouldin_score(X_pca, gmm_labels)
)

print(
    "Calinski Harabasz:",
    calinski_harabasz_score(X_pca, gmm_labels)
)

# =====================================================
# SAVE CLUSTERS
# =====================================================

df['KMeans_Cluster'] = kmeans_labels
df['GMM_Cluster'] = gmm_labels

df.to_csv(
    "customer_segmented_output.csv",
    index=False
)

print("\nClustered Dataset Saved")

# =====================================================
# SAVE MODELS
# =====================================================

joblib.dump(
    scaler,
    "scaler.joblib"
)

joblib.dump(
    pca,
    "pca.joblib"
)

joblib.dump(
    kmeans,
    "kmeans_model.joblib"
)

joblib.dump(
    gmm,
    "gmm_model.joblib"
)

print("\nModels Saved Successfully")

print("\nSaved Files:")
print("- scaler.joblib")
print("- pca.joblib")
print("- kmeans_model.joblib")
print("- gmm_model.joblib")
print("- customer_segmented_output.csv")