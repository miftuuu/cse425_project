import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

IN_CLUSTERED = "data/processed/msd_clustered.csv"   # your clustered file
IN_FEATURES  = "data/processed/msd_features.csv"    # raw features file
OUT_METRICS  = "results/task2/metrics.csv"
OUT_TSNE     = "results/task2/tsne_clusters.png"
OUT_PCA      = "results/task2/pca_clusters.png"

FEATURE_COLS = ["tempo", "loudness", "duration"]
CLUSTER_COL  = "cluster"


def ensure_dirs():
    os.makedirs("results/task2", exist_ok=True)


def load_X(df):
    return df[FEATURE_COLS].astype(float).values


def save_metrics_row(rows, name, X, labels):
    sil = silhouette_score(X, labels)
    ch  = calinski_harabasz_score(X, labels)
    rows.append({"method": name, "silhouette": sil, "calinski_harabasz": ch})


def main():
    ensure_dirs()
    rows = []

    # 1) Evaluate your existing clustering (msd_clustered.csv)
    dfc = pd.read_csv(IN_CLUSTERED)
    Xc = load_X(dfc)
    Xc_scaled = StandardScaler().fit_transform(Xc)
    labels_existing = dfc[CLUSTER_COL].values
    save_metrics_row(rows, "your_existing_clusters", Xc_scaled, labels_existing)

    # 2) Baseline: PCA + KMeans (same K as your file uses)
    dff = pd.read_csv(IN_FEATURES)
    X = load_X(dff)
    X_scaled = StandardScaler().fit_transform(X)

    K = int(pd.Series(labels_existing).nunique())
    pca = PCA(n_components=2, random_state=42)
    X_pca2 = pca.fit_transform(X_scaled)

    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels_pca_km = km.fit_predict(X_pca2)

    save_metrics_row(rows, "baseline_PCA2_plus_KMeans", X_pca2, labels_pca_km)

    # Save metrics
    pd.DataFrame(rows).to_csv(OUT_METRICS, index=False)
    print("Saved metrics to:", OUT_METRICS)

    # 3) t-SNE visualization using your existing clusters
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
    X_tsne = tsne.fit_transform(Xc_scaled)

    plt.figure()
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_existing, s=5)
    plt.title("t-SNE of MSD features (colored by cluster)")
    plt.savefig(OUT_TSNE, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved t-SNE plot to:", OUT_TSNE)

    # 4) PCA visualization (2D) using your existing clusters
    pca2 = PCA(n_components=2, random_state=42)
    X_pca_vis = pca2.fit_transform(Xc_scaled)

    plt.figure()
    plt.scatter(X_pca_vis[:, 0], X_pca_vis[:, 1], c=labels_existing, s=5)
    plt.title("PCA(2) of MSD features (colored by cluster)")
    plt.savefig(OUT_PCA, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved PCA plot to:", OUT_PCA)


if __name__ == "__main__":
    main()
