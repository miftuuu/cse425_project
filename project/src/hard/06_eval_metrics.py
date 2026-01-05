import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

DATA = "data/processed/hard_multimodal.csv"
CLUSTERS = "results/hard/cluster_labels_all.csv"
OUT = "results/hard/hard_metrics.csv"

def purity_score(y_true, y_pred):
    
    df = pd.DataFrame({"y": y_true, "c": y_pred})
    total = 0
    for c, g in df.groupby("c"):
        total += g["y"].value_counts().iloc[0]
    return total / len(df)

def main():
    os.makedirs("results/hard", exist_ok=True)

    df = pd.read_csv(DATA)
    cl = pd.read_csv(CLUSTERS).merge(df[["song_id"]], on="song_id", how="inner")

    
    drop = {"song_id", "artist_name", "title", "genre", "genre_clean", "language", "text"}
    feat_cols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    X = StandardScaler().fit_transform(df[feat_cols].astype(float).values)

    y_true = df["genre_clean"].astype(str).values  
    methods = ["cluster_direct", "cluster_pca", "cluster_ae", "cluster_cvae"]
    rows = []
    for m in methods:
        y_pred = pd.read_csv(CLUSTERS)[m].values
        rows.append({
            "method": m.replace("cluster_", ""),
            "silhouette": float(silhouette_score(X, y_pred)),
            "ARI": float(adjusted_rand_score(y_true, y_pred)),
            "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
            "purity": float(purity_score(y_true, y_pred)),
        })

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("Saved:", OUT)

if __name__ == "__main__":
    main()
