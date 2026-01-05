import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def clustering_metrics(Z: np.ndarray, labels: np.ndarray):
    k = len(set(labels))
    if k < 2:
        return {"silhouette": None, "calinski_harabasz": None}

    sil = silhouette_score(Z, labels)
    ch = calinski_harabasz_score(Z, labels)
    return {"silhouette": float(sil), "calinski_harabasz": float(ch)}
