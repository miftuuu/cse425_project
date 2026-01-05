import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def kmeans_cluster(Z: np.ndarray, k: int, seed: int = 42):
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(Z)
    return labels

def pca_features(X: np.ndarray, latent_dim: int = 16, seed: int = 42):
    pca = PCA(n_components=latent_dim, random_state=seed)
    Zp = pca.fit_transform(X)
    return Zp
