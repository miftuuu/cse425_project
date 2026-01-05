import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATA = "data/processed/hard_multimodal.csv"
LATENTS = "results/hard/cvae_latents.csv"

OUT_CLUSTERS = "results/hard/cluster_labels_all.csv"

K = 10  

class AE(nn.Module):
    def __init__(self, in_dim, z_dim=8):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, z_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(z_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, in_dim),
        )
    def forward(self, x):
        z = self.enc(x)
        xhat = self.dec(z)
        return xhat, z

def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def kmeans_labels(X, k):
    return KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)

def main():
    os.makedirs("results/hard", exist_ok=True)
    df = pd.read_csv(DATA)

    drop = {"song_id", "artist_name", "title", "genre", "genre_clean", "language", "text"}
    feat_cols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feat_cols].astype(float).values
    Xs = StandardScaler().fit_transform(X)

    audio_cols = ["tempo", "loudness", "duration"] + \
        [f"timbre_mean_{i}" for i in range(1,13)] + \
        [f"pitches_mean_{i}" for i in range(1,13)]
    audio_cols = [c for c in audio_cols if c in df.columns]
    Xa = StandardScaler().fit_transform(df[audio_cols].astype(float).values)
    labels_direct = kmeans_labels(Xa, K)

    X_pca = PCA(n_components=10, random_state=42).fit_transform(Xs)
    labels_pca = kmeans_labels(X_pca, K)

    device = pick_device()
    ae = AE(in_dim=Xs.shape[1], z_dim=8).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xs.astype(np.float32))),
                        batch_size=256, shuffle=True)

    ae.train()
    for ep in range(1, 21):
        tot = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            xhat, z = ae(xb)
            loss = ((xb - xhat) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"AE epoch {ep:02d} loss={tot/len(loader):.4f}")

    ae.eval()
    with torch.no_grad():
        Z_ae = ae.enc(torch.from_numpy(Xs.astype(np.float32)).to(device)).cpu().numpy()
    labels_ae = kmeans_labels(StandardScaler().fit_transform(Z_ae), K)

    lat = pd.read_csv(LATENTS)
    zcols = [c for c in lat.columns if c.startswith("z")]
    Zc = StandardScaler().fit_transform(lat[zcols].astype(float).values)
    labels_cvae = kmeans_labels(Zc, K)

    out = pd.DataFrame({
        "song_id": df["song_id"].values,
        "genre_clean": df["genre_clean"].values,
        "language": df["language"].values,
        "cluster_direct": labels_direct,
        "cluster_pca": labels_pca,
        "cluster_ae": labels_ae,
        "cluster_cvae": labels_cvae,
    })
    out.to_csv(OUT_CLUSTERS, index=False)
    print("Saved:", OUT_CLUSTERS)

if __name__ == "__main__":
    main()
