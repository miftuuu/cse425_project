import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

LAT = "results/hard/cvae_latents.csv"
CL  = "results/hard/cluster_labels_all.csv"
RECON = "results/hard/cvae_recon_examples.csv"


META = "data/processed/hard_multimodal.csv"

OUT_DIR = "results/hard/plots"

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def scatter_tsne(Z, labels, title, outpath):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
    P = tsne.fit_transform(Z)
    plt.figure()
    plt.scatter(P[:, 0], P[:, 1], c=labels, s=6)
    plt.title(title)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    lat = pd.read_csv(LAT)
    cl = pd.read_csv(CL)

    if ("genre_clean" not in lat.columns) or ("language" not in lat.columns):
        meta = pd.read_csv(META)
        gcol_m = pick_col(meta, ["genre_clean", "genre"])
        lcol_m = pick_col(meta, ["language"])
        if gcol_m is None or lcol_m is None:
            raise ValueError(f"META file missing genre/language columns. META cols: {list(meta.columns)}")

        meta = meta[["song_id", gcol_m, lcol_m]].copy()
        meta = meta.rename(columns={gcol_m: "genre_clean", lcol_m: "language"})
        lat = lat.merge(meta, on="song_id", how="left")

    zcols = [c for c in lat.columns if c.startswith("z")]
    Z = StandardScaler().fit_transform(lat[zcols].astype(float).values)

    labels_cluster = cl["cluster_cvae"].values
    scatter_tsne(Z, labels_cluster,
                 "t-SNE of CVAE latents (colored by cluster)",
                 f"{OUT_DIR}/tsne_latent_by_cluster.png")

    genre_col = pick_col(lat, ["genre_clean", "genre_clean_x", "genre_clean_y"])
    if genre_col is None:
        raise ValueError(f"No genre column found in lat. lat cols: {list(lat.columns)}")

    genre = lat[genre_col].astype(str).fillna("UNKNOWN").values
    uniq = sorted(set(genre))
    g2i = {g: i for i, g in enumerate(uniq)}
    genre_ids = np.array([g2i[g] for g in genre])
    scatter_tsne(Z, genre_ids,
                 "t-SNE of CVAE latents (colored by genre)",
                 f"{OUT_DIR}/tsne_latent_by_genre.png")

    tmp = cl.copy()

    if "genre_clean" not in tmp.columns:
        tmp = tmp.merge(lat[["song_id", "genre_clean"]], on="song_id", how="left", suffixes=("", "_lat"))
    if "language" not in tmp.columns:
        tmp = tmp.merge(lat[["song_id", "language"]], on="song_id", how="left", suffixes=("", "_lat"))

    gcol = pick_col(tmp, ["genre_clean", "genre_clean_lat", "genre_clean_x", "genre_clean_y"])
    lcol = pick_col(tmp, ["language", "language_lat", "language_x", "language_y"])
    if gcol is None or lcol is None:
        raise ValueError(f"Could not find genre/language in tmp. tmp cols: {list(tmp.columns)}")

    pivot = pd.crosstab(tmp["cluster_cvae"], tmp[gcol])
    ax = pivot.plot(kind="bar", stacked=True, figsize=(12, 5), legend=False)
    ax.set_title("Cluster distribution over genres (CVAE clusters)")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/cluster_vs_genre.png", dpi=200)
    plt.close(fig)

    pivot2 = pd.crosstab(tmp["cluster_cvae"], tmp[lcol])
    ax2 = pivot2.plot(kind="bar", stacked=True, figsize=(10, 5))
    ax2.set_title("Cluster distribution over language (CVAE clusters)")
    fig2 = ax2.get_figure()
    fig2.tight_layout()
    fig2.savefig(f"{OUT_DIR}/cluster_vs_language.png", dpi=200)
    plt.close(fig2)

    if os.path.exists(RECON):
        r = pd.read_csv(RECON)
        orig_cols = [c for c in r.columns if c.startswith("orig_")]
        recon_cols = [c for c in r.columns if c.startswith("recon_")]

        if len(orig_cols) > 0 and len(recon_cols) > 0:
            n = min(10, len(r))
            d = min(20, len(orig_cols), len(recon_cols))
            Xo = r[orig_cols].values[:n, :d]
            Xr = r[recon_cols].values[:n, :d]

            plt.figure(figsize=(10, 4))
            plt.imshow(np.abs(Xo - Xr), aspect="auto")
            plt.title("Reconstruction error (abs(orig - recon)) - first 10 samples")
            plt.xlabel("feature index")
            plt.ylabel("sample index")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f"{OUT_DIR}/recon_error_heatmap.png", dpi=200)
            plt.close()

    print("Saved plots to:", OUT_DIR)

if __name__ == "__main__":
    main()
