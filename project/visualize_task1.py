import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE


def pick_label_file(results_dir: Path, prefix: str) -> Path:
    files = sorted(results_dir.glob(f"{prefix}_k*.npy"))
    if not files:
        raise FileNotFoundError(f"No label files found like {prefix}_k*.npy in {results_dir}")
    return files[-1]


def tsne_plot(Z: np.ndarray, labels: np.ndarray, outpath: Path, title: str):
    n = Z.shape[0]
    perplexity = min(30, max(5, (n - 1) // 3))

    emb = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42
    ).fit_transform(Z)

    plt.figure()
    plt.scatter(emb[:, 0], emb[:, 1], c=labels, s=30)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    base = Path(__file__).resolve().parent
    results_dir = base / "results"
    viz_dir = results_dir / "latent_visualization"
    viz_dir.mkdir(parents=True, exist_ok=True)

    Z_vae = np.load(results_dir / "Z_vae.npy")
    Z_pca = np.load(results_dir / "Z_pca.npy")

    labels_vae_path = pick_label_file(results_dir, "labels_vae")
    labels_pca_path = pick_label_file(results_dir, "labels_pca")

    labels_vae = np.load(labels_vae_path)
    labels_pca = np.load(labels_pca_path)

    tsne_plot(
        Z_vae, labels_vae,
        viz_dir / "tsne_vae.png",
        f"t-SNE (VAE latent) - {labels_vae_path.name}"
    )

    tsne_plot(
        Z_pca, labels_pca,
        viz_dir / "tsne_pca.png",
        f"t-SNE (PCA features) - {labels_pca_path.name}"
    )

    print("Saved plots to:", viz_dir)


if __name__ == "__main__":
    main()
