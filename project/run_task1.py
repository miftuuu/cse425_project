import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from src.dataset import load_dataframe, make_tfidf
from src.vae import VAE, vae_loss
from src.clustering import kmeans_cluster, pca_features
from src.evaluation import clustering_metrics


def main():
    base = Path(__file__).resolve().parent
    xlsx_path = base / "data" / "dataset.xlsx"

    results_dir = base / "results"
    viz_dir = results_dir / "latent_visualization"
    results_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / "clustering_metrics_task1.csv"

    df = load_dataframe(str(xlsx_path))
    X, _ = make_tfidf(df, max_features=2000)

    input_dim = X.shape[1]
    latent_dim = 16
    vae = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=256)

    optimizer = tf.keras.optimizers.Adam(1e-3)
    X_tf = tf.convert_to_tensor(X)

    epochs = 80
    batch_size = 16
    n = X.shape[0]

    for epoch in range(epochs):
        idx = np.random.permutation(n)
        X_shuf = X_tf.numpy()[idx]

        losses = []
        for i in range(0, n, batch_size):
            xb = tf.convert_to_tensor(X_shuf[i:i + batch_size])

            with tf.GradientTape() as tape:
                x_hat, z_mean, z_logvar = vae(xb, training=True)
                total, recon, kl = vae_loss(xb, x_hat, z_mean, z_logvar)

            grads = tape.gradient(total, vae.trainable_variables)
            optimizer.apply_gradients(zip(grads, vae.trainable_variables))
            losses.append([float(total), float(recon), float(kl)])

        if (epoch + 1) % 10 == 0:
            avg = np.mean(losses, axis=0)
            print(f"Epoch {epoch+1:03d} | total={avg[0]:.4f} recon={avg[1]:.4f} kl={avg[2]:.4f}")

    _, Z_mean, _ = vae(X_tf, training=False)
    Z = Z_mean.numpy()
    Zp = pca_features(X, latent_dim=latent_dim)

    np.save(results_dir / "Z_vae.npy", Z)
    np.save(results_dir / "Z_pca.npy", Zp)

    rows = []
    best = {
        "VAE+KMeans": {"k": None, "sil": -1e9},
        "PCA+KMeans": {"k": None, "sil": -1e9},
    }

    for k in range(2, 11):
        labels_vae = kmeans_cluster(Z, k)
        m_vae = clustering_metrics(Z, labels_vae)
        rows.append({"method": "VAE+KMeans", "k": k, **m_vae})
        if m_vae["silhouette"] is not None and m_vae["silhouette"] > best["VAE+KMeans"]["sil"]:
            best["VAE+KMeans"] = {"k": k, "sil": m_vae["silhouette"]}

        labels_pca = kmeans_cluster(Zp, k)
        m_pca = clustering_metrics(Zp, labels_pca)
        rows.append({"method": "PCA+KMeans", "k": k, **m_pca})
        if m_pca["silhouette"] is not None and m_pca["silhouette"] > best["PCA+KMeans"]["sil"]:
            best["PCA+KMeans"] = {"k": k, "sil": m_pca["silhouette"]}

    out = pd.DataFrame(rows).sort_values(["method", "k"])
    out.to_csv(metrics_path, index=False)

    k_vae = best["VAE+KMeans"]["k"]
    k_pca = best["PCA+KMeans"]["k"]

    np.save(results_dir / f"labels_vae_k{k_vae}.npy", kmeans_cluster(Z, k_vae))
    np.save(results_dir / f"labels_pca_k{k_pca}.npy", kmeans_cluster(Zp, k_pca))

    best_txt = results_dir / "best_k.txt"
    best_txt.write_text(
        f"Best VAE+KMeans k={k_vae} silhouette={best['VAE+KMeans']['sil']}\n"
        f"Best PCA+KMeans k={k_pca} silhouette={best['PCA+KMeans']['sil']}\n"
    )

    print("\nSaved metrics to:", metrics_path)
    print("Saved:", results_dir / "Z_vae.npy")
    print("Saved:", results_dir / "Z_pca.npy")
    print("Saved:", results_dir / f"labels_vae_k{k_vae}.npy")
    print("Saved:", results_dir / f"labels_pca_k{k_pca}.npy")
    print("Saved:", best_txt)
    print(out)


if __name__ == "__main__":
    main()
