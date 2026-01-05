import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

IN_CSV = "data/processed/hard_multimodal.csv"

OUT_LATENTS = "results/hard/cvae_latents.csv"
OUT_RECON = "results/hard/cvae_recon_examples.csv"
OUT_LOSS = "results/hard/cvae_train_log.csv"

LATENT_DIM = 8
BATCH = 256
EPOCHS = 30
LR = 1e-3
BETA = 2.0  

ID_COLS = ["song_id", "artist_name", "title", "genre_clean", "language"]

def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

class CVAE(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(x_dim + y_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.mu = nn.Linear(128, z_dim)
        self.logvar = nn.Linear(128, z_dim)

        self.dec = nn.Sequential(
            nn.Linear(z_dim + y_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, x_dim),
        )

    def encode(self, x, y):
        h = self.enc(torch.cat([x, y], dim=1))
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        return self.dec(torch.cat([z, y], dim=1))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparam(mu, logvar)
        xhat = self.decode(z, y)
        return xhat, mu, logvar

def loss_fn(x, xhat, mu, logvar):
    recon = ((x - xhat) ** 2).mean()
    kl = -0.5 * torch.mean(1 + logvar - mu**2 - torch.exp(logvar))
    return recon + BETA * kl, recon.item(), kl.item()

def main():
    os.makedirs("results/hard", exist_ok=True)

    df = pd.read_csv(IN_CSV)

    
    drop = set(ID_COLS + ["genre", "text", "language"])
    feat_cols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]

    
    genres = sorted(df["genre_clean"].astype(str).unique().tolist())
    g2i = {g:i for i,g in enumerate(genres)}
    y = np.zeros((len(df), len(genres)), dtype=np.float32)
    for r, g in enumerate(df["genre_clean"].astype(str).tolist()):
        y[r, g2i[g]] = 1.0

    X = df[feat_cols].astype(float).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)

    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        Xs, y, np.arange(len(df)), test_size=0.15, random_state=42, stratify=df["genre_clean"]
    )

    device = pick_device()
    model = CVAE(x_dim=Xs.shape[1], y_dim=y.shape[1], z_dim=LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=BATCH, shuffle=True)

    log_rows = []
    model.train()
    for ep in range(1, EPOCHS + 1):
        total, rc, klc = 0.0, 0.0, 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            xhat, mu, logvar = model(xb, yb)
            loss, recon, kl = loss_fn(xb, xhat, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            rc += recon
            klc += kl
        log_rows.append({"epoch": ep, "loss": total/len(train_loader), "recon": rc/len(train_loader), "kl": klc/len(train_loader)})
        print(f"epoch {ep:02d} loss={total/len(train_loader):.4f} recon={rc/len(train_loader):.4f} kl={klc/len(train_loader):.4f}")

    pd.DataFrame(log_rows).to_csv(OUT_LOSS, index=False)

    
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(Xs).to(device)
        Yt = torch.from_numpy(y).to(device)
        mu, logvar = model.encode(Xt, Yt)
        Z = mu.cpu().numpy()

      
        z = model.reparam(mu, logvar)
        xhat = model.decode(z, Yt).cpu().numpy()

    lat = pd.DataFrame(Z, columns=[f"z{i+1}" for i in range(LATENT_DIM)])
    lat.insert(0, "song_id", df["song_id"].values)
    lat["genre_clean"] = df["genre_clean"].values
    lat["language"] = df["language"].values
    lat.to_csv(OUT_LATENTS, index=False)

    
    n = min(50, len(df))
    recon_df = pd.DataFrame({
        "song_id": df["song_id"].values[:n],
        "genre_clean": df["genre_clean"].values[:n],
        "language": df["language"].values[:n],
    })
    for i, c in enumerate(feat_cols):
        recon_df[f"orig_{c}"] = Xs[:n, i]
        recon_df[f"recon_{c}"] = xhat[:n, i]
    recon_df.to_csv(OUT_RECON, index=False)

    print("Saved latents:", OUT_LATENTS)
    print("Saved recon examples:", OUT_RECON)
    print("Saved train log:", OUT_LOSS)

if __name__ == "__main__":
    main()
