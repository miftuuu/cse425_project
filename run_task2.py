from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def main():
    # This file is in src/, so project root is one folder up
    base = Path(__file__).resolve().parents[1]

    in_path = base / "data" / "processed" / "msd_features.csv"
    out_path = base / "data" / "processed" / "msd_clustered.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(
            f"Can't find {in_path}. Run your MSD feature extraction first to create msd_features.csv."
        )

    df = pd.read_csv(in_path)

    # Drop obvious non-feature columns if they exist
    ignore = {
        "track_id", "song_id", "title", "artist_name", "artist", "release", "genre", "language"
    }

    # Use all numeric columns as features
    num_cols = []
    for c in df.columns:
        if c in ignore:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)

    if not num_cols:
        raise ValueError(
            "No numeric feature columns found in msd_features.csv. "
            "Open the CSV and check what columns it contains."
        )

    X = df[num_cols].fillna(0)
    X = StandardScaler().fit_transform(X)

    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X)

    df.to_csv(out_path, index=False)
    print(f"✅ Saved clustered results to: {out_path}")
    print(f"Rows: {len(df)} | Features used: {len(num_cols)}")
    print("First few features:", ", ".join(num_cols[:12]))


if __name__ == "__main__":
    main()
