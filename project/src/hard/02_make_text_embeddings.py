import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

IN_CSV = "data/processed/hard_msd_raw.csv"
OUT_CSV = "data/processed/hard_text_emb.csv"

N_SVD = 50

def main():
    os.makedirs("data/processed", exist_ok=True)
    df = pd.read_csv(IN_CSV)
    texts = df["text"].fillna("").astype(str).tolist()

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=2)
    X = tfidf.fit_transform(texts)

    svd = TruncatedSVD(n_components=N_SVD, random_state=42)
    Z = svd.fit_transform(X)

    out = pd.DataFrame(Z, columns=[f"text_{i+1}" for i in range(N_SVD)])
    out.insert(0, "song_id", df["song_id"].values)
    out.to_csv(OUT_CSV, index=False)

    print("Saved:", OUT_CSV)

if __name__ == "__main__":
    main()
