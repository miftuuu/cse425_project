import os
import pandas as pd

RAW = "data/processed/hard_msd_raw.csv"
TXT = "data/processed/hard_text_emb.csv"
OUT = "data/processed/hard_multimodal.csv"

TOP_GENRES = 15  

def main():
    os.makedirs("data/processed", exist_ok=True)
    df = pd.read_csv(RAW)
    te = pd.read_csv(TXT)

    df = df.merge(te, on="song_id", how="inner")

     
    vc = df["genre"].fillna("unknown").value_counts()
    keep = set(vc.head(TOP_GENRES).index.tolist())

    def map_genre(g):
        if pd.isna(g):
            return "unknown"
        return g if g in keep else "other"

    df["genre_clean"] = df["genre"].apply(map_genre)

    df.to_csv(OUT, index=False)
    print("Saved:", OUT)
    print("Rows:", len(df))
    print("Genres:", df["genre_clean"].nunique())

if __name__ == "__main__":
    main()
