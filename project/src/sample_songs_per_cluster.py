import pandas as pd

INP = "data/processed/msd_clustered.csv"
OUT = "results/task2_cluster_examples.csv"

df = pd.read_csv(INP)

examples = (
    df.sort_values(["cluster", "duration"])
      .groupby("cluster")
      .head(10)[["cluster", "artist_name", "title", "tempo", "loudness", "duration"]]
)

examples.to_csv(OUT, index=False)
print("Saved:", OUT)
