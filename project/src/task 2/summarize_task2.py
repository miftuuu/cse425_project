import pandas as pd

INP = "data/processed/msd_clustered.csv"
OUT = "results/task2_cluster_summary.csv"

df = pd.read_csv(INP)

summary = df.groupby("cluster")[["tempo", "loudness", "duration"]].agg(["count", "mean", "std", "min", "max"])
summary.to_csv(OUT)

print("Saved:", OUT)
