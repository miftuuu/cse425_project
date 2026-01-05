import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/msd_clustered.csv")

plt.figure()
plt.scatter(df["tempo"], df["loudness"], c=df["cluster"])
plt.xlabel("Tempo")
plt.ylabel("Loudness")
plt.title("MSD clusters (tempo vs loudness)")
plt.savefig("results/task2_tempo_loudness.png", dpi=200)
print("Saved: results/task2_tempo_loudness.png")
