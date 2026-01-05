import os, h5py, pandas as pd
from pathlib import Path

def read_h5(path):
    with h5py.File(path, "r") as f:
        a = f["analysis/songs"][0]
        m = f["metadata/songs"][0]

       
        print("analysis fields:", list(a.dtype.names))
        print("metadata fields:", list(m.dtype.names))

        def clean(x):
            return x.decode("utf-8", "ignore") if isinstance(x, (bytes, bytearray)) else str(x)

        return {
            "song_id": clean(m["song_id"]),
            "artist_name": clean(m["artist_name"]),
            "title": clean(m["title"]),
            "tempo": float(a["tempo"]),
            "loudness": float(a["loudness"]),
            "duration": float(a["duration"]),
        }



def main():
    base = Path("data/msd_subset")
    rows = []
    for root, _, files in os.walk(base):
        for f in files:
            if f.endswith(".h5"):
                try:
                    rows.append(read_h5(os.path.join(root, f)))
                except Exception as e:
                    print("skip", f, e)
    df = pd.DataFrame(rows)
    df.to_csv("data/processed/msd_features.csv", index=False)
    print("Saved:", len(df), "songs")

if __name__ == "__main__":
    main()
