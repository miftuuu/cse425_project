import os
import glob
import numpy as np
import pandas as pd
import h5py


MSD_ROOT = "data/msd_subset/A"

OUT_CSV = "data/processed/hard_msd_raw.csv"

def safe_decode(x):
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", "ignore")
    return str(x)

def get_artist_terms(f):
    if "metadata" in f and "artist_terms" in f["metadata"]:
        terms = f["metadata/artist_terms"][:]
        terms = [safe_decode(t) for t in terms]
        return terms
    return []

def get_segments_mean(f, path, dim_expected):
    if path in f:
        arr = f[path][:]
        if arr.ndim == 2 and arr.shape[1] == dim_expected and arr.shape[0] > 0:
            return arr.mean(axis=0).astype(float)
    return np.zeros((dim_expected,), dtype=float)

def infer_language_from_text(text):
    for ch in text:
        if "\u0980" <= ch <= "\u09FF":
            return "bn"
    return "en"

def read_one_h5(h5_path, print_fields_once=False):
    with h5py.File(h5_path, "r") as f:
        a = f["analysis/songs"][0] if "analysis" in f and "songs" in f["analysis"] else None
        m = f["metadata/songs"][0] if "metadata" in f and "songs" in f["metadata"] else None

        if print_fields_once:
            if a is not None:
                print("analysis/songs fields:", list(a.dtype.names))
            if m is not None:
                print("metadata/songs fields:", list(m.dtype.names))
            print("top-level keys:", list(f.keys()))

        song_id = safe_decode(m["song_id"]) if m is not None and "song_id" in m.dtype.names else os.path.basename(h5_path)
        artist_name = safe_decode(m["artist_name"]) if m is not None and "artist_name" in m.dtype.names else ""
        title = safe_decode(m["title"]) if m is not None and "title" in m.dtype.names else ""

        tempo = float(a["tempo"]) if a is not None and "tempo" in a.dtype.names else np.nan
        loudness = float(a["loudness"]) if a is not None and "loudness" in a.dtype.names else np.nan
        duration = float(a["duration"]) if a is not None and "duration" in a.dtype.names else np.nan

        timbre_mean = get_segments_mean(f, "analysis/segments_timbre", 12)
        pitches_mean = get_segments_mean(f, "analysis/segments_pitches", 12)

        terms = get_artist_terms(f)
        top_term = terms[0] if len(terms) > 0 else "unknown"
        terms_joined = " ".join(terms[:10]) if len(terms) > 0 else "unknown"

        text = f"{title} {artist_name} {terms_joined}".strip()
        language = infer_language_from_text(text)

        row = {
            "song_id": song_id,
            "artist_name": artist_name,
            "title": title,
            "tempo": tempo,
            "loudness": loudness,
            "duration": duration,
            "genre": top_term,
            "language": language,
            "text": text,
        }

        for i in range(12):
            row[f"timbre_mean_{i+1}"] = float(timbre_mean[i])
            row[f"pitches_mean_{i+1}"] = float(pitches_mean[i])

        return row

def main():
    os.makedirs("data/processed", exist_ok=True)
    pattern = os.path.join(MSD_ROOT, "**", "*.h5")
    files = glob.glob(pattern, recursive=True)
    if len(files) == 0:
        raise FileNotFoundError(f"No .h5 found under: {MSD_ROOT}")

    rows = []
    for idx, fp in enumerate(files):
        rows.append(read_one_h5(fp, print_fields_once=(idx == 0)))

    df = pd.DataFrame(rows)

    df = df.dropna(subset=["tempo", "loudness", "duration"]).reset_index(drop=True)

    df.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV)
    print("Rows:", len(df))

if __name__ == "__main__":
    main()
