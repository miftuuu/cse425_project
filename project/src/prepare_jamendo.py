from pathlib import Path
import pandas as pd
from datasets import load_dataset

def main():
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "jamendo.csv"

    ds = load_dataset("jamendolyrics/jamendolyrics", split="test", streaming=True)

    # IMPORTANT: remove audio column so torchcodec is never needed
    if "audio" in ds.column_names:
        ds = ds.remove_columns(["audio"])

    rows = []
    for i, ex in enumerate(ds):
        # Try common field names (dataset versions vary)
        lyrics = ex.get("lyrics") or ex.get("text") or ""
        rows.append({
            "id": ex.get("id", i),
            "title": ex.get("title") or ex.get("track") or "",
            "artist": ex.get("artist") or "",
            "language": ex.get("language") or "",
            "lyrics": lyrics,
        })

        # keep it small at first so you can confirm it works
        if i >= 999:
            break

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print("Saved:", out_csv)

if __name__ == "__main__":
    main()
