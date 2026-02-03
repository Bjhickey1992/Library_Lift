"""
Remove from exhibition data and embeddings any rows where location is 'IMDB Trending (MOVIE)'.
Close upcoming_exhibitions.xlsx in Excel before running if you get Permission denied.
"""
import shutil
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

EXHIBITIONS_PATH = "upcoming_exhibitions.xlsx"
EMBEDDINGS_NPY = "upcoming_exhibitions_embeddings.npy"
EMBEDDINGS_XLSX = "upcoming_exhibitions_embeddings.xlsx"
LOCATION_TO_REMOVE = "IMDB Trending (MOVIE)"


def main():
    path = Path(EXHIBITIONS_PATH)
    if not path.exists():
        print(f"Not found: {path}")
        return 1
    df = pd.read_excel(path)
    if "location" not in df.columns:
        print("No 'location' column in exhibition file.")
        return 1
    before = len(df)
    mask = df["location"].astype(str).str.strip() != LOCATION_TO_REMOVE
    df_filtered = df[mask].copy()
    removed = before - len(df_filtered)
    if removed == 0:
        print(f"No rows with location '{LOCATION_TO_REMOVE}' found. Nothing to remove.")
        return 0
    try:
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        df_filtered.to_excel(tmp_path, index=False)
        shutil.move(str(tmp_path), path)
    except PermissionError:
        out_alt = Path("upcoming_exhibitions_no_imdb_trending.xlsx")
        df_filtered.to_excel(out_alt, index=False)
        print(f"Could not overwrite {EXHIBITIONS_PATH} (file may be open). Wrote {out_alt} instead.")
        print("Close the Excel file, then replace it with upcoming_exhibitions_no_imdb_trending.xlsx.")
    else:
        print(f"Removed {removed} rows from {EXHIBITIONS_PATH} (kept {len(df_filtered)}).")

    keep_indices = np.where(mask.values)[0]
    npy_path = Path(EMBEDDINGS_NPY)
    if npy_path.exists():
        emb = np.load(npy_path)
        if len(emb) != before:
            print(f"Warning: embeddings rows ({len(emb)}) != exhibition rows ({before}). Aligning by index.")
        emb_filtered = emb[keep_indices]
        np.save(npy_path, emb_filtered)
        print(f"Updated {EMBEDDINGS_NPY}: {len(emb)} -> {len(emb_filtered)} rows.")
    else:
        print(f"{EMBEDDINGS_NPY} not found; skipping.")

    xlsx_path = Path(EMBEDDINGS_XLSX)
    if xlsx_path.exists():
        meta = pd.read_excel(xlsx_path)
        if len(meta) == before:
            meta_filtered = meta.iloc[keep_indices].reset_index(drop=True)
            meta_filtered.to_excel(xlsx_path, index=False)
            print(f"Updated {EMBEDDINGS_XLSX}: {len(meta)} -> {len(meta_filtered)} rows.")
        else:
            print(f"Metadata rows ({len(meta)}) != exhibition rows ({before}); skipping metadata update.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
