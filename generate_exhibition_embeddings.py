#!/usr/bin/env python
"""Generate embeddings for existing upcoming_exhibitions.xlsx file (includes 'need' if present)."""

import os
import pandas as pd
import numpy as np
from typing import List
from openai import OpenAI

from config import get_openai_api_key


def _row_to_embedding_text(row: pd.Series) -> str:
    """Build text for embedding from one exhibition row (includes need if present)."""
    parts = []
    if pd.notna(row.get("title")):
        parts.append(f"Title: {row['title']}")
    if pd.notna(row.get("release_year")):
        parts.append(f"Year: {int(row['release_year'])}")
    if pd.notna(row.get("director")):
        parts.append(f"Director: {row['director']}")
    if pd.notna(row.get("writers")):
        parts.append(f"Writers: {row['writers']}")
    if pd.notna(row.get("producers")):
        parts.append(f"Producers: {row['producers']}")
    if pd.notna(row.get("cinematographers")):
        parts.append(f"Cinematographer: {row['cinematographers']}")
    if pd.notna(row.get("production_designers")):
        parts.append(f"Production Designer: {row['production_designers']}")
    if pd.notna(row.get("cast")):
        parts.append(f"Cast: {row['cast']}")
    if pd.notna(row.get("lead_gender")):
        parts.append(f"Lead actor gender: {row['lead_gender']}")
    if pd.notna(row.get("genres")):
        parts.append(f"Genres: {row['genres']}")
    if pd.notna(row.get("keywords")):
        parts.append(f"Keywords: {row['keywords']}")
    if pd.notna(row.get("tagline")):
        parts.append(f"Tagline: {row['tagline']}")
    if pd.notna(row.get("overview")):
        parts.append(f"Plot: {row['overview']}")
    if pd.notna(row.get("thematic_descriptors")):
        parts.append(f"Themes: {row['thematic_descriptors']}")
    if pd.notna(row.get("stylistic_descriptors")):
        parts.append(f"Style: {row['stylistic_descriptors']}")
    if pd.notna(row.get("emotional_tone")):
        parts.append(f"Tone: {row['emotional_tone']}")
    if pd.notna(row.get("need")):
        parts.append(f"Viewer needs: {row['need']}")
    return "\n".join(parts) if parts else str(row.get("title", ""))


def generate_exhibition_embeddings(
    exhibitions_path: str = "upcoming_exhibitions.xlsx",
    npy_path: str = "upcoming_exhibitions_embeddings.npy",
    metadata_xlsx_path: str = "upcoming_exhibitions_embeddings.xlsx",
) -> np.ndarray:
    """
    Load exhibition file, build text (including need), call OpenAI embeddings, save .npy and metadata .xlsx.
    Returns the embeddings array.
    """
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") or get_openai_api_key()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    ex_df = pd.read_excel(exhibitions_path)
    if len(ex_df) == 0:
        print("[generate_exhibition_embeddings] No rows; skipping.")
        return np.array([])

    film_texts = [_row_to_embedding_text(row) for _, row in ex_df.iterrows()]
    embeddings: List[List[float]] = []
    batch_size = 100
    for i in range(0, len(film_texts), batch_size):
        batch = film_texts[i : i + batch_size]
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch,
                dimensions=1536,
            )
            embeddings.extend([item.embedding for item in response.data])
            print(f"  Generated embeddings batch {i // batch_size + 1}/{(len(film_texts) - 1) // batch_size + 1}...")
        except Exception as e:
            print(f"  Error creating embeddings for batch: {e}")
            embeddings.extend([[0.0] * 1536 for _ in batch])

    arr = np.array(embeddings)
    np.save(npy_path, arr)
    meta_cols = ["tmdb_id", "title", "release_year", "country", "location"]
    meta_df = ex_df[[c for c in meta_cols if c in ex_df.columns]].copy()
    meta_df.to_excel(metadata_xlsx_path, index=False)
    return arr


if __name__ == "__main__":
    print("=" * 80)
    print("GENERATING EMBEDDINGS FOR EXISTING EXHIBITIONS FILE")
    print("=" * 80)
    ex_path = "upcoming_exhibitions.xlsx"
    print(f"\nLoading exhibitions from: {ex_path}")
    ex_df = pd.read_excel(ex_path)
    print(f"Loaded {len(ex_df)} exhibition films")
    print("\nConverting films to text for embedding...")
    arr = generate_exhibition_embeddings(
        exhibitions_path=ex_path,
        npy_path="upcoming_exhibitions_embeddings.npy",
        metadata_xlsx_path="upcoming_exhibitions_embeddings.xlsx",
    )
    print(f"\n[SUCCESS] Embeddings generated and saved! Shape: {arr.shape}")
