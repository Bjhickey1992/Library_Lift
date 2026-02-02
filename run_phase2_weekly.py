#!/usr/bin/env python
"""
Phase 2 weekly: Scrape independent/nonprofit cinemas, build exhibitions Excel, generate embeddings.

Run this script (or trigger from the app on Mondays) to:
1. Scrape all enabled cinemas from cinemas.yaml (aggressive: JSON-LD + LLM fallback)
2. Enrich screenings with TMDb and write upcoming_exhibitions.xlsx
3. Generate embeddings and save upcoming_exhibitions_embeddings.npy

Used by the Streamlit app for the autonomous weekly Monday run.
"""

import os
import sys
from pathlib import Path
from typing import List

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_openai_api_key, get_tmdb_api_key
from film_agent import ExhibitionScrapingAgent

# Optional: OpenAI for embeddings
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def _generate_exhibition_embeddings(exhibitions_path: str = "upcoming_exhibitions.xlsx") -> None:
    """Generate embeddings for exhibition rows and save .npy and metadata .xlsx."""
    import pandas as pd
    import numpy as np

    if OpenAI is None:
        raise RuntimeError("openai package is required for embedding generation")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for embedding generation")

    client = OpenAI(api_key=api_key)
    ex_path = PROJECT_ROOT / exhibitions_path
    if not ex_path.exists():
        raise FileNotFoundError(f"Exhibitions file not found: {ex_path}")

    ex_df = pd.read_excel(ex_path)
    if len(ex_df) == 0:
        print("[Phase2 weekly] No exhibition rows; skipping embeddings.")
        return

    # Same text construction as generate_exhibition_embeddings.py
    film_texts: List[str] = []
    for _, row in ex_df.iterrows():
        parts = []
        if pd.notna(row.get("title")):
            parts.append(f"Title: {row['title']}")
        if pd.notna(row.get("release_year")):
            parts.append(f"Year: {int(row['release_year'])}")
        if pd.notna(row.get("director")):
            parts.append(f"Director: {row['director']}")
        if pd.notna(row.get("writers")):
            parts.append(f"Writers: {row['writers']}")
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
        film_texts.append("\n".join(parts) if parts else str(row.get("title", "")))

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
        except Exception as e:
            print(f"  [WARN] Embedding batch error: {e}")
            embeddings.extend([[0.0] * 1536 for _ in batch])

    arr = np.array(embeddings)
    npy_path = PROJECT_ROOT / "upcoming_exhibitions_embeddings.npy"
    np.save(npy_path, arr)
    print(f"[Phase2 weekly] Saved embeddings to {npy_path} (shape {arr.shape})")


def run_phase2_weekly(
    cinemas_yaml_path: str = "cinemas.yaml",
    weeks_ahead: int = 4,
    output_path: str = "upcoming_exhibitions.xlsx",
) -> None:
    """Scrape cinemas, build exhibitions Excel, generate embeddings."""
    os.environ["TMDB_API_KEY"] = get_tmdb_api_key()
    os.environ["OPENAI_API_KEY"] = get_openai_api_key()

    agent = ExhibitionScrapingAgent(openai_api_key=os.getenv("OPENAI_API_KEY"))

    cinemas_path = PROJECT_ROOT / cinemas_yaml_path
    if not cinemas_path.exists():
        raise FileNotFoundError(f"Cinemas config not found: {cinemas_path}")

    out_path = PROJECT_ROOT / output_path
    exhibitions_df = agent.build_exhibitions_progressively(
        cinemas_yaml_path=str(cinemas_path),
        weeks_ahead=weeks_ahead,
        output_path=str(out_path),
    )

    if exhibitions_df is not None and len(exhibitions_df) > 0:
        _generate_exhibition_embeddings(exhibitions_path=output_path)
    else:
        print("[Phase2 weekly] No exhibitions written; skipping embeddings.")


if __name__ == "__main__":
    print("Phase 2 weekly: scraping cinemas, building exhibitions, generating embeddings...")
    run_phase2_weekly()
    print("Phase 2 weekly complete.")
