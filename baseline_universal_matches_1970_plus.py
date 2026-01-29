#!/usr/bin/env python
"""
Baseline similarity: Universal library (1970–today) vs upcoming exhibitions.

This script:
- Loads `universal_pictures_library.xlsx` and `upcoming_exhibitions.xlsx`
- Uses precomputed embedding arrays:
    - `universal_pictures_library_embeddings.npy`
    - `upcoming_exhibitions_embeddings.npy`
- Filters the Universal library to titles with release_year >= 1970
- Computes cosine similarity between Universal and exhibition embeddings
- Adds lightweight weights for thematic / stylistic / emotional fields
- Writes the top 100 matches (highest overall score) to a temporary Excel file:
    - `universal_baseline_top100_temp.xlsx`

This is intentionally simpler than the production MatchingAgent logic so it can
serve as a baseline for evaluating the current matching algorithm.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Set, Tuple


PROJECT_ROOT = Path(__file__).parent


def _load_data() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load Universal library, exhibitions, and their embeddings."""
    lib_path = PROJECT_ROOT / "universal_pictures_library.xlsx"
    ex_path = PROJECT_ROOT / "upcoming_exhibitions.xlsx"
    lib_emb_path = PROJECT_ROOT / "universal_pictures_library_embeddings.npy"
    ex_emb_path = PROJECT_ROOT / "upcoming_exhibitions_embeddings.npy"

    library_df = pd.read_excel(lib_path)
    exhibitions_df = pd.read_excel(ex_path)

    lib_embeddings = np.load(lib_emb_path)
    ex_embeddings = np.load(ex_emb_path)

    if lib_embeddings.shape[0] != len(library_df):
        raise RuntimeError(
            f"Library embeddings rows ({lib_embeddings.shape[0]}) "
            f"do not match library rows ({len(library_df)})."
        )
    if ex_embeddings.shape[0] != len(exhibitions_df):
        raise RuntimeError(
            f"Exhibition embeddings rows ({ex_embeddings.shape[0]}) "
            f"do not match exhibition rows ({len(exhibitions_df)})."
        )

    return library_df, exhibitions_df, lib_embeddings, ex_embeddings


def _normalize_embeddings(arr: np.ndarray) -> np.ndarray:
    """L2-normalize embedding matrix row-wise."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms


def _tokenize_field(value: str) -> Set[str]:
    """Tokenize a descriptor string into a set of lowercase tokens."""
    if not isinstance(value, str) or not value.strip():
        return set()
    # Split first on commas, then on whitespace for robustness
    tokens: List[str] = []
    for part in value.split(","):
        part = part.strip().lower()
        if not part:
            continue
        tokens.extend(part.split())
    return {t for t in tokens if t}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Simple Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a | b)
    return inter / union if union else 0.0


def compute_baseline_scores() -> pd.DataFrame:
    """
    Compute baseline similarity between Universal library (1970+) and exhibitions.

    Overall score per (library, exhibition) pair:
        base_cosine
        + w_theme   * Jaccard(thematic_descriptors_lib, thematic_descriptors_ex)
        + w_style   * Jaccard(stylistic_descriptors_lib, stylistic_descriptors_ex)
        + w_emotion * I[emotional_tone_lib == emotional_tone_ex]
    """
    library_df, exhibitions_df, lib_emb, ex_emb = _load_data()

    # Filter Universal library to 1970+
    lib_mask = library_df["release_year"].fillna(0).astype(int) >= 1970
    lib_1970_df = library_df[lib_mask].reset_index(drop=False)  # keep original idx
    lib_indices = lib_1970_df["index"].to_numpy()

    if len(lib_indices) == 0:
        raise RuntimeError("No Universal library titles with release_year >= 1970.")

    lib_emb_1970 = lib_emb[lib_indices]

    # Normalize embeddings for cosine similarity
    lib_norm = _normalize_embeddings(lib_emb_1970)
    ex_norm = _normalize_embeddings(ex_emb)

    # Cosine similarity matrix: exhibitions x library
    # shape: (E, L_1970)
    cosine_matrix = ex_norm @ lib_norm.T

    # Pre-tokenize thematic / stylistic descriptors and emotional tone
    lib_theme_tokens = [
        _tokenize_field(v) for v in library_df.loc[lib_indices, "thematic_descriptors"].fillna("")
    ]
    lib_style_tokens = [
        _tokenize_field(v) for v in library_df.loc[lib_indices, "stylistic_descriptors"].fillna("")
    ]
    lib_emotion = library_df.loc[lib_indices, "emotional_tone"].fillna("").str.lower().tolist()

    ex_theme_tokens = [
        _tokenize_field(v) for v in exhibitions_df["thematic_descriptors"].fillna("")
    ]
    ex_style_tokens = [
        _tokenize_field(v) for v in exhibitions_df["stylistic_descriptors"].fillna("")
    ]
    ex_emotion = exhibitions_df["emotional_tone"].fillna("").str.lower().tolist()

    # Weights for additional components (tune as needed)
    w_theme = 0.15
    w_style = 0.10
    w_emotion = 0.05

    rows: List[dict] = []
    num_ex = len(exhibitions_df)
    num_lib = len(lib_1970_df)

    for ex_idx in range(num_ex):
        for local_lib_idx in range(num_lib):
            base_cos = float(cosine_matrix[ex_idx, local_lib_idx])

            theme_sim = _jaccard(lib_theme_tokens[local_lib_idx], ex_theme_tokens[ex_idx])
            style_sim = _jaccard(lib_style_tokens[local_lib_idx], ex_style_tokens[ex_idx])
            emotion_sim = 1.0 if lib_emotion[local_lib_idx] and lib_emotion[local_lib_idx] == ex_emotion[ex_idx] else 0.0

            score = base_cos + w_theme * theme_sim + w_style * style_sim + w_emotion * emotion_sim

            lib_row = lib_1970_df.iloc[local_lib_idx]
            ex_row = exhibitions_df.iloc[ex_idx]

            rows.append(
                {
                    "library_title": lib_row.get("title"),
                    "library_release_year": lib_row.get("release_year"),
                    "exhibition_title": ex_row.get("title"),
                    "exhibition_release_year": ex_row.get("release_year"),
                    "exhibition_country": ex_row.get("country"),
                    "exhibition_location": ex_row.get("location"),
                    "base_cosine": base_cos,
                    "theme_jaccard": theme_sim,
                    "style_jaccard": style_sim,
                    "emotion_match": emotion_sim,
                    "overall_score": score,
                }
            )

    df = pd.DataFrame(rows)
    # Sort by overall_score descending and keep top 100
    df = df.sort_values("overall_score", ascending=False).head(100).reset_index(drop=True)
    return df


def main() -> None:
    print("Computing baseline Universal (1970–today) vs exhibitions similarity...")
    df = compute_baseline_scores()
    out_path = PROJECT_ROOT / "universal_baseline_top100_temp.xlsx"
    df.to_excel(out_path, index=False)
    print(f"[DONE] Wrote top 100 baseline matches to: {out_path}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()

