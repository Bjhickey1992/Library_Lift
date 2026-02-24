"""
Debug script: Trace every step of scoring for
  Library: 12 Rounds 3: Lockdown
  Exhibition: Safe (at Northwest Film Forum)
"""
import sys
from pathlib import Path

_app_root = Path(__file__).resolve().parent

import pandas as pd
import numpy as np

from chatbot_agent import ChatbotAgent
from film_agent import MatchingAgent
from recommendation_scoring import (
    compute_query_sims,
    NUDGE_CAP, NUDGE_BONUS_ABOVE, NUDGE_BONUS_RANGE,
    NUDGE_PENALTY_BELOW, NUDGE_PENALTY_RANGE,
)

QUERY = "show me the films in our library that are the best thematic matches for movies in theaters"
LIB_TITLE = "12 Rounds 3: Lockdown"
EX_TITLE = "Safe"
EX_LOCATION_HINT = "Northwest Film Forum"


def main():
    agent = ChatbotAgent(studio_name="Lionsgate", app_root=_app_root)
    intent = agent.query_parser.parse(QUERY)

    library_df = agent._load_library()
    exhibitions_df = agent._load_exhibitions()
    lib_embeddings, ex_embeddings = agent._load_embeddings(required=True)

    # Apply same filters as get_dynamic_recommendations
    filtered_library_df = agent._apply_library_filters(library_df, intent)
    filtered_exhibitions_df = agent._apply_exhibition_filters(exhibitions_df, intent)
    if len(filtered_exhibitions_df) == 0:
        filtered_exhibitions_df = agent._filter_exhibitions_by_unstructured_query(exhibitions_df, QUERY)
    if len(filtered_exhibitions_df) == 0:
        filtered_exhibitions_df = exhibitions_df

    lib_rows = filtered_library_df.to_dict(orient="records")
    ex_rows = filtered_exhibitions_df.to_dict(orient="records")
    lib_indices = filtered_library_df.index.tolist()
    ex_indices = filtered_exhibitions_df.index.tolist()
    filtered_lib_emb = lib_embeddings[lib_indices]
    filtered_ex_emb = ex_embeddings[ex_indices]

    # Find our pair
    lib_idx = None
    ex_idx = None
    for i, r in enumerate(lib_rows):
        if r.get("title", "").strip() == LIB_TITLE:
            lib_idx = i
            break
    for i, r in enumerate(ex_rows):
        t = r.get("title", "").strip()
        loc = str(r.get("location", "") or "")
        if t == EX_TITLE and EX_LOCATION_HINT in loc:
            ex_idx = i
            break
    if ex_idx is None:
        for i, r in enumerate(ex_rows):
            if r.get("title", "").strip() == EX_TITLE:
                ex_idx = i
                break

    if lib_idx is None:
        print(f"ERROR: Library film '{LIB_TITLE}' not found in filtered library ({len(lib_rows)} films)")
        return 1
    if ex_idx is None:
        print(f"ERROR: Exhibition '{EX_TITLE}' not found in filtered exhibitions ({len(ex_rows)} exhibitions)")
        return 1

    lib_film = lib_rows[lib_idx]
    ex_film = ex_rows[ex_idx]

    print("=" * 70)
    print("SCORING BREAKDOWN: 12 Rounds 3: Lockdown  <->  Safe")
    print("=" * 70)

    # --- 1. Intent weights (thematic query) ---
    print("\n--- 1. INTENT WEIGHTS (thematic-focused query) ---")
    print(f"  director_weight = {intent.director_weight}")
    print(f"  writer_weight   = {intent.writer_weight}")
    print(f"  cast_weight     = {intent.cast_weight}")
    print(f"  thematic_weight = {intent.thematic_weight}")
    print(f"  stylistic_weight= {intent.stylistic_weight}")
    extra = getattr(intent, "column_weights", None) or {}
    print(f"  extra_weights   = {extra}")

    # --- 2. Base embedding similarity ---
    lib_norm = filtered_lib_emb / (np.linalg.norm(filtered_lib_emb, axis=1, keepdims=True) + 1e-8)
    ex_norm = filtered_ex_emb / (np.linalg.norm(filtered_ex_emb, axis=1, keepdims=True) + 1e-8)
    base_sim = float(np.dot(ex_norm[ex_idx], lib_norm[lib_idx]))
    print("\n--- 2. BASE EMBEDDING SIMILARITY (cosine) ---")
    print(f"  base_similarity = {base_sim:.6f}")

    # --- 3. Query similarity ---
    query_sims = compute_query_sims(agent.openai_client, QUERY, lib_norm)
    q_sim = float(query_sims[lib_idx])
    print("\n--- 3. QUERY SIMILARITY (library film vs query embedding) ---")
    print(f"  query_similarity = {q_sim:.6f}")

    # --- 4. Enhanced similarity components (film_agent) ---
    ma = MatchingAgent()
    thematic_sim = ma._calculate_thematic_similarity(lib_film, ex_film)
    stylistic_sim = ma._calculate_stylistic_similarity(lib_film, ex_film)
    director_sim, writer_sim, cast_sim = ma._calculate_personnel_components(lib_film, ex_film)

    print("\n--- 4. COMPONENT SIMILARITIES (0–1) ---")
    print(f"  thematic_sim  = {thematic_sim:.6f}")
    print(f"  stylistic_sim = {stylistic_sim:.6f}")
    print(f"  director_sim  = {director_sim:.6f}")
    print(f"  writer_sim   = {writer_sim:.6f}")
    print(f"  cast_sim      = {cast_sim:.6f}")

    # Data used for thematic/stylistic
    print("\n--- 4b. RAW DATA USED ---")
    print(f"  Lib thematic_descriptors: {lib_film.get('thematic_descriptors', '')[:120]}...")
    print(f"  Ex  thematic_descriptors: {ex_film.get('thematic_descriptors', '')[:120]}...")
    print(f"  Lib stylistic_descriptors: {lib_film.get('stylistic_descriptors', '')[:120]}...")
    print(f"  Ex  stylistic_descriptors: {ex_film.get('stylistic_descriptors', '')[:120]}...")
    print(f"  Lib director: {lib_film.get('director')} | Ex director: {ex_film.get('director')}")
    print(f"  Lib writers: {lib_film.get('writers', '')[:80]}...")
    print(f"  Ex  writers: {ex_film.get('writers', '')[:80]}...")

    # --- 5. Weight normalization ---
    total_weight = intent.director_weight + intent.writer_weight + intent.cast_weight + intent.thematic_weight + intent.stylistic_weight
    extra_sum = sum(extra.values())
    total = total_weight + extra_sum
    scale = 1.0 / total if total > 0 else 1.0
    dw = intent.director_weight * scale
    ww = intent.writer_weight * scale
    cw = intent.cast_weight * scale
    tw = intent.thematic_weight * scale
    sw = intent.stylistic_weight * scale
    extra_scaled = {k: v * scale for k, v in extra.items()}

    print("\n--- 5. NORMALIZED WEIGHTS (sum=1) ---")
    print(f"  director={dw:.4f}, writer={ww:.4f}, cast={cw:.4f}, thematic={tw:.4f}, stylistic={sw:.4f}")
    print(f"  extra (scaled) = {extra_scaled}")

    # --- 6. Component factors (boost-only) ---
    director_factor = 1.0 + (director_sim * dw * 0.5) if dw > 0 else 1.0
    writer_factor = 1.0 + (writer_sim * ww * 0.5) if ww > 0 else 1.0
    cast_factor = 1.0 + (cast_sim * cw * 0.5) if cw > 0 else 1.0
    thematic_factor = 1.0 + (thematic_sim * tw * 0.5) if tw > 0 else 1.0
    stylistic_factor = 1.0 + (stylistic_sim * sw * 0.5) if sw > 0 else 1.0

    extra_factor = 1.0
    for col, w in extra_scaled.items():
        if col not in lib_film or col not in ex_film or w <= 0:
            continue
        sim = ma._column_similarity(lib_film, ex_film, col)
        extra_factor *= 1.0 + (sim * w * 0.5)

    print("\n--- 6. COMPONENT FACTORS (boost-only) ---")
    print(f"  director_factor  = {director_factor:.6f}")
    print(f"  writer_factor    = {writer_factor:.6f}")
    print(f"  cast_factor      = {cast_factor:.6f}")
    print(f"  thematic_factor  = {thematic_factor:.6f}")
    print(f"  stylistic_factor = {stylistic_factor:.6f}")
    print(f"  extra_factor     = {extra_factor:.6f}")

    # --- 7. Enhanced similarity ---
    enhanced_sim_before_boost = base_sim * director_factor * writer_factor * cast_factor * thematic_factor * stylistic_factor * extra_factor
    boost = ma._calculate_similarity_boost(lib_film, ex_film)
    exhibition_similarity = min(1.0, max(0.0, enhanced_sim_before_boost + boost))

    print("\n--- 7. ENHANCED EXHIBITION SIMILARITY ---")
    print(f"  base * factors = {enhanced_sim_before_boost:.6f}")
    print(f"  boost (actors/themes/keywords) = {boost:.6f}")
    print(f"  exhibition_similarity (final) = {exhibition_similarity:.6f}")

    # --- 8. Nudge (deep_gate_tie_nudge) ---
    if q_sim >= NUDGE_BONUS_ABOVE:
        nudge = NUDGE_CAP * (q_sim - NUDGE_BONUS_ABOVE) / NUDGE_BONUS_RANGE
        nudge = min(NUDGE_CAP, nudge)
    elif q_sim <= NUDGE_PENALTY_BELOW:
        nudge = -NUDGE_CAP * (NUDGE_PENALTY_BELOW - q_sim) / NUDGE_PENALTY_RANGE
    else:
        nudge = 0.0

    relevance_score = exhibition_similarity + nudge
    relevance_score = max(0.0, min(1.0, relevance_score))

    print("\n--- 8. RELEVANCE SCORE (deep_gate_tie_nudge) ---")
    print(f"  nudge from query_sim = {nudge:.6f}")
    print(f"  relevance_score = exhibition_similarity + nudge = {relevance_score:.6f}")

    # --- 9. Verification: call matching_agent directly ---
    ex_sim_verify = agent.matching_agent._calculate_enhanced_similarity(
        lib_film, ex_film, base_sim,
        director_weight=intent.director_weight,
        writer_weight=intent.writer_weight,
        cast_weight=intent.cast_weight,
        thematic_weight=intent.thematic_weight,
        stylistic_weight=intent.stylistic_weight,
        extra_weights=extra,
    )
    print("\n--- 9. VERIFICATION ---")
    print(f"  _calculate_enhanced_similarity() = {ex_sim_verify:.6f}")

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
