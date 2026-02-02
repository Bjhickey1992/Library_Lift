"""
Modular recommendation scoring: exhibition + query.
Strategies:
  - legacy: composite score (w_query * query_sim + w_ex * exhibition_sim) with intent-based weights.
  - deep_gate_tie_nudge: exhibition primary with deep-factor emphasis; query as gate + tie-breaker + nudge.
"""
import re
from typing import List, Dict, Optional, Tuple, Any

import numpy as np

# QueryIntent from parser
try:
    from query_intent_parser import QueryIntent
except ImportError:
    QueryIntent = Any


def _normalize_title(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _title_signature(t: str) -> str:
    words = _normalize_title(t).split()
    if not words:
        return ""
    if words[0] == "the":
        return " ".join(words[:3])
    return " ".join(words[:2])


def _is_exact_same_film(lib: Dict, ex: Dict) -> bool:
    lib_id = lib.get("tmdb_id")
    ex_id = ex.get("tmdb_id")
    if lib_id and ex_id and str(lib_id).isdigit() and str(ex_id).isdigit():
        try:
            return int(lib_id) == int(ex_id)
        except Exception:
            pass
    lib_title = _normalize_title(lib.get("title", ""))
    ex_title = _normalize_title(ex.get("title", ""))
    lib_year = str(lib.get("release_year") or "").strip()
    ex_year = str(ex.get("release_year") or "").strip()
    return bool(lib_title and ex_title and lib_title == ex_title and lib_year and ex_year and lib_year == ex_year)


def _location_key(loc: str) -> str:
    if not loc or not isinstance(loc, str):
        return loc or ""
    m = re.search(r"\(([^)]+)\)\s*$", loc.strip())
    return m.group(1).strip() if m else loc.strip()


def compute_query_sims(openai_client, query: str, lib_norm: np.ndarray) -> np.ndarray:
    """Query-to-library relevance [0, 1] per library film."""
    query_sims = np.zeros(lib_norm.shape[0], dtype=float)
    if not openai_client:
        return query_sims
    try:
        q_emb = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query,
            dimensions=1536,
        )
        q_vec = np.array(q_emb.data[0].embedding, dtype=float)
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-8)
        q_cos = np.dot(lib_norm, q_norm)
        query_sims = (q_cos + 1.0) / 2.0
    except Exception as e:
        print(f"[recommendation_scoring] Query embedding failed: {e}")
    return query_sims


# ---------- Legacy strategy (composite w_query * query + w_ex * exhibition) ----------

def _legacy_weights(query: str, intent: QueryIntent) -> Tuple[float, float]:
    q_lower = query.lower()
    w_query = 0.8
    if getattr(intent, "match_to_specific_film", None):
        w_query = 0.25
        return w_query, 0.75
    if "theme" in q_lower or "themes" in q_lower:
        w_query = 0.88
    if "existential" in q_lower:
        w_query = max(w_query, 0.92)
    if intent.genres or intent.lead_gender or intent.year_start or intent.year_end:
        w_query = max(w_query, 0.85)
    if any(kw in q_lower for kw in ["in theaters", "in theatres", "current exhibitions", "what's showing", "now showing"]):
        w_query = max(0.6, w_query - 0.15)
    return w_query, 1.0 - w_query


def run_legacy(
    lib_rows: List[Dict],
    ex_rows: List[Dict],
    filtered_lib_embeddings: np.ndarray,
    filtered_ex_embeddings: np.ndarray,
    query: str,
    intent: QueryIntent,
    matching_agent,
    openai_client,
    top_n: int,
    min_exhibition_similarity: float = 0.5,
) -> List[Dict]:
    """
    Legacy scoring: relevance_score = w_query * query_sim + w_ex * exhibition_similarity.
    Exhibition similarity uses intent weights (director, writer, cast, thematic, stylistic).
    """
    lib_norm = filtered_lib_embeddings / (np.linalg.norm(filtered_lib_embeddings, axis=1, keepdims=True) + 1e-8)
    ex_norm = filtered_ex_embeddings / (np.linalg.norm(filtered_ex_embeddings, axis=1, keepdims=True) + 1e-8)
    base_matrix = np.dot(ex_norm, lib_norm.T)

    query_sims = compute_query_sims(openai_client, query, lib_norm)
    w_query, w_ex = _legacy_weights(query, intent)

    topk = min(12, len(ex_rows)) if ex_rows else 0
    lib_candidates = []
    for lib_idx, lib_row in enumerate(lib_rows):
        if topk <= 0:
            continue
        col = base_matrix[:, lib_idx]
        cand = np.argpartition(-col, min(topk, len(col)) - 1)[:topk]
        cand = cand[np.argsort(-col[cand])]
        candidates_for_lib = []
        for ex_i in cand:
            ex_row = ex_rows[int(ex_i)]
            if _is_exact_same_film(lib_row, ex_row):
                continue
            base_sim = float(col[int(ex_i)])
            exhibition_similarity = matching_agent._calculate_enhanced_similarity(
                lib_row,
                ex_row,
                base_sim,
                director_weight=intent.director_weight,
                writer_weight=intent.writer_weight,
                cast_weight=intent.cast_weight,
                thematic_weight=intent.thematic_weight,
                stylistic_weight=intent.stylistic_weight,
                extra_weights=getattr(intent, "column_weights", None),
            )
            relevance_score = (w_query * float(query_sims[lib_idx])) + (w_ex * float(exhibition_similarity))
            candidates_for_lib.append({
                "exhibition_film": ex_row,
                "exhibition_similarity": float(exhibition_similarity),
                "relevance_score": relevance_score,
            })
            if len(candidates_for_lib) >= 10:
                break
        if not candidates_for_lib:
            continue
        best_relevance = max(c["relevance_score"] for c in candidates_for_lib)
        lib_candidates.append((lib_row, float(query_sims[lib_idx]), candidates_for_lib, best_relevance))

    lib_candidates.sort(key=lambda x: x[3], reverse=True)
    unique_matches = _build_unique_matches(
        lib_candidates, top_n, min_exhibition_similarity,
        _location_key, _title_signature,
    )
    return unique_matches


# ---------- Deep + gate/tie/nudge strategy ----------
#
# Exhibition similarity weights (all normalized to sum 1.0 inside _calculate_enhanced_similarity):
#   - director, writer, cast, thematic, stylistic: from intent (query-driven; defaults below from QueryIntent).
#   - emotional_tone, need: from intent.column_weights if set by user, else DEEP_* defaults.
# Default intent weights: director 0.2, writer 0.15, cast 0.15, thematic 0.25, stylistic 0.2.
# Default extra: emotional_tone 0.15, need 0.15.
#
# Director/writer/cast/thematic/stylistic weights come from intent (query-driven), same as legacy.
# emotional_tone and need are included in extra_weights; weight from intent.column_weights when set by user prompt, else default below.
DEEP_EMOTIONAL_TONE_WEIGHT = 0.15  # Default emotional_tone weight when user prompt does not set it.
DEEP_NEED_WEIGHT = 0.15  # Default need (viewer desires/needs) weight when user prompt does not set it.

# Query: gate threshold (exclude lib films below this query_sim).
QUERY_GATE_THRESHOLD = 0.25
# Nudge: cap on how much query can add/subtract from exhibition score.
NUDGE_CAP = 0.15
# Nudge: query_sim above this adds up to NUDGE_CAP; below NUDGE_PENALTY_BELOW subtracts.
NUDGE_BONUS_ABOVE = 0.6
# Range over which bonus ramps to full cap (steeper = more nudge for typical q_sim 0.61–0.65).
NUDGE_BONUS_RANGE = 0.2  # full boost at q_sim=0.8 (was 0.4 → full at 1.0)
NUDGE_PENALTY_BELOW = 0.4
NUDGE_PENALTY_RANGE = 0.4  # full penalty at q_sim=0 (ramp over 0→0.4)
# Tie-breaker: exhibition scores within this band use query_sim to order.
TIE_EPSILON = 0.03


def run_deep_gate_tie_nudge(
    lib_rows: List[Dict],
    ex_rows: List[Dict],
    filtered_lib_embeddings: np.ndarray,
    filtered_ex_embeddings: np.ndarray,
    query: str,
    intent: QueryIntent,
    matching_agent,
    openai_client,
    top_n: int,
    min_exhibition_similarity: float = 0.5,
) -> List[Dict]:
    """
    Exhibition-led ranking; query as gate + nudge + tie-breaker.
    - Exhibition similarity uses intent weights (query-driven), same as legacy—user query
      can dynamically increase or decrease director/writer/cast/thematic/stylistic weights.
    - emotional_tone and need are included in extra_weights; their weights come from intent.column_weights
      when the user prompt sets them, otherwise default to DEEP_EMOTIONAL_TONE_WEIGHT and DEEP_NEED_WEIGHT.
    - Gate: drop library films with query_sim < QUERY_GATE_THRESHOLD.
    - Rank by: exhibition_similarity + nudge(query_sim) where nudge is capped.
    - Tie-break: when exhibition scores within TIE_EPSILON, order by query_sim.
    """
    lib_norm = filtered_lib_embeddings / (np.linalg.norm(filtered_lib_embeddings, axis=1, keepdims=True) + 1e-8)
    ex_norm = filtered_ex_embeddings / (np.linalg.norm(filtered_ex_embeddings, axis=1, keepdims=True) + 1e-8)
    base_matrix = np.dot(ex_norm, lib_norm.T)

    query_sims = compute_query_sims(openai_client, query, lib_norm)

    topk = min(12, len(ex_rows)) if ex_rows else 0
    lib_candidates = []
    for lib_idx, lib_row in enumerate(lib_rows):
        # Gate: exclude if query relevance too low
        if float(query_sims[lib_idx]) < QUERY_GATE_THRESHOLD:
            continue
        if topk <= 0:
            continue
        col = base_matrix[:, lib_idx]
        cand = np.argpartition(-col, min(topk, len(col)) - 1)[:topk]
        cand = cand[np.argsort(-col[cand])]
        candidates_for_lib = []
        for ex_i in cand:
            ex_row = ex_rows[int(ex_i)]
            if _is_exact_same_film(lib_row, ex_row):
                continue
            base_sim = float(col[int(ex_i)])
            # Exhibition similarity: use intent weights (query-driven) same as legacy; emotional_tone and need from intent or default
            deep_extra = dict(getattr(intent, "column_weights", None) or {})
            if "emotional_tone" not in deep_extra:
                deep_extra["emotional_tone"] = DEEP_EMOTIONAL_TONE_WEIGHT
            if "need" not in deep_extra:
                deep_extra["need"] = DEEP_NEED_WEIGHT
            exhibition_similarity = matching_agent._calculate_enhanced_similarity(
                lib_row,
                ex_row,
                base_sim,
                director_weight=intent.director_weight,
                writer_weight=intent.writer_weight,
                cast_weight=intent.cast_weight,
                thematic_weight=intent.thematic_weight,
                stylistic_weight=intent.stylistic_weight,
                extra_weights=deep_extra,
            )
            ex_sim = float(exhibition_similarity)
            q_sim = float(query_sims[lib_idx])
            # relevance_score = exhibition_similarity + nudge(query_similarity); exhibition-led, query only adds ±NUDGE_CAP.
            # Nudge: bounded adjustment from query (steeper ramp so typical q_sim 0.61–0.65 gets a noticeable boost).
            if q_sim >= NUDGE_BONUS_ABOVE:
                nudge = NUDGE_CAP * (q_sim - NUDGE_BONUS_ABOVE) / NUDGE_BONUS_RANGE
                nudge = min(NUDGE_CAP, nudge)
            elif q_sim <= NUDGE_PENALTY_BELOW:
                nudge = -NUDGE_CAP * (NUDGE_PENALTY_BELOW - q_sim) / NUDGE_PENALTY_RANGE
            else:
                nudge = 0.0
            relevance_score = ex_sim + nudge
            relevance_score = max(0.0, min(1.0, relevance_score))
            candidates_for_lib.append({
                "exhibition_film": ex_row,
                "exhibition_similarity": ex_sim,
                "relevance_score": relevance_score,
            })
            if len(candidates_for_lib) >= 10:
                break
        if not candidates_for_lib:
            continue
        best_relevance = max(c["relevance_score"] for c in candidates_for_lib)
        lib_candidates.append((lib_row, float(query_sims[lib_idx]), candidates_for_lib, best_relevance))

    # Sort by relevance_score; when two films are within TIE_EPSILON, use query_sim as tie-breaker.
    # Use a band key so any pair within 0.03 is ordered by query_sim (not just vs the leader).
    def sort_key(item):
        lib_row, query_sim, candidates_for_lib, best_rel = item
        band = round(best_rel / TIE_EPSILON) * TIE_EPSILON
        return (band, query_sim)

    lib_candidates.sort(key=sort_key, reverse=True)

    unique_matches = _build_unique_matches(
        lib_candidates, top_n, min_exhibition_similarity,
        _location_key, _title_signature,
    )
    return unique_matches


def _build_unique_matches(
    lib_candidates: List[Tuple],
    top_n: int,
    min_exhibition_similarity: float,
    location_key_fn,
    title_sig_fn,
) -> List[Dict]:
    """Build unique_matches with location diversity; filter by min_exhibition_similarity."""
    seen_sig = set()
    used_location_keys = set()
    unique_matches = []
    for lib_row, query_sim, candidates_for_lib, _ in lib_candidates:
        sig = title_sig_fn(lib_row.get("title", ""))
        if sig and sig in seen_sig:
            continue
        if len(unique_matches) >= top_n:
            break
        best_item = None
        best_score = -1.0
        best_diverse_item = None
        best_diverse_score = -1.0
        for c in candidates_for_lib:
            loc_key = location_key_fn(c["exhibition_film"].get("location") or "")
            is_new_location = loc_key and loc_key not in used_location_keys
            score = c["relevance_score"]
            if is_new_location and score > best_diverse_score:
                best_diverse_item = c
                best_diverse_score = score
            if score > best_score:
                best_item = c
                best_score = score
        if best_diverse_item is not None:
            best_item = best_diverse_item
        elif best_item is None:
            best_item = candidates_for_lib[0]
        if sig:
            seen_sig.add(sig)
        loc_key = location_key_fn(best_item["exhibition_film"].get("location") or "")
        if loc_key:
            used_location_keys.add(loc_key)
        exhibition_matches_list = sorted(
            candidates_for_lib,
            key=lambda c: c["exhibition_similarity"],
            reverse=True,
        )[:5]
        exhibition_matches = [
            {
                "title": c["exhibition_film"].get("title", ""),
                "location": c["exhibition_film"].get("location", ""),
                "similarity": c["exhibition_similarity"],
            }
            for c in exhibition_matches_list
        ]
        unique_matches.append({
            "library_film": lib_row,
            "exhibition_film": best_item["exhibition_film"],
            "relevance_score": best_item["relevance_score"],
            "exhibition_similarity": best_item["exhibition_similarity"],
            "query_similarity": float(query_sim),
            "exhibition_matches": exhibition_matches,
        })
    unique_matches = [m for m in unique_matches if (m.get("exhibition_similarity") or 0) >= min_exhibition_similarity]
    return unique_matches


def compute_ranked_matches(
    lib_rows: List[Dict],
    ex_rows: List[Dict],
    filtered_lib_embeddings: np.ndarray,
    filtered_ex_embeddings: np.ndarray,
    query: str,
    intent: QueryIntent,
    matching_agent,
    openai_client,
    top_n: int = 5,
    min_exhibition_similarity: float = 0.5,
    strategy: str = "legacy",
) -> List[Dict]:
    """
    Compute ranked unique_matches for dynamic recommendations.
    strategy: "legacy" | "deep_gate_tie_nudge"
    """
    if strategy == "deep_gate_tie_nudge":
        return run_deep_gate_tie_nudge(
            lib_rows, ex_rows,
            filtered_lib_embeddings, filtered_ex_embeddings,
            query, intent, matching_agent, openai_client,
            top_n, min_exhibition_similarity,
        )
    return run_legacy(
        lib_rows, ex_rows,
        filtered_lib_embeddings, filtered_ex_embeddings,
        query, intent, matching_agent, openai_client,
        top_n, min_exhibition_similarity,
    )
