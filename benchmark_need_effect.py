"""
Benchmark: effect of need (and emotional_tone) on recommendation scores.
Compares:
  - Legacy: composite query+exhibition; no explicit need/emotional_tone in exhibition similarity.
  - Deep (no need): deep_gate_tie_nudge with need and emotional_tone weight = 0.
  - Deep (with need): deep_gate_tie_nudge with default need + emotional_tone weights.

Uses same embeddings (which include need in text) for all; only the explicit per-factor
need/emotional_tone in enhanced_similarity differs between deep no-need vs with-need.
"""
import os
import json
import numpy as np
from dataclasses import replace

from chatbot_agent import ChatbotAgent
from recommendation_scoring import compute_ranked_matches

# Fixed prompts that typically yield recommendations (diverse; avoid narrow filters that zero exhibitions)
BENCHMARK_PROMPTS = [
    "What do we have that pairs with Eternal Sunshine of the Spotless Mind?",
    "Something dark and existential for current exhibitions.",
    "Best library matches for what's showing at American Cinematheque.",
    "What can we recommend for the Love Film Festival?",
    "Indie or drama that matches the vibe of Punch-Drunk Love.",
    "Show me dramas with strong thematic depth for current exhibitions.",
]


def _get_filtered_inputs(agent: ChatbotAgent, query: str):
    """Run parse + filters (with fallbacks). Return (intent, lib_rows, ex_rows, filtered_lib_emb, filtered_ex_emb) or None."""
    intent = agent.query_parser.parse(query, history_prompts=[], previous_intent=None)
    library_df = agent._load_library()
    exhibitions_df = agent._load_exhibitions()
    lib_embeddings, ex_embeddings = agent._load_embeddings(required=True)

    filtered_library_df = agent._apply_library_filters(library_df, intent)
    if len(filtered_library_df) == 0 and intent.genres:
        fallback_intent = replace(intent, genres=None)
        filtered_no_genre = agent._apply_library_filters(library_df, fallback_intent)
        filtered_library_df = agent._filter_library_by_text_terms(filtered_no_genre, intent.genres)
    if len(filtered_library_df) == 0:
        unstructured_candidates = agent._filter_library_by_unstructured_query(library_df, query)
        if len(unstructured_candidates) > 0:
            filtered_library_df = unstructured_candidates
    if len(filtered_library_df) == 0:
        return None

    filtered_exhibitions_df = agent._apply_exhibition_filters(exhibitions_df, intent)
    if len(filtered_exhibitions_df) == 0 and intent.territory:
        fallback_intent = replace(intent, territory=None)
        filtered_exhibitions_df = agent._apply_exhibition_filters(exhibitions_df, fallback_intent)
        if len(filtered_exhibitions_df) > 0:
            intent = fallback_intent
    if len(filtered_exhibitions_df) == 0 and (getattr(intent, "exhibition_date_start", None) or getattr(intent, "exhibition_date_end", None)):
        fallback_intent = replace(intent, exhibition_date_start=None, exhibition_date_end=None)
        filtered_exhibitions_df = agent._apply_exhibition_filters(exhibitions_df, fallback_intent)
        if len(filtered_exhibitions_df) > 0:
            intent = fallback_intent
    if len(filtered_exhibitions_df) == 0 and getattr(intent, "match_to_specific_film", None):
        fallback_intent = replace(intent, match_to_specific_film=None)
        filtered_exhibitions_df = agent._apply_exhibition_filters(exhibitions_df, fallback_intent)
        if len(filtered_exhibitions_df) > 0:
            intent = fallback_intent
    if len(filtered_exhibitions_df) == 0:
        return None

    library_indices = filtered_library_df.index.tolist()
    exhibition_indices = filtered_exhibitions_df.index.tolist()
    filtered_lib_embeddings = lib_embeddings[library_indices]
    filtered_ex_embeddings = ex_embeddings[exhibition_indices]
    lib_rows = filtered_library_df.to_dict(orient="records")
    ex_rows = filtered_exhibitions_df.to_dict(orient="records")
    return (intent, lib_rows, ex_rows, filtered_lib_embeddings, filtered_ex_embeddings)


def _collect_scores(matches, top_n=10):
    """Extract relevance_score, exhibition_similarity, query_similarity from match list."""
    out = []
    for m in matches[:top_n]:
        out.append({
            "relevance_score": float(m.get("relevance_score", 0)),
            "exhibition_similarity": float(m.get("exhibition_similarity", 0)),
            "query_similarity": float(m.get("query_similarity", 0)),
        })
    return out


def _stats(arr):
    a = np.array(arr, dtype=float)
    if len(a) == 0:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0, "n": 0}
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "median": float(np.median(a)),
        "n": len(a),
    }


def run_benchmark():
    agent = ChatbotAgent()
    openai_client = agent.openai_client
    matching_agent = agent.matching_agent
    top_n = 10  # collect up to 10 per prompt for better stats

    all_legacy = []
    all_deep_no_need = []
    all_deep_with_need = []

    per_prompt = []

    for i, query in enumerate(BENCHMARK_PROMPTS):
        print(f"[{i+1}/{len(BENCHMARK_PROMPTS)}] {query[:55]}...")
        inputs = _get_filtered_inputs(agent, query)
        if inputs is None:
            print("    Skip: no library or exhibitions after filters.")
            continue
        intent, lib_rows, ex_rows, filtered_lib_emb, filtered_ex_emb = inputs

        kwargs = dict(
            lib_rows=lib_rows,
            ex_rows=ex_rows,
            filtered_lib_embeddings=filtered_lib_emb,
            filtered_ex_embeddings=filtered_ex_emb,
            query=query,
            intent=intent,
            matching_agent=matching_agent,
            openai_client=openai_client,
            top_n=top_n,
            min_exhibition_similarity=0.5,
        )

        matches_legacy = compute_ranked_matches(**kwargs, strategy="legacy")
        matches_deep_no = compute_ranked_matches(**kwargs, strategy="deep_gate_tie_nudge", exclude_need=True)
        matches_deep_yes = compute_ranked_matches(**kwargs, strategy="deep_gate_tie_nudge", exclude_need=False)

        scores_legacy = _collect_scores(matches_legacy, top_n)
        scores_deep_no = _collect_scores(matches_deep_no, top_n)
        scores_deep_yes = _collect_scores(matches_deep_yes, top_n)

        all_legacy.extend(scores_legacy)
        all_deep_no_need.extend(scores_deep_no)
        all_deep_with_need.extend(scores_deep_yes)

        per_prompt.append({
            "prompt": query,
            "n_legacy": len(scores_legacy),
            "n_deep_no_need": len(scores_deep_no),
            "n_deep_with_need": len(scores_deep_yes),
            "legacy": {k: _stats([s[k] for s in scores_legacy]) for k in ("relevance_score", "exhibition_similarity", "query_similarity")},
            "deep_no_need": {k: _stats([s[k] for s in scores_deep_no]) for k in ("relevance_score", "exhibition_similarity", "query_similarity")},
            "deep_with_need": {k: _stats([s[k] for s in scores_deep_yes]) for k in ("relevance_score", "exhibition_similarity", "query_similarity")},
        })

    # Overall stats
    def overall(name, scores):
        return {
            "relevance_score": _stats([s["relevance_score"] for s in scores]),
            "exhibition_similarity": _stats([s["exhibition_similarity"] for s in scores]),
            "query_similarity": _stats([s["query_similarity"] for s in scores]),
        }

    results = {
        "prompts_used": len(per_prompt),
        "total_matches_legacy": len(all_legacy),
        "total_matches_deep_no_need": len(all_deep_no_need),
        "total_matches_deep_with_need": len(all_deep_with_need),
        "overall": {
            "legacy": overall("legacy", all_legacy),
            "deep_no_need": overall("deep_no_need", all_deep_no_need),
            "deep_with_need": overall("deep_with_need", all_deep_with_need),
        },
        "per_prompt": per_prompt,
    }

    return results


def print_analysis(results: dict) -> str:
    o = results["overall"]
    lines = []
    lines.append("=" * 70)
    lines.append("NEED EMBEDDING BENCHMARK: Legacy vs Deep (no need) vs Deep (with need)")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Note: All use the same embeddings (which include 'need' in the embedding text).")
    lines.append("Legacy = composite relevance; no explicit need/emotional_tone in enhanced_similarity.")
    lines.append("Deep = base scoring weights of deep data fields (in exhibition similarity) + query gate + tie-break + nudge.")
    lines.append("      Query intent dynamically boosts/diminishes those weights. Deep data fields do NOT have their own gate/tie-break/nudge.")
    lines.append("Deep no need = deep with need & emotional_tone weight = 0 in exhibition similarity.")
    lines.append("Deep with need = deep with default need (0.15) & emotional_tone (0.15) in exhibition similarity.")
    lines.append("")

    for metric in ("exhibition_similarity", "relevance_score", "query_similarity"):
        lines.append(f"--- {metric} ---")
        for strategy in ("legacy", "deep_no_need", "deep_with_need"):
            s = o[strategy][metric]
            lines.append(f"  {strategy:18}  mean={s['mean']:.4f}  std={s['std']:.4f}  min={s['min']:.4f}  max={s['max']:.4f}  n={s['n']}")
        lines.append("")

    lines.append("--- Analysis ---")
    # Exhibition similarity: are scores higher with need? is spread wider?
    ex_legacy = o["legacy"]["exhibition_similarity"]
    ex_deep_no = o["deep_no_need"]["exhibition_similarity"]
    ex_deep_yes = o["deep_with_need"]["exhibition_similarity"]

    lines.append(f"Exhibition similarity:")
    lines.append(f"  Legacy mean = {ex_legacy['mean']:.4f}; Deep (no need) mean = {ex_deep_no['mean']:.4f}; Deep (with need) mean = {ex_deep_yes['mean']:.4f}.")
    if ex_deep_yes["mean"] > ex_deep_no["mean"]:
        lines.append(f"  Adding need/emotional_tone raises mean exhibition similarity by {ex_deep_yes['mean'] - ex_deep_no['mean']:.4f}.")
    else:
        lines.append(f"  Adding need/emotional_tone changes mean exhibition similarity by {ex_deep_yes['mean'] - ex_deep_no['mean']:.4f}.")

    lines.append(f"  Spread (std): Legacy = {ex_legacy['std']:.4f}, Deep no need = {ex_deep_no['std']:.4f}, Deep with need = {ex_deep_yes['std']:.4f}.")
    if ex_deep_yes["std"] > ex_deep_no["std"]:
        lines.append(f"  With need, spread is WIDER (std +{ex_deep_yes['std'] - ex_deep_no['std']:.4f}) — more discrimination between high and lower matches.")
    elif ex_deep_yes["std"] < ex_deep_no["std"]:
        lines.append(f"  With need, spread is NARROWER (std {ex_deep_yes['std'] - ex_deep_no['std']:.4f}) — scores more compressed.")
    else:
        lines.append(f"  Spread is similar with vs without need.")

    lines.append("")
    lines.append("Relevance score (final ranking score):")
    rel_legacy = o["legacy"]["relevance_score"]
    rel_deep_no = o["deep_no_need"]["relevance_score"]
    rel_deep_yes = o["deep_with_need"]["relevance_score"]
    lines.append(f"  Legacy mean = {rel_legacy['mean']:.4f}; Deep (no need) = {rel_deep_no['mean']:.4f}; Deep (with need) = {rel_deep_yes['mean']:.4f}.")
    lines.append(f"  Spread (std): Legacy = {rel_legacy['std']:.4f}, Deep no need = {rel_deep_no['std']:.4f}, Deep with need = {rel_deep_yes['std']:.4f}.")
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    print("Running need-effect benchmark (legacy vs deep no-need vs deep with-need)...")
    results = run_benchmark()
    analysis = print_analysis(results)
    print(analysis)
    out_path = "benchmark_need_effect_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_path}")
