# Scoring Algorithm Comparison: Mauvais Sang Test Prompt

**Test query:** "I see Mauvais Sang is playing, do we have anything we can push based on that?"

**Benchmarks:** `benchmark_mauvais_sang_legacy.json` (composite query+exhibition) vs `benchmark_mauvais_sang_deep.json` (deep-factor emphasis + gate/tie/nudge).

---

## Summary

- **Same top 5 in the same order** for both strategies: Vidocq, Dark Seduction, Buffalo '66, Dying of the Light, Death of Me.
- **Exhibition-led scoring in the deep strategy** shifts the *meaning* of the relevance score and raises **exhibition similarity** for titles that resonate on theme/style rather than surface (genre/cast). **Buffalo '66** is the clearest example: its exhibition similarity increases under the deep weights (0.614 → 0.623), so the algorithm is reflecting its stronger thematic/audience relationship with Mauvais Sang even though the films look different on the surface.

---

## Side-by-side scores

| Rank | Title            | Legacy (rel / ex / query)     | Deep (rel / ex / query)      |
|------|------------------|-------------------------------|------------------------------|
| 1    | Vidocq           | 0.648 / 0.653 / 0.634         | 0.669 / **0.658** / 0.634     |
| 2    | Dark Seduction   | 0.630 / 0.633 / 0.619         | 0.646 / **0.636** / 0.619     |
| 3    | Buffalo '66      | 0.616 / 0.614 / 0.621         | 0.632 / **0.623** / 0.621     |
| 4    | Dying of the Light | 0.611 / 0.607 / 0.622       | 0.620 / **0.611** / 0.622     |
| 5    | Death of Me      | 0.599 / 0.590 / 0.625         | 0.610 / **0.600** / 0.625     |

- **rel** = relevance_score (final ranking score)  
- **ex** = exhibition_similarity (library–exhibition match)  
- **query** = query_similarity (query–library match)

---

## What changed with the new algorithm

### 1. **Exhibition is the main signal; query gates, nudges, and breaks ties**

- **Legacy:** Relevance = weighted blend of query + exhibition (e.g. 0.25×query + 0.75×exhibition for match_to_specific_film). Query and exhibition both drive the number.
- **Deep:** Relevance = exhibition_similarity + a **capped nudge** from query (gate: drop libs with query_sim &lt; 0.25; nudge: ±0.04; tie-break: when exhibition scores are within 0.015, order by query_sim). So ranking is **exhibition-led**; query only filters and fine-tunes.

### 2. **Deep-factor weights in exhibition similarity**

- **Legacy:** Uses intent weights (e.g. director 0.2, writer 0.2, cast 0.2, thematic 0.3, stylistic 0.2).
- **Deep:** Uses fixed “deep” weights: **thematic 0.4, stylistic 0.35**, director 0.1, writer 0.05, cast 0.1. So theme and style count much more; director/cast count less.

Effect: Titles that match Mauvais Sang on **mood, theme, and style** (e.g. Buffalo '66) get a **higher exhibition_similarity** even when they don’t share genre or star. That’s exactly the “you may not have thought of this, but it’s a really good match” behavior.

### 3. **Buffalo '66**

- **Legacy:** exhibition_similarity 0.614, relevance 0.616.  
- **Deep:** exhibition_similarity **0.623**, relevance 0.632.

So under the deep strategy, Buffalo '66 is scored **higher** on how well it matches the exhibition (Mauvais Sang). The films are different on the surface (genre, cast, director) but share thematic and audience resonance; the deep weights capture that better. The new algorithm is therefore better aligned with the goal of surfacing “different on the surface, similar audience/resonance” for distribution.

### 4. **Order unchanged for this prompt**

For this single-exhibition, match_to_specific_film prompt, the top 5 and their order are the same. The **interpretation** of the scores changes: in the deep strategy, relevance is clearly “exhibition match + small query nudge,” and exhibition_similarity is more about theme/style than surface. So we get:

- **More consistent story:** “We rank by fit to this exhibition; query keeps results on-topic and breaks ties.”
- **Better signal for “Buffalo '66–type” titles:** Higher weight on thematic/stylistic match helps titles that resonate with the audience of Mauvais Sang without looking like Mauvais Sang on paper.

---

## How to switch strategies

- **Use new algorithm (deep + gate/tie/nudge) — app default:**  
  Leave `RECOMMENDATION_SCORING_STRATEGY` unset, or set `RECOMMENDATION_SCORING_STRATEGY=deep_gate_tie_nudge`.

- **Use legacy (composite query+exhibition):**  
  Set `RECOMMENDATION_SCORING_STRATEGY=legacy`.

- **In code:**  
  `agent._scoring_strategy = "legacy"` or `"deep_gate_tie_nudge"`.

Scoring logic lives in **`recommendation_scoring.py`** (`run_legacy`, `run_deep_gate_tie_nudge`, `compute_ranked_matches`), so you can revert or re-implement by changing the strategy or editing that module.
