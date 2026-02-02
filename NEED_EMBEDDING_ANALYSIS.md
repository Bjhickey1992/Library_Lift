# Need Embedding Benchmark Analysis

## Setup

- **Legacy**: Composite relevance (`w_query * query_sim + w_ex * exhibition_sim`). Exhibition similarity uses director, writer, cast, thematic, stylistic only (no explicit need/emotional_tone).
- **Deep**: Base scoring weights of the deep data fields (director, writer, cast, thematic, stylistic, emotional_tone, need) in exhibition similarity, **plus** query gate + tie-break + nudge. **Query intent dynamically boosts and diminishes** these weights (e.g. user emphasizes themes → thematic_weight boosted). If a data field is **not** modified by the query, it keeps its **base weight**; all weights (base or query-adjusted) are then **normalized together** for the entire exhibition calculation. The deep data fields do **not** have their own gate/tie-break/nudge—only the query does.
- **Deep (no need)**: Same as deep but need and emotional_tone weight set to 0 in exhibition similarity.
- **Deep (with need)**: Same as deep with default need (0.15) and emotional_tone (0.15) in exhibition similarity.

All runs use the **same embeddings** (library and exhibition embeddings include "Viewer needs" in the embedding text). So the base cosine similarity already reflects need; the only difference between "deep no need" and "deep with need" is the explicit per-factor need/emotional_tone term in `_calculate_enhanced_similarity`.

Benchmark: 6 prompts × up to 10 matches each → **60 matches per strategy**.

---

## Results (summary)

| Metric                  | Legacy   | Deep (no need) | Deep (with need) |
|-------------------------|----------|----------------|------------------|
| **Exhibition similarity** mean | 0.6769   | 0.6939         | 0.6940           |
| **Exhibition similarity** std  | 0.0454   | 0.0374         | 0.0374           |
| **Relevance score** mean      | 0.6890   | 0.7567         | 0.7569           |
| **Relevance score** std       | 0.0212   | 0.0417         | 0.0414           |
| **Query similarity** mean      | 0.6918   | 0.6838         | 0.6838           |

---

## Findings

### 1. Need is not inflating scores

- **Exhibition similarity**: Deep with need (0.6940) vs deep no need (0.6939) → difference **+0.0001**. The explicit need/emotional_tone weight barely moves the needle.
- **Relevance score**: Deep with need (0.7569) vs deep no need (0.7567) → **+0.0002**. Again negligible.

So the “higher” scores you see are **not** from the need factor; they come from the **deep formula** (exhibition-led score + nudge) vs the legacy composite formula.

### 2. Legacy vs deep: formula effect

- **Exhibition similarity**: Legacy mean 0.677 vs deep ~0.694 → deep is ~0.017 higher. That’s from the deep formula (and possibly from how weights are normalized with extra_weights), not from need.
- **Relevance score**: Legacy mean 0.689 vs deep ~0.757 → deep is ~0.07 higher. Legacy blends query and exhibition; deep is exhibition-led with a bounded nudge, so top matches get a higher final score.

### 3. Spread: more resonance with deep

- **Relevance score std**: Legacy **0.021** vs deep **~0.041**. Deep has about **2× the spread** in the final ranking score.
- So under deep, high matches score higher and lower matches score lower—**more discrimination** and “more resonance” in the sense that the ranking score separates good vs weaker matches more clearly than legacy.
- **Exhibition similarity std**: Legacy 0.045 vs deep 0.037. Legacy has slightly more spread in raw exhibition similarity; deep compresses it a bit, but the **relevance score** (what we rank on) has wider spread under deep.

### 4. Embedding need vs explicit need

- The **embedding** already encodes need (it’s in the text used to build library/exhibition embeddings). So base cosine similarity is “with need” for both legacy and deep.
- The **explicit** need (and emotional_tone) weight in deep only adds a small extra term. Benchmark shows that turning that term off (deep no need) barely changes means. So:
  - Most of the “need” effect is in the **embedding**, shared by all strategies.
  - The **explicit need weight** in deep does not materially inflate scores.

---

## Conclusion

- **Scores are not artificially higher because of the new need embeddings** in the sense of the explicit need weight: deep with need vs deep without need is effectively the same.
- **Higher scores vs legacy** come from the **deep algorithm** (exhibition-led + nudge), not from need.
- **Spread of high-scoring matches**: Under deep, **relevance_score** has **wider spread** than under legacy (std ~0.041 vs ~0.021), so films that match well score more clearly higher and weaker matches lower—i.e. **more resonance** in the ranking.

Full numeric results: `benchmark_need_effect_results.json`.  
To re-run: `python benchmark_need_effect.py`.
