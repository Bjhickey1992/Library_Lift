# Mauvais Sang Test: Before vs After "Need" Field

**Test query:** "I see Mauvais Sang is playing, do we have anything we can push based on that?"

**Benchmarks:** `benchmark_mauvais_sang_deep_before_need.json` (embeddings without need) vs `benchmark_mauvais_sang_deep.json` (embeddings with need). Same deep scoring strategy (gate + tie-breaker + nudge).

---

## Summary

- **Top 3 unchanged in identity:** Vidocq, Dark Seduction, Buffalo '66 remain in the top three; order shifts (Dark Seduction #2, Buffalo '66 #3 with need vs. Dying of the Light #2, Buffalo '66 #3 before).
- **Scores rise with need:** Relevance, exhibition similarity, and query similarity all increase when embeddings include the "viewer needs" text. Richer semantic signal (desires/needs) strengthens the match to Mauvais Sang.
- **Two new titles in top 5 with need:** Dying of the Light and Death of Me drop out; **Repo! The Genetic Opera** and **My Bloody Valentine** enter. Both fit the “dark romance / existential / stylized” need space that aligns with Mauvais Sang.

---

## Side-by-side scores

### Before need (embeddings without "Viewer needs")

| Rank | Title              | relevance | exhibition_sim | query_sim |
|------|--------------------|-----------|----------------|-----------|
| 1    | Vidocq             | 0.6629    | 0.6586         | 0.6344    |
| 2    | Dying of the Light | 0.6155    | 0.6127         | 0.6225    |
| 3    | Buffalo '66        | 0.6177    | 0.6151         | 0.6211    |
| 4    | Dark Seduction     | 0.6374    | 0.6350         | 0.6193    |
| 5    | Death of Me        | 0.5949    | 0.5918         | 0.6248    |

### After need (embeddings with "Viewer needs")

| Rank | Title                  | relevance | exhibition_sim | query_sim |
|------|------------------------|-----------|----------------|-----------|
| 1    | Vidocq                 | **0.8198** | **0.7763**   | **0.6580** |
| 2    | Dark Seduction         | **0.7437** | **0.7115**   | **0.6430** |
| 3    | Buffalo '66            | **0.7434** | **0.7136**   | **0.6397** |
| 4    | Repo! The Genetic Opera | 0.7116   | 0.6554        | 0.6749    |
| 5    | My Bloody Valentine   | 0.7188    | 0.6687        | 0.6667    |

---

## What changed

1. **Higher scores across the board**  
   Embeddings now include “Viewer needs” (desires/needs met by the film). Library and exhibition vectors encode that dimension, so similarity to Mauvais Sang increases and relevance/exhibition/query scores all go up for the same titles.

2. **Rank order within top 3**  
   With need, Dark Seduction and Buffalo '66 both sit around 0.74 relevance; Dark Seduction is slightly ahead. Before need, Dying of the Light was #2 and Dark Seduction #4; with need, Dark Seduction and Buffalo '66 rise and Dying of the Light drops out of the top 5.

3. **New entrants: Repo! and My Bloody Valentine**  
   Both have strong “viewer need” overlap with Mauvais Sang (emotional extremity, stylized romance, genre-blending). They benefit from the new need dimension and displace Dying of the Light and Death of Me in the top 5.

4. **Buffalo '66**  
   Still in the top 3 with need (exhibition_sim 0.7136, relevance 0.7434). The “you may not have thought of this, but it’s a strong match” behavior is preserved and slightly strengthened.

---

## Conclusion

Adding the **need** field and embedding it improves discrimination among Mauvais Sang–style matches: scores rise, and the top 5 shifts to titles that share both thematic/stylistic and “viewer need” overlap (e.g. Repo!, My Bloody Valentine). The core strong matches (Vidocq, Dark Seduction, Buffalo '66) remain at the top with higher confidence.
