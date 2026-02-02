# Mauvais Sang Test: Before vs After Exhibition Rebuild

**Test prompt:** "I see Mauvais Sang is playing, do we have anything we can push based on that?"  
**Scoring:** Deep (gate + tie-breaker + nudge) in both runs.

- **Before rebuild:** Last successful run with previous exhibition file (with need data).  
- **After rebuild:** Current run with recreated exhibition file (525 rows, need + embeddings regenerated).

---

## Before rebuild (benchmark_mauvais_sang_before_rebuild.json)

| Rank | Title | Relevance | Ex. sim | Query sim |
|------|--------|-----------|---------|-----------|
| 1 | Vidocq | 0.8226 | 0.7791 | 0.6580 |
| 2 | Dark Seduction | 0.7450 | 0.7127 | 0.6430 |
| 3 | Buffalo '66 | 0.7463 | 0.7165 | 0.6397 |
| 4 | Repo! The Genetic Opera | 0.7135 | 0.6573 | 0.6749 |
| 5 | My Bloody Valentine | 0.7199 | 0.6698 | 0.6667 |

---

## After rebuild (benchmark_mauvais_sang_deep.json – current run)

| Rank | Title | Relevance | Ex. sim | Query sim |
|------|--------|-----------|---------|-----------|
| 1 | Buffalo '66 | 0.8105 | 0.7807 | 0.6397 |
| 2 | Vidocq | 0.7886 | 0.7451 | 0.6580 |
| 3 | Dark Seduction | 0.7569 | 0.7246 | 0.6430 |
| 4 | Peacock | 0.7375 | 0.7058 | 0.6422 |
| 5 | My Bloody Valentine | 0.7195 | 0.6694 | 0.6667 |

---

## Comparison summary

1. **Order change:** Buffalo '66 moved from #3 to #1. Its exhibition similarity increased (0.7165 → 0.7807) with the new embeddings, so the tie-breaker/nudge favored it over Vidocq.
2. **Vidocq** dropped to #2; relevance and exhibition similarity are slightly lower (0.8226→0.7886, 0.7791→0.7451), likely due to embedding differences with the new need-inclusive exhibition data.
3. **Dark Seduction** stayed in the top three (#2→#3) with a small relevance gain (0.7450→0.7569).
4. **Repo! The Genetic Opera** left the top 5; **Peacock** entered at #4 with strong exhibition similarity (0.7058).
5. **My Bloody Valentine** stayed #5 with almost the same scores.

**Conclusion:** The same core “Mauvais Sang” matches (Buffalo '66, Vidocq, Dark Seduction, My Bloody Valentine) remain in the top 5. The new exhibition embeddings (with need) changed exhibition similarities enough to reorder the top (Buffalo '66 first) and swap Repo for Peacock. The test still finds Mauvais Sang in the exhibition file and returns strong, comparable recommendations.
