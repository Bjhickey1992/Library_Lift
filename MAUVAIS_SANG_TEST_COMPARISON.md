# Mauvais Sang Test: Current Run vs Last Benchmark

**Test query:** "I see Mauvais Sang is playing, do we have anything we can push based on that?"

---

## Important: Exhibition filter difference

- **Last benchmark** (benchmark_mauvais_sang_deep.json): Filtering by `match_to_specific_film=Mauvais Sang` found **1 exhibition** (Mauvais Sang). All 5 recommendations were scored against that single exhibition.
- **Current run**: Filtering by `match_to_specific_film=Mauvais Sang` yielded **0 exhibitions** (from 37 total in this run’s exhibition set). The fallback removed the match-to-film filter and used **all exhibitions**, so recommendations were matched to various titles (Blue Moon, All That Heaven Allows, Hamlet, etc.), not specifically to Mauvais Sang.

So the two runs are not strictly apples-to-apples: the previous run was “match library to Mauvais Sang”; the current run was “match library to current market (no Mauvais Sang in filtered set).” If Mauvais Sang is no longer in the exhibition file or the title differs slightly, that would explain 0 exhibitions.

---

## Top 5 comparison

| Rank | **Last benchmark (deep, match to Mauvais Sang)** | **Current run (deep, fallback to all exhibitions)** |
|------|--------------------------------------------------|-----------------------------------------------------|
| 1 | Vidocq (2001) — Mauvais Sang | Daybreakers (2010) — Blue Moon |
| 2 | Dark Seduction (2010) — Mauvais Sang | Love in Taipei (2023) — All That Heaven Allows |
| 3 | Buffalo '66 (1998) — Mauvais Sang | The Haunting in Connecticut (2009) — Hamlet |
| 4 | Repo! The Genetic Opera (2008) — Mauvais Sang | Vampire Assassin (2005) — Deepfaking Sam Altman |
| 5 | My Bloody Valentine (2009) — Mauvais Sang | Buffalo '66 (1998) — All That Heaven Allows |

**Overlap:** Only **Buffalo '66** appears in both top 5s (rank 3 in the previous run, rank 5 in the current run).

---

## Score comparison (where comparable)

### Last benchmark (all matched to Mauvais Sang)

| Title | Relevance | Exhibition sim | Query sim |
|-------|-----------|----------------|-----------|
| Vidocq | 0.8226 | 0.7791 | 0.6580 |
| Dark Seduction | 0.7450 | 0.7127 | 0.6430 |
| Buffalo '66 | 0.7463 | 0.7165 | 0.6397 |
| Repo! The Genetic Opera | 0.7135 | 0.6573 | 0.6749 |
| My Bloody Valentine | 0.7199 | 0.6698 | 0.6667 |

**Means:** relevance 0.749, exhibition_sim 0.707, query_sim 0.656.

### Current run (matched to various exhibitions)

| Title | Relevance | Exhibition sim | Query sim | Matched exhibition |
|-------|-----------|----------------|-----------|---------------------|
| Daybreakers | 0.7730 | 0.7226 | 0.6673 | Blue Moon |
| Love in Taipei | 0.7671 | 0.7671 | 0.5889 | All That Heaven Allows |
| The Haunting in Connecticut | 0.7454 | 0.7173 | 0.6374 | Hamlet |
| Vampire Assassin | 0.7086 | 0.6576 | 0.6679 | Deepfaking Sam Altman |
| Buffalo '66 | 0.7097 | 0.6799 | 0.6397 | All That Heaven Allows |

**Means:** relevance 0.741, exhibition_sim 0.721, query_sim 0.640.

---

## Analysis

1. **Exhibition filter:** The current run did not find Mauvais Sang in the exhibition set (0 after filter), so the pipeline used the fallback and matched to the full exhibition set. To compare “match to Mauvais Sang” again, ensure Mauvais Sang is present in `upcoming_exhibitions.xlsx` (or equivalent) with a title that matches the filter (e.g. “Mauvais Sang” or “mauvais sang”).

2. **Scores:** With fallback, mean relevance (0.741) is close to the previous run (0.749); mean exhibition_similarity is slightly higher in the current run (0.721 vs 0.707) because matches are to the best-matching exhibition in the full set, not only to Mauvais Sang.

3. **Buffalo '66:** Present in both top 5s. In the last benchmark it had exhibition_sim 0.7165 (vs Mauvais Sang); in the current run 0.6799 (vs All That Heaven Allows). So when the target is Mauvais Sang, Buffalo '66 scores higher on exhibition similarity than when it is matched to All That Heaven Allows—consistent with stronger thematic/style overlap with Mauvais Sang.

4. **Reasoning:** Current run uses quoted film titles (e.g. "Daybreakers", "Mauvais Sang") in reasoning, matching the requested format (no bold markdown for titles).

---

## Conclusion

- The **only like-for-like comparison** is Buffalo '66: same film, different matched exhibition (Mauvais Sang vs All That Heaven Allows). Exhibition similarity is higher when the match is to Mauvais Sang (0.7165 vs 0.6799).
- For a true re-run of the “match to Mauvais Sang” test, ensure Mauvais Sang appears in the exhibition data and that the title string matches what the filter uses (e.g. “Mauvais Sang”). Then re-run; the top 5 and scores should again be comparable to the last benchmark.

Current run results saved in: `benchmark_mauvais_sang_current_run.json`.
