# Latency Reduction: User Prompt → Response

Current end-to-end time is **~30–50 seconds** per prompt. Most time is spent in **sequential API calls**. Below are concrete ways to reduce it, ordered by impact and effort.

---

## Where Time Is Spent (per request)

| Step | What | Approx. time | API |
|------|------|--------------|-----|
| 1 | Intent: new-search vs refinement | ~2–5 s | 1× LLM (gpt-4o, 120 tokens) |
| 2 | Intent: full parse (filters, weights) | ~3–8 s | 1× LLM (gpt-4o, 600 tokens) |
| 3 | Query embedding | ~0.5–2 s | 1× Embedding (text-embedding-3-small) |
| 4 | Scoring (NumPy + enhanced similarity) | &lt;1 s | Local |
| 5 | Reasoning (×5) | ~15–35 s | **5× LLM (gpt-4o-mini, 380 tokens) — sequential** |
| 6 | Posters (×5) | ~2–10 s | **5× TMDB — sequential** |

So the bulk of latency is **5 sequential reasoning calls** and **5 sequential poster fetches**, plus **2 sequential intent LLM calls**.

---

## High-Impact, Low-Effort

### 1. **Parallelize reasoning generation** ✅ (implemented)

- **Current:** 5 × `_generate_dynamic_reasoning()` in a loop → 5× single-call latency.
- **Change:** Run all 5 in parallel with `concurrent.futures.ThreadPoolExecutor`.
- **Effect:** Wall time for reasoning ≈ one call (~4–8 s) instead of five (~20–35 s). **Saves ~15–25 s.**

### 2. **Parallelize poster fetches** ✅ (implemented)

- **Current:** 5 × `_get_poster_url(tmdb_id)` in a loop → 5× TMDB latency.
- **Change:** Fetch all 5 poster URLs in parallel (same executor or separate).
- **Effect:** Poster phase ≈ one round-trip (~0.5–2 s) instead of five (~2–10 s). **Saves ~2–8 s.**

### 3. **Use a faster model for intent**

- **Current:** `get_intent_model()` returns `gpt-4o` (config + `INTENT_MODEL`).
- **Change:** Default to `gpt-4o-mini` for intent, or set `INTENT_MODEL=gpt-4o-mini` in env.
- **Effect:** Intent parsing ~2–3× faster with only small risk to edge-case parsing. **Saves ~3–8 s.**

---

## Medium-Impact, Medium-Effort

### 4. **Run query embedding in parallel with intent parse**

- **Current:** Parse intent (2 LLM calls), then in scoring call `compute_query_sims()` (1 embedding call).
- **Change:** Start both as soon as the user sends the prompt: thread/coroutine A = `query_parser.parse(query)`, thread/coroutine B = `openai.embeddings.create(input=query)`. After both complete, run filtering + scoring (pass precomputed query vector into `compute_ranked_matches`).
- **Effect:** Embedding no longer adds its full delay after intent; overlap saves ~0.5–2 s.

### 5. **Merge the two intent LLM calls into one**

- **Current:** `_classify_new_search()` (1 LLM) then `_llm_parse_query()` (1 LLM).
- **Change:** Single prompt that returns `{ "is_new_search": bool, "reason": str, ...all current intent fields }` in one JSON.
- **Effect:** One round-trip instead of two. **Saves ~2–5 s.**

### 6. **Cache query embedding**

- **Current:** Every request calls `embeddings.create(input=query)`.
- **Change:** In-memory (or Redis) cache keyed by normalized `query.strip().lower()`; reuse vector when present (with optional TTL).
- **Effect:** Repeat queries or near-duplicates are much faster (no embedding call). **Saves ~0.5–2 s for cache hits.**

---

## Lower-Impact / UX

### 7. **Return recommendations first, reasoning later (streaming / lazy)**

- **Current:** Wait for all 5 reasonings (and posters) before returning the response.
- **Change:** Return the list of recommendations immediately with placeholder reasoning (e.g. “Loading…”). In the UI, stream or poll for each reasoning (and poster) as it completes; or fetch reasoning in one background task and append when ready.
- **Effect:** Time to first meaningful content drops to ~10–15 s (intent + embedding + scoring + one reasoning or none). **Perceived latency improves a lot.**

### 8. **Fewer reasonings or shorter replies**

- **Current:** 5 reasonings, ~380 max tokens each.
- **Change:** e.g. top 3 with full reasoning, rest with one line; or reduce `max_tokens` to 200.
- **Effect:** Fewer/smaller LLM calls. **Saves a few seconds** if reasoning stays sequential; if already parallel, saves less.

### 9. **Poster cache**

- **Current:** Every request fetches poster from TMDB by `tmdb_id`.
- **Change:** Cache `tmdb_id → poster_url` in memory or on disk; only call TMDB on cache miss.
- **Effect:** Repeat films return instantly for poster. **Saves ~2–8 s on cache hits for same titles.**

---

## Summary

- **Implemented:** Parallel reasoning (5→1 wall time) and parallel poster fetches (5→1). **Expected savings ~15–30 s** per request.
- **Quick win:** Set `INTENT_MODEL=gpt-4o-mini` for faster intent.
- **Next:** Parallel intent + query embedding, then single-call intent JSON; then streaming/lazy reasoning and caching for further gains.

Target: bring typical response time from **~35 s** to **~10–15 s** with the above changes.
