# Similarity Matching Improvements

## Problem Statement

The current matching system sometimes produces good thematic/stylistic matches (like "Women in Film" ↔ "Othello" in row 399) but assigns them low cosine similarity scores (0.4266). This happens because the current embedding representation focuses primarily on factual metadata (director, cast, genres) rather than deeper semantic content like themes, style, and emotional tone.

## Solution: Enhanced Semantic Enrichment

### 1. Added TMDB Fields
- **Keywords**: Extracted from TMDB API (`/movie/{id}/keywords`)
- **Tagline**: Film tagline from TMDB (often captures thematic essence)

### 2. LLM-Generated Semantic Descriptors
For each film, we now use OpenAI to generate:
- **Thematic Descriptors**: 3-5 key themes (e.g., "alienation, urban isolation, existential crisis, betrayal")
- **Stylistic Descriptors**: 1-2 sentences about cinematic style (e.g., "minimalist cinematography, slow-paced, contemplative")
- **Emotional Tone**: 1 sentence describing emotional atmosphere (e.g., "melancholic, introspective, existential")

### 3. Enhanced Embedding Text Representation

**Before:**
```
Title: Buffalo '66
Year: 1998
Director: Vincent Gallo
Genres: Drama, Comedy
Overview: A man kidnaps a woman...
```

**After:**
```
Film: Buffalo '66
Released: 1998
Tagline: Love is a stranger...
Directed by: Vincent Gallo
Genres: Drama, Comedy
Keywords: alienation, urban life, dysfunctional family
Plot: A man kidnaps a woman...
Themes: alienation, urban isolation, existential crisis, psychological complexity
Style: minimalist cinematography, slow-paced, contemplative, non-linear narrative
Tone: melancholic, introspective, darkly humorous, existential
```

## How This Improves Matching

1. **Thematic Similarity**: Films with similar themes (betrayal, isolation, etc.) will have higher similarity even if they have different directors/cast
2. **Stylistic Similarity**: Films with similar cinematic styles will cluster together
3. **Emotional Resonance**: Films with similar emotional tones will be matched
4. **Better Context**: The richer text representation gives the embedding model more semantic information to work with

## Implementation Details

### Phase 1 (Library Building)
- Fetches keywords and tagline from TMDB
- Uses LLM to generate semantic descriptors for each film
- Includes all fields in embedding text representation

### Phase 2 (Exhibition Scraping)
- Same enrichment process for exhibition films
- Ensures consistent representation for matching

### Phase 3 (Matching)
- Uses enriched embeddings for both library and exhibition films
- Cosine similarity now captures deeper semantic relationships

## Expected Results

Matches like "Women in Film" ↔ "Othello" should now have:
- **Higher cosine similarity** (closer to 0.6-0.7 instead of 0.4)
- **Better ranking** in top 3 matches
- **More accurate thematic connections** in LLM reasoning

## Next Steps

1. Re-run Phase 1 to regenerate library with semantic descriptors
2. Re-run Phase 2 to regenerate exhibitions with semantic descriptors  
3. Re-run Phase 3 to see improved matches
4. Compare results - matches like row 399 should now rank higher
