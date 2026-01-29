# Enhanced Similarity Framework

## Problem
Matches like "Buffalo '66" ↔ "The Killing of a Chinese Bookie" have good thematic/stylistic connections but only score 0.5158 cosine similarity. We need to boost these types of matches.

## Current Issues
1. Single embedding captures everything but doesn't emphasize specific connections
2. No weighting for different types of similarities (thematic vs. personnel vs. style)
3. No post-processing boost for explicit shared attributes (common actors, explicit theme matches)

## Solution: Multi-Dimensional Weighted Similarity

### Approach 1: Enhanced Embedding Text with Explicit Connections
- Emphasize notable cast members more prominently
- Add explicit "connection hints" in embedding text
- Example: "Notable cast: Ben Gazzara. Themes: alienation, moral ambiguity, urban decay..."

### Approach 2: Multi-Dimensional Similarity Calculation
Calculate separate similarity scores for:
1. **Thematic Similarity** (40% weight): Based on thematic_descriptors, emotional_tone, keywords
2. **Stylistic Similarity** (30% weight): Based on stylistic_descriptors
3. **Personnel Similarity** (20% weight): Based on cast, director, cinematographer
4. **Base Embedding Similarity** (10% weight): Current cosine similarity

Final score = weighted sum of all components

### Approach 3: Post-Processing Similarity Boost
After calculating base similarity, apply boosts for:
- **Common Notable Actors**: +0.05 to +0.10 per shared actor (especially if mentioned in both)
- **Explicit Theme Matches**: +0.03 to +0.08 if themes overlap significantly
- **Explicit Style Matches**: +0.02 to +0.05 if stylistic descriptors overlap
- **Shared Keywords**: +0.01 to +0.03 per significant keyword match

### Approach 4: Separate Embeddings for Different Aspects
Create 3 separate embeddings per film:
1. **Thematic Embedding**: themes, tone, keywords, overview
2. **Stylistic Embedding**: style descriptors, cinematography, production design
3. **Personnel Embedding**: cast, director, writers

Then calculate:
- Thematic similarity (40%)
- Stylistic similarity (30%)
- Personnel similarity (20%)
- Combined base similarity (10%)

## Recommended Implementation
**Hybrid Approach**: Combine Approach 2 (multi-dimensional) + Approach 3 (post-processing boost)

This gives us:
1. Better semantic understanding through weighted components
2. Explicit recognition of shared attributes
3. Maintains efficiency (no need for 3x embeddings)
