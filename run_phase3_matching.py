#!/usr/bin/env python
"""Phase 3: Matching Library and Exhibition Films (embeddings + cosine similarity + LLM reasoning)"""

import os
from film_agent import MatchingAgent
from config import get_openai_api_key

# Load API keys from environment variables or .env file
openai_key = get_openai_api_key()

# Set in environment for agent initialization
os.environ["OPENAI_API_KEY"] = openai_key

# Initialize matching agent
matching_agent = MatchingAgent()

print("Starting Phase 3: Matching Library and Exhibition Films")
print("This will:")
print("  1. Load prebuilt library Excel (with semantic descriptors)")
print("  2. Load exhibition Excel (with semantic descriptors)")
print("  3. Load pre-saved embeddings for both (from .npy files)")
print("  4. Calculate cosine similarity (using enriched embeddings)")
print("  5. Find top 3 matches per exhibition film")
print("  6. Generate LLM reasoning for each match")
print()

# File paths
studio_name = "Lionsgate"
library_path = f"{studio_name.lower().replace(' ', '_')}_library.xlsx"
exhibitions_path = "upcoming_exhibitions.xlsx"
library_embeddings_path = f"{studio_name.lower().replace(' ', '_')}_library_embeddings.npy"
exhibition_embeddings_path = "upcoming_exhibitions_embeddings.npy"

# Run matching with pre-saved embeddings
matches_df = matching_agent.match_library_and_exhibitions(
    library_path=library_path,
    exhibitions_path=exhibitions_path,
    studio_name=studio_name,
    library_embeddings_path=library_embeddings_path,
    exhibition_embeddings_path=exhibition_embeddings_path,
)

print(f"\n[SUCCESS] Phase 3 complete!")
print(f"  Total matches: {len(matches_df)}")
print(f"  Output file: {studio_name.lower().replace(' ', '_')}_matches.xlsx")
print(f"  (Top 3 library matches for each exhibition film)")
