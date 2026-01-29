#!/usr/bin/env python
"""Phase 1: Build Studio Library from TMDb (decade chunks)"""

import os
from film_agent import StudioLibraryAgent
from config import get_openai_api_key, get_tmdb_api_key

# Load API keys from environment variables or .env file
# Keys are loaded via config.py which handles .env file automatically
tmdb_key = get_tmdb_api_key()
openai_key = get_openai_api_key()

# Set in environment for agent initialization
os.environ["TMDB_API_KEY"] = tmdb_key
os.environ["OPENAI_API_KEY"] = openai_key

studio_name = "Universal Pictures"
lib_agent = StudioLibraryAgent(openai_api_key=openai_key)

print("Starting Phase 1: Building studio library...")
print("This will:")
print("  1. Fetch films in decade chunks from TMDb")
print("  2. Add overview, keywords, and tagline for each film")
print("  3. Generate LLM semantic descriptors (themes, style, tone)")
print("  4. Generate embeddings for all films")
print("  5. Save library and embeddings to Excel files\n")

library_df, company_id = lib_agent.build_library_decade_chunks(
    studio_name=studio_name,
    start_year=1940,            # Start from 1940 instead of 1920
    end_year=None,              # defaults to current year
    chunk_size=10,              # 10-year increments
    include_adult=False,
    max_pages_per_chunk=None,   # None = fetch all pages per chunk
    output_prefix=studio_name.lower().replace(" ", "_"),
)

print(f"\n[SUCCESS] Phase 1 complete!")
print(f"  Total films: {len(library_df)}")
print(f"  Company ID: {company_id}")
print(f"  Library saved to: {studio_name.lower().replace(' ', '_')}_library.xlsx")
print(f"  Embeddings saved to: {studio_name.lower().replace(' ', '_')}_library_embeddings.xlsx and .npy")
