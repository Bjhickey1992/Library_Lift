#!/usr/bin/env python
"""Phase 2: Exhibition Scraping (progressive, cinema by cinema)"""

import os
from film_agent import ExhibitionScrapingAgent

# Load API keys from environment variables or .env file
from config import get_openai_api_key, get_tmdb_api_key

tmdb_key = get_tmdb_api_key()
openai_key = get_openai_api_key()

# Set in environment for compatibility
os.environ["TMDB_API_KEY"] = tmdb_key
os.environ["OPENAI_API_KEY"] = openai_key

exhibition_agent = ExhibitionScrapingAgent(openai_api_key=os.getenv("OPENAI_API_KEY"))

print("Starting Phase 2: Exhibition Scraping...")
print("This will:")
print("  1. Scrape each cinema one by one")
print("  2. Scrape IMDB top 50 trending films and TV")
print("  3. Scrape Nielsen top 10 streaming data (streaming only)")
print("  4. Add overview data from TMDB API")
print("  5. Generate embeddings for all exhibition films")
print("  6. Save exhibitions and embeddings to files\n")

# Scrape cinema exhibitions
exhibitions_df = exhibition_agent.build_exhibitions_progressively(
    cinemas_yaml_path="cinemas.yaml",
    weeks_ahead=4,
    output_path="upcoming_exhibitions.xlsx",
)

# Scrape IMDB trending
print("\n" + "="*60)
print("SCRAPING IMDB TRENDING")
print("="*60)
imdb_records = exhibition_agent.scrape_imdb_trending(top_n=50)

# Scrape Nielsen streaming
print("\n" + "="*60)
print("SCRAPING NIELSEN STREAMING")
print("="*60)
nielsen_records = exhibition_agent.scrape_nielsen_streaming()

# Append trending data to exhibitions
if imdb_records or nielsen_records:
    print("\n" + "="*60)
    print("APPENDING TRENDING DATA TO EXHIBITIONS")
    print("="*60)
    from add_trending_data import append_to_exhibitions, regenerate_embeddings_for_all
    
    all_trending_records = imdb_records + nielsen_records
    updated_df = append_to_exhibitions(all_trending_records)
    
    # Regenerate embeddings for all
    regenerate_embeddings_for_all(updated_df, exhibition_agent)
    exhibitions_df = updated_df

print(f"\n[SUCCESS] Phase 2 complete!")
print(f"  Total exhibition films: {len(exhibitions_df)}")
print(f"  Saved to: upcoming_exhibitions.xlsx")
print(f"  Embeddings saved to: upcoming_exhibitions_embeddings.xlsx and .npy")
