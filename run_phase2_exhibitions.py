#!/usr/bin/env python
"""Phase 2: Exhibition Scraping (cinema only). After scraping: add 'need' via Claude, then OpenAI embeddings."""

import os
import pandas as pd
from pathlib import Path

from film_agent import ExhibitionScrapingAgent
from config import get_openai_api_key, get_tmdb_api_key

EXHIBITIONS_PATH = "upcoming_exhibitions.xlsx"

tmdb_key = get_tmdb_api_key()
openai_key = get_openai_api_key()
os.environ["TMDB_API_KEY"] = tmdb_key
os.environ["OPENAI_API_KEY"] = openai_key

exhibition_agent = ExhibitionScrapingAgent(openai_api_key=os.getenv("OPENAI_API_KEY"))

print("Starting Phase 2: Exhibition Scraping (cinema only)...")
print("This will:")
print("  1. Scrape each cinema one by one from cinemas.yaml, save upcoming_exhibitions.xlsx")
print("  2. Add 'need' field via Claude LLM (viewer desires/needs per title)")
print("  3. Generate embeddings via OpenAI for the full file, save .npy and metadata .xlsx\n")

# Step 1: Scrape cinema exhibitions only. Resumable: re-run to continue from existing file.
exhibitions_df = exhibition_agent.build_exhibitions_progressively(
    cinemas_yaml_path="cinemas.yaml",
    weeks_ahead=4,
    output_path=EXHIBITIONS_PATH,
)

if exhibitions_df is None or len(exhibitions_df) == 0:
    print("\n[Phase 2] No exhibition data; skipping need and embeddings.")
else:
    # Step 2: Add 'need' field via Claude
    try:
        from config import get_anthropic_api_key
        from add_need_field import add_need_to_dataframe
        from anthropic import Anthropic

        try:
            api_key = get_anthropic_api_key()
        except ValueError:
            api_key = None
        if api_key:
            print("\n" + "=" * 60)
            print("ADDING 'NEED' FIELD (Claude)")
            print("=" * 60)
            claude = Anthropic(api_key=api_key)
            ex_path = Path(EXHIBITIONS_PATH)
            ex_df = pd.read_excel(ex_path)
            ex_df = add_need_to_dataframe(
                ex_df, "title", claude, "exhibitions",
                save_path=ex_path, save_every=50,
            )
            ex_df.to_excel(ex_path, index=False)
            print(f"   Saved {ex_path}")
        else:
            print("\n[Phase 2] ANTHROPIC_API_KEY not set; skipping 'need' field.")
    except Exception as e:
        print(f"\n[Phase 2] Need step failed: {e} (skipping)")

    # Step 3: Generate embeddings via OpenAI for the entire file
    try:
        if get_openai_api_key():
            print("\n" + "=" * 60)
            print("GENERATING EMBEDDINGS (OpenAI)")
            print("=" * 60)
            from generate_exhibition_embeddings import generate_exhibition_embeddings
            generate_exhibition_embeddings(
                exhibitions_path=EXHIBITIONS_PATH,
                npy_path="upcoming_exhibitions_embeddings.npy",
                metadata_xlsx_path="upcoming_exhibitions_embeddings.xlsx",
            )
            print("   Saved upcoming_exhibitions_embeddings.npy and .xlsx")
        else:
            print("\n[Phase 2] OPENAI_API_KEY not set; skipping embeddings.")
    except Exception as e:
        print(f"\n[Phase 2] Embeddings step failed: {e}")

print(f"\n[SUCCESS] Phase 2 complete!")
print(f"  Exhibition file: {EXHIBITIONS_PATH} ({len(exhibitions_df) if exhibitions_df is not None else 0} films)")
print(f"  Embeddings: upcoming_exhibitions_embeddings.npy and .xlsx")
