#!/usr/bin/env python
"""Phase 2: Exhibition Scraping (cinema only). After scraping: add 'need' via Claude, then OpenAI embeddings."""

import argparse
import os
import pandas as pd
from pathlib import Path

from film_agent import ExhibitionScrapingAgent
from config import get_openai_api_key, get_tmdb_api_key

EXHIBITIONS_PATH = "upcoming_exhibitions.xlsx"

parser = argparse.ArgumentParser(description="Phase 2: Scrape exhibitions, add need, generate embeddings.")
parser.add_argument(
    "--fresh",
    action="store_true",
    help="Generate a completely new exhibition file (do not resume from existing file).",
)
parser.add_argument(
    "--scrape-only",
    action="store_true",
    help="Only run the scrape; do not add 'need' or generate embeddings. Pause for review.",
)
parser.add_argument(
    "--add-need",
    action="store_true",
    help="Run the expensive 'need' (Claude) step. Omit unless you have reviewed the file and want to populate need.",
)
args = parser.parse_args()

tmdb_key = get_tmdb_api_key()
openai_key = get_openai_api_key()
os.environ["TMDB_API_KEY"] = tmdb_key
os.environ["OPENAI_API_KEY"] = openai_key

exhibition_agent = ExhibitionScrapingAgent(openai_api_key=os.getenv("OPENAI_API_KEY"))

if args.fresh:
    print("Phase 2: FRESH RUN (completely new file, no resume).\n")
    for p in [EXHIBITIONS_PATH, "upcoming_exhibitions_embeddings.npy", "upcoming_exhibitions_embeddings.xlsx"]:
        if Path(p).exists():
            Path(p).unlink()
            print(f"  Removed existing {p}")

print("Starting Phase 2: Exhibition Scraping (cinema only)...")
print("This will:")
print("  1. Scrape each cinema one by one from cinemas.yaml, save upcoming_exhibitions.xlsx")
if args.add_need:
    print("  2. Add 'need' field via Claude LLM (viewer desires/needs per title) [--add-need]")
    print("  3. Generate embeddings via OpenAI for the full file, save .npy and metadata .xlsx")
else:
    print("  2. Generate embeddings via OpenAI (no 'need' step; use --add-need when ready)")
print()

# Step 1: Scrape cinema exhibitions. Use --fresh for a completely new file; otherwise resumable.
exhibitions_df = exhibition_agent.build_exhibitions_progressively(
    cinemas_yaml_path="cinemas.yaml",
    weeks_ahead=4,
    output_path=EXHIBITIONS_PATH,
    fresh=args.fresh,
)

if exhibitions_df is None or len(exhibitions_df) == 0:
    print("\n[Phase 2] No exhibition data; skipping need and embeddings.")
elif args.scrape_only:
    print(f"\n[Phase 2] Scrape complete (--scrape-only). Review {EXHIBITIONS_PATH} ({len(exhibitions_df)} films).")
    print("When ready, run without --scrape-only to generate embeddings; add --add-need only when you want to run the expensive need step.")
    raise SystemExit(0)
else:
    # Step 2: Add 'need' field via Claude (only when --add-need; expensive)
    if args.add_need:
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
    else:
        print("\n[Phase 2] Skipping 'need' step (use --add-need when ready).")

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
