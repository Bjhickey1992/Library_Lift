#!/usr/bin/env python
"""Run full Phase 2 pipeline for a specified exhibitions file."""

import argparse
import os
from pathlib import Path

import pandas as pd

from add_need_field import add_need_to_dataframe
from config import (
    get_anthropic_api_key,
    get_openai_api_key,
    get_tmdb_api_key,
)
from film_agent import ExhibitionScrapingAgent
from generate_exhibition_embeddings import generate_exhibition_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape exhibitions, add need, and generate embeddings for a target file."
    )
    parser.add_argument(
        "--exhibitions-path",
        required=True,
        help="Target exhibitions xlsx file path to create/update.",
    )
    parser.add_argument(
        "--embeddings-npy-path",
        required=True,
        help="Target .npy path for exhibition embeddings.",
    )
    parser.add_argument(
        "--embeddings-xlsx-path",
        required=True,
        help="Target metadata .xlsx path for exhibition embeddings.",
    )
    parser.add_argument(
        "--weeks-ahead",
        type=int,
        default=4,
        help="Number of weeks to scrape ahead (default: 4).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Force fresh rebuild of target exhibitions file.",
    )
    args = parser.parse_args()

    os.environ["TMDB_API_KEY"] = get_tmdb_api_key()
    os.environ["OPENAI_API_KEY"] = get_openai_api_key()

    ex_path = Path(args.exhibitions_path)
    print(f"[Phase2 custom] Target exhibitions file: {ex_path}")

    agent = ExhibitionScrapingAgent(openai_api_key=os.getenv("OPENAI_API_KEY"))
    exhibitions_df = agent.build_exhibitions_progressively(
        cinemas_yaml_path="cinemas.yaml",
        weeks_ahead=args.weeks_ahead,
        output_path=str(ex_path),
        fresh=args.fresh,
    )

    if exhibitions_df is None or len(exhibitions_df) == 0:
        print("[Phase2 custom] No exhibition rows returned. Stopping.")
        return

    print(f"[Phase2 custom] Scrape + TMDB enrichment complete: {len(exhibitions_df)} rows")

    # Add "need" field using Claude.
    anthropic_key = get_anthropic_api_key()
    if anthropic_key:
        from anthropic import Anthropic

        print("[Phase2 custom] Populating 'need' with Claude...")
        claude = Anthropic(api_key=anthropic_key)
        ex_df = pd.read_excel(ex_path)
        ex_df = add_need_to_dataframe(
            ex_df,
            "title",
            claude,
            "exhibitions",
            save_path=ex_path,
            save_every=50,
        )
        ex_df.to_excel(ex_path, index=False)
        print(f"[Phase2 custom] Saved need updates to {ex_path}")
    else:
        print("[Phase2 custom] ANTHROPIC_API_KEY missing; skipping need step.")

    # Generate embeddings for this exact file.
    print("[Phase2 custom] Generating embeddings...")
    arr = generate_exhibition_embeddings(
        exhibitions_path=str(ex_path),
        npy_path=args.embeddings_npy_path,
        metadata_xlsx_path=args.embeddings_xlsx_path,
    )
    print(
        f"[Phase2 custom] Embeddings saved: {args.embeddings_npy_path} and "
        f"{args.embeddings_xlsx_path} (shape={arr.shape})"
    )
    print("[Phase2 custom] Complete.")


if __name__ == "__main__":
    main()
