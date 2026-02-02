#!/usr/bin/env python
"""Run only the 'need' (Claude) and embeddings (OpenAI) steps on upcoming_exhibitions.xlsx. No scraping."""

import pandas as pd
from pathlib import Path

EXHIBITIONS_PATH = "upcoming_exhibitions.xlsx"

def main():
    ex_path = Path(EXHIBITIONS_PATH)
    if not ex_path.exists():
        print(f"[ERROR] {EXHIBITIONS_PATH} not found. Run Phase 2 scraping first.")
        return

    ex_df = pd.read_excel(ex_path)
    if len(ex_df) == 0:
        print("[ERROR] Exhibition file is empty.")
        return

    # Step 1: Add 'need' via Claude (resumable: skips rows that already have need)
    try:
        from config import get_anthropic_api_key
        from add_need_field import add_need_to_dataframe
        from anthropic import Anthropic

        try:
            api_key = get_anthropic_api_key()
        except ValueError:
            api_key = None
        if api_key:
            print("=" * 60)
            print("ADDING 'NEED' FIELD (Claude)")
            print("=" * 60)
            claude = Anthropic(api_key=api_key)
            ex_df = add_need_to_dataframe(
                ex_df, "title", claude, "exhibitions",
                save_path=ex_path, save_every=50,
            )
            ex_df.to_excel(ex_path, index=False)
            print(f"   Saved {ex_path}")
        else:
            print("[WARN] ANTHROPIC_API_KEY not set; skipping 'need' field.")
    except Exception as e:
        print(f"[WARN] Need step failed: {e}")

    # Step 2: Generate embeddings via OpenAI
    try:
        from config import get_openai_api_key
        from generate_exhibition_embeddings import generate_exhibition_embeddings

        if get_openai_api_key():
            print("\n" + "=" * 60)
            print("GENERATING EMBEDDINGS (OpenAI)")
            print("=" * 60)
            generate_exhibition_embeddings(
                exhibitions_path=EXHIBITIONS_PATH,
                npy_path="upcoming_exhibitions_embeddings.npy",
                metadata_xlsx_path="upcoming_exhibitions_embeddings.xlsx",
            )
            print("   Saved upcoming_exhibitions_embeddings.npy and .xlsx")
        else:
            print("[WARN] OPENAI_API_KEY not set; skipping embeddings.")
    except Exception as e:
        print(f"[ERROR] Embeddings step failed: {e}")
        return

    print("\n[SUCCESS] Need and embeddings complete.")


if __name__ == "__main__":
    main()
