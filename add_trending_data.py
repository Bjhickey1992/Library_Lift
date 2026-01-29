#!/usr/bin/env python
"""
Add IMDB trending and Nielsen streaming data to existing exhibition file.
This script scrapes the new data sources and appends them to the existing file.
"""

import os
import pandas as pd
import numpy as np
from typing import List
from datetime import datetime
from film_agent import ExhibitionScrapingAgent, FilmRecord, asdict
from config import get_openai_api_key, get_tmdb_api_key

def append_to_exhibitions(
    new_records: List[FilmRecord],
    existing_path: str = "upcoming_exhibitions.xlsx",
    output_path: str = "upcoming_exhibitions.xlsx"
) -> pd.DataFrame:
    """
    Append new FilmRecords to existing exhibition file.
    
    Args:
        new_records: List of new FilmRecord objects to add
        existing_path: Path to existing exhibition file
        output_path: Path to save updated file
    
    Returns:
        Updated DataFrame
    """
    # Load existing exhibitions
    if os.path.exists(existing_path):
        print(f"\nLoading existing exhibitions from: {existing_path}")
        existing_df = pd.read_excel(existing_path)
        print(f"  Found {len(existing_df)} existing exhibition films")
    else:
        print(f"\nNo existing exhibition file found. Creating new file.")
        existing_df = pd.DataFrame()
    
    # Convert new records to DataFrame
    new_data = [asdict(r) for r in new_records]
    new_df = pd.DataFrame(new_data)
    
    if len(new_df) == 0:
        print("  No new records to add.")
        return existing_df
    
    print(f"\nAdding {len(new_df)} new records...")
    
    # Combine DataFrames
    if len(existing_df) > 0:
        # Check for duplicates (same title and tmdb_id)
        existing_titles = set(existing_df["title"].astype(str).str.lower())
        existing_tmdb_ids = set(existing_df["tmdb_id"].dropna().astype(int).astype(str))
        
        new_records_filtered = []
        duplicates = 0
        for _, row in new_df.iterrows():
            title_lower = str(row.get("title", "")).lower()
            tmdb_id_str = str(int(row.get("tmdb_id"))) if pd.notna(row.get("tmdb_id")) else None
            
            # Check if duplicate
            is_duplicate = False
            if title_lower in existing_titles:
                if tmdb_id_str and tmdb_id_str in existing_tmdb_ids:
                    is_duplicate = True
                elif title_lower in [str(t).lower() for t in existing_df["title"].values]:
                    is_duplicate = True
            
            if not is_duplicate:
                new_records_filtered.append(row)
            else:
                duplicates += 1
        
        if duplicates > 0:
            print(f"  Skipped {duplicates} duplicate records")
        
        new_df_filtered = pd.DataFrame(new_records_filtered)
        
        if len(new_df_filtered) > 0:
            # Combine
            combined_df = pd.concat([existing_df, new_df_filtered], ignore_index=True)
        else:
            print("  All new records were duplicates. No updates needed.")
            return existing_df
    else:
        combined_df = new_df
    
    # Aggregate by title, country, location, programme_url (same as phase 2)
    def _sorted_unique_dates(series: pd.Series) -> str:
        dates = []
        for v in series.dropna().astype(str).tolist():
            for part in v.split(","):
                p = part.strip()
                if p and p not in dates:
                    dates.append(p)
        dates = sorted(set(dates))
        return ", ".join(dates)
    
    # Get all columns that should be aggregated
    agg_dict = {
        "release_year": "first",
        "director": "first",
        "writers": "first",
        "producers": "first",
        "cinematographers": "first",
        "production_designers": "first",
        "cast": "first",
        "genres": "first",
        "tmdb_id": "first",
        "overview": "first",
        "keywords": "first",
        "tagline": "first",
        "thematic_descriptors": "first",
        "stylistic_descriptors": "first",
        "emotional_tone": "first",
        "scheduled_dates": _sorted_unique_dates,
    }
    
    # Add lead_gender if it exists in the DataFrame
    if "lead_gender" in combined_df.columns:
        agg_dict["lead_gender"] = "first"
    
    grouped = (
        combined_df.groupby(["title", "country", "location", "programme_url"], dropna=False, as_index=False)
        .agg(agg_dict)
    )
    
    # Compute start/end dates
    def _min_date(dates_str: str) -> str:
        parts = [p.strip() for p in (dates_str or "").split(",") if p.strip()]
        return min(parts) if parts else None
    
    def _max_date(dates_str: str) -> str:
        parts = [p.strip() for p in (dates_str or "").split(",") if p.strip()]
        return max(parts) if parts else None
    
    grouped["start_date"] = grouped["scheduled_dates"].apply(_min_date)
    grouped["end_date"] = grouped["scheduled_dates"].apply(_max_date)
    
    # Save updated file
    grouped.to_excel(output_path, index=False)
    print(f"\n[SUCCESS] Updated exhibition file: {output_path}")
    print(f"  Total films: {len(grouped)} (was {len(existing_df)}, added {len(grouped) - len(existing_df)})")
    
    return grouped


def regenerate_embeddings_for_all(exhibitions_df: pd.DataFrame, agent: ExhibitionScrapingAgent):
    """
    Regenerate embeddings for all exhibition films including new additions.
    
    Args:
        exhibitions_df: DataFrame with all exhibition films
        agent: ExhibitionScrapingAgent instance
    """
    if not agent.openai_client:
        print("\n[WARNING] OpenAI client not available - skipping embedding regeneration")
        return
    
    print(f"\n[ExhibitionAgent] Regenerating embeddings for {len(exhibitions_df)} exhibition films...")
    embeddings_path = agent._generate_exhibition_embeddings(exhibitions_df, "upcoming_exhibitions.xlsx")
    print(f"[ExhibitionAgent] Embeddings saved to: {embeddings_path}")


def main():
    """Add IMDB trending and Nielsen streaming data to exhibitions."""
    print("="*80)
    print("ADDING TRENDING DATA TO EXHIBITIONS")
    print("="*80)
    print("This will:")
    print("  1. Scrape IMDB top 50 trending films and TV")
    print("  2. Scrape Nielsen top 10 streaming data")
    print("  3. Append new data to existing upcoming_exhibitions.xlsx")
    print("  4. Regenerate embeddings for all exhibition films")
    print()
    
    # Initialize agent
    tmdb_key = get_tmdb_api_key()
    openai_key = get_openai_api_key()
    os.environ["TMDB_API_KEY"] = tmdb_key
    os.environ["OPENAI_API_KEY"] = openai_key
    
    agent = ExhibitionScrapingAgent(
        tmdb_api_key=tmdb_key,
        openai_api_key=openai_key
    )
    
    # Scrape IMDB trending
    print("\n" + "="*60)
    print("SCRAPING IMDB TRENDING")
    print("="*60)
    imdb_records = agent.scrape_imdb_trending(top_n=50)
    
    # Scrape Nielsen streaming
    print("\n" + "="*60)
    print("SCRAPING NIELSEN STREAMING")
    print("="*60)
    nielsen_records = agent.scrape_nielsen_streaming()
    
    # Combine all new records
    all_new_records = imdb_records + nielsen_records
    print(f"\n[SUMMARY] Scraped {len(imdb_records)} IMDB items and {len(nielsen_records)} Nielsen items")
    print(f"  Total new records: {len(all_new_records)}")
    
    # Append to existing file
    print("\n" + "="*60)
    print("APPENDING TO EXISTING EXHIBITIONS")
    print("="*60)
    updated_df = append_to_exhibitions(all_new_records)
    
    # Regenerate embeddings
    print("\n" + "="*60)
    print("REGENERATING EMBEDDINGS")
    print("="*60)
    regenerate_embeddings_for_all(updated_df, agent)
    
    print("\n" + "="*80)
    print("[SUCCESS] Trending data added to exhibitions!")
    print("="*80)


if __name__ == "__main__":
    main()
