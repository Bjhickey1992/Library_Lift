"""
Enrich existing library and exhibition files with lead gender information.
This script adds the 'lead_gender' field to existing Excel files without rebuilding everything.
"""

import os
import pandas as pd
from typing import Optional
from film_agent import TMDbClient
from config import get_tmdb_api_key

def get_lead_gender_from_tmdb(tmdb_id: Optional[int], tmdb_client: TMDbClient) -> Optional[str]:
    """
    Get the gender of the lead actor (first cast member) from TMDb.
    
    Returns:
        "female", "male", or None if not available
    """
    if not tmdb_id or pd.isna(tmdb_id):
        return None
    
    try:
        # Get movie details with credits
        details = tmdb_client.get_movie_details_with_credits(int(tmdb_id))
        credits = details.get("credits", {})
        cast = credits.get("cast", [])
        
        if not cast:
            return None
        
        # Get first cast member (the lead)
        lead_actor = cast[0]
        person_id = lead_actor.get("id")
        
        if not person_id:
            return None
        
        # Get person details to check gender
        person_details = tmdb_client.get_person_details(person_id)
        person_gender = person_details.get("gender", 0)
        
        # TMDb gender codes: 0 = not specified, 1 = female, 2 = male
        if person_gender == 1:
            return "female"
        elif person_gender == 2:
            return "male"
        else:
            return None
            
    except Exception as e:
        print(f"  Warning: Could not get lead gender for TMDb ID {tmdb_id}: {e}")
        return None


def enrich_file_with_lead_gender(file_path: str, tmdb_client: TMDbClient, output_path: Optional[str] = None):
    """
    Add lead_gender column to an existing Excel file.
    
    Args:
        file_path: Path to the Excel file to enrich
        tmdb_client: TMDb client for API calls
        output_path: Optional output path (defaults to overwriting input file)
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return
    
    print(f"\nLoading {file_path}...")
    df = pd.read_excel(file_path)
    
    # Check if lead_gender already exists
    if "lead_gender" in df.columns:
        print(f"  File already has 'lead_gender' column. Checking for missing values...")
        missing_count = df["lead_gender"].isna().sum()
        if missing_count == 0:
            print(f"  All films already have lead_gender data. Skipping.")
            return
        print(f"  Found {missing_count} films with missing lead_gender. Enriching...")
    else:
        print(f"  Adding 'lead_gender' column...")
        df["lead_gender"] = None
        missing_count = len(df)
    
    # Enrich films with missing lead_gender
    enriched = 0
    skipped = 0
    
    for idx, row in df.iterrows():
        if pd.notna(row.get("lead_gender")):
            continue  # Already has gender data
        
        tmdb_id = row.get("tmdb_id")
        lead_gender = get_lead_gender_from_tmdb(tmdb_id, tmdb_client)
        
        if lead_gender:
            df.at[idx, "lead_gender"] = lead_gender
            enriched += 1
        else:
            skipped += 1
        
        # Progress indicator
        if (enriched + skipped) % 50 == 0:
            print(f"  Processed {enriched + skipped}/{missing_count} films... (enriched: {enriched}, skipped: {skipped})")
    
    # Save enriched file
    output_path = output_path or file_path
    df.to_excel(output_path, index=False)
    print(f"\n[SUCCESS] Enriched {enriched} films with lead_gender")
    print(f"  Skipped {skipped} films (no TMDb ID or gender unavailable)")
    print(f"  Saved to: {output_path}")


def main():
    """Enrich library and exhibition files with lead gender."""
    try:
        tmdb_client = TMDbClient(tmdb_api_key=get_tmdb_api_key())
    except Exception as e:
        print(f"Error initializing TMDb client: {e}")
        return
    
    # Enrich library file
    library_file = "universal_pictures_library.xlsx"
    if os.path.exists(library_file):
        print(f"\n{'='*60}")
        print(f"Enriching library file: {library_file}")
        print(f"{'='*60}")
        enrich_file_with_lead_gender(library_file, tmdb_client)
    else:
        print(f"Library file not found: {library_file}")
    
    # Enrich exhibition file
    exhibition_file = "upcoming_exhibitions.xlsx"
    if os.path.exists(exhibition_file):
        print(f"\n{'='*60}")
        print(f"Enriching exhibition file: {exhibition_file}")
        print(f"{'='*60}")
        enrich_file_with_lead_gender(exhibition_file, tmdb_client)
    else:
        print(f"Exhibition file not found: {exhibition_file}")
    
    print(f"\n{'='*60}")
    print("Enrichment complete!")
    print("Next step: Regenerate embeddings to include lead_gender in the embedding text.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
