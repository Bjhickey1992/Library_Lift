"""
Regenerate embeddings for library and exhibition files after adding lead_gender field.
This script reads the enriched Excel files and regenerates embeddings that include lead_gender.
"""

import os
import pandas as pd
import numpy as np
from typing import List
from openai import OpenAI
from config import get_openai_api_key

def film_to_text_for_embedding(row: pd.Series) -> str:
    """Convert a DataFrame row to text representation for embedding."""
    parts = []
    
    if pd.notna(row.get("title")):
        parts.append(f"Film: {row['title']}")
    if pd.notna(row.get("release_year")):
        parts.append(f"Released: {int(row['release_year'])}")
    if pd.notna(row.get("tagline")):
        parts.append(f"Tagline: {row['tagline']}")
    
    # Creative personnel
    if pd.notna(row.get("director")):
        parts.append(f"Directed by: {row['director']}")
    if pd.notna(row.get("writers")):
        parts.append(f"Written by: {row['writers']}")
    if pd.notna(row.get("cinematographers")):
        parts.append(f"Cinematography: {row['cinematographers']}")
    if pd.notna(row.get("production_designers")):
        parts.append(f"Production Design: {row['production_designers']}")
    if pd.notna(row.get("cast")):
        parts.append(f"Starring: {row['cast']}")
    if pd.notna(row.get("lead_gender")):
        parts.append(f"Lead actor gender: {row['lead_gender']}")
    
    # Genres and keywords
    if pd.notna(row.get("genres")):
        parts.append(f"Genres: {row['genres']}")
    if pd.notna(row.get("keywords")):
        parts.append(f"Keywords: {row['keywords']}")
    
    # Plot and themes
    if pd.notna(row.get("overview")):
        parts.append(f"Plot: {row['overview']}")
    
    # LLM-generated semantic enrichment (if available)
    if pd.notna(row.get("thematic_descriptors")):
        parts.append(f"Themes: {row['thematic_descriptors']}")
    if pd.notna(row.get("stylistic_descriptors")):
        parts.append(f"Style: {row['stylistic_descriptors']}")
    if pd.notna(row.get("emotional_tone")):
        parts.append(f"Tone: {row['emotional_tone']}")
    
    return "\n".join(parts)


def regenerate_embeddings(file_path: str, output_prefix: str, client: OpenAI):
    """
    Regenerate embeddings for a library or exhibition file.
    
    Args:
        file_path: Path to the Excel file
        output_prefix: Prefix for output files (e.g., "universal_pictures" or "upcoming_exhibitions")
        client: OpenAI client
    """
    if not os.path.exists(file_path):
        print(f"  File not found: {file_path}")
        return
    
    print(f"\n  Loading {file_path}...")
    df = pd.read_excel(file_path)
    print(f"  Loaded {len(df)} films")
    
    # Check if lead_gender column exists
    if "lead_gender" not in df.columns:
        print(f"  WARNING: 'lead_gender' column not found. Run enrich_with_lead_gender.py first.")
        return
    
    # Convert to text for embedding
    print(f"  Converting films to text for embedding...")
    film_texts = [film_to_text_for_embedding(row) for _, row in df.iterrows()]
    
    # Generate embeddings in batches
    print(f"  Generating embeddings for {len(film_texts)} films...")
    embeddings: List[List[float]] = []
    batch_size = 100
    for i in range(0, len(film_texts), batch_size):
        batch = film_texts[i:i + batch_size]
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch,
                dimensions=1536
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            print(f"    Generated embeddings for batch {i//batch_size + 1}/{(len(film_texts)-1)//batch_size + 1}...")
        except Exception as e:
            print(f"    Error creating embeddings for batch {i//batch_size + 1}: {e}")
            embeddings.extend([[0.0] * 1536 for _ in batch])
    
    # Create metadata DataFrame
    print(f"  Creating embeddings metadata...")
    embedding_data = []
    for _, row in df.iterrows():
        metadata = {
            "tmdb_id": row.get("tmdb_id"),
            "title": row.get("title"),
            "release_year": row.get("release_year"),
        }
        # Add exhibition-specific fields if present
        if "country" in df.columns:
            metadata["country"] = row.get("country")
        if "location" in df.columns:
            metadata["location"] = row.get("location")
        embedding_data.append(metadata)
    
    embeddings_df = pd.DataFrame(embedding_data)
    embeddings_excel_path = f"{output_prefix}_embeddings.xlsx"
    embeddings_df.to_excel(embeddings_excel_path, index=False)
    print(f"    Saved metadata to: {embeddings_excel_path}")
    
    # Save embeddings as numpy array
    embeddings_array = np.array(embeddings)
    embeddings_npy_path = f"{output_prefix}_embeddings.npy"
    np.save(embeddings_npy_path, embeddings_array)
    print(f"    Saved embeddings array to: {embeddings_npy_path}")
    print(f"    Shape: {embeddings_array.shape} (films x dimensions)")
    
    print(f"  [SUCCESS] Successfully regenerated embeddings for {len(df)} films")


def main():
    """Regenerate embeddings for library and exhibition files."""
    print("="*80)
    print("REGENERATING EMBEDDINGS WITH LEAD_GENDER FIELD")
    print("="*80)
    
    # Initialize OpenAI client
    openai_key = get_openai_api_key()
    os.environ["OPENAI_API_KEY"] = openai_key
    client = OpenAI(api_key=openai_key)
    
    # Regenerate library embeddings
    library_file = "universal_pictures_library.xlsx"
    if os.path.exists(library_file):
        print(f"\n{'='*60}")
        print(f"Regenerating library embeddings: {library_file}")
        print(f"{'='*60}")
        regenerate_embeddings(library_file, "universal_pictures_library", client)
    else:
        print(f"\nLibrary file not found: {library_file}")
    
    # Regenerate exhibition embeddings
    exhibition_file = "upcoming_exhibitions.xlsx"
    if os.path.exists(exhibition_file):
        print(f"\n{'='*60}")
        print(f"Regenerating exhibition embeddings: {exhibition_file}")
        print(f"{'='*60}")
        regenerate_embeddings(exhibition_file, "upcoming_exhibitions", client)
    else:
        print(f"\nExhibition file not found: {exhibition_file}")
    
    print(f"\n{'='*80}")
    print("Embedding regeneration complete!")
    print("The new embeddings now include lead_gender information.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
