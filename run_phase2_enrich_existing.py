#!/usr/bin/env python
"""Phase 2: Enrich Existing Exhibition Excel with Keywords, Tagline, and Semantic Descriptors"""

import os
import pandas as pd
import numpy as np
import json
from typing import List, Dict
from openai import OpenAI
from film_agent import TMDbClient, FilmRecord, film_record_from_tmdb_details
import requests

# Load API keys from environment variables or .env file
from config import get_openai_api_key, get_tmdb_api_key

tmdb_key = get_tmdb_api_key()
openai_key = get_openai_api_key()

# Set in environment for compatibility
os.environ["TMDB_API_KEY"] = tmdb_key
os.environ["OPENAI_API_KEY"] = openai_key

tmdb = TMDbClient(tmdb_api_key=tmdb_key)
client = OpenAI(api_key=openai_key)

print("="*80)
print("PHASE 2: ENRICHING EXISTING EXHIBITION DATA")
print("="*80)
print("This will:")
print("  1. Load existing upcoming_exhibitions.xlsx")
print("  2. Fetch keywords and tagline from TMDB for each film")
print("  3. Generate LLM semantic descriptors (themes, style, tone)")
print("  4. Update Excel with new fields")
print("  5. Regenerate embeddings with enriched data")
print()

# Load existing exhibitions
exhibitions_path = "upcoming_exhibitions.xlsx"
print(f"Loading exhibitions from: {exhibitions_path}")
ex_df = pd.read_excel(exhibitions_path)
print(f"Loaded {len(ex_df)} exhibition films\n")

# Ensure new columns exist
for col in ["keywords", "tagline", "thematic_descriptors", "stylistic_descriptors", "emotional_tone"]:
    if col not in ex_df.columns:
        ex_df[col] = None

# Function to enrich a single film
def enrich_film(row_idx: int, row: pd.Series) -> Dict:
    """Enrich a single film with keywords, tagline, and semantic descriptors."""
    result = {"row_idx": row_idx}
    
    tmdb_id = row.get("tmdb_id")
    if pd.isna(tmdb_id):
        print(f"  [SKIP] Row {row_idx}: No TMDB ID for '{row.get('title', 'Unknown')}'")
        return result
    
    tmdb_id = int(tmdb_id)
    title = row.get("title", "Unknown")
    
    # Fetch keywords and tagline from TMDB
    try:
        details = tmdb.get_movie_details_with_credits(tmdb_id)
        
        # Get tagline
        tagline = details.get("tagline") or ""
        
        # Get keywords
        keywords_list = []
        if "keywords" in details and "keywords" in details["keywords"]:
            keywords_list = [kw.get("name", "") for kw in details["keywords"]["keywords"] if kw.get("name")]
        keywords_str = ", ".join(keywords_list) if keywords_list else None
        
        result["keywords"] = keywords_str
        result["tagline"] = tagline if tagline else None
        
        # Build film description for LLM
        film_desc_parts = []
        film_desc_parts.append(f"Title: {title}")
        if pd.notna(row.get("release_year")):
            film_desc_parts.append(f"Year: {int(row['release_year'])}")
        if pd.notna(row.get("director")):
            film_desc_parts.append(f"Director: {row['director']}")
        if pd.notna(row.get("genres")):
            film_desc_parts.append(f"Genres: {row['genres']}")
        if pd.notna(row.get("overview")):
            film_desc_parts.append(f"Plot: {row['overview']}")
        if keywords_str:
            film_desc_parts.append(f"Keywords: {keywords_str}")
        if tagline:
            film_desc_parts.append(f"Tagline: {tagline}")
        
        film_description = "\n".join(film_desc_parts)
        
        # Generate semantic descriptors with LLM
        prompt = f"""Analyze this film and provide concise descriptors:

{film_description}

Provide a JSON response with:
1. "thematic_descriptors": 3-5 key themes (e.g., "alienation, urban isolation, existential crisis, betrayal, psychological complexity")
2. "stylistic_descriptors": 1-2 sentences describing cinematic style (e.g., "minimalist cinematography, slow-paced, contemplative, non-linear narrative")
3. "emotional_tone": 1 sentence describing emotional atmosphere (e.g., "melancholic, introspective, existential, darkly humorous")

Return ONLY valid JSON: {{"thematic_descriptors": "...", "stylistic_descriptors": "...", "emotional_tone": "..."}}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a film analysis expert. Provide concise, accurate descriptors."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=300
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            
            result["thematic_descriptors"] = data.get("thematic_descriptors", "")
            result["stylistic_descriptors"] = data.get("stylistic_descriptors", "")
            result["emotional_tone"] = data.get("emotional_tone", "")
            
        except Exception as e:
            print(f"  [WARN] LLM enrichment failed for {title}: {e}")
            result["thematic_descriptors"] = None
            result["stylistic_descriptors"] = None
            result["emotional_tone"] = None
            
    except requests.HTTPError as e:
        print(f"  [ERROR] TMDB fetch failed for {title} (ID: {tmdb_id}): {e}")
        result["keywords"] = None
        result["tagline"] = None
        result["thematic_descriptors"] = None
        result["stylistic_descriptors"] = None
        result["emotional_tone"] = None
    
    return result

# Enrich all films
print("Enriching films with keywords, tagline, and semantic descriptors...")
enrichment_results = []
for idx, (row_idx, row) in enumerate(ex_df.iterrows(), 1):
    if idx % 10 == 0:
        print(f"  Processing {idx}/{len(ex_df)} films...")
    result = enrich_film(row_idx, row)
    enrichment_results.append(result)

# Update DataFrame with enrichment results
print("\nUpdating DataFrame with enrichment data...")
for result in enrichment_results:
    row_idx = result["row_idx"]
    if "keywords" in result:
        ex_df.at[row_idx, "keywords"] = result["keywords"]
    if "tagline" in result:
        ex_df.at[row_idx, "tagline"] = result["tagline"]
    if "thematic_descriptors" in result:
        ex_df.at[row_idx, "thematic_descriptors"] = result["thematic_descriptors"]
    if "stylistic_descriptors" in result:
        ex_df.at[row_idx, "stylistic_descriptors"] = result["stylistic_descriptors"]
    if "emotional_tone" in result:
        ex_df.at[row_idx, "emotional_tone"] = result["emotional_tone"]

# Save updated Excel
print(f"\nSaving enriched exhibitions to: {exhibitions_path}")
ex_df.to_excel(exhibitions_path, index=False)
print("  [OK] Excel file updated")

# Regenerate embeddings with enriched data
print("\nRegenerating embeddings with enriched data...")
film_texts = []
for _, row in ex_df.iterrows():
    parts = []
    if pd.notna(row.get("title")):
        parts.append(f"Film: {row['title']}")
    if pd.notna(row.get("release_year")):
        parts.append(f"Released: {int(row['release_year'])}")
    if pd.notna(row.get("tagline")):
        parts.append(f"Tagline: {row['tagline']}")
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
    if pd.notna(row.get("genres")):
        parts.append(f"Genres: {row['genres']}")
    if pd.notna(row.get("keywords")):
        parts.append(f"Keywords: {row['keywords']}")
    if pd.notna(row.get("overview")):
        parts.append(f"Plot: {row['overview']}")
    if pd.notna(row.get("thematic_descriptors")):
        parts.append(f"Themes: {row['thematic_descriptors']}")
    if pd.notna(row.get("stylistic_descriptors")):
        parts.append(f"Style: {row['stylistic_descriptors']}")
    if pd.notna(row.get("emotional_tone")):
        parts.append(f"Tone: {row['emotional_tone']}")
    film_texts.append("\n".join(parts))

# Generate embeddings in batches
print(f"Generating embeddings for {len(film_texts)} films...")
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
        print(f"  Generated embeddings for batch {i//batch_size + 1}/{(len(film_texts)-1)//batch_size + 1}...")
    except Exception as e:
        print(f"  Error creating embeddings for batch {i//batch_size + 1}: {e}")
        embeddings.extend([[0.0] * 1536 for _ in batch])

# Save embeddings metadata
print("\nSaving embeddings...")
embedding_data = []
for i, (_, row) in enumerate(ex_df.iterrows()):
    embedding_data.append({
        "tmdb_id": row.get("tmdb_id"),
        "title": row.get("title"),
        "release_year": row.get("release_year"),
        "country": row.get("country"),
        "location": row.get("location"),
    })

embeddings_df = pd.DataFrame(embedding_data)
embeddings_excel_path = "upcoming_exhibitions_embeddings.xlsx"
embeddings_df.to_excel(embeddings_excel_path, index=False)
print(f"  Saved metadata to: {embeddings_excel_path}")

# Save embeddings as numpy array
embeddings_array = np.array(embeddings)
embeddings_npy_path = "upcoming_exhibitions_embeddings.npy"
np.save(embeddings_npy_path, embeddings_array)
print(f"  Saved embeddings array to: {embeddings_npy_path}")
print(f"  Shape: {embeddings_array.shape} (films × dimensions)")

print(f"\n[SUCCESS] Phase 2 (Enrichment) complete!")
print(f"  Total films enriched: {len(ex_df)}")
print(f"  Updated file: {exhibitions_path}")
print(f"  Embeddings saved to: {embeddings_excel_path} and {embeddings_npy_path}")
