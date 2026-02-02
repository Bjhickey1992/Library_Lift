#!/usr/bin/env python
"""Generate embeddings for existing library xlsx file"""

import os
import pandas as pd
import numpy as np
from typing import List
from openai import OpenAI

# Load API key from environment variables or .env file
from config import get_openai_api_key

openai_key = get_openai_api_key()

# Set in environment for compatibility
os.environ["OPENAI_API_KEY"] = openai_key

client = OpenAI(api_key=openai_key)

print("="*80)
print("GENERATING EMBEDDINGS FOR EXISTING LIBRARY FILE")
print("="*80)

# Load existing library
library_path = "lionsgate_library.xlsx"
print(f"\nLoading library from: {library_path}")
lib_df = pd.read_excel(library_path)
print(f"Loaded {len(lib_df)} library films")

# Convert to text for embedding (using same format as MatchingAgent._film_to_text)
print("\nConverting films to text for embedding...")
film_texts = []
for _, row in lib_df.iterrows():
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
    if pd.notna(row.get("need")):
        parts.append(f"Viewer needs: {row['need']}")
    film_texts.append("\n".join(parts))

# Generate embeddings in batches
print(f"\nGenerating embeddings for {len(film_texts)} films...")
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

# Create metadata DataFrame
print("\nCreating embeddings metadata...")
embedding_data = []
for i, (_, row) in enumerate(lib_df.iterrows()):
    embedding_data.append({
        "tmdb_id": row.get("tmdb_id"),
        "title": row.get("title"),
        "release_year": row.get("release_year"),
    })

embeddings_df = pd.DataFrame(embedding_data)
embeddings_excel_path = "lionsgate_library_embeddings.xlsx"
embeddings_df.to_excel(embeddings_excel_path, index=False)
print(f"  Saved metadata to: {embeddings_excel_path}")

# Save embeddings as numpy array
embeddings_array = np.array(embeddings)
embeddings_npy_path = "lionsgate_library_embeddings.npy"
np.save(embeddings_npy_path, embeddings_array)
print(f"  Saved embeddings array to: {embeddings_npy_path}")
print(f"  Shape: {embeddings_array.shape} (films × dimensions)")

print(f"\n[SUCCESS] Embeddings generated and saved!")
print(f"  Total films: {len(lib_df)}")
print(f"  Embeddings: {embeddings_array.shape}")
