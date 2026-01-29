#!/usr/bin/env python
"""Regenerate exhibition embeddings for current exhibition data"""

import os
import pandas as pd
import numpy as np
from film_agent import MatchingAgent
from config import get_openai_api_key

# Load API keys
openai_key = get_openai_api_key()
os.environ["OPENAI_API_KEY"] = openai_key

# Initialize matching agent (has the embedding methods we need)
matching_agent = MatchingAgent(openai_api_key=openai_key)

print("Regenerating exhibition embeddings...")
print("=" * 60)

# Load current exhibition data
exhibitions_path = "upcoming_exhibitions.xlsx"
print(f"\nLoading exhibitions from: {exhibitions_path}")
exhibitions_df = pd.read_excel(exhibitions_path)
print(f"Loaded {len(exhibitions_df)} exhibition films")

# Convert to records for text conversion
ex_rows = exhibitions_df.to_dict(orient="records")

# Convert films to text using matching agent's method
print("\nConverting films to text for embedding...")
film_texts = []
for ex_row in ex_rows:
    text = matching_agent._film_to_text(ex_row)
    film_texts.append(text)

print(f"Converted {len(film_texts)} films to text")

# Generate embeddings in batches
print("\nGenerating embeddings...")
embeddings = matching_agent._create_embeddings(film_texts)
print(f"Generated {len(embeddings)} embeddings")

# Create metadata DataFrame
print("\nCreating metadata file...")
embedding_data = []
for i, (_, row) in enumerate(exhibitions_df.iterrows()):
    embedding_data.append({
        "tmdb_id": row.get("tmdb_id"),
        "title": row.get("title"),
        "release_year": row.get("release_year"),
        "country": row.get("country"),
        "location": row.get("location"),
    })

embeddings_df = pd.DataFrame(embedding_data)

# Save files
base_path = "upcoming_exhibitions"
embeddings_excel_path = f"{base_path}_embeddings.xlsx"
embeddings_npy_path = f"{base_path}_embeddings.npy"

embeddings_df.to_excel(embeddings_excel_path, index=False)
print(f"Saved metadata to: {embeddings_excel_path}")

embeddings_array = np.array(embeddings)
np.save(embeddings_npy_path, embeddings_array)
print(f"Saved embeddings to: {embeddings_npy_path}")
print(f"  Shape: {embeddings_array.shape} (films × dimensions)")

print("\n" + "=" * 60)
print("SUCCESS: Exhibition embeddings regenerated!")
print(f"  - {len(embeddings)} embeddings generated")
print(f"  - Files saved: {embeddings_excel_path}, {embeddings_npy_path}")
