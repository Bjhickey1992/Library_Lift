# Lead Gender Enrichment Guide

This guide explains how to add lead gender information to your library and exhibition data files, and regenerate embeddings to include this information.

## Overview

The system now supports filtering by lead actor gender (e.g., "films with female leads"). To enable this feature:

1. **Enrich existing data files** with `lead_gender` field
2. **Regenerate embeddings** to include lead gender in the embedding text

## Step 1: Enrich Data Files with Lead Gender

Run the enrichment script to add `lead_gender` to your existing Excel files:

```bash
python enrich_with_lead_gender.py
```

This script:
- Reads `universal_pictures_library.xlsx` and `upcoming_exhibitions.xlsx`
- For each film, looks up the first cast member (lead actor) from TMDb
- Gets the person's gender from TMDb API
- Adds a `lead_gender` column with values: `"female"`, `"male"`, or `None`

**Note:** This makes TMDb API calls for each film. For large libraries, this may take some time.

## Step 2: Regenerate Embeddings

After enriching the data files, regenerate embeddings to include lead gender in the embedding text:

```bash
python regenerate_embeddings_with_lead_gender.py
```

This script:
- Reads the enriched Excel files
- Regenerates embeddings that include "Lead actor gender: female/male" in the text
- Saves new `.npy` and `.xlsx` embedding files

## What Changed

### Data Model
- Added `lead_gender: Optional[str]` field to `FilmRecord` dataclass
- Values: `"female"`, `"male"`, or `None`

### Embedding Text
The embedding text now includes:
```
Lead actor gender: female
```
or
```
Lead actor gender: male
```

This allows the similarity algorithm to consider lead gender when matching films.

### Filtering Logic
The `_apply_library_filters()` method now:
- Checks if `lead_gender` column exists in the DataFrame
- Filters directly using: `filtered_df[filtered_df["lead_gender"] == intent.lead_gender]`
- No longer requires on-the-fly TMDb API calls

## Usage

After completing both steps, queries like:
- "what films with female leads are relevant today?"
- "show me films with male leads from the 80s"

Will properly filter the library to only films matching the requested lead gender.

## Files Modified

1. **film_agent.py**
   - Added `lead_gender` field to `FilmRecord`
   - Updated `_film_to_text_for_embedding()` to include lead gender
   - Updated `_generate_exhibition_embeddings()` to include lead gender
   - Updated `MatchingAgent._film_to_text()` to include lead gender

2. **chatbot_agent.py**
   - Updated `_apply_library_filters()` to use stored `lead_gender` field instead of API calls

3. **query_intent_parser.py**
   - Already supports `lead_gender` in `QueryIntent` (from previous changes)

4. **generate_exhibition_embeddings.py**
   - Updated to include lead gender in embedding text

## New Scripts

1. **enrich_with_lead_gender.py**
   - Enriches existing Excel files with lead gender data

2. **regenerate_embeddings_with_lead_gender.py**
   - Regenerates embeddings after lead gender has been added

## Performance

- **Before:** Filtering by lead gender required TMDb API calls for each film during query time (very slow)
- **After:** Filtering uses pre-computed data from the Excel file (instant)

The initial enrichment step makes API calls, but this is a one-time operation. Subsequent queries use the stored data.
