# Streamlit Chatbot Setup Guide

## Overview

The Streamlit chatbot provides a fast, interactive interface for getting film library recommendations based on exhibition schedules.

## Key Features

### 1. Low Latency Optimizations
- **Pre-computed embeddings**: Uses existing `.npy` files (no regeneration)
- **Cached data**: Library and exhibition data loaded once and cached
- **No re-scraping**: Exhibition data is cached for 7 days
- **Fast similarity calculation**: Uses pre-loaded embeddings

### 2. Smart Filtering
- **Excludes exact matches**: Filters out similarity > 0.9 (likely same film)
- **Focuses on sweet spot**: Emphasizes 0.5-0.7 range (unintuitive but logical matches)
- **Territory filtering**: Returns recommendations for specific countries

### 3. Optimized Recommendations
- **Top 3-5 per territory**: Returns best recommendations only
- **Deduplication**: One recommendation per library film
- **Sorted by relevance**: Highest similarity first

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Ensure Pre-built Data Exists
The chatbot requires:
- `lionsgate_library.xlsx` - Pre-built library
- `lionsgate_library_embeddings.npy` - Pre-computed embeddings
- `upcoming_exhibitions.xlsx` - Exhibition data
- `upcoming_exhibitions_embeddings.npy` - Pre-computed embeddings

### 3. Run the App
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### Example Queries
- "What library titles should we emphasize this month in the US?"
- "Show me recommendations for the UK"
- "What films should we promote in France this month?"

### Configuration
Use the sidebar to adjust:
- **Minimum Similarity**: Lower bound (default: 0.5)
- **Maximum Similarity**: Upper bound (default: 0.7)
- **Number of Recommendations**: How many to return (default: 5)
- **Exclude Exact Matches**: Filter out >0.9 similarity (default: enabled)

## How It Works

1. **User asks a question** about recommendations for a territory
2. **Territory extraction**: LLM extracts country code from query
3. **Fast matching**: Uses pre-computed embeddings to find matches
4. **Filtering**: Excludes exact matches, focuses on 0.5-0.7 range
5. **Top N selection**: Returns best 3-5 recommendations per territory
6. **Display**: Shows formatted recommendations with details

## Latency Optimizations

### Current Optimizations
- ✅ Pre-computed embeddings (saves ~30 seconds)
- ✅ Cached data loading (saves ~5 seconds)
- ✅ No re-scraping (saves ~10-20 minutes)
- ✅ Fast similarity calculation (vectorized NumPy operations)

### Response Time
- **First query**: ~2-5 seconds (data loading)
- **Subsequent queries**: ~1-3 seconds (cached data)

### Future Optimizations (if needed)
- Background refresh of exhibition data
- Pre-compute all territory matches
- Use vector database (FAISS) for faster similarity search
- Parallel processing for multiple territories

## Filtering Logic

### Similarity Range Focus
- **0.5-0.7 range**: "Sweet spot" for unintuitive but logical matches
- **< 0.5**: Too dissimilar, not useful
- **> 0.7**: Likely obvious matches or same film
- **> 0.9**: Exact matches (excluded by default)

### Example
- "Buffalo '66" ↔ "The Killing of a Chinese Bookie": 0.65 (✅ included)
- "The Godfather" ↔ "The Godfather": 0.95 (❌ excluded - same film)
- "Action Movie A" ↔ "Romance Movie B": 0.3 (❌ excluded - too dissimilar)

## Troubleshooting

**Error: "Library file not found"**
- Run Phase 1 to build the library
- Or use pre-built library from team package

**Error: "Exhibition file not found"**
- Run Phase 2 to build exhibitions
- Or use pre-built exhibitions from team package

**Slow responses:**
- Check that `.npy` embedding files exist
- Verify data files are not corrupted
- Consider reducing `top_n` parameter

**No recommendations:**
- Check territory code is correct (US, UK, FR, CA, MX)
- Adjust similarity range (try 0.4-0.8)
- Verify exhibition data exists for that territory
