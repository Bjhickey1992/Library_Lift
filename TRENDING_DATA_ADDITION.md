# Trending Data Addition to Phase 2

This document describes the addition of IMDB trending and Nielsen streaming data sources to the Phase 2 exhibition scraping process.

## Overview

Two new data sources have been added to Phase 2:

1. **IMDB Top 50 Trending** - Scrapes the top 50 trending films and TV shows from IMDB's trending charts
2. **Nielsen Top 10 Streaming** - Scrapes the top 10 streaming titles from Nielsen (streaming only, excludes linear TV)

## Files Added/Modified

### New Files

1. **`trending_scrapers.py`**
   - Contains scrapers for IMDB and Nielsen
   - Functions:
     - `scrape_imdb_trending(top_n=50)` - Scrapes IMDB trending charts
     - `scrape_nielsen_streaming_top10()` - Scrapes Nielsen streaming top 10
     - `convert_imdb_to_film_record()` - Converts IMDB items to FilmRecord format
     - `convert_nielsen_to_film_record()` - Converts Nielsen items to FilmRecord format

2. **`add_trending_data.py`**
   - Standalone script to add trending data to existing exhibition file
   - Can be run independently to update exhibitions with new trending data
   - Handles duplicate detection and embedding regeneration

### Modified Files

1. **`film_agent.py`**
   - Added `scrape_imdb_trending()` method to `ExhibitionScrapingAgent`
   - Added `scrape_nielsen_streaming()` method to `ExhibitionScrapingAgent`

2. **`run_phase2_exhibitions.py`**
   - Updated to include IMDB and Nielsen scraping
   - Automatically appends trending data to exhibitions
   - Regenerates embeddings after adding trending data

## Usage

### Option 1: Run Full Phase 2 (includes trending data)

```bash
python run_phase2_exhibitions.py
```

This will:
1. Scrape all cinemas from `cinemas.yaml`
2. Scrape IMDB top 50 trending
3. Scrape Nielsen top 10 streaming
4. Combine all data
5. Generate embeddings

### Option 2: Add Trending Data to Existing Exhibitions

If you just want to add trending data to an existing exhibition file without re-scraping cinemas:

```bash
python add_trending_data.py
```

This will:
1. Load existing `upcoming_exhibitions.xlsx`
2. Scrape IMDB top 50 trending
3. Scrape Nielsen top 10 streaming
4. Append new data (skipping duplicates)
5. Regenerate embeddings for all films

## Data Source Details

### IMDB Trending

- **Source**: https://www.imdb.com/chart/moviemeter/ and /chart/tvmeter/
- **Data**: Top 50 trending films and TV shows
- **Update Frequency**: Weekly (IMDB updates these charts regularly)
- **Fields Captured**:
  - Title, year, IMDB ID, type (movie/tv), rank
  - Enriched with TMDb metadata (director, cast, genres, etc.)

### Nielsen Streaming

- **Source**: https://www.nielsen.com/data-center/top-ten/
- **Data**: Top 10 streaming titles (streaming only, excludes linear TV)
- **Update Frequency**: Weekly (Nielsen publishes weekly streaming charts)
- **Fields Captured**:
  - Title, platform, rank
  - Enriched with TMDb metadata
- **Note**: Nielsen's website structure may change. The scraper attempts to find streaming-specific content but may need adjustment if the site structure changes.

## Duplicate Detection

The system automatically detects and skips duplicates when appending trending data:
- Checks for same title (case-insensitive)
- Checks for same TMDb ID
- Only adds truly new films

## Embeddings

After adding trending data, embeddings are automatically regenerated for ALL exhibition films (not just new ones) to ensure consistency.

## Troubleshooting

### IMDB Scraping Issues

**Current Status**: IMDB appears to be blocking programmatic access to their trending charts (returns 202 status with empty content). This is a common anti-scraping measure.

**Potential Solutions**:
1. **Use a headless browser** (Selenium/Playwright) to render JavaScript content
2. **Use IMDB's official API** (requires AWS Data Exchange subscription)
3. **Use third-party APIs** like Apify's IMDB Trending actor
4. **Manual data entry** for critical trending titles

The scraper code is in place and will work if IMDB allows access or if you implement one of the solutions above.

### Nielsen Scraping Issues

### Nielsen Scraping Issues

- Nielsen's website structure may change frequently
- If scraping fails, you may need to:
  1. Check the Nielsen website manually
  2. Update the `scrape_nielsen_streaming_top10()` function in `trending_scrapers.py`
  3. Consider using Nielsen's official API if available (requires subscription)

### TMDb Lookup Failures

- Some titles may not be found in TMDb
- These will be skipped with a warning message
- The script continues processing remaining items

## Future Enhancements

Potential improvements:
- Add caching for TMDb lookups to reduce API calls
- Add support for more trending data sources
- Add historical trending data tracking
- Improve Nielsen scraper robustness
- Add configuration for which data sources to include
