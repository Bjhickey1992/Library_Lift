# TMDB API Function in This Project

## TL;DR: TMDB is NOT used for trends

**TMDB API is used for movie metadata enrichment, NOT for pulling trends.**

---

## What TMDB API Actually Does Here

### 1. **Building Studio Library** (Phase 1)
**Purpose:** Find all films owned/distributed by a studio (e.g., Lionsgate)

**How it works:**
```python
# Step 1: Search for studio company
tmdb.search_company("Lionsgate")  # Returns company ID

# Step 2: Discover all movies by that company
tmdb.discover_movies_for_company(company_id)  
# Returns list of all movies associated with Lionsgate

# Step 3: Get detailed metadata for each movie
tmdb.get_movie_details_with_credits(movie_id)
# Returns: title, year, director, cast, crew, genres, overview, tagline, keywords, etc.
```

**What data is extracted:**
- Title, release year, director, writers, producers
- Cast, cinematographers, production designers
- Genres, overview (plot summary), tagline
- Keywords, TMDB ID (for poster images)
- Reviews (optional)

**Output:** `lionsgate_library.xlsx` - Complete studio film catalog

---

### 2. **Enriching Cinema Screenings** (Phase 2)
**Purpose:** When scraping cinema websites, enrich scraped titles with full metadata

**How it works:**
```python
# Cinema scraper finds: "Halloween (1978)" playing at Film Forum

# Step 1: Search TMDB for the title
tmdb.search_movie("Halloween", year=1978)
# Returns matching movie IDs

# Step 2: Get full details
tmdb.get_movie_details_with_credits(movie_id)
# Returns complete metadata (same as above)

# Step 3: Store enriched data
# Now we have: title, director, cast, genres, tmdb_id, etc.
```

**What this enables:**
- Scraped cinema schedules only have basic info (title, date, venue)
- TMDB enriches them with: director, cast, genres, poster images, etc.
- This enriched data is needed for semantic similarity matching

**Output:** `upcoming_exhibitions.xlsx` - Enriched cinema schedule

---

### 3. **Poster Images** (Dashboard/Display)
**Purpose:** Get movie poster images for display

**How it works:**
```python
# Use tmdb_id from library or exhibitions
poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
# Returns: https://image.tmdb.org/t/p/w500/abc123.jpg
```

**What this enables:**
- Display movie posters in dashboard
- Show visual thumbnails for recommendations
- Better UX than text-only lists

---

## What TMDB API Does NOT Do

### ❌ **NOT Used For:**
1. **Trending Movies** - TMDB doesn't provide "trending" or "popular now" data
2. **Current Box Office** - No real-time box office data
3. **OTT Platform Trends** - No Netflix/Disney+/Hulu trending data
4. **Social Media Trends** - No Twitter/Reddit mention counts
5. **Genre Popularity Trends** - No "Horror is up 68%" type data

### ⚠️ **What TMDB DOES Have (but we're not using):**
- `popularity` score (static, not trend-based)
- `vote_average` (rating, not trend)
- `vote_count` (total votes, not recent activity)

These are **static metrics**, not **trending indicators**.

---

## Where Trends Would Come From

### Current Implementation:
**Trends are HARDCODED** in the dashboard:
```python
# In streamlit_app.py - line 431-435
genres_data = [
    {"name": "Horror", "value": 68, "color": "orange"},  # Hardcoded!
    {"name": "Sci-Fi", "value": 52, "color": "blue"},     # Hardcoded!
    {"name": "Comedy", "value": 35, "color": "blue"}      # Hardcoded!
]
```

### How to Get REAL Trends:

#### Option 1: **Analyze Exhibition Data** (Can implement now)
```python
# Count genres in current exhibitions
if exhibitions_df is not None:
    all_genres = exhibitions_df['genres'].str.split(',').explode().str.strip()
    genre_counts = all_genres.value_counts()
    
    # Calculate percentage of total
    total = len(exhibitions_df)
    horror_pct = (genre_counts.get('Horror', 0) / total) * 100
    # This shows current genre distribution, not "trends"
```

**Limitation:** This shows **current distribution**, not **trends over time** (would need historical data).

#### Option 2: **External Trend APIs** (Future implementation)
- **Google Trends API** - Real search trend data
- **Twitter API** - Hashtag mentions, engagement
- **IMDb API** - Search trends, popularity changes
- **JustWatch API** - Streaming platform trends
- **Box Office Mojo API** - Box office trends

#### Option 3: **Compare Historical Data** (Requires data storage)
```python
# Store exhibition data over time
# Compare current month vs previous month
# Calculate: (current - previous) / previous * 100
# This gives real trend percentages
```

---

## TMDB API Endpoints Used

| Endpoint | Purpose | Used In |
|----------|---------|---------|
| `/search/company` | Find studio by name | Phase 1: Build library |
| `/discover/movie` | Get all movies by studio | Phase 1: Build library |
| `/movie/{id}` | Get movie details | Phase 1 & 2: Enrich data |
| `/movie/{id}/credits` | Get cast & crew | Phase 1 & 2: Enrich data |
| `/movie/{id}/keywords` | Get keywords | Phase 1 & 2: Enrich data |
| `/movie/{id}/reviews` | Get user reviews | Optional enrichment |
| `/search/movie` | Find movie by title | Phase 2: Match scraped titles |
| `image.tmdb.org` | Get poster images | Dashboard display |

---

## Data Flow Summary

```
Phase 1: Build Library
├── TMDB: Search for "Lionsgate" company
├── TMDB: Discover all Lionsgate movies
├── TMDB: Get details for each movie
└── Save: lionsgate_library.xlsx

Phase 2: Scrape Exhibitions
├── Scrape: Cinema websites (Film Forum, etc.)
├── Get: Basic info (title, date, venue)
├── TMDB: Search for each title
├── TMDB: Get full details (director, cast, genres, etc.)
└── Save: upcoming_exhibitions.xlsx

Phase 3: Matching
├── Use: Pre-computed embeddings
├── Match: Library films to exhibitions
└── Output: Recommendations

Dashboard
├── Load: library_df, exhibitions_df
├── TMDB: Get poster images (via tmdb_id)
└── Display: Recommendations + metadata
```

---

## Key Takeaway

**TMDB = Movie Database (Metadata)**
- ✅ Provides: Movie information, cast, crew, genres, posters
- ❌ Does NOT provide: Trends, popularity changes, real-time data

**For trends, you need:**
- Historical data comparison (store exhibitions over time)
- External trend APIs (Google Trends, Twitter, etc.)
- Or analyze current exhibition distribution (shows current state, not trends)

---

## Example: What TMDB Returns

```python
# TMDB API Response Example
{
    "id": 694,  # TMDB ID
    "title": "The Shining",
    "release_date": "1980-05-23",
    "genres": [{"id": 27, "name": "Horror"}, {"id": 53, "name": "Thriller"}],
    "overview": "A family heads to an isolated hotel...",
    "tagline": "The tide of terror that swept America IS HERE",
    "poster_path": "/9fgh3Ns1iRzlQNYuJyK0ARQZU7w.jpg",
    "credits": {
        "cast": [{"name": "Jack Nicholson", ...}, ...],
        "crew": [{"name": "Stanley Kubrick", "job": "Director", ...}, ...]
    },
    "keywords": {
        "keywords": [{"name": "hotel"}, {"name": "isolation"}, ...]
    }
}
```

**This is static metadata, not trend data!**
