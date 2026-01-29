# Solution: Pulling Actual Trending Films in Theaters

## Available Data Sources

From `exhibitions_df` (upcoming_exhibitions.xlsx), we have:
- `title` - Film title
- `genres` - Comma-separated genres (e.g., "Horror, Thriller")
- `country` - Country code (US, GB, FR, etc.)
- `location` - Cinema name/location
- `start_date` - When exhibition starts (ISO format)
- `end_date` - When exhibition ends (ISO format)
- `release_year` - Original film release year
- `tmdb_id` - For fetching poster images
- `scheduled_dates` - Comma-separated list of screening dates

---

## Solution 1: Calculate Current Genre Distribution (Immediate)

**What we can do NOW:**
- Count genres in current exhibitions
- Show which genres are most common
- Calculate percentage of total exhibitions

**Implementation:**
```python
# Count genres in exhibitions
all_genres = exhibitions_df['genres'].str.split(',').explode().str.strip()
genre_counts = all_genres.value_counts()

# Calculate percentages
total_exhibitions = len(exhibitions_df)
top_genres = []
for genre, count in genre_counts.head(3).items():
    pct = (count / total_exhibitions) * 100
    top_genres.append({"name": genre, "value": int(pct)})
```

**Limitation:** Shows current distribution, not "trends" (would need historical comparison)

---

## Solution 2: Identify Most Popular Films (By Venue Count)

**What we can do:**
- Count how many different venues are showing each film
- Films showing at more venues = more popular/trending
- Sort by venue count to get top trending films

**Implementation:**
```python
# Count unique venues per film
film_popularity = exhibitions_df.groupby('title').agg({
    'location': 'nunique',  # Number of unique venues
    'country': lambda x: ', '.join(x.unique()),  # Countries showing it
    'tmdb_id': 'first',  # For poster
    'release_year': 'first',
    'genres': 'first'
}).reset_index()
film_popularity.columns = ['title', 'venue_count', 'countries', 'tmdb_id', 'release_year', 'genres']
film_popularity = film_popularity.sort_values('venue_count', ascending=False)

# Top trending films
top_films = film_popularity.head(5)
```

---

## Solution 3: Identify "NOW SHOWING" vs "RE-RELEASE"

**What we can do:**
- Compare `release_year` to current year
- Recent releases (within 2-3 years) = "NOW SHOWING"
- Older films (5+ years) = "RE-RELEASE HIT"

**Implementation:**
```python
from datetime import datetime
current_year = datetime.now().year

# Categorize films
exhibitions_df['film_type'] = exhibitions_df['release_year'].apply(
    lambda year: 'NOW SHOWING' if year and (current_year - year) <= 3 
    else 'RE-RELEASE HIT' if year and (current_year - year) > 5 
    else 'OTHER'
)

# Get top "NOW SHOWING" film
now_showing = exhibitions_df[exhibitions_df['film_type'] == 'NOW SHOWING']
if len(now_showing) > 0:
    top_now_showing = now_showing.groupby('title').agg({
        'location': 'nunique',
        'tmdb_id': 'first'
    }).sort_values('location', ascending=False).head(1)

# Get top "RE-RELEASE HIT"
rerelease = exhibitions_df[exhibitions_df['film_type'] == 'RE-RELEASE HIT']
if len(rerelease) > 0:
    top_rerelease = rerelease.groupby('title').agg({
        'location': 'nunique',
        'tmdb_id': 'first'
    }).sort_values('location', ascending=False).head(1)
```

---

## Solution 4: Get Poster Images

**What we can do:**
- Use `tmdb_id` to fetch poster URLs
- Display actual movie posters instead of placeholders

**Implementation:**
```python
# In ChatbotAgent, there's already a method:
def _get_poster_url(self, tmdb_id: Optional[int]) -> Optional[str]:
    """Get poster image URL from TMDB."""
    if not tmdb_id or not self.tmdb_client:
        return None
    try:
        details = self.tmdb_client.get_movie_details_with_credits(tmdb_id)
        poster_path = details.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        return None

# Use it:
poster_url = agent._get_poster_url(tmdb_id)
```

---

## Solution 5: Historical Trends (Future Enhancement)

**To get REAL trends (percentage changes):**

1. **Store historical data:**
   ```python
   # Save exhibitions_df with timestamp
   exhibitions_df['scraped_date'] = datetime.now().isoformat()
   exhibitions_df.to_excel(f'exhibitions_{datetime.now().strftime("%Y%m%d")}.xlsx')
   ```

2. **Compare periods:**
   ```python
   # Load current and previous period
   current = load_exhibitions('exhibitions_20250122.xlsx')
   previous = load_exhibitions('exhibitions_20250115.xlsx')
   
   # Calculate genre trends
   current_genres = current['genres'].str.split(',').explode().str.strip().value_counts()
   previous_genres = previous['genres'].str.split(',').explode().str.strip().value_counts()
   
   # Calculate percentage change
   for genre in current_genres.index:
       current_count = current_genres[genre]
       previous_count = previous_genres.get(genre, 0)
       if previous_count > 0:
           trend_pct = ((current_count - previous_count) / previous_count) * 100
       else:
           trend_pct = 100  # New genre
   ```

---

## Implementation Plan

### Phase 1: Immediate (Can do now)
1. ✅ Calculate genre distribution from exhibitions_df
2. ✅ Identify most popular films (by venue count)
3. ✅ Categorize "NOW SHOWING" vs "RE-RELEASE"
4. ✅ Fetch and display poster images

### Phase 2: Enhanced (Next iteration)
1. Store historical exhibition data
2. Calculate percentage changes (real trends)
3. Add time-based filtering (current week, month, etc.)

### Phase 3: Advanced (Future)
1. Integrate external APIs (Box Office Mojo, etc.)
2. Add social media sentiment analysis
3. Real-time updates via scheduled scraping

---

## Code Structure

```python
def calculate_trending_genres(exhibitions_df):
    """Calculate genre distribution from exhibitions."""
    # Split genres and count
    # Return top 3 genres with percentages
    
def get_trending_films(exhibitions_df, top_n=5):
    """Get most popular films by venue count."""
    # Group by title, count venues
    # Return top N films with metadata
    
def categorize_films(exhibitions_df):
    """Categorize films as NOW SHOWING or RE-RELEASE."""
    # Compare release_year to current year
    # Return categorized dataframe
    
def get_poster_images(agent, tmdb_ids):
    """Fetch poster URLs from TMDB."""
    # Use agent._get_poster_url()
    # Return dict of tmdb_id -> poster_url
```

---

## Expected Results

After implementation:
- **Trending Genres:** Real percentages from current exhibitions (e.g., "Horror: 25%", "Drama: 20%")
- **Trending Films:** Actual films showing at most venues (e.g., "The Shining" at 5 venues)
- **Posters:** Real movie poster images from TMDB
- **NOW SHOWING:** Recent releases currently in theaters
- **RE-RELEASE HIT:** Classic films being re-released
