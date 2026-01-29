# How Trends Are Currently "Computed" in the Dashboard

## TL;DR: They're NOT computed - they're **hardcoded static values**

There is **zero computation** happening. All trend values are hardcoded in the code.

---

## 1. Trending Genres (Cinema Trends Card)

**Location:** `streamlit_app.py`, lines 430-435

**Current Implementation:**
```python
# Trending genres with progress bars
genres_data = [
    {"name": "Horror", "value": 68, "color": "orange"},
    {"name": "Sci-Fi", "value": 52, "color": "blue"},
    {"name": "Comedy", "value": 35, "color": "blue"}
]

for genre in genres_data:
    st.markdown(f"""
    <div class="progress-container">
        <div class="progress-label">
            <span>{genre['name']}</span>
            <span>+{genre['value']}%</span>  <!-- Hardcoded: 68, 52, 35 -->
        </div>
        <div class="progress-bar">
            <div class="progress-fill progress-{genre['color']}" style="width: {genre['value']}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
```

**What this means:**
- ❌ **No calculation** from `exhibitions_df`
- ❌ **No analysis** of genre frequencies
- ❌ **No comparison** to historical data
- ✅ **Just static values**: 68%, 52%, 35% - always the same

**Display:**
- Shows: "Horror +68%", "Sci-Fi +52%", "Comedy +35%"
- These values **never change** - they're hardcoded

---

## 2. OTT Platform Trends

**Location:** `streamlit_app.py`, lines 500-505

**Current Implementation:**
```python
# Platform trends
platforms = [
    {"name": "Netflix", "trend": "#1 Rising Horror Picks"},
    {"name": "IMDb", "trend": "Top Sci-Fi Searches", "icon": "📈"},
    {"name": "Google Trends", "trend": "80s Movies Surge"}
]

for platform in platforms:
    st.markdown(f"""
    <div class="platform-item">
        <div class="platform-logo">{platform['name'][0]}</div>
        <div style="flex: 1;">
            <div style="font-weight: 600; color: #FFFFFF; margin-bottom: 0.25rem;">{platform['name']}</div>
            <div style="font-size: 0.85rem; color: #B0B0B0;">{platform['trend']}</div>  <!-- Hardcoded strings -->
        </div>
    </div>
    """, unsafe_allow_html=True)
```

**What this means:**
- ❌ **No API calls** to Netflix, IMDb, or Google Trends
- ❌ **No web scraping** of platform data
- ❌ **No real-time data** fetching
- ✅ **Just hardcoded strings**: "#1 Rising Horror Picks", "Top Sci-Fi Searches", "80s Movies Surge"

**Display:**
- Shows: Static text labels that never change

---

## 3. Real-time Data Feeds

**Location:** `streamlit_app.py`, lines 528-534

**Current Implementation:**
```python
# Data feed panels
feeds = [
    {"platform": "IMDb", "metric": "48K Mentions"},
    {"platform": "JustWatch", "metric": "Sci-Fi Rentals Up 62%"},
    {"platform": "Twitter", "metric": "125K #TimeTravel"},
    {"platform": "Variety News", "metric": "Horror Films in Demand"}
]

for idx, feed in enumerate(feeds):
    # Display hardcoded metrics
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.9rem; color: #FFFFFF; font-weight: 600;">{feed['metric']}</div>
    </div>
    """, unsafe_allow_html=True)
```

**What this means:**
- ❌ **No API calls** to IMDb, JustWatch, Twitter, or Variety
- ❌ **No web scraping** of social media or news sites
- ❌ **No database queries** for historical data
- ✅ **Just hardcoded strings**: "48K Mentions", "Sci-Fi Rentals Up 62%", etc.

**Display:**
- Shows: Static metrics that never update

---

## Summary: Zero Computation

### What's Actually Happening:

```
Dashboard Loads
    ↓
Reads hardcoded arrays:
    - genres_data = [{"name": "Horror", "value": 68}, ...]
    - platforms = [{"name": "Netflix", "trend": "#1..."}, ...]
    - feeds = [{"platform": "IMDb", "metric": "48K..."}, ...]
    ↓
Displays static values
    ↓
Done. No computation. No data analysis. No API calls.
```

### Code Flow:

1. **Dashboard loads** → `if st.session_state.current_tab == "Dashboard":`
2. **Defines hardcoded arrays** → Lines 431-435, 501-505, 529-534
3. **Loops through arrays** → `for genre in genres_data:`
4. **Renders HTML** → `st.markdown(f"...")`
5. **Displays static values** → Always shows the same numbers/text

### No Data Sources Used:

- ❌ `exhibitions_df` - **Not used** for trends
- ❌ `library_df` - **Not used** for trends
- ❌ `recent_recs` - **Not used** for trends
- ❌ External APIs - **Not called**
- ❌ Database queries - **Not executed**
- ❌ File reads - **Not performed** (except for recommendations)

---

## Comparison: What Real Computation Would Look Like

### If trends were actually computed:

```python
# REAL computation (NOT in current code)
if exhibitions_df is not None:
    # Count genres in exhibitions
    all_genres = exhibitions_df['genres'].str.split(',').explode().str.strip()
    genre_counts = all_genres.value_counts()
    
    # Calculate percentages
    total_films = len(exhibitions_df)
    horror_count = genre_counts.get('Horror', 0)
    horror_pct = (horror_count / total_films) * 100
    
    # Compare to previous period (would need historical data)
    # previous_horror_pct = get_previous_period_genre_pct('Horror')
    # trend = horror_pct - previous_horror_pct
    
    genres_data = [
        {"name": "Horror", "value": int(horror_pct), "color": "orange"},
        # ... calculated from actual data
    ]
```

**But this code doesn't exist!** The current code just uses:
```python
genres_data = [
    {"name": "Horror", "value": 68, "color": "orange"},  # Hardcoded!
]
```

---

## Why This Design?

The dashboard was built to match a **UI mockup/design concept**. The design showed:
- Trending genres with percentages
- OTT platform trends
- Real-time data feeds

But the **data sources** for these trends weren't implemented yet. So the values were hardcoded to:
1. **Match the design** visually
2. **Show what the UI would look like** with data
3. **Allow development** of the UI without waiting for data sources

---

## Current State Summary

| Trend Type | Computation? | Data Source | Updates? |
|------------|--------------|-------------|----------|
| Genre Trends | ❌ No | Hardcoded array | ❌ Never |
| OTT Trends | ❌ No | Hardcoded strings | ❌ Never |
| Data Feeds | ❌ No | Hardcoded metrics | ❌ Never |
| Recommendations | ✅ Yes | Real matching algorithm | ✅ Yes (when data exists) |

---

## Key Takeaway

**Trends are NOT computed - they're static placeholder values.**

The dashboard displays:
- **Hardcoded genre percentages** (68%, 52%, 35%)
- **Hardcoded platform trends** (static text)
- **Hardcoded data feed metrics** (static numbers)

**No analysis, no calculation, no data processing** - just displaying predefined values to match the UI design.
