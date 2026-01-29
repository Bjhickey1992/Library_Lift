# Film Library Recommendation Chatbot

## 🎯 Overview

A Streamlit-based chatbot that provides fast film library recommendations based on upcoming cinema exhibition schedules. Optimized for low latency with intelligent filtering to find the best unintuitive but logical matches.

## ✨ Key Features

### 1. **Low Latency** (< 3 seconds per query)
- ✅ Pre-computed embeddings (no regeneration)
- ✅ Cached data loading
- ✅ Vectorized similarity calculations
- ✅ No re-scraping (uses cached exhibition data)

### 2. **Smart Filtering**
- ✅ Excludes exact matches (similarity > 0.9)
- ✅ Focuses on 0.5-0.7 similarity range (sweet spot)
- ✅ Territory-specific filtering
- ✅ Deduplication (one recommendation per library film)

### 3. **Optimized Recommendations**
- ✅ Returns 3-5 best recommendations per territory
- ✅ Sorted by relevance (highest similarity first)
- ✅ Includes exhibition context (location, dates)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Ensure Pre-built Data Exists
- `lionsgate_library.xlsx`
- `lionsgate_library_embeddings.npy`
- `upcoming_exhibitions.xlsx`
- `upcoming_exhibitions_embeddings.npy`

### 3. Run the App
```bash
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`

## 💬 Example Queries

- "What library titles should we emphasize this month in the US?"
- "Show me recommendations for the UK"
- "What films should we promote in France?"
- "Which titles should we highlight in Canada this month?"

## ⚙️ Configuration

Use the sidebar to adjust:
- **Minimum Similarity**: 0.5 (default) - Lower bound for matches
- **Maximum Similarity**: 0.7 (default) - Upper bound for matches
- **Number of Recommendations**: 5 (default) - How many to return
- **Exclude Exact Matches**: Enabled (default) - Filters out >0.9 similarity

## 🏗️ Architecture

### Components

1. **`chatbot_agent.py`** - Fast recommendation engine
   - Caches data and embeddings
   - Vectorized similarity calculations
   - Smart filtering logic

2. **`streamlit_app.py`** - User interface
   - Chat interface
   - Real-time recommendations
   - Interactive configuration

3. **`mcp_server.py`** - MCP/Function calling integration
   - Tool definitions for LLMs
   - OpenAI function calling support
   - Context management

### Data Flow

```
User Query → Territory Extraction → Fast Matching → Filtering → Top N Selection → Display
```

## 📊 Performance

### Latency Breakdown
- **Data Loading** (first time): ~2 seconds
- **Similarity Calculation**: ~0.5-1 second (vectorized)
- **Filtering & Ranking**: ~0.1 seconds
- **Total Response Time**: ~1-3 seconds

### Optimizations Applied
1. **Pre-computed Embeddings**: Saves ~30 seconds per query
2. **Cached Data**: Saves ~5 seconds per query
3. **Vectorized Operations**: NumPy matrix operations (10-100x faster)
4. **Pre-filtering**: Only calculates enhanced similarity for candidates
5. **No Re-scraping**: Exhibition data cached for 7 days

## 🎯 Filtering Logic

### Similarity Ranges
- **< 0.5**: Too dissimilar (excluded)
- **0.5-0.7**: **Sweet spot** - Unintuitive but logical matches (included)
- **0.7-0.9**: Likely obvious matches (excluded by default)
- **> 0.9**: Exact matches/same film (excluded)

### Example Matches
- ✅ "Buffalo '66" ↔ "The Killing of a Chinese Bookie": 0.65 (included)
- ❌ "The Godfather" ↔ "The Godfather": 0.95 (excluded - same film)
- ❌ "Action Movie" ↔ "Romance Movie": 0.3 (excluded - too dissimilar)

## 🔧 MCP Integration

The chatbot uses MCP-style tool definitions for OpenAI function calling:

### Available Tools
1. **`get_film_recommendations`**
   - Parameters: territory, min_similarity, max_similarity, top_n
   - Returns: List of recommendations

2. **`parse_recommendation_query`**
   - Parameters: query (natural language)
   - Returns: Parsed query with recommendations

### Usage
The Streamlit app automatically uses function calling when available, falling back to direct query parsing if needed.

## 📝 Troubleshooting

**"Library file not found"**
- Run Phase 1 to build library, or use pre-built files

**"Exhibition file not found"**
- Run Phase 2 to build exhibitions, or use pre-built files

**"No recommendations found"**
- Check territory code (US, UK, FR, CA, MX)
- Adjust similarity range (try 0.4-0.8)
- Verify exhibition data exists for that territory

**Slow responses**
- Ensure `.npy` embedding files exist
- Check data files aren't corrupted
- Reduce `top_n` parameter

## 🚀 Future Enhancements

Potential improvements:
- Background refresh of exhibition data
- Pre-compute all territory matches
- Vector database (FAISS) for even faster search
- Multi-territory queries
- Export recommendations to Excel

## 📚 Related Files

- `CHATBOT_SETUP.md` - Detailed setup guide
- `chatbot_agent.py` - Core recommendation engine
- `streamlit_app.py` - User interface
- `mcp_server.py` - MCP/function calling integration
