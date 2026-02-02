# Streamlit Recommendation App

The dashboard (`streamlit run streamlit_app.py`) provides an interactive interface for film library recommendations based on exhibition schedules and natural-language queries.

## What you need

- Pre-built library and exhibition Excel files
- Pre-computed `.npy` embeddings for both
- `OPENAI_API_KEY` and `TMDB_API_KEY` (see [SETUP_API_KEYS.md](SETUP_API_KEYS.md))

## Running the app

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501`. Use the **Recommendations** tab to ask questions; the **Trends** tab for trend-based queries.

## Example queries

- "Show me spy movies to emphasize in the US."
- "What can we push in Berlin that feels like late-night European art house?"
- "Films similar to whatever is doing well at Film Forum this month."
- "Nothing like comedy—drama from the 80s."
- "Narrow to the 80s." / "Drop the romance filter." (refinement)
- After seeing results: click **More like this** on a title, or say "like the second one."

## How it works

1. Your query is parsed for intent (territory, venue, genre, decade, exclusions, refinement).
2. Library and exhibitions are filtered (with fallbacks when filters are too strict).
3. Recommendations are ranked using exhibition similarity and query relevance (gate/tie/nudge).
4. Reasoning and posters are generated and shown.

## Configuration

- **Minimum / maximum similarity** and **number of recommendations** are in the sidebar.
- Keys are read from the environment or Streamlit Cloud Secrets; users never enter keys in the app.

## Deployment

For **Streamlit Community Cloud**: connect the repo, then set `OPENAI_API_KEY` and `TMDB_API_KEY` in the app’s **Secrets** in the Cloud dashboard. See [SETUP_API_KEYS.md](SETUP_API_KEYS.md).
