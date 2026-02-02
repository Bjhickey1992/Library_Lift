# Library Lift – Film Library Recommendation Dashboard

An intelligent dashboard that matches studio film libraries with upcoming cinema exhibitions using semantic similarity. Ask questions in natural language and get pointed recommendations with reasoning.

## Features

- **Streamlit dashboard**: Recommendations and trends tabs; natural-language queries; “More like this” and refinement.
- **Intent-aware matching**: Parses territory, venue, genre, decade, tone, and “nothing like X” from your questions.
- **Pre-computed embeddings**: Library and exhibition embeddings (with thematic, stylistic, emotional, and “need” fields) for fast responses.
- **Data pipeline**: Phase 1 (library), Phase 2 (exhibitions + enrichment), Phase 3 (matching); optional weekly refresh.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API keys

Create a `.env` file in the project root (see [SETUP_API_KEYS.md](SETUP_API_KEYS.md)):

```
TMDB_API_KEY=your_key
OPENAI_API_KEY=your_key
```

Optional: `ANTHROPIC_API_KEY` for “need” field generation; `INTENT_MODEL` to override the intent model (default: gpt-4o-mini).

### 3. Run the app

```bash
streamlit run streamlit_app.py
```

The app runs at `http://localhost:8501`. You need pre-built data and embeddings (see **Data pipeline** below).

## Data pipeline

If you’re starting from scratch or refreshing data:

| Step | Script | Purpose |
|------|--------|---------|
| Phase 1 | `run_phase1.py` | Build studio library from TMDB |
| Phase 2a | `run_phase2_exhibitions.py` | Scrape cinema exhibitions, add “need”, generate exhibition embeddings |
| Phase 2b | `run_phase2_enrich_existing.py` | Enrich existing library (keywords, descriptors) |
| Library embeddings | `generate_library_embeddings.py` | Build library embeddings (run after library + optional `add_need_field.py`) |
| Phase 3 | `run_phase3_matching.py` | Generate match outputs (optional; dashboard uses embeddings directly) |

See [QUICK_START.md](QUICK_START.md) for a short walkthrough. For deployment and secrets (e.g. Streamlit Cloud), see [SETUP_API_KEYS.md](SETUP_API_KEYS.md).

## Project structure

| Path | Purpose |
|------|---------|
| `streamlit_app.py` | Main dashboard (recommendations, trends) |
| `chatbot_agent.py` | Query handling, intent, filtering, scoring |
| `query_intent_parser.py` | Intent parsing (filters, weights, negative filters, refinement) |
| `recommendation_scoring.py` | Ranking (legacy and deep gate/tie/nudge strategies) |
| `film_agent.py` | Film/exhibition data, TMDB, embeddings, similarity |
| `config.py` | API keys and config (env / `.env`) |
| `cinema_scrapers.py` | Cinema site scraping |
| `cinemas.yaml` | Cinema list and URLs |
| `run_phase*.py` | Data pipeline scripts |
| `generate_*_embeddings.py` | Embedding generation |

## Documentation

- **[SETUP_API_KEYS.md](SETUP_API_KEYS.md)** – API keys (local `.env` and Streamlit Cloud Secrets)
- **[QUICK_START.md](QUICK_START.md)** – Install, keys, and first run
- **[CHATBOT_SETUP.md](CHATBOT_SETUP.md)** – How the Streamlit recommendation app works

## Requirements

- Python 3.8+
- TMDB API key (free)
- OpenAI API key

## License

[Add your license here]
