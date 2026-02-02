# Quick Start

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Set API keys

Use a `.env` file in the project root (see [SETUP_API_KEYS.md](SETUP_API_KEYS.md)):

```
TMDB_API_KEY=your_tmdb_key_here
OPENAI_API_KEY=your_openai_key_here
```

Or set the same variables in your environment. Get keys from [TMDB](https://www.themoviedb.org/settings/api) and [OpenAI](https://platform.openai.com/api-keys).

## 3. Run the app

```bash
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`. It expects:

- `lionsgate_library.xlsx` (or your studio library)
- `lionsgate_library_embeddings.npy`
- `upcoming_exhibitions.xlsx`
- `upcoming_exhibitions_embeddings.npy`

If you don’t have these, use the data pipeline below.

## 4. Data pipeline (if starting from scratch)

**Option A – Use prebuilt data**  
If you have `*.xlsx` and `*.npy` from the team, place them in the project root and run the app.

**Option B – Build from scratch**

```bash
# Phase 1: Build library
python run_phase1.py

# Optional: add "need" field (requires ANTHROPIC_API_KEY)
python add_need_field.py

# Library embeddings
python generate_library_embeddings.py

# Phase 2: Exhibitions (scrapes cinemas, adds need, builds exhibition embeddings)
python run_phase2_exhibitions.py
```

Then run the app as in step 3.

## Troubleshooting

- **"API key not found"** – Ensure `.env` exists and keys are set; see [SETUP_API_KEYS.md](SETUP_API_KEYS.md).
- **"Library/Exhibition file not found"** – Run the data pipeline or add the expected files.
- **"Embeddings not found"** – Run `generate_library_embeddings.py` and Phase 2 (which creates exhibition embeddings).
- **Slow or no recommendations** – Check that embeddings and data files exist and match the app’s expected names.
