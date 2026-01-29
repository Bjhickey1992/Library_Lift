# Film Library Cinema Agent

An intelligent agent that matches studio film libraries with upcoming cinema exhibition schedules using enhanced semantic similarity analysis.

## Overview

This agent:
1. **Builds studio film libraries** from TMDB API with semantic enrichment
2. **Scrapes cinema exhibition schedules** from cinema websites
3. **Matches films** using enhanced multi-dimensional similarity analysis

## Features

- **Semantic Enrichment**: LLM-generated thematic, stylistic, and emotional descriptors
- **Enhanced Similarity**: Multi-dimensional matching (thematic, stylistic, personnel)
- **Progressive Processing**: Handles large datasets efficiently
- **Parallel Processing**: Fast LLM reasoning generation

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Keys
```bash
# Windows PowerShell
$env:TMDB_API_KEY = "your_key"
$env:OPENAI_API_KEY = "your_key"

# macOS/Linux
export TMDB_API_KEY="your_key"
export OPENAI_API_KEY="your_key"
```

### 3. Run Phases

**Phase 1: Build Library**
```bash
python run_phase1.py
```

**Phase 2: Enrich Exhibitions**
```bash
python run_phase2_enrich_existing.py
```

**Phase 3: Generate Matches**
```bash
python run_phase3_matching.py
```

## Project Structure

- `film_agent.py` - Main agent implementation
- `cinema_scrapers.py` - Cinema scraping utilities
- `cinemas.yaml` - Cinema configuration
- `run_phase*.py` - Phase execution scripts

## Documentation

- `COLLABORATION_GUIDE.md` - Team onboarding guide
- `SESSION_SUMMARY.md` - Recent improvements
- `SIMILARITY_IMPROVEMENTS.md` - Similarity framework details
- `ENHANCED_SIMILARITY_FRAMEWORK.md` - Framework design

## Requirements

- Python 3.8+
- TMDB API key (free)
- OpenAI API key

## License

[Add your license here]

## Contributors

[Add team members here]
