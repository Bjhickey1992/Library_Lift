# Collaboration Guide - Film Library Cinema Agent

## Getting Started for Team Members

### 1. Prerequisites

Team members need:
- Python 3.8+ installed
- Access to the project repository/files
- API keys for:
  - **TMDB API** (free at https://www.themoviedb.org/settings/api)
  - **OpenAI API** (requires account at https://platform.openai.com/)

### 2. Initial Setup

#### Step 1: Clone/Download the Project
```bash
# If using Git:
git clone <repository-url>
cd GenAI-App-Project

# Or download the project folder from shared location
```

#### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- numpy
- openpyxl (for Excel files)
- requests
- pyyaml
- openai
- beautifulsoup4

#### Step 3: Set Up API Keys

**Option A: .env File (Recommended - Most Secure)**

1. Copy the template file:
   ```bash
   # Windows
   copy .env.example .env
   
   # macOS/Linux
   cp .env.example .env
   ```

2. Edit `.env` file and add your actual API keys:
   ```
   TMDB_API_KEY=your_actual_tmdb_key_here
   OPENAI_API_KEY=your_actual_openai_key_here
   ```

3. Install python-dotenv (if not already installed):
   ```bash
   pip install python-dotenv
   ```

4. Done! The scripts will automatically load keys from `.env`

**Option B: Environment Variables**
```bash
# Windows (PowerShell)
$env:TMDB_API_KEY = "your_tmdb_key_here"
$env:OPENAI_API_KEY = "your_openai_key_here"

# Windows (Command Prompt)
set TMDB_API_KEY=your_tmdb_key_here
set OPENAI_API_KEY=your_openai_key_here

# macOS/Linux
export TMDB_API_KEY="your_tmdb_key_here"
export OPENAI_API_KEY="your_openai_key_here"
```

**Option B: Update Scripts Directly**
Edit the phase scripts (`run_phase1.py`, `run_phase2_enrich_existing.py`, `run_phase3_matching.py`) and replace the API keys in the `os.environ` lines.

**Option C: Use .env File (Most Secure)**
1. Create a `.env` file in the project root:
```
TMDB_API_KEY=your_tmdb_key_here
OPENAI_API_KEY=your_openai_key_here
```

2. Install python-dotenv:
```bash
pip install python-dotenv
```

3. Add to the top of each script:
```python
from dotenv import load_dotenv
load_dotenv()
```

### 3. Project Structure

```
GenAI-App-Project/
├── film_agent.py              # Main agent code (all phases)
├── cinema_scrapers.py          # Cinema scraping utilities
├── cinemas.yaml               # Cinema configuration
├── run_phase1.py              # Phase 1: Build library
├── run_phase2_enrich_existing.py  # Phase 2: Enrich exhibitions
├── run_phase3_matching.py     # Phase 3: Match films
├── requirements.txt           # Python dependencies
├── SESSION_SUMMARY.md         # Recent improvements
├── SIMILARITY_IMPROVEMENTS.md # Similarity framework docs
├── ENHANCED_SIMILARITY_FRAMEWORK.md  # Framework design
└── COLLABORATION_GUIDE.md     # This file
```

### 4. Running the Agent

#### Phase 1: Build Studio Library
```bash
python run_phase1.py
```
- Fetches films from TMDB API
- Generates semantic descriptors (themes, style, tone)
- Creates embeddings
- Output: `lionsgate_library.xlsx` and `lionsgate_library_embeddings.npy`

#### Phase 2: Enrich Exhibition Data
```bash
python run_phase2_enrich_existing.py
```
- Enriches existing `upcoming_exhibitions.xlsx`
- Adds keywords, tagline, semantic descriptors
- Regenerates embeddings
- Output: Updated `upcoming_exhibitions.xlsx` and embeddings

#### Phase 3: Match Films
```bash
python run_phase3_matching.py
```
- Loads library and exhibition data
- Calculates enhanced similarity scores
- Generates matches with reasoning
- Output: `lionsgate_matches.xlsx`

### 5. Key Concepts

#### Enhanced Similarity Framework
The agent uses a multi-dimensional similarity calculation:
- **Base similarity**: Full embedding cosine similarity (foundation)
- **Component similarities**: Thematic, stylistic, personnel matches
- **Post-processing boosts**: Explicit shared attributes (actors, themes, styles)

#### Data Flow
1. **Library Building**: TMDB API → Excel + Embeddings
2. **Exhibition Scraping**: Cinema websites → Excel + Embeddings
3. **Matching**: Library + Exhibitions → Enhanced Similarity → Matches

### 6. Configuration Files

#### `cinemas.yaml`
Contains cinema sources for scraping. Each entry includes:
- Cinema name and location
- Programme URL
- Country and city
- Scraper type

To add new cinemas, edit this file following the existing format.

### 7. Common Tasks

#### Adding a New Studio
1. Update `studio_name` in `run_phase1.py`
2. Run Phase 1 to build the library
3. Run Phase 3 to generate matches

#### Adding New Cinemas
1. Edit `cinemas.yaml`
2. Add cinema entry with `enabled: true`
3. Run Phase 2 to scrape new cinemas

#### Adjusting Similarity Weights
Edit `_calculate_enhanced_similarity()` in `film_agent.py`:
- Modify multiplicative factors
- Adjust boost amounts
- Change component weights

### 8. Troubleshooting

#### API Key Errors
- Verify keys are set correctly
- Check API key validity
- Ensure keys have proper permissions

#### File Permission Errors
- Close Excel files before running scripts
- Check file write permissions

#### Import Errors
- Run `pip install -r requirements.txt`
- Verify Python version (3.8+)

#### Low Similarity Scores
- Check if semantic descriptors are populated
- Verify embeddings were generated correctly
- Review similarity calculation parameters

### 9. Best Practices

1. **Version Control**: Use Git to track changes
2. **API Keys**: Never commit API keys to repository
3. **Testing**: Test on small datasets first
4. **Documentation**: Document any changes you make
5. **Backup**: Keep backups of important Excel files

### 10. Getting Help

- Review `SESSION_SUMMARY.md` for recent changes
- Check `SIMILARITY_IMPROVEMENTS.md` for framework details
- Read `ENHANCED_SIMILARITY_FRAMEWORK.md` for design decisions
- Check code comments in `film_agent.py`

### 11. Sharing the Project

#### Option A: Git Repository (Recommended)
1. Create a repository (GitHub, GitLab, etc.)
2. Add `.gitignore` to exclude:
   - `*.xlsx` (data files)
   - `*.npy` (embedding files)
   - `.env` (API keys)
   - `__pycache__/`
3. Share repository link with team

#### Option B: Shared Drive/Folder
1. Share project folder via OneDrive, Google Drive, etc.
2. Team members download/copy the folder
3. Each member sets up their own API keys

#### Option C: Cloud Development Environment
- Use GitHub Codespaces, GitPod, or similar
- Share environment configuration
- Team members work in isolated cloud environments

### 12. Team Workflow Suggestions

1. **Feature Branches**: Create branches for new features
2. **Code Reviews**: Review changes before merging
3. **Documentation**: Update docs when making changes
4. **Testing**: Test changes before committing
5. **Communication**: Discuss major changes with team

### 13. Data Files to Share (Optional)

If team members need existing data:
- `lionsgate_library.xlsx` - Pre-built library
- `upcoming_exhibitions.xlsx` - Exhibition data
- `*.npy` files - Pre-generated embeddings (large files)

**Note**: These files can be large. Consider sharing via:
- Cloud storage (OneDrive, Google Drive)
- Shared network drive
- Git LFS (Large File Storage)

---

## Quick Start Checklist

- [ ] Python 3.8+ installed
- [ ] Project files downloaded/cloned
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] TMDB API key obtained and set
- [ ] OpenAI API key obtained and set
- [ ] Test run Phase 1 (small dataset first)
- [ ] Review documentation files
- [ ] Understand project structure

---

**Welcome to the team!** 🎬
