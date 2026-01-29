# Team Setup Package - Film Library Cinema Agent

## 📦 What's Included

This package contains everything your team needs to start working on the agent immediately, including prebuilt data files so they don't have to rebuild from scratch.

## 📁 Files to Share with Team

### Core Code Files (Required)
- ✅ `film_agent.py` - Main agent implementation
- ✅ `cinema_scrapers.py` - Cinema scraping utilities
- ✅ `cinemas.yaml` - Cinema configuration
- ✅ `config.py` - API key configuration module
- ✅ `run_phase1.py` - Phase 1 execution script
- ✅ `run_phase2_enrich_existing.py` - Phase 2 enrichment script
- ✅ `run_phase2_exhibitions.py` - Phase 2 exhibition scraping script
- ✅ `run_phase3_matching.py` - Phase 3 matching script
- ✅ `generate_exhibition_embeddings.py` - Embedding generation utility
- ✅ `requirements.txt` - Python dependencies

### Configuration & Documentation (Required)
- ✅ `.env.example` - API key template (safe to share)
- ✅ `README.md` - Project overview
- ✅ `COLLABORATION_GUIDE.md` - Team onboarding guide
- ✅ `SETUP_API_KEYS.md` - API key setup instructions
- ✅ `SESSION_SUMMARY.md` - Recent improvements summary
- ✅ `SIMILARITY_IMPROVEMENTS.md` - Similarity framework docs
- ✅ `ENHANCED_SIMILARITY_FRAMEWORK.md` - Framework design
- ✅ `API_KEY_MIGRATION.md` - API key migration notes
- ✅ `.gitignore` - Git ignore rules

### Prebuilt Data Files (Optional but Recommended)
- ✅ `lionsgate_library.xlsx` - Pre-built library (421 films)
- ✅ `lionsgate_library_embeddings.npy` - Pre-built library embeddings
- ✅ `lionsgate_library_embeddings.xlsx` - Library embeddings metadata
- ✅ `upcoming_exhibitions.xlsx` - Pre-built exhibition data (158 films)
- ✅ `upcoming_exhibitions_embeddings.npy` - Pre-built exhibition embeddings
- ✅ `upcoming_exhibitions_embeddings.xlsx` - Exhibition embeddings metadata
- ✅ `lionsgate_matches.xlsx` - Example matches output

### Files to EXCLUDE (Do Not Share)
- ❌ `.env` - Contains actual API keys (each member creates their own)
- ❌ `*.ipynb` - Jupyter notebooks (optional, can share if needed)
- ❌ `__pycache__/` - Python cache files
- ❌ `*.pyc` - Compiled Python files
- ❌ Any temporary test/debug scripts

## 🚀 Quick Start for Team Members

### Step 1: Receive the Package
Team members should receive all the files listed above (except `.env`).

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Set Up API Keys
1. Copy `.env.example` to `.env`:
   ```bash
   # Windows
   copy .env.example .env
   
   # macOS/Linux
   cp .env.example .env
   ```

2. Edit `.env` and add your own API keys:
   ```
   TMDB_API_KEY=your_tmdb_key_here
   OPENAI_API_KEY=your_openai_key_here
   ```

3. Get API keys:
   - **TMDB**: https://www.themoviedb.org/settings/api (free)
   - **OpenAI**: https://platform.openai.com/api-keys

### Step 4: Verify Setup
Run a quick test:
```bash
python -c "from config import get_tmdb_api_key, get_openai_api_key; print('Keys loaded:', get_tmdb_api_key()[:10] + '...')"
```

### Step 5: Start Working!
- **Use prebuilt data**: Skip Phase 1 & 2, go straight to Phase 3
- **Or rebuild**: Run Phase 1 & 2 to rebuild from scratch

## 📋 File Sizes (for sharing considerations)

- `lionsgate_library.xlsx` - ~160 KB
- `lionsgate_library_embeddings.npy` - ~5 MB
- `upcoming_exhibitions.xlsx` - ~50 KB
- `upcoming_exhibitions_embeddings.npy` - ~2 MB
- Total data files: ~7-8 MB

## 📤 Sharing Methods

### Option 1: Git Repository (Recommended)
1. Create a repository
2. Commit all files except `.env`
3. Use Git LFS for `.npy` files (or commit directly if < 100MB)
4. Share repository link

### Option 2: Zip File
1. Create a zip with all files (except `.env`)
2. Upload to cloud storage (OneDrive, Google Drive, Dropbox)
3. Share download link
4. Team members extract and set up `.env`

### Option 3: Shared Folder
1. Upload to shared cloud folder
2. Team members download/copy
3. Each creates their own `.env`

## ✅ Verification Checklist

Before sharing, verify:
- [ ] All code files are included
- [ ] `.env` is NOT included
- [ ] `.env.example` IS included
- [ ] Prebuilt data files are included
- [ ] Documentation files are included
- [ ] `.gitignore` is included
- [ ] No hardcoded API keys in any files
- [ ] All scripts use `config.py` for keys

## 🎯 Team Workflow

### For New Team Members:
1. Receive package
2. Set up `.env` with their own keys
3. Test with Phase 3 (uses prebuilt data)
4. Start contributing!

### For Development:
- Use prebuilt data for testing
- Rebuild Phase 1/2 only when needed
- Share improvements via Git/version control

## 📝 Notes

- **Prebuilt data is optional**: Team members can rebuild if they prefer
- **API keys are personal**: Each member uses their own
- **Data files are large**: Consider Git LFS or cloud storage
- **Version control**: Use Git for code, not for data files (or use LFS)
