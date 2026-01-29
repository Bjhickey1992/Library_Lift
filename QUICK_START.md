# Quick Start Guide for Team Members

## 🎯 Get Up and Running in 5 Minutes

### 1. Extract the Package
Unzip the team package to a folder on your computer.

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Your API Keys

**Create `.env` file:**
```bash
# Windows
copy .env.example .env

# macOS/Linux  
cp .env.example .env
```

**Edit `.env` and add your keys:**
```
TMDB_API_KEY=your_tmdb_key_here
OPENAI_API_KEY=your_openai_key_here
```

**Get your API keys:**
- TMDB (free): https://www.themoviedb.org/settings/api
- OpenAI: https://platform.openai.com/api-keys

### 4. Test the Setup

**Option A: Use Prebuilt Data (Fastest)**
```bash
# Test Phase 3 with prebuilt data
python run_phase3_matching.py
```

**Option B: Rebuild Everything**
```bash
# Phase 1: Build library
python run_phase1.py

# Phase 2: Enrich exhibitions
python run_phase2_enrich_existing.py

# Phase 3: Generate matches
python run_phase3_matching.py
```

### 5. You're Ready!

The agent is now set up and ready to use. Check the output files:
- `lionsgate_matches.xlsx` - Film matches with similarity scores

## 📚 Next Steps

- Read `COLLABORATION_GUIDE.md` for detailed information
- Review `SESSION_SUMMARY.md` to understand recent improvements
- Check `ENHANCED_SIMILARITY_FRAMEWORK.md` for framework details

## ❓ Troubleshooting

**"API key not found" error:**
- Make sure `.env` file exists in the project root
- Check that keys are spelled correctly (no extra spaces)
- Verify `python-dotenv` is installed: `pip install python-dotenv`

**"Module not found" error:**
- Run: `pip install -r requirements.txt`

**File permission errors:**
- Close any Excel files that might be open
- Check file write permissions

## 💡 Tips

- Start with Phase 3 to test (uses prebuilt data)
- Prebuilt data files are large (~7-8 MB total)
- Each team member needs their own API keys
- Never commit `.env` to version control

---

**Welcome to the team!** 🎬
