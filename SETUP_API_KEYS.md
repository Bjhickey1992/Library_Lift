# Setting Up API Keys

## Quick Setup

### Option 1: Using .env File (Recommended)

1. **Install python-dotenv** (if not already installed):
   ```bash
   pip install python-dotenv
   ```

2. **Create your .env file**:
   - Copy `.env.example` to `.env`:
     ```bash
     # Windows
     copy .env.example .env
     
     # macOS/Linux
     cp .env.example .env
     ```

3. **Edit .env file** and add your actual API keys:
   ```
   TMDB_API_KEY=your_actual_tmdb_key_here
   OPENAI_API_KEY=your_actual_openai_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   ```
   `ANTHROPIC_API_KEY` is used only for generating the "need" field (viewer desires/needs) for Lionsgate library and exhibitions via Claude.

4. **Done!** The scripts will automatically load keys from `.env`

### Option 2: Using Environment Variables

**Windows (PowerShell):**
```powershell
$env:TMDB_API_KEY = "your_tmdb_key"
$env:OPENAI_API_KEY = "your_openai_key"
$env:ANTHROPIC_API_KEY = "your_anthropic_key"
```

**Windows (Command Prompt):**
```cmd
set TMDB_API_KEY=your_tmdb_key
set OPENAI_API_KEY=your_openai_key
set ANTHROPIC_API_KEY=your_anthropic_key
```

**macOS/Linux:**
```bash
export TMDB_API_KEY="your_tmdb_key"
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
```

## Getting API Keys

### TMDB API Key (Free)
1. Go to https://www.themoviedb.org/
2. Create an account (free)
3. Go to Settings → API
4. Request an API key
5. Copy your API key (v3 auth)

### OpenAI API Key
1. Go to https://platform.openai.com/
2. Create an account or sign in
3. Go to API Keys section
4. Create a new secret key
5. Copy your API key (starts with `sk-`)

### Anthropic API Key (optional – for "need" field generation)
1. Go to https://console.anthropic.com/
2. Create an account or sign in
3. Create an API key
4. Set `ANTHROPIC_API_KEY` in your environment or `.env`
5. Required only when running `add_need_field.py` (Lionsgate library and exhibitions viewer-needs)

## Streamlit Cloud deployment

When deploying to **Streamlit Community Cloud**, the app does not read `.env` from the repo. Add secrets in the Cloud dashboard:

1. Open your app on share.streamlit.io → **Settings** (or **Manage app**) → **Secrets**.
2. Add key-value pairs, e.g.:
   - `OPENAI_API_KEY` = your OpenAI key
   - `TMDB_API_KEY` = your TMDB key
   - Optionally: `INTENT_MODEL` = `gpt-4o-mini` (or another model)

The app copies these into the environment so the rest of the code works unchanged. Never commit real keys to the repository.

## Security notes

- ✅ **DO**: Use `.env` file locally (gitignored)
- ✅ **DO**: Use Streamlit Cloud Secrets for deployed apps
- ❌ **DON'T**: Commit `.env` or real keys to version control
- ❌ **DON'T**: Hardcode keys in source code

## Verification

To verify your keys are loaded correctly, run:
```python
from config import get_tmdb_api_key, get_openai_api_key

try:
    tmdb_key = get_tmdb_api_key()
    openai_key = get_openai_api_key()
    print("✅ API keys loaded successfully!")
    print(f"TMDB key: {tmdb_key[:10]}...")
    print(f"OpenAI key: {openai_key[:10]}...")
except ValueError as e:
    print(f"❌ Error: {e}")
```

## Troubleshooting

**Error: "TMDB_API_KEY not found"**
- Make sure `.env` file exists in the project root
- Check that keys are spelled correctly in `.env`
- Verify `.env` file is in the same directory as the scripts

**Error: "ModuleNotFoundError: No module named 'dotenv'"**
- Run: `pip install python-dotenv`

**Keys not loading:**
- Check that `.env` file has no extra spaces
- Ensure keys are on separate lines
- Verify file encoding is UTF-8
