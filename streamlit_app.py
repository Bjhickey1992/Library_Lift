"""
TakeOne - Film Library Monetization Dashboard
Modern dashboard UI for film library recommendations and market insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import html
from chatbot_agent import ChatbotAgent
from config import get_openai_api_key
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

# Streamlit Cloud: expose Secrets as environment variables so config.py works unchanged.
# (Do not overwrite any variables already set in the environment.)
for _k in ("OPENAI_API_KEY", "TMDB_API_KEY", "INTENT_MODEL"):
    try:
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = str(st.secrets[_k])
    except Exception:
        # st.secrets may not be configured locally; ignore
        pass

# Page config
st.set_page_config(
    page_title="TakeOne - Monetize Your Library Content",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Auto-generate embeddings if missing or invalid (for deployment)
@st.cache_resource
def check_embeddings_status():
    """Check if embeddings exist and are valid. Returns status dict."""
    from pathlib import Path
    
    lib_path = Path("lionsgate_library_embeddings.npy")
    ex_path = Path("upcoming_exhibitions_embeddings.npy")
    
    lib_valid = lib_path.exists() and not np.allclose(np.load(lib_path), 0)
    ex_valid = ex_path.exists() and not np.allclose(np.load(ex_path), 0)
    
    return {
        "lib_valid": lib_valid,
        "ex_valid": ex_valid,
        "both_valid": lib_valid and ex_valid,
        "lib_path": lib_path,
        "ex_path": ex_path,
    }

# Check embeddings on startup
embeddings_status = check_embeddings_status()

# Show warning if embeddings are missing/invalid (but don't allow public generation)
if not embeddings_status["both_valid"]:
    st.error("⚠️ **Embeddings Missing or Invalid**")
    
    st.markdown("""
    **Recommendations require embedding files that are not currently available.**
    
    The embedding files (`lionsgate_library_embeddings.npy` and `upcoming_exhibitions_embeddings.npy`) 
    need to be generated and committed to the repository by the app administrators.
    
    **For App Administrators:**
    
    To generate embeddings, run these commands in your terminal or Streamlit Cloud console:
    
    ```bash
    python generate_library_embeddings.py
    python generate_exhibition_embeddings.py
    ```
    
    Then commit the generated `.npy` files to git:
    
    ```bash
    git add lionsgate_library_embeddings.npy upcoming_exhibitions_embeddings.npy
    git commit -m "Add embedding files"
    git push
    ```
    
    After pushing, Streamlit Cloud will automatically redeploy with the embeddings included.
    """)
    
    st.warning("⚠️ **Recommendations will not work until embeddings are generated and committed to the repository.**")

# Helper functions for calculating trends from real data
def calculate_trending_genres(exhibitions_df: pd.DataFrame, top_n: int = 3) -> List[Dict]:
    """
    Calculate trending genres from exhibition data.
    Returns list of genres with their percentage of total exhibitions.
    """
    if exhibitions_df is None or len(exhibitions_df) == 0:
        return []
    
    # Split genres (comma-separated) and count
    all_genres = exhibitions_df['genres'].astype(str).str.split(',').explode().str.strip()
    # Remove empty strings and 'nan'
    all_genres = all_genres[all_genres.notna() & (all_genres != '') & (all_genres != 'nan')]
    
    if len(all_genres) == 0:
        return []
    
    genre_counts = all_genres.value_counts()
    total_genre_mentions = len(all_genres)
    
    # Calculate percentages and get top N
    genres_data = []
    color_map = {"Horror": "orange", "Sci-Fi": "blue", "Science Fiction": "blue", 
                 "Comedy": "blue", "Drama": "green", "Action": "orange", 
                 "Thriller": "orange", "Romance": "green"}
    
    for genre, count in genre_counts.head(top_n).items():
        pct = (count / total_genre_mentions) * 100
        color = color_map.get(genre, "blue")
        genres_data.append({
            "name": genre,
            "value": int(pct),
            "color": color
        })
    
    return genres_data

def _recommendations_intro(count: int, territory: str, query_type: str, trends: Optional[Dict] = None) -> str:
    """Short intro line for recommendations. No per-film text; details go in poster rows only."""
    if count == 0:
        return f"No recommendations found for {territory}."
    territory_str = f" ({territory})" if territory and territory != "Unknown" else ""
    if query_type == "trend":
        return f"**Based on current trends in theaters{territory_str}:** Here are **{count}** library titles that match your query."
    return f"**Recommendations for {territory}:** Here are **{count}** library titles that match your query."


def get_trending_films(exhibitions_df: pd.DataFrame, top_n: int = 2) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Get most popular films by venue count.
    Returns (now_showing_film, rerelease_film) tuples with metadata.
    """
    if exhibitions_df is None or len(exhibitions_df) == 0:
        return None, None
    
    current_year = datetime.now().year
    
    # Count venues per film
    film_stats = exhibitions_df.groupby('title').agg({
        'location': 'nunique',  # Number of unique venues
        'tmdb_id': 'first',
        'release_year': 'first',
        'genres': 'first',
        'country': lambda x: ', '.join(x.unique()[:2])  # First 2 countries
    }).reset_index()
    film_stats.columns = ['title', 'venue_count', 'tmdb_id', 'release_year', 'genres', 'countries']
    
    # Categorize films
    film_stats['film_type'] = film_stats['release_year'].apply(
        lambda year: 'NOW SHOWING' if pd.notna(year) and (current_year - int(year)) <= 3
        else 'RE-RELEASE HIT' if pd.notna(year) and (current_year - int(year)) > 5
        else 'OTHER'
    )
    
    # Get top NOW SHOWING film
    now_showing = film_stats[film_stats['film_type'] == 'NOW SHOWING'].sort_values('venue_count', ascending=False)
    now_showing_film = None
    if len(now_showing) > 0:
        top = now_showing.iloc[0]
        now_showing_film = {
            'title': top['title'],
            'tmdb_id': top['tmdb_id'],
            'venue_count': top['venue_count'],
            'release_year': int(top['release_year']) if pd.notna(top['release_year']) else None
        }
    
    # Get top RE-RELEASE HIT film
    rerelease = film_stats[film_stats['film_type'] == 'RE-RELEASE HIT'].sort_values('venue_count', ascending=False)
    rerelease_film = None
    if len(rerelease) > 0:
        top = rerelease.iloc[0]
        rerelease_film = {
            'title': top['title'],
            'tmdb_id': top['tmdb_id'],
            'venue_count': top['venue_count'],
            'release_year': int(top['release_year']) if pd.notna(top['release_year']) else None
        }
    
    return now_showing_film, rerelease_film

# Custom CSS for Ivory theme and modern styling
st.markdown("""
<style>
    /* Ivory base theme */
    :root {
        --ivory: #FFFFF0;
        --ivory-warm: #F5F3E8;
        --ivory-card: #FFFEF4;
        /* Higher-contrast borders/dividers while staying warm */
        --ivory-border: #CFCFBE;
        --ivory-divider: #BDBDAE;
        --primary-orange: #E85D04;
        --primary-blue: #2563EB;
        --primary-green: #059669;
        --text-primary: #1A1A1A;
        --text-secondary: #3F3F38;
        --text-muted: #5A5A52;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    /* Background: Ivory */
    .stApp {
        background: var(--ivory);
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, var(--ivory-card) 0%, var(--ivory-warm) 100%);
        padding: 1.5rem 2rem;
        border-bottom: 2px solid var(--primary-orange);
        margin-bottom: 2rem;
        border: 1px solid var(--ivory-border);
        border-radius: 0 0 12px 12px;
        box-shadow: 0 3px 14px rgba(0,0,0,0.08);
    }
    
    .logo-text {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
        letter-spacing: 2px;
    }
    
    .tagline {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
        letter-spacing: 1px;
    }
    
    /* Navigation tabs */
    .nav-tabs {
        display: flex;
        gap: 0;
        margin-top: 1rem;
        border-bottom: 1px solid var(--ivory-divider);
    }
    
    .nav-tab {
        padding: 0.75rem 1.5rem;
        color: var(--text-secondary);
        cursor: pointer;
        border-bottom: 3px solid transparent;
        transition: all 0.3s;
        font-weight: 500;
    }
    
    .nav-tab:hover {
        color: var(--text-primary);
    }
    
    .nav-tab.active {
        color: var(--primary-blue);
        border-bottom-color: var(--primary-blue);
    }
    
    /* Card styling */
    .dashboard-card {
        background: var(--ivory-card);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid var(--ivory-border);
        box-shadow: 0 3px 14px rgba(0,0,0,0.08);
    }
    
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .see-all-btn {
        color: var(--primary-blue);
        font-size: 0.85rem;
        cursor: pointer;
        text-decoration: none;
    }
    
    /* Progress bars */
    .progress-container {
        margin: 0.75rem 0;
    }
    
    .progress-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.25rem;
        font-size: 0.85rem;
        color: var(--text-secondary);
    }
    
    .progress-bar {
        height: 8px;
        background: var(--ivory-divider);
        border-radius: 4px;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s;
    }
    
    .progress-orange { background: var(--primary-orange); }
    .progress-blue { background: var(--primary-blue); }
    .progress-green { background: var(--primary-green); }
    
    /* Metric cards */
    .metric-card {
        background: var(--ivory-warm);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid var(--ivory-border);
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Movie poster container */
    .poster-container {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .poster-card {
        flex: 1;
        position: relative;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--ivory-border);
    }
    
    .poster-label {
        position: absolute;
        top: 0.5rem;
        left: 0.5rem;
        background: rgba(0, 0, 0, 0.75);
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        color: white;
    }
    
    .label-now-showing { background: var(--primary-orange); }
    .label-rerelease { background: var(--primary-blue); }
    
    /* Recommendation item */
    .rec-item {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.75rem;
        background: var(--ivory-warm);
        border-radius: 8px;
        margin-bottom: 0.75rem;
        border: 1px solid var(--ivory-border);
        border-left: 3px solid var(--primary-orange);
        transition: all 0.2s;
    }
    
    .rec-item:hover {
        background: var(--ivory-card);
        transform: translateX(4px);
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    
    .rec-thumbnail {
        width: 60px;
        height: 80px;
        border-radius: 4px;
        object-fit: cover;
    }
    
    .rec-content {
        flex: 1;
    }
    
    .rec-title {
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }
    
    .rec-descriptor {
        font-size: 0.8rem;
        color: var(--text-secondary);
    }
    
    .rec-tag {
        display: inline-block;
        background: var(--primary-orange);
        color: white;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    
    /* Platform logo container */
    .platform-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem;
        background: var(--ivory-warm);
        border-radius: 8px;
        margin-bottom: 0.75rem;
        border: 1px solid var(--ivory-border);
    }
    
    .platform-logo {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: var(--ivory-divider);
        color: var(--text-primary);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
    }
    
    /* Global text colors for Ivory theme */
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: var(--text-primary);
    }
    
    /* Streamlit native widgets: light surfaces */
    .stButton > button {
        color: var(--text-primary) !important;
        background-color: var(--ivory-card) !important;
        border: 1px solid var(--ivory-border) !important;
    }
    
    .stButton > button:hover {
        background-color: var(--ivory-warm) !important;
        border-color: var(--primary-blue) !important;
        color: var(--primary-blue) !important;
    }
    
    /* Chat message bubbles, inputs, etc. */
    [data-testid="stChatMessage"] {
        background: var(--ivory-card);
        border: 1px solid var(--ivory-border);
        border-radius: 12px;
    }
    
    [data-testid="stSidebar"] {
        background: var(--ivory-warm);
    }
    
    [data-testid="stSidebar"] * {
        color: var(--text-primary);
    }

    /* Floating chatbot button (Dashboard) */
    .floating-chatbot-btn {
        position: fixed;
        right: 24px;
        bottom: 24px;
        z-index: 9999;
        background: var(--primary-blue);
        color: white !important;
        padding: 12px 16px;
        border-radius: 999px;
        text-decoration: none;
        font-weight: 700;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 16px rgba(37,99,235,0.35);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .floating-chatbot-btn:hover {
        background: #1d4ed8;
        color: white !important;
        text-decoration: none;
        box-shadow: 0 6px 20px rgba(37,99,235,0.4);
    }

    /* Why this film: clean outline, subtle background, lifts off page */
    .reasoning-block {
        background: var(--ivory-card);
        border: 1px solid var(--ivory-border);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        box-shadow: 0 3px 14px rgba(0,0,0,0.08);
    }
    .reasoning-block .reasoning-title {
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 0.5rem 0;
        font-size: 0.95rem;
    }
    .reasoning-block .reasoning-text {
        color: var(--text-secondary);
        margin: 0;
        font-size: 0.9rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

STUDIO_OPTIONS = ["Universal Pictures", "Lionsgate"]


def _init_chatbot_agent(studio_name: str) -> None:
    """(Re)initialize the agent for the selected library/studio."""
    try:
        st.session_state.chatbot_agent = ChatbotAgent(studio_name=studio_name)
        st.session_state.initialized = True
        st.session_state.init_error = None
        st.session_state.agent_studio = studio_name
    except Exception as e:
        st.session_state.initialized = False
        st.session_state.init_error = str(e)
        st.session_state.agent_studio = studio_name


def _on_library_change() -> None:
    """Handle switching between Universal/Lionsgate libraries."""
    # Clear chat histories to avoid mixing libraries
    st.session_state.messages = []
    if "dashboard_messages" in st.session_state:
        st.session_state.dashboard_messages = []
    _init_chatbot_agent(st.session_state.selected_studio)


# Initialize session state
if "selected_studio" not in st.session_state:
    st.session_state.selected_studio = "Universal Pictures"
elif st.session_state.selected_studio not in STUDIO_OPTIONS:
    st.session_state.selected_studio = "Universal Pictures"

if "agent_studio" not in st.session_state:
    st.session_state.agent_studio = None

if "chatbot_agent" not in st.session_state or st.session_state.agent_studio != st.session_state.selected_studio:
    _init_chatbot_agent(st.session_state.selected_studio)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Dashboard"

# Check initialization
if not st.session_state.get("initialized", False):
    st.error(f"❌ Failed to initialize: {st.session_state.get('init_error', 'Unknown error')}")
    st.info("Make sure you have API keys configured and data files ready.")
    st.stop()

# Header Section
st.markdown("""
<div class="header-container">
    <div>
        <h1 class="logo-text">TakeOne</h1>
        <p class="tagline">MONETIZE YOUR LIBRARY CONTENT</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation Tabs
col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 10])
with col1:
    if st.button("Dashboard", key="nav_dashboard", use_container_width=True):
        st.session_state.current_tab = "Dashboard"
        st.rerun()
with col2:
    if st.button("Recommendations", key="nav_recommendations", use_container_width=True):
        st.session_state.current_tab = "Recommendations"
        st.rerun()
with col3:
    if st.button("OTT Trends", key="nav_ott", use_container_width=True):
        st.session_state.current_tab = "OTT Trends"
        st.rerun()
with col4:
    if st.button("Market Insights", key="nav_insights", use_container_width=True):
        st.session_state.current_tab = "Market Insights"
        st.rerun()
with col5:
    st.selectbox(
        "Library",
        options=STUDIO_OPTIONS,
        key="selected_studio",
        on_change=_on_library_change,
        help="Switch between Universal and Lionsgate library data/embeddings.",
    )

st.markdown("<hr style='border-color: #D4D4C8; margin: 1rem 0;'>", unsafe_allow_html=True)

# Main Content based on selected tab
if st.session_state.current_tab == "Dashboard":
    # Load data for dashboard
    agent = st.session_state.chatbot_agent
    library_df = None
    exhibitions_df = None
    data_files_missing = False
    
    # Try to load library data (gracefully handle missing files)
    try:
        library_df = agent._load_library()
    except FileNotFoundError as e:
        library_df = None
        data_files_missing = True
    except Exception as e:
        st.warning(f"Could not load library data: {str(e)}")
        library_df = None
    
    # Try to load exhibitions data (gracefully handle missing files)
    try:
        exhibitions_df = agent._load_exhibitions()
    except FileNotFoundError as e:
        exhibitions_df = None
        data_files_missing = True
    except Exception as e:
        st.warning(f"Could not load exhibition data: {str(e)}")
        exhibitions_df = None
    
    # Show info message if data files are missing
    if data_files_missing:
        st.info("ℹ️ **Data files not found.** The dashboard is showing placeholder data. To see real data, run the phase scripts: `python run_phase1.py` to build the library, then `python run_phase2_exhibitions.py` to scrape exhibitions.")
    
    # Metrics: library/exhibition counts only. Skip heavy sample recommendations on load
    # so Dashboard and tab navigation stay fast. Use Recommendations tab for queries.
    total_titles = len(library_df) if library_df is not None else 0
    total_exhibitions = len(exhibitions_df) if exhibitions_df is not None else 0
    vod_opportunities = 0
    
    # Top row: Cinema Trends, OTT Trends
    col1, col2 = st.columns([1, 1])
    
    with col1:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">CURRENT CINEMA TRENDS</div>', unsafe_allow_html=True)
            
            # Get trending films from real data
            now_showing_film, rerelease_film = get_trending_films(exhibitions_df, top_n=2)
            
            # Movie posters - use real data if available
            poster_col1, poster_col2 = st.columns(2)
            with poster_col1:
                if now_showing_film and agent.tmdb_client:
                    try:
                        poster_url = agent._get_poster_url(now_showing_film.get('tmdb_id'))
                        if poster_url:
                            st.markdown(f"""
                            <div class="poster-card">
                                <div class="poster-label label-now-showing">NOW SHOWING</div>
                                <img src="{poster_url}" style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px;" />
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="poster-card">
                                <div class="poster-label label-now-showing">NOW SHOWING</div>
                                <div style="background: #E5E5DA; height: 200px; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #5A5A52; padding: 1rem; text-align: center;">
                                    {now_showing_film['title']}<br/>
                                    <small style="color: #7A7A72;">{now_showing_film.get('release_year', '')}</small>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    except:
                        st.markdown("""
                        <div class="poster-card">
                            <div class="poster-label label-now-showing">NOW SHOWING</div>
                            <div style="background: #E5E5DA; height: 200px; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #5A5A52;">
                                Movie Poster
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="poster-card">
                        <div class="poster-label label-now-showing">NOW SHOWING</div>
                        <div style="background: #E5E5DA; height: 200px; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #5A5A52;">
                            Movie Poster
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with poster_col2:
                if rerelease_film and agent.tmdb_client:
                    try:
                        poster_url = agent._get_poster_url(rerelease_film.get('tmdb_id'))
                        if poster_url:
                            st.markdown(f"""
                            <div class="poster-card">
                                <div class="poster-label label-rerelease">RE-RELEASE HIT</div>
                                <img src="{poster_url}" style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px;" />
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="poster-card">
                                <div class="poster-label label-rerelease">RE-RELEASE HIT</div>
                                <div style="background: #E5E5DA; height: 200px; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #5A5A52; padding: 1rem; text-align: center;">
                                    {rerelease_film['title']}<br/>
                                    <small style="color: #7A7A72;">{rerelease_film.get('release_year', '')}</small>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    except:
                        st.markdown("""
                        <div class="poster-card">
                            <div class="poster-label label-rerelease">RE-RELEASE HIT</div>
                            <div style="background: #E5E5DA; height: 200px; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #5A5A52;">
                                Movie Poster
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="poster-card">
                        <div class="poster-label label-rerelease">RE-RELEASE HIT</div>
                        <div style="background: #E5E5DA; height: 200px; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #5A5A52;">
                            Movie Poster
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('<div style="margin-top: 1.5rem;"><strong style="color: #5A5A52; font-size: 0.9rem;">Trending Genres</strong></div>', unsafe_allow_html=True)
            
            # Calculate trending genres from real data
            genres_data = calculate_trending_genres(exhibitions_df, top_n=3)
            
            # Fallback to placeholder if no data
            if not genres_data:
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
                        <span>{genre['value']}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill progress-{genre['color']}" style="width: {genre['value']}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">OTT TREND ANALYSIS</div>', unsafe_allow_html=True)
            
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
                        <div style="font-weight: 600; color: #1A1A1A; margin-bottom: 0.25rem;">{platform['name']}</div>
                        <div style="font-size: 0.85rem; color: #5A5A52;">{platform['trend']}</div>
                    </div>
                    {f"<div style='color: #E85D04; font-size: 1.2rem;'>{platform.get('icon', '')}</div>" if platform.get('icon') else ""}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Bottom row: Real-time Data Feeds, Catalog Match Overview
    col4, col5 = st.columns([1, 1])
    
    with col4:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">REAL-TIME DATA FEEDS</div>', unsafe_allow_html=True)
            
            # Data feed panels
            feeds = [
                {"platform": "IMDb", "metric": "48K Mentions"},
                {"platform": "JustWatch", "metric": "Sci-Fi Rentals Up 62%"},
                {"platform": "Twitter", "metric": "125K #TimeTravel"},
                {"platform": "Variety News", "metric": "Horror Films in Demand"}
            ]
            
            feed_cols = st.columns(2)
            for idx, feed in enumerate(feeds):
                with feed_cols[idx % 2]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="platform-logo" style="margin: 0 auto 0.5rem;">{feed['platform'][0]}</div>
                        <div style="font-size: 0.75rem; color: #5A5A52; margin-bottom: 0.5rem;">{feed['platform']}</div>
                        <div style="font-size: 0.9rem; color: #1A1A1A; font-weight: 600;">{feed['metric']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">CATALOG MATCH OVERVIEW</div>', unsafe_allow_html=True)
            
            # Metrics
            st.markdown("""
            <div style="margin-bottom: 1.5rem;">
                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                    <div style="display: flex; gap: 0.25rem;">
                        <div style="width: 20px; height: 8px; background: #E85D04; border-radius: 2px;"></div>
                        <div style="width: 20px; height: 8px; background: #E85D04; border-radius: 2px;"></div>
                        <div style="width: 20px; height: 8px; background: #E85D04; border-radius: 2px;"></div>
                        <div style="width: 20px; height: 8px; background: #E85D04; border-radius: 2px;"></div>
                        <div style="width: 20px; height: 8px; background: #E85D04; border-radius: 2px;"></div>
                    </div>
                    <div style="color: #1A1A1A; font-weight: 600;">12 Titles Identified</div>
                </div>
                <div style="color: #5A5A52; font-size: 0.85rem; margin-left: 6rem;">Top VOD Opportunities</div>
            </div>
            
            <div style="margin-bottom: 1.5rem;">
                <div class="progress-label">
                    <span>High ROI Potential</span>
                    <span style="color: #E85D04; font-weight: 600;">$215%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill progress-orange" style="width: 85%;"></div>
                </div>
            </div>
            
            <div>
                <div class="progress-label">
                    <span>License Availability</span>
                    <span style="color: #059669; font-weight: 600;">Ready for OTT Pitch</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill progress-green" style="width: 95%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

    # Floating button (jumps to the assistant section on the Dashboard)
    st.markdown(
        '<a class="floating-chatbot-btn" href="#ask-reel-insight">Ask TakeOne</a>',
        unsafe_allow_html=True
    )

    # --- Producer-friendly assistant (simple, guided) -----------------------------
    st.markdown('<div id="ask-reel-insight"></div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">ASK TakeOne <span style="color:#5A5A52; font-weight:500;">(Your assistant)</span></div>', unsafe_allow_html=True)

    st.markdown(
        "Ask in plain English. If you’re not sure what to ask, use one of the buttons below."
    )

    if "dashboard_messages" not in st.session_state:
        st.session_state.dashboard_messages = []

    # Big, guided controls
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        dashboard_mode = st.radio(
            "What do you want?",
            options=["What’s trending (no embeddings needed)", "Match our library to what’s showing (needs embeddings)"],
            index=0,
            key="dashboard_mode_radio",
        )
    with c2:
        territory_choice = st.selectbox(
            "Where?",
            options=["US", "UK", "FR", "CA", "MX"],
            index=0,
            key="dashboard_territory_select",
        )
    with c3:
        if st.button("Clear chat", use_container_width=True, key="dashboard_clear_chat"):
            st.session_state.dashboard_messages = []
            st.rerun()

    # One-click questions (easy for less technical users)
    q1, q2, q3 = st.columns([1, 1, 1])
    if q1.button("What’s trending right now?", use_container_width=True, key="dash_q_trending"):
        st.session_state.dashboard_prefill = f"What's trending in theaters right now in the {territory_choice}?"
    if q2.button("What should we emphasize?", use_container_width=True, key="dash_q_emphasize"):
        st.session_state.dashboard_prefill = f"What films should we emphasize based on current trends in theaters in the {territory_choice}?"
    if q3.button("Give me 5 titles", use_container_width=True, key="dash_q_five"):
        st.session_state.dashboard_prefill = f"Which 5 library titles should we emphasize based on current trends in {territory_choice}?"

    # Show chat history
    for message in st.session_state.dashboard_messages[-12:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input (Dashboard only)
    default_prompt = st.session_state.pop("dashboard_prefill", "")
    prompt = st.chat_input("Type your question here…", key="dashboard_chat_input")
    if default_prompt and not prompt:
        prompt = default_prompt

    if prompt:
        # Provide up to the last 10 user prompts for context continuity (refinements).
        history_prompts = [m["content"] for m in st.session_state.dashboard_messages if m.get("role") == "user"][-10:]
        st.session_state.dashboard_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                # If user picked a mode, gently steer query phrasing
                final_prompt = prompt
                if dashboard_mode.startswith("What’s trending"):
                    # Ensure it reads like a trend query
                    if "trend" not in final_prompt.lower() and "trending" not in final_prompt.lower():
                        final_prompt = f"Based on current trends in theaters, {final_prompt}"
                else:
                    # Encourage similarity/matching phrasing (requires embeddings)
                    if "match" not in final_prompt.lower() and "similar" not in final_prompt.lower():
                        final_prompt = f"What library titles should we emphasize this month in the {territory_choice}? {final_prompt}"

                result = st.session_state.chatbot_agent.get_recommendations_for_query(
                    final_prompt,
                    top_n=5,
                    history_prompts=history_prompts,
                )

                if "error" in result:
                    st.error(result["error"])
                    if "suggestion" in result:
                        st.info(f"Suggestion: {result['suggestion']}")
                    response_text = result["error"]
                else:
                    recommendations = result.get("recommendations", [])
                    query_type = result.get("query_type", "similarity")
                    territory = result.get("territory") or territory_choice

                    if query_type == "trend":
                        trends = result.get("trends", {})
                        response_text = f"**Here’s what’s trending in {territory} (and what to emphasize):**\n\n"
                        if trends.get("trending_genres"):
                            top3 = ", ".join([g["name"] for g in trends["trending_genres"][:3]])
                            response_text += f"**Top genres right now:** {top3}\n\n"
                        response_text += st.session_state.chatbot_agent.format_recommendations_for_chat(recommendations)
                    else:
                        response_text = f"**Recommended library titles for {territory}:**\n\n"
                        response_text += st.session_state.chatbot_agent.format_recommendations_for_chat(recommendations)

                    st.markdown(response_text)

                st.session_state.dashboard_messages.append({"role": "assistant", "content": response_text})

    st.markdown('</div>', unsafe_allow_html=True)
    
    # Note: If data files are missing, dashboard will show with placeholder data
    # To see real data, run the phase scripts to generate library and exhibition files

elif st.session_state.current_tab == "Recommendations":
    st.header("📊 Recommendations & Trends Chatbot")
    
    # Sidebar for filters
    with st.sidebar:
        st.header("⚙️ Filters")
        st.write(f"**Library:** {st.session_state.selected_studio}")
        min_sim = st.slider("Minimum Similarity", 0.0, 1.0, 0.5, 0.01)
        max_sim = st.slider("Maximum Similarity", 0.0, 1.0, 0.7, 0.01)
        top_n = st.slider("Number of Recommendations", 1, 10, 5)

        # Embeddings availability hint (per selected library)
        agent = st.session_state.chatbot_agent
        if not os.path.exists(agent.library_embeddings_path):
            st.info(
                f"Embeddings file not found for **{st.session_state.selected_studio}**: "
                f"`{agent.library_embeddings_path}`. "
                "Recommendations require embeddings. Generate them (e.g. run `python generate_library_embeddings.py`)."
            )
    
    # Chat interface for recommendations
    st.markdown("""
    **Ask questions like:**
    - *"What films should we emphasize based on current trends in theaters?"*
    - *"What library titles should we emphasize this month in the US?"*
    - *"What's trending in theaters right now?"*
    - *"Which films match current trends in the UK?"*
    """)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if "recommendations" in message and message["recommendations"]:
                recs = message["recommendations"]
                query_type = message.get("query_type", "similarity")
                
                # Show trend information if it's a trend query
                if query_type == "trend" and "trends" in message:
                    trends = message["trends"]
                    with st.expander("📈 Current Trends in Theaters", expanded=True):
                        if trends.get("trending_genres"):
                            st.write("**Top Genres:**")
                            for genre in trends["trending_genres"][:5]:
                                st.write(f"- {genre['name']}: {genre['percentage']}% ({genre['count']} films)")
                        if trends.get("popular_films"):
                            st.write("**Popular Films:**")
                            for film in trends["popular_films"][:5]:
                                st.write(f"- {film['title']} ({film['release_year']}) - Showing at {film['venue_count']} venues")
                
                for i, rec in enumerate(recs, 1):
                    with st.container():
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if rec.get("poster_url"):
                                st.image(rec["poster_url"], use_column_width=True)
                            else:
                                st.info("Poster not available")
                        with col2:
                            st.subheader(f"{i}. {rec['title']} ({rec['year']})")
                            st.write(f"**Director:** {rec['director']}")
                            st.write(f"**Genres:** {rec['genres']}")
                            
                            # Show trend score or similarity score based on query type
                            if query_type == "trend":
                                if rec.get("trend_score") is not None:
                                    st.write(f"**Trend Alignment Score:** {rec['trend_score']:.3f}")
                                if rec.get("trending_genres"):
                                    st.write(f"**Matches Trending Genres:** {', '.join(rec['trending_genres'])}")
                            else:
                                score = rec.get("relevance_score", rec.get("similarity"))
                                if score is not None:
                                    st.write(f"**Relevance Score:** {float(score):.3f}")
                                if rec.get("exhibition_similarity") is not None and rec.get("query_similarity") is not None:
                                    st.caption(
                                        f"Exhibition match: {float(rec['exhibition_similarity']):.3f} · "
                                        f"Query match: {float(rec['query_similarity']):.3f}"
                                    )
                                if rec.get("matched_exhibition"):
                                    st.write(f"**Matched Exhibition:** {rec['matched_exhibition']} at {rec.get('exhibition_location', 'N/A')}")
                                if rec.get('exhibition_dates'):
                                    st.write(f"**Exhibition Dates:** {rec['exhibition_dates']}")
                            
                            if rec.get('reasoning'):
                                st.markdown("---")
                                r_esc = html.escape(rec['reasoning']).replace('\n', '<br/>')
                                st.markdown(
                                    f'<div class="reasoning-block">'
                                    f'<p class="reasoning-title"><strong>Why this film is recommended</strong></p>'
                                    f'<p class="reasoning-text">{r_esc}</p></div>',
                                    unsafe_allow_html=True
                                )
                        st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask about library recommendations or current trends..."):
        # Provide up to the last 10 user prompts for context continuity (refinements).
        history_prompts = [m["content"] for m in st.session_state.messages if m.get("role") == "user"][-10:]
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            spinner_text = "Analyzing trends and finding recommendations..."
            with st.spinner(spinner_text):
                try:
                    result = st.session_state.chatbot_agent.get_recommendations_for_query(
                        prompt,
                        min_similarity=min_sim,
                        max_similarity=max_sim,
                        top_n=top_n,
                        history_prompts=history_prompts
                    )
                    
                    if "error" in result:
                        st.markdown(result["error"])
                        if "suggestion" in result:
                            st.info(f"💡 **Suggestion:** {result['suggestion']}")
                        # Show instructions for generating embeddings
                        if "embeddings" in result.get("error", "").lower():
                            with st.expander("📝 How to Generate Embeddings", expanded=False):
                                st.markdown("""
                                **To generate embeddings, run these commands in your terminal:**
                                
                                1. **Generate Library Embeddings:**
                                   ```bash
                                   python generate_library_embeddings.py
                                   ```
                                
                                2. **Generate Exhibition Embeddings:**
                                   ```bash
                                   python generate_exhibition_embeddings.py
                                   ```
                                
                                **Note:** Trend-based queries work without embeddings! Try asking:
                                - "What films should we emphasize based on current trends?"
                                - "What's trending in theaters right now?"
                                """)
                        response_text = result["error"]
                        recommendations = []
                        query_type = "error"
                    else:
                        recommendations = result["recommendations"]
                        query_type = result.get("query_type", "similarity")
                        territory = result.get("territory", "Unknown")
                        trends = result.get("trends") if query_type == "trend" else None
                        
                        # Short intro only; details live in poster rows below
                        response_text = _recommendations_intro(
                            len(recommendations), territory, query_type, trends
                        )
                        st.markdown(response_text)
                        
                        # Trend expander (summary only; no film list)
                        if query_type == "trend" and trends:
                            with st.expander("📈 Current Trends in Theaters", expanded=True):
                                if trends.get("trending_genres"):
                                    st.write("**Top Genres:**")
                                    for genre in trends["trending_genres"][:5]:
                                        st.write(f"- {genre['name']}: {genre['percentage']}% ({genre['count']} films)")
                                if trends.get("popular_films"):
                                    st.write("**Popular Films:**")
                                    for film in trends["popular_films"][:5]:
                                        st.write(f"- {film['title']} ({film['release_year']}) - Showing at {film['venue_count']} venues")
                        
                        # One row per recommendation: poster + details only
                        if recommendations:
                            for i, rec in enumerate(recommendations, 1):
                                with st.container():
                                    col1, col2 = st.columns([1, 3])
                                    with col1:
                                        if rec.get("poster_url"):
                                            st.image(rec["poster_url"], use_column_width=True)
                                        else:
                                            st.info("Poster not available")
                                    with col2:
                                        st.subheader(f"{i}. {rec['title']} ({rec['year']})")
                                        st.write(f"**Director:** {rec['director']}")
                                        st.write(f"**Genres:** {rec['genres']}")
                                        
                                        # Show appropriate score based on query type
                                        if query_type == "trend":
                                            if rec.get("trend_score") is not None:
                                                st.write(f"**Trend Alignment Score:** {rec['trend_score']:.3f}")
                                            if rec.get("trending_genres"):
                                                st.write(f"**Matches Trending Genres:** {', '.join(rec['trending_genres'])}")
                                        else:
                                            score = rec.get("relevance_score", rec.get("similarity"))
                                            if score is not None:
                                                st.write(f"**Relevance Score:** {float(score):.3f}")
                                            if rec.get("exhibition_similarity") is not None and rec.get("query_similarity") is not None:
                                                st.caption(
                                                    f"Exhibition match: {float(rec['exhibition_similarity']):.3f} · "
                                                    f"Query match: {float(rec['query_similarity']):.3f}"
                                                )
                                            if rec.get("matched_exhibition"):
                                                st.write(f"**Matched Exhibition:** {rec['matched_exhibition']} at {rec.get('exhibition_location', 'N/A')}")
                                            if rec.get('exhibition_dates'):
                                                st.write(f"**Exhibition Dates:** {rec['exhibition_dates']}")
                                        
                                        if rec.get('reasoning'):
                                            st.markdown("---")
                                            r_esc = html.escape(rec['reasoning']).replace('\n', '<br/>')
                                            st.markdown(
                                                f'<div class="reasoning-block">'
                                                f'<p class="reasoning-title"><strong>Why this film is recommended</strong></p>'
                                                f'<p class="reasoning-text">{r_esc}</p></div>',
                                                unsafe_allow_html=True
                                            )
                                    st.markdown("---")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "recommendations": recommendations if 'recommendations' in locals() else [],
                        "query_type": query_type,
                        "trends": result.get("trends") if query_type == "trend" else None
                    })
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

elif st.session_state.current_tab == "OTT Trends":
    st.header("📺 OTT Trend Analysis")
    st.info("OTT trend analysis features coming soon...")

elif st.session_state.current_tab == "Market Insights":
    st.header("💡 Market Insights")
    st.info("Market insights features coming soon...")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #5A5A52; padding: 2rem 0; font-size: 0.85rem;">
    TakeOne Dashboard | Powered by AI-Powered Film Library Matching
</div>
""", unsafe_allow_html=True)
