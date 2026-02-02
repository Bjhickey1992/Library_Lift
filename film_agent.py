import os
import json
import datetime as dt
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import pandas as pd
import requests
import yaml

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from cinema_scrapers import (
    CinemaSource,
    Screening,
    filter_screenings_to_window,
    normalize_title_for_lookup,
    scrape_programme,
)


TMDB_API_BASE = "https://api.themoviedb.org/3"


@dataclass
class FilmRecord:
    """Represents a single film in a studio library or exhibition schedule."""

    title: str
    release_year: Optional[int]
    director: str
    writers: str
    producers: str
    cinematographers: str
    production_designers: str
    cast: str
    genres: str
    tmdb_id: Optional[int] = None
    lead_gender: Optional[str] = None  # "female", "male", or None - gender of first cast member (lead actor)
    overview: Optional[str] = None  # Film overview/plot summary from TMDB
    reviews: Optional[str] = None  # User reviews from TMDB (concatenated)
    keywords: Optional[str] = None  # Keywords from TMDB (comma-separated)
    tagline: Optional[str] = None  # Tagline from TMDB
    thematic_descriptors: Optional[str] = None  # LLM-generated thematic descriptors
    stylistic_descriptors: Optional[str] = None  # LLM-generated stylistic descriptors
    emotional_tone: Optional[str] = None  # LLM-generated emotional tone description

    # Exhibition-specific fields (optional for library records)
    country: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = None  # ISO format YYYY-MM-DD
    end_date: Optional[str] = None    # ISO format YYYY-MM-DD
    scheduled_dates: Optional[str] = None  # comma-separated ISO dates
    programme_url: Optional[str] = None


@dataclass
class MatchRecord:
    """Represents a similarity match between a library film and an exhibition programme."""

    library_title: str
    exhibition_title: str
    country: str
    location: str
    relevance_start: str
    relevance_end: str
    similarity_reason: str
    programme_url: str
    cosine_similarity: float


class TMDbClient:
    """
    Lightweight TMDb client.

    You must provide a TMDb API key (v3) via:
    - tmdb_api_key argument, or
    - TMDB_API_KEY environment variable.
    """

    def __init__(self, tmdb_api_key: Optional[str] = None, session: Optional[requests.Session] = None):
        self.api_key = tmdb_api_key or os.getenv("TMDB_API_KEY")
        if not self.api_key:
            raise ValueError("TMDB API key is required. Set TMDB_API_KEY env var or pass tmdb_api_key.")
        self.session = session or requests.Session()

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        params = params or {}
        params.setdefault("api_key", self.api_key)
        url = f"{TMDB_API_BASE}{path}"
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # ---- Company / studio helpers -------------------------------------------------

    def search_company(self, query: str, limit: int = 5) -> List[Dict]:
        data = self._get("/search/company", {"query": query})
        return data.get("results", [])[:limit]

    def discover_movies_for_company(
        self,
        company_id: int,
        max_pages: Optional[int] = None,
        include_adult: bool = False,
        min_release_date: str = "1920-01-01",
        max_release_date: Optional[str] = None,
    ) -> List[Dict]:
        """
        Discover movies associated with a company (studio / distributor).
        Fetches ALL pages unless max_pages is specified.

        Note: This is an approximation of "rights owned". TMDb models
        production and distribution relationships, but it does not track
        current contractual rights.
        """
        if max_release_date is None:
            max_release_date = dt.datetime.now().strftime("%Y-%m-%d")
        
        movies: List[Dict] = []
        page = 1

        while True:
            params = {
                "with_companies": company_id,
                "include_adult": str(include_adult).lower(),
                "sort_by": "primary_release_date.asc",
                "page": page,
                "primary_release_date.gte": min_release_date,
                "primary_release_date.lte": max_release_date,
            }
            
            data = self._get("/discover/movie", params)
            results = data.get("results", [])
            movies.extend(results)
            
            total_pages = data.get("total_pages", 1)
            if page >= total_pages:
                break
            if max_pages is not None and page >= max_pages:
                break
            page += 1
            
            # Progress indicator for large libraries
            if page % 10 == 0:
                print(f"  Fetched {len(movies)} films so far (page {page}/{total_pages})...")

        return movies

    def get_movie_details_with_credits(self, movie_id: int) -> Dict:
        return self._get(
            f"/movie/{movie_id}",
            {"append_to_response": "credits,keywords"},
        )
    
    def get_person_details(self, person_id: int) -> Dict:
        """Get person details including gender from TMDb API."""
        return self._get(f"/person/{person_id}")
    
    def get_movie_keywords(self, movie_id: int) -> List[str]:
        """Get keywords for a movie from TMDB API."""
        try:
            data = self._get(f"/movie/{movie_id}/keywords")
            keywords = data.get("keywords", [])
            return [kw.get("name", "") for kw in keywords if kw.get("name")]
        except requests.HTTPError:
            return []
    
    def get_movie_reviews(self, movie_id: int, max_pages: int = 2) -> List[Dict]:
        """Get user reviews for a movie. Returns up to max_pages of reviews."""
        reviews: List[Dict] = []
        page = 1
        
        while page <= max_pages:
            try:
                data = self._get(f"/movie/{movie_id}/reviews", {"page": page})
                results = data.get("results", [])
                if not results:
                    break
                reviews.extend(results)
                total_pages = data.get("total_pages", 1)
                if page >= total_pages:
                    break
                page += 1
            except requests.HTTPError:
                break
        
        return reviews

    # ---- Movie lookup helpers (for scraped exhibition titles) ---------------------

    def search_movie(self, query: str, year: Optional[int] = None, limit: int = 5) -> List[Dict]:
        params: Dict = {"query": query}
        if year:
            params["year"] = year
        data = self._get("/search/movie", params)
        return data.get("results", [])[:limit]


# ---- Parsing helpers --------------------------------------------------------------

def _names_from_crew(crew: Iterable[Dict], jobs: Iterable[str]) -> List[str]:
    jobs_lower = {j.lower() for j in jobs}
    out: List[str] = []
    for person in crew:
        job = (person.get("job") or "").lower()
        if job in jobs_lower:
            name = person.get("name")
            if name and name not in out:
                out.append(name)
    return out


def _genres_to_str(genres: Iterable[Dict]) -> str:
    return ", ".join(g.get("name") for g in genres if g.get("name"))


def _cast_to_str(cast_list: Iterable[Dict], max_cast: int = 10) -> str:
    names = []
    for person in cast_list:
        name = person.get("name")
        if name and name not in names:
            names.append(name)
        if len(names) >= max_cast:
            break
    return ", ".join(names)


def film_record_from_tmdb_details(
    details: Dict,
    *,
    country: Optional[str] = None,
    location: Optional[str] = None,
    default_run_days: int = 28,
    scheduled_dates: Optional[str] = None,
    programme_url: Optional[str] = None,
) -> FilmRecord:
    title = details.get("title") or details.get("original_title") or ""
    release_date_str = details.get("release_date")
    release_year: Optional[int]
    if release_date_str:
        try:
            release_year = int(release_date_str[:4])
        except ValueError:
            release_year = None
    else:
        release_year = None

    credits = details.get("credits") or {}
    cast = credits.get("cast") or []
    crew = credits.get("crew") or []

    director_names = _names_from_crew(crew, ["Director"])
    writer_names = _names_from_crew(
        crew,
        ["Writer", "Screenplay", "Story", "Teleplay"],
    )
    producer_names = _names_from_crew(crew, ["Producer", "Executive Producer"])
    cinematographer_names = _names_from_crew(
        crew,
        ["Director of Photography", "Cinematography"],
    )
    production_designer_names = _names_from_crew(
        crew,
        ["Production Design", "Production Designer"],
    )

    genres_str = _genres_to_str(details.get("genres") or [])
    cast_str = _cast_to_str(cast)

    # Exhibition window: simple heuristic from release_date
    start_date_str: Optional[str] = None
    end_date_str: Optional[str] = None
    if country is not None:
        # treat as exhibition data point when country is provided
        if release_date_str:
            try:
                start_date = dt.datetime.strptime(release_date_str, "%Y-%m-%d").date()
                end_date = start_date + dt.timedelta(days=default_run_days)
                start_date_str = start_date.isoformat()
                end_date_str = end_date.isoformat()
            except ValueError:
                start_date_str = release_date_str
                end_date_str = None

    # Get overview, tagline, and keywords from details
    overview = details.get("overview") or ""
    tagline = details.get("tagline") or ""
    
    # Get keywords
    keywords_list = []
    if "keywords" in details and "keywords" in details["keywords"]:
        keywords_list = [kw.get("name", "") for kw in details["keywords"]["keywords"] if kw.get("name")]
    keywords_str = ", ".join(keywords_list) if keywords_list else None
    
    # Reviews will be fetched separately if needed (not in details by default)
    reviews_str = None  # Will be populated when reviews are fetched
    
    return FilmRecord(
        title=title,
        release_year=release_year,
        director=", ".join(director_names),
        writers=", ".join(writer_names),
        producers=", ".join(producer_names),
        cinematographers=", ".join(cinematographer_names),
        production_designers=", ".join(production_designer_names),
        cast=cast_str,
        genres=genres_str,
        tmdb_id=details.get("id"),
        overview=overview,
        reviews=reviews_str,
        keywords=keywords_str,
        tagline=tagline if tagline else None,
        thematic_descriptors=None,  # Will be populated by LLM enrichment
        stylistic_descriptors=None,  # Will be populated by LLM enrichment
        emotional_tone=None,  # Will be populated by LLM enrichment
        country=country,
        location=location,
        start_date=start_date_str,
        end_date=end_date_str,
        scheduled_dates=scheduled_dates,
        programme_url=programme_url,
    )


# ---- Public agent API -------------------------------------------------------------

class FilmExhibitionAgent:
    """
    High-level agent wrapper.

    Responsibilities:
    - Build a studio / distributor film library from TMDb.
    - Scrape upcoming cinema programmes from a configurable list of cinema URLs.
    - Compare both datasets to surface relevant overlaps.
    - Write all outputs to cleanly formatted Excel files.
    """

    def __init__(self, tmdb_api_key: Optional[str] = None, openai_api_key: Optional[str] = None):
        self.tmdb = TMDbClient(tmdb_api_key=tmdb_api_key)
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_client = None
        if self.openai_api_key and OpenAI:
            self.openai_client = OpenAI(api_key=self.openai_api_key)

    # -- Step 1: Studio library -----------------------------------------------------

    def build_studio_library(
        self,
        studio_name: str,
        *,
        company_id: Optional[int] = None,
        max_pages: Optional[int] = None,
        min_year: int = 1920,
    ) -> Tuple[pd.DataFrame, int]:
        """
        Build a film library DataFrame for a given studio / distributor.
        Fetches ALL films from 1920 to today unless max_pages is specified.

        Returns:
            (library_df, resolved_company_id)
        """
        if company_id is None:
            companies = self.tmdb.search_company(studio_name, limit=5)
            if not companies:
                raise ValueError(f"No TMDb company found for studio name '{studio_name}'.")
            # Naive choice: first result
            company_id = companies[0]["id"]

        print(f"\nBuilding library for {studio_name} (TMDb company ID: {company_id})...")
        print(f"Fetching all films from {min_year} to today...")
        
        min_date = f"{min_year}-01-01"
        max_date = dt.datetime.now().strftime("%Y-%m-%d")
        
        raw_movies = self.tmdb.discover_movies_for_company(
            company_id,
            max_pages=max_pages,
            min_release_date=min_date,
            max_release_date=max_date,
        )

        print(f"Found {len(raw_movies)} films. Enriching with full metadata...")
        records: List[FilmRecord] = []
        for i, m in enumerate(raw_movies, 1):
            movie_id = m.get("id")
            if not movie_id:
                continue
            try:
                details = self.tmdb.get_movie_details_with_credits(movie_id)
                record = film_record_from_tmdb_details(details)
                records.append(record)
                if i % 50 == 0:
                    print(f"  Enriched {i}/{len(raw_movies)} films...")
            except requests.HTTPError:
                # Skip individual failures but continue
                continue

        print(f"Successfully built library with {len(records)} films")
        df = pd.DataFrame([asdict(r) for r in records])
        # Studio library does not need exhibition columns, but we keep them;
        # they'll simply be NaN for these rows.
        return df, company_id

    # -- Step 2: Exhibition schedules (scraped from web) ----------------------------

    @staticmethod
    def load_cinema_sources(cinemas_yaml_path: str) -> List[CinemaSource]:
        with open(cinemas_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        items = data.get("cinemas") or []
        sources: List[CinemaSource] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            sources.append(
                CinemaSource(
                    id=str(item.get("id") or ""),
                    name=str(item.get("name") or ""),
                    country=str(item.get("country") or "").upper(),
                    city=item.get("city"),
                    type=item.get("type"),
                    programme_url=str(item.get("programme_url") or ""),
                    enabled=bool(item.get("enabled", True)),
                    scraper=str(item.get("scraper") or "auto"),
                )
            )
        # Basic validation
        return [s for s in sources if s.enabled and s.id and s.name and s.country and s.programme_url]

    def _enrich_screening_via_tmdb(self, screening: Screening) -> Optional[FilmRecord]:
        """
        Convert a scraped screening into a FilmRecord by resolving the title
        to a TMDb movie and pulling credits.
        """
        normalized_title = normalize_title_for_lookup(screening.title)
        candidates = self.tmdb.search_movie(normalized_title, limit=5)
        if not candidates:
            return None

        # Heuristic: pick the top result.
        movie_id = candidates[0].get("id")
        if not movie_id:
            return None

        try:
            details = self.tmdb.get_movie_details_with_credits(int(movie_id))
        except requests.HTTPError:
            return None

        location = screening.venue_name
        if screening.city:
            location = f"{screening.venue_name} ({screening.city})"

        date_iso = screening.start_dt.date().isoformat()
        return film_record_from_tmdb_details(
            details,
            country=screening.country,
            location=location,
            scheduled_dates=date_iso,
            programme_url=screening.source_url,
        )

    def build_exhibition_schedule_from_cinemas(
        self,
        cinemas_yaml_path: str,
        *,
        weeks_ahead: int = 4,
    ) -> pd.DataFrame:
        """
        Scrape upcoming screenings from a list of specific cinema programme URLs.

        Output shape:
        - one row per (film, venue) aggregated over the next N weeks
        - scheduled_dates: comma-separated ISO dates
        - start_date / end_date: min/max scheduled date for that film at that venue
        """
        sources = self.load_cinema_sources(cinemas_yaml_path)
        print(f"\nScraping {len(sources)} enabled cinemas for next {weeks_ahead} weeks...")

        all_screenings: List[Screening] = []
        for source in sources:
            try:
                screenings = scrape_programme(source)
            except Exception as e:
                print(f"[ERROR] {source.name}: Scraping failed: {e}")
                screenings = []
            screenings = filter_screenings_to_window(screenings, weeks_ahead=weeks_ahead)
            if screenings:
                print(f"  -> {source.name}: {len(screenings)} screenings within window")
            all_screenings.extend(screenings)
        
        print(f"\nTotal screenings scraped: {len(all_screenings)}")

        # Enrich each screening to the full film schema via TMDb (metadata only)
        print(f"\nEnriching {len(all_screenings)} screenings with TMDb metadata...")
        enriched_rows: List[FilmRecord] = []
        for i, s in enumerate(all_screenings, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(all_screenings)}")
            rec = self._enrich_screening_via_tmdb(s)
            if rec is not None:
                enriched_rows.append(rec)
        
        print(f"Successfully enriched {len(enriched_rows)}/{len(all_screenings)} screenings")

        if not enriched_rows:
            return pd.DataFrame(columns=[f.name for f in FilmRecord.__dataclass_fields__.values()])

        df = pd.DataFrame([asdict(r) for r in enriched_rows])

        # Aggregate multiple screenings per (title, country, location, programme_url)
        def _sorted_unique_dates(series: pd.Series) -> str:
            dates = []
            for v in series.dropna().astype(str).tolist():
                for part in v.split(","):
                    p = part.strip()
                    if p and p not in dates:
                        dates.append(p)
            # Sort ISO dates safely as strings
            dates = sorted(set(dates))
            return ", ".join(dates)

        grouped = (
            df.groupby(["title", "country", "location", "programme_url"], dropna=False, as_index=False)
            .agg(
                {
                    "release_year": "first",
                    "director": "first",
                    "writers": "first",
                    "producers": "first",
                    "cinematographers": "first",
                    "production_designers": "first",
                    "cast": "first",
                    "genres": "first",
                    "tmdb_id": "first",
                    "scheduled_dates": _sorted_unique_dates,
                }
            )
        )

        # Compute start/end from scheduled_dates
        def _min_date(dates_str: str) -> Optional[str]:
            parts = [p.strip() for p in (dates_str or "").split(",") if p.strip()]
            return min(parts) if parts else None

        def _max_date(dates_str: str) -> Optional[str]:
            parts = [p.strip() for p in (dates_str or "").split(",") if p.strip()]
            return max(parts) if parts else None

        grouped["start_date"] = grouped["scheduled_dates"].apply(_min_date)
        grouped["end_date"] = grouped["scheduled_dates"].apply(_max_date)

        return grouped

    # -- Step 3: Similarity matching (using embeddings) ---------------------------

    def _film_to_text(self, film_dict: Dict) -> str:
        """
        Convert a film record dictionary to a text representation for embedding.
        Includes all relevant metadata fields.
        """
        parts = []
        if film_dict.get("title"):
            parts.append(f"Title: {film_dict['title']}")
        if film_dict.get("release_year"):
            parts.append(f"Year: {film_dict['release_year']}")
        if film_dict.get("director"):
            parts.append(f"Director: {film_dict['director']}")
        if film_dict.get("writers"):
            parts.append(f"Writers: {film_dict['writers']}")
        if film_dict.get("producers"):
            parts.append(f"Producers: {film_dict['producers']}")
        if film_dict.get("cinematographers"):
            parts.append(f"Cinematographer: {film_dict['cinematographers']}")
        if film_dict.get("production_designers"):
            parts.append(f"Production Designer: {film_dict['production_designers']}")
        if film_dict.get("cast"):
            parts.append(f"Cast: {film_dict['cast']}")
        if film_dict.get("genres"):
            parts.append(f"Genres: {film_dict['genres']}")
        return "\n".join(parts)

    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of texts using OpenAI's text-embedding-3-small model.
        Returns a list of embedding vectors (1536 dimensions).
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized. Set OPENAI_API_KEY.")

        embeddings = []
        # Process in batches to avoid rate limits
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch,
                    dimensions=1536
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error creating embeddings for batch {i//batch_size + 1}: {e}")
                # Fallback: create zero vectors
                embeddings.extend([[0.0] * 1536 for _ in batch])
        return embeddings

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))

    def _generate_match_reasoning(
        self, library_film: Dict, exhibition_film: Dict, cosine_sim: float
    ) -> str:
        """
        Use LLM to generate reasoning text explaining why a library film matches an exhibition film.
        """
        if not self.openai_client:
            return f"Cosine similarity: {cosine_sim:.4f}"

        lib_text = self._film_to_text(library_film)
        ex_text = self._film_to_text(exhibition_film)

        prompt = (
            f"Library Film:\n{lib_text}\n\n"
            f"Exhibition Film:\n{ex_text}\n\n"
            f"These films have a cosine similarity of {cosine_sim:.4f} based on their embeddings.\n"
            "Explain in 2-3 sentences why this library film is a relevant match for the exhibition film. "
            "Focus on shared creative personnel (directors, writers, cast, cinematographers, etc.), "
            "similar genres, themes, or time periods. Be specific about the connections."
        )

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a film analysis assistant that explains connections between films."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            reasoning = response.choices[0].message.content.strip()
            return f"{reasoning} (Similarity: {cosine_sim:.4f})"
        except Exception as e:
            print(f"Error generating reasoning: {e}")
            return f"Cosine similarity: {cosine_sim:.4f}"

    def compare_library_and_exhibitions(
        self,
        library_df: pd.DataFrame,
        exhibitions_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compare a studio library and exhibition schedule using OpenAI embeddings and cosine similarity.

        For each exhibition film, finds the top 3 most similar library films based on:
        - Embeddings of all film metadata (title, year, director, writers, producers,
          cinematographers, production designers, cast, genres)
        - Cosine similarity between embeddings
        - LLM-generated reasoning for each match

        Returns matches ordered by cosine similarity (highest first).
        """
        if not self.openai_client:
            raise ValueError("OpenAI client required for embedding-based matching. Set OPENAI_API_KEY.")

        print("\nCreating embeddings for library films...")
        lib_rows = library_df.to_dict(orient="records")
        lib_texts = [self._film_to_text(lib) for lib in lib_rows]
        lib_embeddings = self._create_embeddings(lib_texts)
        print(f"Created embeddings for {len(lib_embeddings)} library films")

        print("\nCreating embeddings for exhibition films...")
        ex_rows = exhibitions_df.to_dict(orient="records")
        ex_texts = [self._film_to_text(ex) for ex in ex_rows]
        ex_embeddings = self._create_embeddings(ex_texts)
        print(f"Created embeddings for {len(ex_embeddings)} exhibition films")

        print("\nCalculating cosine similarities and finding top matches...")
        matches: List[MatchRecord] = []

        for ex_idx, ex in enumerate(ex_rows):
            ex_embedding = ex_embeddings[ex_idx]
            similarities = []

            # Calculate similarity to all library films
            for lib_idx, lib in enumerate(lib_rows):
                lib_embedding = lib_embeddings[lib_idx]
                similarity = self._cosine_similarity(ex_embedding, lib_embedding)
                similarities.append((lib_idx, similarity))

            # Sort by similarity (highest first) and take top 3
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_3 = similarities[:3]

            # Generate matches with LLM reasoning
            for lib_idx, cosine_sim in top_3:
                lib = lib_rows[lib_idx]
                reasoning = self._generate_match_reasoning(lib, ex, cosine_sim)

                match = MatchRecord(
                    library_title=str(lib.get("title") or ""),
                    exhibition_title=str(ex.get("title") or ""),
                    country=str(ex.get("country") or ""),
                    location=str(ex.get("location") or ""),
                    relevance_start=str(ex.get("start_date") or ""),
                    relevance_end=str(ex.get("end_date") or ""),
                    similarity_reason=reasoning,
                    programme_url=str(ex.get("programme_url") or ""),
                    cosine_similarity=cosine_sim,
                )
                matches.append(match)

            if (ex_idx + 1) % 10 == 0:
                print(f"  Processed {ex_idx + 1}/{len(ex_rows)} exhibition films...")

        # Sort all matches by cosine similarity (highest first)
        matches.sort(key=lambda m: m.cosine_similarity, reverse=True)

        print(f"\nGenerated {len(matches)} total matches (top 3 per exhibition film)")
        return pd.DataFrame([asdict(m) for m in matches])

    # -- Orchestrator --------------------------------------------------------------

    def run(
        self,
        studio_name: str,
        *,
        weeks_ahead: int = 4,
        max_pages: Optional[int] = None,
        output_prefix: Optional[str] = None,
        cinemas_yaml_path: str = "cinemas.yaml",
        min_year: int = 1920,
        use_prebuilt_library: bool = False,
        prebuilt_library_path: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Full pipeline:
        1. Build studio library (ALL films from 1920 to today by default).
        2. Scrape exhibition schedules from a list of cinema programme URLs.
        3. Compare and surface relevant overlaps.
        4. Write three Excel files:
           - {prefix}_library.xlsx
           - upcoming_exhibitions.xlsx
           - {prefix}_matches.xlsx

        Returns:
            dict with keys: 'library', 'exhibitions', 'matches'
        """
        output_prefix = output_prefix or studio_name.lower().replace(" ", "_")

        # Step 1: studio library
        if use_prebuilt_library:
            # Load an existing library Excel instead of calling TMDb
            library_path = prebuilt_library_path or f"{output_prefix}_library.xlsx"
            print(f"\nLoading prebuilt library from: {library_path}")
            library_df = pd.read_excel(library_path)
            company_id = None
        else:
            # Build library from TMDb (fetches ALL films unless max_pages is set)
            library_df, company_id = self.build_studio_library(
                studio_name, max_pages=max_pages, min_year=min_year
            )

        # Step 2: exhibition schedules
        exhibitions_df = self.build_exhibition_schedule_from_cinemas(
            cinemas_yaml_path, weeks_ahead=weeks_ahead
        )

        # Step 3: similarity matches
        matches_df = self.compare_library_and_exhibitions(
            library_df, exhibitions_df
        )

        # Step 4: Excel outputs
        exhibitions_path = "upcoming_exhibitions.xlsx"  # Fixed filename for all exhibitions
        matches_path = f"{output_prefix}_matches.xlsx"

        # Only write library if we built it (not if using prebuilt)
        if not use_prebuilt_library:
            library_path = f"{output_prefix}_library.xlsx"
            library_df.to_excel(library_path, index=False)
            print(f"Studio library saved to: {library_path}")
        else:
            print(f"(Using prebuilt library - not overwriting)")

        exhibitions_df.to_excel(exhibitions_path, index=False)
        matches_df.to_excel(matches_path, index=False)

        print(f"Exhibition schedule saved to: {exhibitions_path}")
        print(f"Matches saved to: {matches_path}")
        if company_id:
            print(f"(Resolved TMDb company id for '{studio_name}': {company_id})")
        print(f"(Cinemas config used: {cinemas_yaml_path})")

        return {
            "library": library_df,
            "exhibitions": exhibitions_df,
            "matches": matches_df,
        }


class StudioLibraryAgent:
    """
    Separate agent responsible ONLY for building a studio/distributor library from TMDb
    and writing it to Excel.

    It fetches films in decade-sized chunks (e.g., 1920–1929, 1930–1939, …) to
    spread out TMDb discover queries and give clearer progress feedback.
    """

    def __init__(self, tmdb_api_key: Optional[str] = None, openai_api_key: Optional[str] = None):
        self.tmdb = TMDbClient(tmdb_api_key=tmdb_api_key)
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_client = None
        if self.openai_api_key and OpenAI:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
    
    def _enrich_film_with_semantic_descriptors(self, film_record: FilmRecord) -> FilmRecord:
        """Use LLM to generate thematic, stylistic, and emotional descriptors for a film."""
        if not self.openai_client:
            return film_record
        
        # Build film description for LLM
        film_desc_parts = []
        if film_record.title:
            film_desc_parts.append(f"Title: {film_record.title}")
        if film_record.release_year:
            film_desc_parts.append(f"Year: {film_record.release_year}")
        if film_record.director:
            film_desc_parts.append(f"Director: {film_record.director}")
        if film_record.genres:
            film_desc_parts.append(f"Genres: {film_record.genres}")
        if film_record.overview:
            film_desc_parts.append(f"Plot: {film_record.overview}")
        if film_record.keywords:
            film_desc_parts.append(f"Keywords: {film_record.keywords}")
        if film_record.tagline:
            film_desc_parts.append(f"Tagline: {film_record.tagline}")
        
        film_description = "\n".join(film_desc_parts)
        
        prompt = f"""Analyze this film and provide concise descriptors:

{film_description}

Provide a JSON response with:
1. "thematic_descriptors": 3-5 key themes (e.g., "alienation, urban isolation, existential crisis, betrayal, psychological complexity")
2. "stylistic_descriptors": 1-2 sentences describing cinematic style (e.g., "minimalist cinematography, slow-paced, contemplative, non-linear narrative")
3. "emotional_tone": 1 sentence describing emotional atmosphere (e.g., "melancholic, introspective, existential, darkly humorous")

Return ONLY valid JSON: {{"thematic_descriptors": "...", "stylistic_descriptors": "...", "emotional_tone": "..."}}"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a film analysis expert. Provide concise, accurate descriptors."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=300
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            
            film_record.thematic_descriptors = data.get("thematic_descriptors", "")
            film_record.stylistic_descriptors = data.get("stylistic_descriptors", "")
            film_record.emotional_tone = data.get("emotional_tone", "")
        except Exception as e:
            print(f"  [WARN] LLM enrichment failed for {film_record.title}: {e}")
        
        return film_record
    
    def build_library_decade_chunks(
        self,
        studio_name: str,
        *,
        start_year: int = 1920,
        end_year: Optional[int] = None,
        chunk_size: int = 10,
        include_adult: bool = False,
        max_pages_per_chunk: Optional[int] = None,
        output_prefix: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, int]:
        """
        Build a studio library from TMDb in decade chunks and write to Excel.

        - 1920–1929
        - 1930–1939
        - …

        This agent ONLY builds the library; the exhibition/matching agent can then
        be run later with use_prebuilt_library=True to avoid re-hitting TMDb for the library.
        """
        if end_year is None:
            end_year = dt.datetime.now().year

        # Resolve company ID
        companies = self.tmdb.search_company(studio_name, limit=5)
        if not companies:
            raise ValueError(f"No TMDb company found for studio name '{studio_name}'.")
        company_id = companies[0]["id"]

        print(f"\n[LibraryAgent] Building library for {studio_name} (company ID: {company_id})")
        print(f"[LibraryAgent] Year range: {start_year}–{end_year}, chunk size: {chunk_size} years")

        # Collect raw movie stubs across all chunks, de-duplicated by TMDb ID
        raw_movies_by_id: Dict[int, Dict] = {}

        year = start_year
        while year <= end_year:
            chunk_start = year
            chunk_end = min(year + chunk_size - 1, end_year)
            min_date = f"{chunk_start}-01-01"
            max_date = f"{chunk_end}-12-31"

            print(
                f"[LibraryAgent] Discovering movies for {studio_name} "
                f"from {chunk_start} to {chunk_end}..."
            )

            chunk_movies = self.tmdb.discover_movies_for_company(
                company_id,
                max_pages=max_pages_per_chunk,
                include_adult=include_adult,
                min_release_date=min_date,
                max_release_date=max_date,
            )

            print(f"[LibraryAgent]   Retrieved {len(chunk_movies)} movies for {chunk_start}–{chunk_end}")

            for m in chunk_movies:
                mid = m.get("id")
                if not mid:
                    continue
                if mid not in raw_movies_by_id:
                    raw_movies_by_id[mid] = m

            year += chunk_size

        all_movies = list(raw_movies_by_id.values())
        print(f"[LibraryAgent] Total unique movies discovered: {len(all_movies)}")

        # Enrich with full details/credits, overview, keywords, and LLM semantic descriptors
        records: List[FilmRecord] = []
        for i, m in enumerate(all_movies, 1):
            movie_id = m.get("id")
            if not movie_id:
                continue
            try:
                details = self.tmdb.get_movie_details_with_credits(int(movie_id))
                record = film_record_from_tmdb_details(details)
                # Reviews are not fetched - set to None
                record.reviews = None
                
                # Enrich with LLM-generated semantic descriptors
                if self.openai_client:
                    record = self._enrich_film_with_semantic_descriptors(record)
                
                records.append(record)
                if i % 50 == 0:
                    print(f"[LibraryAgent]   Enriched {i}/{len(all_movies)} movies (with overview, keywords, and semantic descriptors)...")
            except requests.HTTPError:
                continue

        print(f"[LibraryAgent] Successfully enriched {len(records)} movies with overview, keywords, and semantic descriptors")

        df = pd.DataFrame([asdict(r) for r in records])
        output_prefix = output_prefix or studio_name.lower().replace(" ", "_")
        library_path = f"{output_prefix}_library.xlsx"
        df.to_excel(library_path, index=False)
        print(f"[LibraryAgent] Library written to: {library_path}")

        # Generate embeddings for all films and save to permanent file
        if self.openai_client:
            print(f"\n[LibraryAgent] Generating embeddings for {len(records)} films...")
            embeddings_path = self._generate_library_embeddings(records, output_prefix)
            print(f"[LibraryAgent] Embeddings saved to: {embeddings_path}")
        else:
            print(f"[LibraryAgent] OpenAI API key not available - skipping embedding generation")

        return df, company_id
    
    def _film_to_text_for_embedding(self, film_record: FilmRecord) -> str:
        """Convert a FilmRecord to text representation for embedding with enhanced semantic richness."""
        parts = []
        
        # Basic metadata
        if film_record.title:
            parts.append(f"Film: {film_record.title}")
        if film_record.release_year:
            parts.append(f"Released: {film_record.release_year}")
        if film_record.tagline:
            parts.append(f"Tagline: {film_record.tagline}")
        
        # Creative personnel
        if film_record.director:
            parts.append(f"Directed by: {film_record.director}")
        if film_record.writers:
            parts.append(f"Written by: {film_record.writers}")
        if film_record.cinematographers:
            parts.append(f"Cinematography: {film_record.cinematographers}")
        if film_record.production_designers:
            parts.append(f"Production Design: {film_record.production_designers}")
        if film_record.cast:
            parts.append(f"Starring: {film_record.cast}")
        if film_record.lead_gender:
            parts.append(f"Lead actor gender: {film_record.lead_gender}")
        
        # Genres and keywords
        if film_record.genres:
            parts.append(f"Genres: {film_record.genres}")
        if film_record.keywords:
            parts.append(f"Keywords: {film_record.keywords}")
        
        # Plot and themes
        if film_record.overview:
            parts.append(f"Plot: {film_record.overview}")
        
        # LLM-generated semantic enrichment (if available)
        if film_record.thematic_descriptors:
            parts.append(f"Themes: {film_record.thematic_descriptors}")
        if film_record.stylistic_descriptors:
            parts.append(f"Style: {film_record.stylistic_descriptors}")
        if film_record.emotional_tone:
            parts.append(f"Tone: {film_record.emotional_tone}")
        
        return "\n".join(parts)
    
    def _generate_library_embeddings(
        self, records: List[FilmRecord], output_prefix: str
    ) -> str:
        """Generate embeddings for all library films and save to Excel."""
        if not self.openai_client:
            raise ValueError("OpenAI client required for embedding generation")
        
        # Convert films to text
        film_texts = [self._film_to_text_for_embedding(record) for record in records]
        
        # Create embeddings in batches
        embeddings: List[List[float]] = []
        batch_size = 100
        for i in range(0, len(film_texts), batch_size):
            batch = film_texts[i:i + batch_size]
            try:
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch,
                    dimensions=1536
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                print(f"[LibraryAgent]   Generated embeddings for batch {i//batch_size + 1}/{(len(film_texts)-1)//batch_size + 1}...")
            except Exception as e:
                print(f"[LibraryAgent] Error creating embeddings for batch {i//batch_size + 1}: {e}")
                embeddings.extend([[0.0] * 1536 for _ in batch])
        
        # Create DataFrame with embeddings
        # Store embeddings as separate columns (Excel has 32K char limit per cell, JSON would exceed it)
        # We'll store as a numpy array in a pickle file, and also save metadata to Excel
        embedding_data = []
        for i, (record, embedding) in enumerate(zip(records, embeddings)):
            embedding_data.append({
                "tmdb_id": record.tmdb_id,
                "title": record.title,
                "release_year": record.release_year,
            })
        
        # Save metadata to Excel
        embeddings_df = pd.DataFrame(embedding_data)
        embeddings_path = f"{output_prefix}_library_embeddings.xlsx"
        embeddings_df.to_excel(embeddings_path, index=False)
        
        # Save actual embeddings as numpy array to a separate .npy file
        embeddings_array = np.array(embeddings)
        embeddings_npy_path = f"{output_prefix}_library_embeddings.npy"
        np.save(embeddings_npy_path, embeddings_array)
        print(f"[LibraryAgent] Embedding vectors saved to: {embeddings_npy_path}")
        print(f"[LibraryAgent]   Shape: {embeddings_array.shape} (films × dimensions)")
        
        return embeddings_path


class ExhibitionScrapingAgent:
    """
    Separate agent responsible ONLY for scraping cinema exhibition schedules
    and building the upcoming_exhibitions.xlsx file.
    
    It processes cinemas one by one and progressively adds results to the Excel file.
    """

    def __init__(self, tmdb_api_key: Optional[str] = None, openai_api_key: Optional[str] = None):
        self.tmdb = TMDbClient(tmdb_api_key=tmdb_api_key)
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_client = None
        if self.openai_api_key and OpenAI:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
    
    def _enrich_film_with_semantic_descriptors(self, film_record: FilmRecord) -> FilmRecord:
        """Use LLM to generate thematic, stylistic, and emotional descriptors for a film."""
        if not self.openai_client:
            return film_record
        
        # Build film description for LLM
        film_desc_parts = []
        if film_record.title:
            film_desc_parts.append(f"Title: {film_record.title}")
        if film_record.release_year:
            film_desc_parts.append(f"Year: {film_record.release_year}")
        if film_record.director:
            film_desc_parts.append(f"Director: {film_record.director}")
        if film_record.genres:
            film_desc_parts.append(f"Genres: {film_record.genres}")
        if film_record.overview:
            film_desc_parts.append(f"Plot: {film_record.overview}")
        if film_record.keywords:
            film_desc_parts.append(f"Keywords: {film_record.keywords}")
        if film_record.tagline:
            film_desc_parts.append(f"Tagline: {film_record.tagline}")
        
        film_description = "\n".join(film_desc_parts)
        
        prompt = f"""Analyze this film and provide concise descriptors:

{film_description}

Provide a JSON response with:
1. "thematic_descriptors": 3-5 key themes (e.g., "alienation, urban isolation, existential crisis, betrayal, psychological complexity")
2. "stylistic_descriptors": 1-2 sentences describing cinematic style (e.g., "minimalist cinematography, slow-paced, contemplative, non-linear narrative")
3. "emotional_tone": 1 sentence describing emotional atmosphere (e.g., "melancholic, introspective, existential, darkly humorous")

Return ONLY valid JSON: {{"thematic_descriptors": "...", "stylistic_descriptors": "...", "emotional_tone": "..."}}"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a film analysis expert. Provide concise, accurate descriptors."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=300
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            
            film_record.thematic_descriptors = data.get("thematic_descriptors", "")
            film_record.stylistic_descriptors = data.get("stylistic_descriptors", "")
            film_record.emotional_tone = data.get("emotional_tone", "")
        except Exception as e:
            print(f"  [WARN] LLM enrichment failed for {film_record.title}: {e}")
        
        return film_record

    @staticmethod
    def load_cinema_sources(cinemas_yaml_path: str) -> List[CinemaSource]:
        """Load enabled cinema sources from YAML config."""
        with open(cinemas_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        items = data.get("cinemas") or []
        sources: List[CinemaSource] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            sources.append(
                CinemaSource(
                    id=str(item.get("id") or ""),
                    name=str(item.get("name") or ""),
                    country=str(item.get("country") or "").upper(),
                    city=item.get("city"),
                    type=item.get("type"),
                    programme_url=str(item.get("programme_url") or ""),
                    enabled=bool(item.get("enabled", True)),
                    scraper=str(item.get("scraper") or "auto"),
                )
            )
        return [s for s in sources if s.enabled and s.id and s.name and s.country and s.programme_url]

    def _enrich_screening_via_tmdb(self, screening: Screening) -> Optional[FilmRecord]:
        """Convert a scraped screening into a FilmRecord by resolving via TMDb."""
        normalized_title = normalize_title_for_lookup(screening.title)
        candidates = self.tmdb.search_movie(normalized_title, limit=5)
        if not candidates:
            return None

        movie_id = candidates[0].get("id")
        if not movie_id:
            return None

        try:
            details = self.tmdb.get_movie_details_with_credits(int(movie_id))
        except requests.HTTPError:
            return None

        location = screening.venue_name
        if screening.city:
            location = f"{screening.venue_name} ({screening.city})"

        date_iso = screening.start_dt.date().isoformat()
        record = film_record_from_tmdb_details(
            details,
            country=screening.country,
            location=location,
            scheduled_dates=date_iso,
            programme_url=screening.source_url,
        )
        
        # Enrich with LLM-generated semantic descriptors
        if self.openai_client:
            record = self._enrich_film_with_semantic_descriptors(record)
        
        return record

    def build_exhibitions_progressively(
        self,
        cinemas_yaml_path: str = "cinemas.yaml",
        *,
        weeks_ahead: int = 4,
        output_path: str = "upcoming_exhibitions.xlsx",
    ) -> pd.DataFrame:
        """
        Scrape cinema exhibitions progressively, adding to Excel as we go.
        
        Processes each cinema one by one:
        1. Scrapes screenings
        2. Enriches with TMDb metadata
        3. Adds to cumulative DataFrame
        4. Writes to Excel after each cinema (or periodically)
        """
        sources = self.load_cinema_sources(cinemas_yaml_path)
        print(f"\n[ExhibitionAgent] Scraping {len(sources)} enabled cinemas for next {weeks_ahead} weeks...")
        print(f"[ExhibitionAgent] Output file: {output_path}\n")

        all_enriched_rows: List[FilmRecord] = []
        
        for idx, source in enumerate(sources, 1):
            print(f"[ExhibitionAgent] Processing cinema {idx}/{len(sources)}: {source.name}")
            
            # Add delay between requests to avoid rate limiting (except for first cinema)
            if idx > 1:
                import time
                time.sleep(2)  # 2 second delay between cinemas
            
            # Scrape screenings
            try:
                screenings = scrape_programme(source)
            except Exception as e:
                print(f"[ExhibitionAgent]   [ERROR] Scraping failed: {e}")
                screenings = []
            
            if screenings:
                print(f"[ExhibitionAgent]   Found {len(screenings)} total screenings before filtering")
            
            screenings = filter_screenings_to_window(screenings, weeks_ahead=weeks_ahead)
            print(f"[ExhibitionAgent]   Found {len(screenings)} screenings within window")
            
            if not screenings:
                continue
            
            # Enrich with TMDb metadata
            print(f"[ExhibitionAgent]   Enriching {len(screenings)} screenings with TMDb metadata...")
            for i, s in enumerate(screenings, 1):
                rec = self._enrich_screening_via_tmdb(s)
                if rec is not None:
                    all_enriched_rows.append(rec)
                if i % 10 == 0:
                    print(f"[ExhibitionAgent]     Progress: {i}/{len(screenings)}")
            
            print(f"[ExhibitionAgent]   Added {len([r for r in all_enriched_rows if r.programme_url == source.programme_url])} enriched films from {source.name}")
            
            # Write to Excel after each cinema (progressive save)
            if all_enriched_rows:
                df = pd.DataFrame([asdict(r) for r in all_enriched_rows])
                
                # Aggregate multiple screenings per (title, country, location, programme_url)
                def _sorted_unique_dates(series: pd.Series) -> str:
                    dates = []
                    for v in series.dropna().astype(str).tolist():
                        for part in v.split(","):
                            p = part.strip()
                            if p and p not in dates:
                                dates.append(p)
                    dates = sorted(set(dates))
                    return ", ".join(dates)

                grouped = (
                    df.groupby(["title", "country", "location", "programme_url"], dropna=False, as_index=False)
                    .agg(
                        {
                            "release_year": "first",
                            "director": "first",
                            "writers": "first",
                            "producers": "first",
                            "cinematographers": "first",
                            "production_designers": "first",
                            "cast": "first",
                            "genres": "first",
                            "tmdb_id": "first",
                            "overview": "first",
                            "keywords": "first",  # Add keywords
                            "tagline": "first",  # Add tagline
                            "thematic_descriptors": "first",  # Add LLM descriptors
                            "stylistic_descriptors": "first",
                            "emotional_tone": "first",
                            "scheduled_dates": _sorted_unique_dates,
                        }
                    )
                )

                # Compute start/end from scheduled_dates
                def _min_date(dates_str: str) -> Optional[str]:
                    parts = [p.strip() for p in (dates_str or "").split(",") if p.strip()]
                    return min(parts) if parts else None

                def _max_date(dates_str: str) -> Optional[str]:
                    parts = [p.strip() for p in (dates_str or "").split(",") if p.strip()]
                    return max(parts) if parts else None

                grouped["start_date"] = grouped["scheduled_dates"].apply(_min_date)
                grouped["end_date"] = grouped["scheduled_dates"].apply(_max_date)
                
                grouped.to_excel(output_path, index=False)
                print(f"[ExhibitionAgent]   Saved {len(grouped)} unique films to {output_path}")

        if not all_enriched_rows:
            print(f"[ExhibitionAgent] No exhibitions found. Creating empty file.")
            empty_df = pd.DataFrame(columns=[f.name for f in FilmRecord.__dataclass_fields__.values()])
            empty_df.to_excel(output_path, index=False)
            return empty_df

        # Final aggregation and save
        df = pd.DataFrame([asdict(r) for r in all_enriched_rows])
        
        def _sorted_unique_dates(series: pd.Series) -> str:
            dates = []
            for v in series.dropna().astype(str).tolist():
                for part in v.split(","):
                    p = part.strip()
                    if p and p not in dates:
                        dates.append(p)
            dates = sorted(set(dates))
            return ", ".join(dates)

        grouped = (
            df.groupby(["title", "country", "location", "programme_url"], dropna=False, as_index=False)
            .agg(
                {
                    "release_year": "first",
                    "director": "first",
                    "writers": "first",
                    "producers": "first",
                    "cinematographers": "first",
                    "production_designers": "first",
                    "cast": "first",
                    "genres": "first",
                    "tmdb_id": "first",
                    "overview": "first",
                    "keywords": "first",  # Add keywords
                    "tagline": "first",  # Add tagline
                    "thematic_descriptors": "first",  # Add LLM descriptors
                    "stylistic_descriptors": "first",
                    "emotional_tone": "first",
                    "scheduled_dates": _sorted_unique_dates,
                }
            )
        )

        def _min_date(dates_str: str) -> Optional[str]:
            parts = [p.strip() for p in (dates_str or "").split(",") if p.strip()]
            return min(parts) if parts else None

        def _max_date(dates_str: str) -> Optional[str]:
            parts = [p.strip() for p in (dates_str or "").split(",") if p.strip()]
            return max(parts) if parts else None

        grouped["start_date"] = grouped["scheduled_dates"].apply(_min_date)
        grouped["end_date"] = grouped["scheduled_dates"].apply(_max_date)
        
        grouped.to_excel(output_path, index=False)
        print(f"\n[ExhibitionAgent] Final save: {len(grouped)} unique exhibition films")
        print(f"[ExhibitionAgent] Exhibition file written to: {output_path}")
        
        # Generate embeddings for all exhibition films
        if self.openai_client and len(grouped) > 0:
            print(f"\n[ExhibitionAgent] Generating embeddings for {len(grouped)} exhibition films...")
            embeddings_path = self._generate_exhibition_embeddings(grouped, output_path)
            print(f"[ExhibitionAgent] Embeddings saved to: {embeddings_path}")
        else:
            if not self.openai_client:
                print(f"[ExhibitionAgent] OpenAI API key not available - skipping embedding generation")
        
        # Generate embeddings for all exhibition films
        if self.openai_client and len(grouped) > 0:
            print(f"\n[ExhibitionAgent] Generating embeddings for {len(grouped)} exhibition films...")
            embeddings_path = self._generate_exhibition_embeddings(grouped, output_path)
            print(f"[ExhibitionAgent] Embeddings saved to: {embeddings_path}")
        else:
            if not self.openai_client:
                print(f"[ExhibitionAgent] OpenAI API key not available - skipping embedding generation")
        
        return grouped
    
    def scrape_imdb_trending(self, top_n: int = 50) -> List[FilmRecord]:
        """
        Scrape IMDB's top trending films and TV shows and convert to FilmRecords.
        
        Args:
            top_n: Number of top items to scrape (default: 50)
        
        Returns:
            List of FilmRecord objects
        """
        import time
        from trending_scrapers import scrape_imdb_trending, convert_imdb_to_film_record
        
        print(f"\n[ExhibitionAgent] Scraping IMDB top {top_n} trending films and TV...")
        imdb_items = scrape_imdb_trending(top_n=top_n)
        
        film_records = []
        for idx, item in enumerate(imdb_items, 1):
            print(f"[ExhibitionAgent]   Processing IMDB item {idx}/{len(imdb_items)}: {item.get('title', 'unknown')}")
            record_dict = convert_imdb_to_film_record(item, self.tmdb)
            if record_dict:
                # Convert dict to FilmRecord
                record = FilmRecord(
                    title=record_dict.get("title", ""),
                    release_year=record_dict.get("release_year"),
                    director=record_dict.get("director", ""),
                    writers=record_dict.get("writers", ""),
                    producers=record_dict.get("producers", ""),
                    cinematographers=record_dict.get("cinematographers", ""),
                    production_designers=record_dict.get("production_designers", ""),
                    cast=record_dict.get("cast", ""),
                    genres=record_dict.get("genres", ""),
                    tmdb_id=record_dict.get("tmdb_id"),
                    overview=record_dict.get("overview"),
                    keywords=record_dict.get("keywords"),
                    tagline=record_dict.get("tagline"),
                    thematic_descriptors=record_dict.get("thematic_descriptors"),
                    stylistic_descriptors=record_dict.get("stylistic_descriptors"),
                    emotional_tone=record_dict.get("emotional_tone"),
                    country=record_dict.get("country"),
                    location=record_dict.get("location"),
                    start_date=record_dict.get("start_date"),
                    end_date=record_dict.get("end_date"),
                    scheduled_dates=record_dict.get("scheduled_dates"),
                    programme_url=record_dict.get("programme_url"),
                    lead_gender=record_dict.get("lead_gender"),
                )
                # Enrich with LLM descriptors if available
                if self.openai_client:
                    record = self._enrich_film_with_semantic_descriptors(record)
                film_records.append(record)
            time.sleep(0.5)  # Small delay between TMDb lookups
        
        print(f"[ExhibitionAgent] Successfully converted {len(film_records)} IMDB trending items to FilmRecords")
        return film_records
    
    def scrape_nielsen_streaming(self) -> List[FilmRecord]:
        """
        Scrape Nielsen's top 10 streaming data and convert to FilmRecords.
        
        Returns:
            List of FilmRecord objects
        """
        import time
        from trending_scrapers import scrape_nielsen_streaming_top10, convert_nielsen_to_film_record
        
        print(f"\n[ExhibitionAgent] Scraping Nielsen top 10 streaming data...")
        nielsen_items = scrape_nielsen_streaming_top10()
        
        film_records = []
        for idx, item in enumerate(nielsen_items, 1):
            print(f"[ExhibitionAgent]   Processing Nielsen item {idx}/{len(nielsen_items)}: {item.get('title', 'unknown')}")
            record_dict = convert_nielsen_to_film_record(item, self.tmdb)
            if record_dict:
                # Convert dict to FilmRecord
                record = FilmRecord(
                    title=record_dict.get("title", ""),
                    release_year=record_dict.get("release_year"),
                    director=record_dict.get("director", ""),
                    writers=record_dict.get("writers", ""),
                    producers=record_dict.get("producers", ""),
                    cinematographers=record_dict.get("cinematographers", ""),
                    production_designers=record_dict.get("production_designers", ""),
                    cast=record_dict.get("cast", ""),
                    genres=record_dict.get("genres", ""),
                    tmdb_id=record_dict.get("tmdb_id"),
                    overview=record_dict.get("overview"),
                    keywords=record_dict.get("keywords"),
                    tagline=record_dict.get("tagline"),
                    thematic_descriptors=record_dict.get("thematic_descriptors"),
                    stylistic_descriptors=record_dict.get("stylistic_descriptors"),
                    emotional_tone=record_dict.get("emotional_tone"),
                    country=record_dict.get("country"),
                    location=record_dict.get("location"),
                    start_date=record_dict.get("start_date"),
                    end_date=record_dict.get("end_date"),
                    scheduled_dates=record_dict.get("scheduled_dates"),
                    programme_url=record_dict.get("programme_url"),
                    lead_gender=record_dict.get("lead_gender"),
                )
                # Enrich with LLM descriptors if available
                if self.openai_client:
                    record = self._enrich_film_with_semantic_descriptors(record)
                film_records.append(record)
            time.sleep(0.5)  # Small delay between TMDb lookups
        
        print(f"[ExhibitionAgent] Successfully converted {len(film_records)} Nielsen streaming items to FilmRecords")
        return film_records
    
    def _generate_exhibition_embeddings(
        self, exhibitions_df: pd.DataFrame, exhibitions_path: str
    ) -> str:
        """Generate embeddings for all exhibition films and save to files."""
        if not self.openai_client:
            raise ValueError("OpenAI client required for embedding generation")
        
        # Convert DataFrame rows to FilmRecord-like dictionaries for text conversion
        film_texts = []
        for _, row in exhibitions_df.iterrows():
            # Create text representation similar to library films
            parts = []
            if pd.notna(row.get("title")):
                parts.append(f"Title: {row['title']}")
            if pd.notna(row.get("release_year")):
                parts.append(f"Year: {int(row['release_year'])}")
            if pd.notna(row.get("director")):
                parts.append(f"Director: {row['director']}")
            if pd.notna(row.get("writers")):
                parts.append(f"Writers: {row['writers']}")
            if pd.notna(row.get("producers")):
                parts.append(f"Producers: {row['producers']}")
            if pd.notna(row.get("cinematographers")):
                parts.append(f"Cinematographer: {row['cinematographers']}")
            if pd.notna(row.get("production_designers")):
                parts.append(f"Production Designer: {row['production_designers']}")
            if pd.notna(row.get("cast")):
                parts.append(f"Cast: {row['cast']}")
            if pd.notna(row.get("lead_gender")):
                parts.append(f"Lead actor gender: {row['lead_gender']}")
            if pd.notna(row.get("genres")):
                parts.append(f"Genres: {row['genres']}")
            if pd.notna(row.get("keywords")):
                parts.append(f"Keywords: {row['keywords']}")
            if pd.notna(row.get("tagline")):
                parts.append(f"Tagline: {row['tagline']}")
            if pd.notna(row.get("overview")):
                parts.append(f"Plot: {row['overview']}")
            thematic = row.get("thematic_descriptors")
            if isinstance(thematic, (list, tuple, np.ndarray)):
                if len(thematic) > 0:
                    parts.append(f"Themes: {thematic}")
            elif pd.notna(thematic):
                parts.append(f"Themes: {thematic}")
            stylistic = row.get("stylistic_descriptors")
            if isinstance(stylistic, (list, tuple, np.ndarray)):
                if len(stylistic) > 0:
                    parts.append(f"Style: {stylistic}")
            elif pd.notna(stylistic):
                parts.append(f"Style: {stylistic}")
            emotional = row.get("emotional_tone")
            if isinstance(emotional, (list, tuple, np.ndarray)):
                if len(emotional) > 0:
                    parts.append(f"Tone: {emotional}")
            elif pd.notna(emotional):
                parts.append(f"Tone: {emotional}")
            film_texts.append("\n".join(parts))
        
        # Create embeddings in batches
        embeddings: List[List[float]] = []
        batch_size = 100
        for i in range(0, len(film_texts), batch_size):
            batch = film_texts[i:i + batch_size]
            try:
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch,
                    dimensions=1536
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                print(f"[ExhibitionAgent]   Generated embeddings for batch {i//batch_size + 1}/{(len(film_texts)-1)//batch_size + 1}...")
            except Exception as e:
                print(f"[ExhibitionAgent] Error creating embeddings for batch {i//batch_size + 1}: {e}")
                embeddings.extend([[0.0] * 1536 for _ in batch])
        
        # Create DataFrame with embeddings metadata
        embedding_data = []
        for i, (_, row) in enumerate(exhibitions_df.iterrows()):
            embedding_data.append({
                "tmdb_id": row.get("tmdb_id"),
                "title": row.get("title"),
                "release_year": row.get("release_year"),
                "country": row.get("country"),
                "location": row.get("location"),
            })
        
        # Save metadata to Excel
        embeddings_df = pd.DataFrame(embedding_data)
        # Use exhibitions path to derive embeddings path
        base_path = exhibitions_path.replace(".xlsx", "")
        embeddings_excel_path = f"{base_path}_embeddings.xlsx"
        embeddings_df.to_excel(embeddings_excel_path, index=False)
        
        # Save actual embeddings as numpy array
        embeddings_array = np.array(embeddings)
        embeddings_npy_path = f"{base_path}_embeddings.npy"
        np.save(embeddings_npy_path, embeddings_array)
        print(f"[ExhibitionAgent] Embedding vectors saved to: {embeddings_npy_path}")
        print(f"[ExhibitionAgent]   Shape: {embeddings_array.shape} (films × dimensions)")
        
        return embeddings_excel_path


class MatchingAgent:
    """
    Separate agent responsible ONLY for matching library and exhibition films
    using embeddings and cosine similarity.
    
    Takes prebuilt Excel files as input:
    - Library Excel (from StudioLibraryAgent)
    - Exhibition Excel (from ExhibitionScrapingAgent)
    
    Outputs matches Excel with top 3 matches per exhibition film.
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_client = None
        if self.openai_api_key and OpenAI:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        if not self.openai_client:
            raise ValueError("OpenAI client required. Set OPENAI_API_KEY.")

    def _film_to_text(self, film_dict: Dict) -> str:
        """Convert a film record dictionary to a text representation for embedding with enhanced semantic richness."""
        parts = []
        
        # Basic metadata
        if film_dict.get("title"):
            parts.append(f"Film: {film_dict['title']}")
        if film_dict.get("release_year"):
            parts.append(f"Released: {film_dict['release_year']}")
        if film_dict.get("tagline"):
            parts.append(f"Tagline: {film_dict['tagline']}")
        
        # Creative personnel
        if film_dict.get("director"):
            parts.append(f"Directed by: {film_dict['director']}")
        if film_dict.get("writers"):
            parts.append(f"Written by: {film_dict['writers']}")
        if film_dict.get("cinematographers"):
            parts.append(f"Cinematography: {film_dict['cinematographers']}")
        if film_dict.get("production_designers"):
            parts.append(f"Production Design: {film_dict['production_designers']}")
        if film_dict.get("cast"):
            parts.append(f"Starring: {film_dict['cast']}")
        if film_dict.get("lead_gender"):
            parts.append(f"Lead actor gender: {film_dict['lead_gender']}")
        
        # Genres and keywords
        if film_dict.get("genres"):
            parts.append(f"Genres: {film_dict['genres']}")
        if film_dict.get("keywords"):
            parts.append(f"Keywords: {film_dict['keywords']}")
        
        # Plot and themes
        if film_dict.get("overview"):
            parts.append(f"Plot: {film_dict['overview']}")
        
        # LLM-generated semantic enrichment (if available)
        if film_dict.get("thematic_descriptors"):
            parts.append(f"Themes: {film_dict['thematic_descriptors']}")
        if film_dict.get("stylistic_descriptors"):
            parts.append(f"Style: {film_dict['stylistic_descriptors']}")
        if film_dict.get("emotional_tone"):
            parts.append(f"Tone: {film_dict['emotional_tone']}")
        if film_dict.get("need"):
            parts.append(f"Viewer needs: {film_dict['need']}")
        
        return "\n".join(parts)

    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of texts using OpenAI's text-embedding-3-small model."""
        embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch,
                    dimensions=1536
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"[MatchingAgent] Error creating embeddings for batch {i//batch_size + 1}: {e}")
                embeddings.extend([[0.0] * 1536 for _ in batch])
        return embeddings

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))
    
    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def _normalize_list_string(self, text: str) -> str:
        """Convert list-format string like "['item1', 'item2']" to comma-separated "item1, item2"."""
        import ast
        text = str(text).strip()
        if text.startswith('[') and text.endswith(']'):
            try:
                # Try to parse as Python list
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return ", ".join(str(item) for item in parsed)
            except:
                pass
        return text
    
    def _calculate_thematic_similarity(self, lib_film: Dict, ex_film: Dict) -> float:
        """Calculate thematic similarity based on themes, tone, and keywords using word overlap."""
        import re
        # Extract thematic elements and normalize list-format strings
        lib_themes = self._normalize_list_string(lib_film.get("thematic_descriptors", "")).lower()
        ex_themes = self._normalize_list_string(ex_film.get("thematic_descriptors", "")).lower()
        lib_tone = str(lib_film.get("emotional_tone", "")).lower()
        ex_tone = str(ex_film.get("emotional_tone", "")).lower()
        lib_keywords = str(lib_film.get("keywords", "")).lower()
        ex_keywords = str(ex_film.get("keywords", "")).lower()
        
        # Combine and tokenize
        lib_words = set(re.findall(r'\b\w+\b', f"{lib_themes} {lib_tone} {lib_keywords}"))
        ex_words = set(re.findall(r'\b\w+\b', f"{ex_themes} {ex_tone} {ex_keywords}"))
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were"}
        lib_words = lib_words - stop_words
        ex_words = ex_words - stop_words
        
        if not lib_words and not ex_words:
            return 0.0
        
        return self._jaccard_similarity(lib_words, ex_words)
    
    def _calculate_stylistic_similarity(self, lib_film: Dict, ex_film: Dict) -> float:
        """Calculate stylistic similarity based on style descriptors using word overlap."""
        import re
        lib_style = self._normalize_list_string(lib_film.get("stylistic_descriptors", "")).lower()
        ex_style = self._normalize_list_string(ex_film.get("stylistic_descriptors", "")).lower()
        
        if not lib_style.strip() or not ex_style.strip():
            return 0.0
        
        # Tokenize style descriptors
        lib_words = set(re.findall(r'\b\w+\b', lib_style))
        ex_words = set(re.findall(r'\b\w+\b', ex_style))
        
        # Remove stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "features", "film"}
        lib_words = lib_words - stop_words
        ex_words = ex_words - stop_words
        
        return self._jaccard_similarity(lib_words, ex_words)
    
    def _calculate_personnel_similarity(self, lib_film: Dict, ex_film: Dict) -> float:
        """Calculate personnel similarity based on cast, director, etc. using exact name matching."""
        lib_cast = str(lib_film.get("cast", "")).lower()
        ex_cast = str(ex_film.get("cast", "")).lower()
        lib_dir = str(lib_film.get("director", "")).lower()
        ex_dir = str(ex_film.get("director", "")).lower()
        
        # Split cast into individual names (comma-separated)
        lib_cast_set = set([name.strip() for name in lib_cast.split(",") if name.strip()])
        ex_cast_set = set([name.strip() for name in ex_cast.split(",") if name.strip()])
        
        # Director matching
        dir_match = 0.0
        if lib_dir and ex_dir:
            lib_dirs = set([d.strip() for d in lib_dir.split(",") if d.strip()])
            ex_dirs = set([d.strip() for d in ex_dir.split(",") if d.strip()])
            if lib_dirs & ex_dirs:  # Any common director
                dir_match = 0.3
        
        # Cast matching (Jaccard similarity)
        cast_sim = self._jaccard_similarity(lib_cast_set, ex_cast_set)
        
        # Combine: 30% director, 70% cast
        return min(1.0, dir_match + 0.7 * cast_sim)
    
    def _calculate_similarity_boost(self, lib_film: Dict, ex_film: Dict) -> float:
        """Calculate post-processing similarity boost based on explicit shared attributes."""
        boost = 0.0
        
        # Check for common actors
        lib_cast = str(lib_film.get("cast", "")).lower()
        ex_cast = str(ex_film.get("cast", "")).lower()
        if lib_cast and ex_cast:
            # Split cast lists and find common actors
            lib_actors = [a.strip() for a in lib_cast.split(",")]
            ex_actors = [a.strip() for a in ex_cast.split(",")]
            common_actors = set(lib_actors) & set(ex_actors)
            if common_actors:
                # Boost: 0.05 per common actor, up to 0.15 max
                boost += min(0.15, len(common_actors) * 0.05)
        
        # Check for explicit theme matches
        lib_themes = self._normalize_list_string(lib_film.get("thematic_descriptors", "")).lower()
        ex_themes = self._normalize_list_string(ex_film.get("thematic_descriptors", "")).lower()
        if lib_themes and ex_themes:
            import re
            # Extract theme keywords
            lib_theme_words = set(re.findall(r'\b\w+\b', lib_themes))
            ex_theme_words = set(re.findall(r'\b\w+\b', ex_themes))
            # Remove common stop words
            stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
            common_themes = (lib_theme_words & ex_theme_words) - stop_words
            if len(common_themes) >= 2:
                boost += 0.08
            elif len(common_themes) == 1:
                boost += 0.03
        
        # Check for explicit style matches
        lib_style = self._normalize_list_string(lib_film.get("stylistic_descriptors", "")).lower()
        ex_style = self._normalize_list_string(ex_film.get("stylistic_descriptors", "")).lower()
        if lib_style and ex_style:
            # Look for common style keywords
            style_keywords = ["gritty", "raw", "minimalist", "contemplative", "character-driven", 
                            "non-linear", "slow-paced", "urban", "realistic", "surreal"]
            lib_has = [kw for kw in style_keywords if kw in lib_style]
            ex_has = [kw for kw in style_keywords if kw in ex_style]
            common_styles = set(lib_has) & set(ex_has)
            if common_styles:
                boost += min(0.05, len(common_styles) * 0.02)
        
        # Check for shared keywords
        lib_keywords = str(lib_film.get("keywords", "")).lower()
        ex_keywords = str(ex_film.get("keywords", "")).lower()
        if lib_keywords and ex_keywords:
            lib_kw_set = set([kw.strip() for kw in lib_keywords.split(",")])
            ex_kw_set = set([kw.strip() for kw in ex_keywords.split(",")])
            common_kw = lib_kw_set & ex_kw_set
            if common_kw:
                boost += min(0.03, len(common_kw) * 0.01)
        
        return min(boost, 0.25)  # Cap total boost at 0.25
    
    def _column_similarity(self, lib_film: Dict, ex_film: Dict, col: str) -> float:
        """
        Compute a 0-1 similarity for a single column between library and exhibition film.
        Used for extra_weights (dynamic column boost). Numeric columns use proximity; text uses overlap.
        """
        lv = lib_film.get(col)
        ev = ex_film.get(col)
        if lv is None or (isinstance(lv, float) and pd.isna(lv)):
            return 0.0
        if ev is None or (isinstance(ev, float) and pd.isna(ev)):
            return 0.0
        try:
            lnum = float(lv)
            enum = float(ev)
            # Numeric: 1 - normalized distance (scale ~50 years)
            dist = abs(lnum - enum) / 50.0
            return max(0.0, 1.0 - min(1.0, dist))
        except (TypeError, ValueError):
            pass
        # Text: tokenize and Jaccard or substring match
        ls = str(lv).strip().lower()
        es = str(ev).strip().lower()
        if not ls or not es:
            return 0.0
        lset = set(_.strip() for _ in ls.replace(",", " ").split() if _.strip())
        eset = set(_.strip() for _ in es.replace(",", " ").split() if _.strip())
        if lset and eset:
            return self._jaccard_similarity(lset, eset)
        return 1.0 if ls == es else (0.5 if ls in es or es in ls else 0.0)

    def _calculate_enhanced_similarity(
        self, lib_film: Dict, ex_film: Dict, base_similarity: float,
        director_weight: float = 0.2,
        writer_weight: float = 0.15,
        cast_weight: float = 0.15,
        thematic_weight: float = 0.3,
        stylistic_weight: float = 0.2,
        extra_weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate enhanced similarity using multiplicative approach with boost.
        
        Uses base similarity as foundation and multiplies by component match factors,
        then adds post-processing boost. extra_weights: optional { column_name: weight }
        to boost additional columns (e.g. genres, release_year) in the match.
        
        Args:
            lib_film: Library film dictionary
            ex_film: Exhibition film dictionary
            base_similarity: Base cosine similarity from embeddings
            director_weight: Weight for director matching (0.0-1.0)
            writer_weight: Weight for writer matching (0.0-1.0)
            cast_weight: Weight for cast matching (0.0-1.0)
            thematic_weight: Weight for thematic matching (0.0-1.0)
            stylistic_weight: Weight for stylistic matching (0.0-1.0)
            extra_weights: Optional dict of column_name -> weight for dynamic column boosting
        """
        # Calculate component similarities
        thematic_sim = self._calculate_thematic_similarity(lib_film, ex_film)
        stylistic_sim = self._calculate_stylistic_similarity(lib_film, ex_film)
        personnel_sim = self._calculate_personnel_similarity(lib_film, ex_film)
        
        # Split personnel similarity into director, writer, cast components
        director_sim, writer_sim, cast_sim = self._calculate_personnel_components(lib_film, ex_film)
        
        # Normalize all weights (fixed + extra) to sum to 1.0
        total_weight = director_weight + writer_weight + cast_weight + thematic_weight + stylistic_weight
        extra = dict(extra_weights) if extra_weights else {}
        extra_sum = sum(extra.values())
        total = total_weight + extra_sum
        if total > 0:
            scale = 1.0 / total
            director_weight = director_weight * scale
            writer_weight = writer_weight * scale
            cast_weight = cast_weight * scale
            thematic_weight = thematic_weight * scale
            stylistic_weight = stylistic_weight * scale
            extra = {k: v * scale for k, v in extra.items()}
        else:
            director_weight = 0.2
            writer_weight = 0.15
            cast_weight = 0.15
            thematic_weight = 0.3
            stylistic_weight = 0.2
            extra = {}
        
        # Calculate weighted component match factors
        director_factor = 1.0 + (director_sim * director_weight * 0.5) if director_weight > 0 else 1.0
        writer_factor = 1.0 + (writer_sim * writer_weight * 0.5) if writer_weight > 0 else 1.0
        cast_factor = 1.0 + (cast_sim * cast_weight * 0.5) if cast_weight > 0 else 1.0
        thematic_factor = 1.0 + (thematic_sim * thematic_weight * 0.5) if thematic_weight > 0 else 1.0
        stylistic_factor = 1.0 + (stylistic_sim * stylistic_weight * 0.5) if stylistic_weight > 0 else 1.0
        
        # Extra column factors (only for columns present in both records)
        extra_factor = 1.0
        for col, w in extra.items():
            if col in lib_film and col in ex_film and w > 0:
                sim = self._column_similarity(lib_film, ex_film, col)
                extra_factor *= 1.0 + (sim * w * 0.5)
        
        # Multiply base similarity by weighted component factors
        enhanced_sim = base_similarity * director_factor * writer_factor * cast_factor * thematic_factor * stylistic_factor * extra_factor
        
        # Apply post-processing boost for explicit shared attributes
        boost = self._calculate_similarity_boost(lib_film, ex_film)
        enhanced_sim = enhanced_sim + boost
        
        # Normalize to [0, 1] range (in case boost pushes it over)
        return min(1.0, max(0.0, enhanced_sim))
    
    def _calculate_personnel_components(self, lib_film: Dict, ex_film: Dict) -> Tuple[float, float, float]:
        """Calculate separate similarity scores for director, writer, and cast."""
        # Director similarity
        lib_dir = str(lib_film.get("director", "")).lower()
        ex_dir = str(ex_film.get("director", "")).lower()
        director_sim = 0.0
        if lib_dir and ex_dir:
            lib_dirs = set([d.strip() for d in lib_dir.split(",") if d.strip()])
            ex_dirs = set([d.strip() for d in ex_dir.split(",") if d.strip()])
            if lib_dirs & ex_dirs:  # Any common director
                director_sim = 1.0
            elif lib_dirs and ex_dirs:
                # Partial match (similar names, etc.) - could be enhanced
                director_sim = 0.0
        
        # Writer similarity
        lib_writers = str(lib_film.get("writers", "")).lower()
        ex_writers = str(ex_film.get("writers", "")).lower()
        writer_sim = 0.0
        if lib_writers and ex_writers:
            lib_writer_set = set([w.strip() for w in lib_writers.split(",") if w.strip()])
            ex_writer_set = set([w.strip() for w in ex_writers.split(",") if w.strip()])
            writer_sim = self._jaccard_similarity(lib_writer_set, ex_writer_set)
        
        # Cast similarity
        lib_cast = str(lib_film.get("cast", "")).lower()
        ex_cast = str(ex_film.get("cast", "")).lower()
        cast_sim = 0.0
        if lib_cast and ex_cast:
            lib_cast_set = set([name.strip() for name in lib_cast.split(",") if name.strip()])
            ex_cast_set = set([name.strip() for name in ex_cast.split(",") if name.strip()])
            cast_sim = self._jaccard_similarity(lib_cast_set, ex_cast_set)
        
        return director_sim, writer_sim, cast_sim

    def _generate_match_reasoning(
        self, library_film: Dict, exhibition_film: Dict, cosine_sim: float
    ) -> str:
        """Use LLM to generate reasoning text explaining why a library film matches an exhibition film."""
        # Skip reasoning for low similarity matches
        if cosine_sim < 0.4:
            return f"Low similarity match (Similarity: {cosine_sim:.4f})"
        
        lib_text = self._film_to_text(library_film)
        ex_text = self._film_to_text(exhibition_film)

        prompt = (
            f"Library Film:\n{lib_text}\n\n"
            f"Exhibition Film:\n{ex_text}\n\n"
            f"These films have a cosine similarity of {cosine_sim:.4f} based on their embeddings.\n"
            "Explain in 2-3 sentences why this library film is a relevant match for the exhibition film. "
            "Focus on shared creative personnel (directors, writers, cast, cinematographers, etc.), "
            "similar genres, themes, or time periods. Be specific about the connections."
        )

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a film analysis assistant that explains connections between films."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            reasoning = response.choices[0].message.content.strip()
            return f"{reasoning} (Similarity: {cosine_sim:.4f})"
        except Exception as e:
            print(f"[MatchingAgent] Error generating reasoning: {e}")
            return f"Cosine similarity: {cosine_sim:.4f}"
    
    def _generate_reasoning_parallel(self, args: Tuple) -> Tuple[int, str]:
        """Wrapper for parallel reasoning generation. Returns (index, reasoning)."""
        lib, ex, cosine_sim, idx = args
        reasoning = self._generate_match_reasoning(lib, ex, cosine_sim)
        return (idx, reasoning)

    def match_library_and_exhibitions(
        self,
        library_path: str,
        exhibitions_path: str,
        *,
        output_path: Optional[str] = None,
        studio_name: Optional[str] = None,
        library_embeddings_path: Optional[str] = None,
        exhibition_embeddings_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load library and exhibition Excel files, use pre-saved embeddings or create new ones,
        calculate cosine similarity, and generate matches with LLM reasoning.
        
        Args:
            library_path: Path to library Excel file
            exhibitions_path: Path to exhibitions Excel file
            library_embeddings_path: Optional path to pre-saved library embeddings .npy file
            exhibition_embeddings_path: Optional path to pre-saved exhibition embeddings .npy file
        
        Returns matches DataFrame and writes to Excel.
        """
        print(f"\n[MatchingAgent] Loading library from: {library_path}")
        library_df = pd.read_excel(library_path)
        print(f"[MatchingAgent]   Loaded {len(library_df)} library films")

        print(f"\n[MatchingAgent] Loading exhibitions from: {exhibitions_path}")
        exhibitions_df = pd.read_excel(exhibitions_path)
        print(f"[MatchingAgent]   Loaded {len(exhibitions_df)} exhibition films")

        # Load or create library embeddings
        if library_embeddings_path and os.path.exists(library_embeddings_path):
            print(f"\n[MatchingAgent] Loading pre-saved library embeddings from: {library_embeddings_path}")
            lib_embeddings_array = np.load(library_embeddings_path)
            lib_embeddings = lib_embeddings_array.tolist()
            print(f"[MatchingAgent]   Loaded embeddings for {len(lib_embeddings)} library films")
            if len(lib_embeddings) != len(library_df):
                print(f"[MatchingAgent]   [WARNING] Embedding count ({len(lib_embeddings)}) doesn't match library count ({len(library_df)})")
        else:
            print("\n[MatchingAgent] Creating embeddings for library films...")
            lib_rows = library_df.to_dict(orient="records")
            lib_texts = [self._film_to_text(lib) for lib in lib_rows]
            lib_embeddings = self._create_embeddings(lib_texts)
            print(f"[MatchingAgent]   Created embeddings for {len(lib_embeddings)} library films")

        # Load or create exhibition embeddings
        if exhibition_embeddings_path and os.path.exists(exhibition_embeddings_path):
            print(f"\n[MatchingAgent] Loading pre-saved exhibition embeddings from: {exhibition_embeddings_path}")
            ex_embeddings_array = np.load(exhibition_embeddings_path)
            ex_embeddings = ex_embeddings_array.tolist()
            print(f"[MatchingAgent]   Loaded embeddings for {len(ex_embeddings)} exhibition films")
            if len(ex_embeddings) != len(exhibitions_df):
                print(f"[MatchingAgent]   [WARNING] Embedding count ({len(ex_embeddings)}) doesn't match exhibition count ({len(exhibitions_df)})")
        else:
            print("\n[MatchingAgent] Creating embeddings for exhibition films...")
            ex_rows = exhibitions_df.to_dict(orient="records")
            ex_texts = [self._film_to_text(ex) for ex in ex_rows]
            ex_embeddings = self._create_embeddings(ex_texts)
            print(f"[MatchingAgent]   Created embeddings for {len(ex_embeddings)} exhibition films")

        print("\n[MatchingAgent] Calculating cosine similarities and finding top matches...")
        matches: List[MatchRecord] = []
        
        # Convert DataFrames to records for matching
        lib_rows = library_df.to_dict(orient="records")
        ex_rows = exhibitions_df.to_dict(orient="records")
        
        # First pass: Calculate all similarities and find top 3 for each exhibition film
        all_top_matches: List[Tuple[int, int, float]] = []  # (ex_idx, lib_idx, similarity)
        
        print("[MatchingAgent] Using enhanced multi-dimensional similarity calculation...")
        print("[MatchingAgent]   Components: Base (10%), Thematic (40%), Stylistic (30%), Personnel (20%) + Boost")
        
        for ex_idx, ex in enumerate(ex_rows):
            ex_embedding = ex_embeddings[ex_idx]
            similarities = []

            # Calculate similarity to all library films
            for lib_idx, lib in enumerate(lib_rows):
                lib_embedding = lib_embeddings[lib_idx]
                base_similarity = self._cosine_similarity(ex_embedding, lib_embedding)
                
                # Calculate enhanced similarity with multi-dimensional approach
                enhanced_similarity = self._calculate_enhanced_similarity(
                    lib, ex, base_similarity
                )
                
                similarities.append((lib_idx, enhanced_similarity))

            # Sort by similarity (highest first) and take top 3
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_3 = similarities[:3]
            
            # Store for parallel processing
            for lib_idx, cosine_sim in top_3:
                all_top_matches.append((ex_idx, lib_idx, cosine_sim))
            
            if (ex_idx + 1) % 20 == 0:
                print(f"[MatchingAgent]   Calculated similarities for {ex_idx + 1}/{len(ex_rows)} exhibition films...")
        
        print(f"\n[MatchingAgent] Generating reasoning for {len(all_top_matches)} matches in parallel...")
        print(f"[MatchingAgent]   Using up to 20 concurrent threads")
        
        # Prepare arguments for parallel reasoning generation
        reasoning_args = []
        for match_idx, (ex_idx, lib_idx, cosine_sim) in enumerate(all_top_matches):
            lib = lib_rows[lib_idx]
            ex = ex_rows[ex_idx]
            reasoning_args.append((lib, ex, cosine_sim, match_idx))
        
        # Generate reasoning in parallel
        reasoning_results: Dict[int, str] = {}
        max_workers = min(20, len(reasoning_args))  # Limit concurrent API calls
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self._generate_reasoning_parallel, args): args[3]
                for args in reasoning_args
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_idx):
                try:
                    idx, reasoning = future.result()
                    reasoning_results[idx] = reasoning
                    completed += 1
                    if completed % 50 == 0:
                        print(f"[MatchingAgent]   Generated reasoning for {completed}/{len(reasoning_args)} matches...")
                except Exception as e:
                    idx = future_to_idx[future]
                    ex_idx, lib_idx, cosine_sim = all_top_matches[idx]
                    print(f"[MatchingAgent]   Error generating reasoning for match {idx}: {e}")
                    reasoning_results[idx] = f"Error generating reasoning (Similarity: {cosine_sim:.4f})"
        
        # Build match records
        for match_idx, (ex_idx, lib_idx, cosine_sim) in enumerate(all_top_matches):
            lib = lib_rows[lib_idx]
            ex = ex_rows[ex_idx]
            reasoning = reasoning_results.get(match_idx, f"Cosine similarity: {cosine_sim:.4f}")
            
            match = MatchRecord(
                library_title=str(lib.get("title") or ""),
                exhibition_title=str(ex.get("title") or ""),
                country=str(ex.get("country") or ""),
                location=str(ex.get("location") or ""),
                relevance_start=str(ex.get("start_date") or ""),
                relevance_end=str(ex.get("end_date") or ""),
                similarity_reason=reasoning,
                programme_url=str(ex.get("programme_url") or ""),
                cosine_similarity=cosine_sim,
            )
            matches.append(match)

        # Sort all matches by cosine similarity (highest first)
        matches.sort(key=lambda m: m.cosine_similarity, reverse=True)

        print(f"\n[MatchingAgent] Generated {len(matches)} total matches (top 3 per exhibition film)")

        matches_df = pd.DataFrame([asdict(m) for m in matches])
        
        # Determine output path
        if output_path is None:
            if studio_name:
                output_path = f"{studio_name.lower().replace(' ', '_')}_matches.xlsx"
            else:
                output_path = "matches.xlsx"
        
        matches_df.to_excel(output_path, index=False)
        print(f"[MatchingAgent] Matches written to: {output_path}")
        
        return matches_df


if __name__ == "__main__":
    """
    Example CLI-style usage:

    1. Set your TMDb API key as an environment variable:
       - On macOS/Linux:
           export TMDB_API_KEY="YOUR_KEY"
       - On Windows (PowerShell):
           $env:TMDB_API_KEY = "YOUR_KEY"

    2. Run this file:
           python film_agent.py
    """
    studio = input("Enter studio / distributor name (e.g., 'Lionsgate'): ").strip()
    if not studio:
        raise SystemExit("Studio name is required.")

    agent = FilmExhibitionAgent()
    agent.run(studio_name=studio)

