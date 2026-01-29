"""
Fast Chatbot Agent for Streamlit - Optimized for low latency
Uses pre-computed data and caching to provide quick recommendations
"""

import os
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path

from film_agent import MatchingAgent, TMDbClient
from config import get_openai_api_key, get_tmdb_api_key
from query_intent_parser import QueryIntentParser, QueryIntent

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class ChatbotAgent:
    """
    Fast chatbot agent optimized for low-latency responses.
    
    Features:
    - Uses pre-computed embeddings (no regeneration)
    - Caches exhibition data (avoids re-scraping)
    - Filters matches: excludes exact matches (>0.9), focuses on 0.5-0.7 range
    - Returns 3-5 best recommendations per territory
    """
    
    def __init__(self, studio_name: str = "Lionsgate"):
        self.studio_name = studio_name
        self.openai_key = get_openai_api_key()
        self.openai_client = OpenAI(api_key=self.openai_key) if OpenAI else None
        
        # File paths
        self.library_path = f"{studio_name.lower().replace(' ', '_')}_library.xlsx"
        self.exhibitions_path = "upcoming_exhibitions.xlsx"
        self.library_embeddings_path = f"{studio_name.lower().replace(' ', '_')}_library_embeddings.npy"
        self.exhibition_embeddings_path = "upcoming_exhibitions_embeddings.npy"
        
        # Cache for loaded data
        self._library_df = None
        self._exhibitions_df = None
        self._library_embeddings = None
        self._exhibition_embeddings = None
        self._last_exhibition_update = None
        
        # Matching agent
        self.matching_agent = MatchingAgent()
        
        # Query intent parser for dynamic matching
        self.query_parser = QueryIntentParser(openai_api_key=self.openai_key)

        # Remember last parsed intent to support multi-turn refinements
        self._last_intent: Optional[QueryIntent] = None
        
        # TMDB client for poster images
        try:
            self.tmdb_client = TMDbClient(tmdb_api_key=get_tmdb_api_key())
        except Exception:
            self.tmdb_client = None
    
    def _load_library(self) -> pd.DataFrame:
        """Load library data (cached)."""
        if self._library_df is None:
            if not os.path.exists(self.library_path):
                raise FileNotFoundError(f"Library file not found: {self.library_path}")
            self._library_df = pd.read_excel(self.library_path)
        return self._library_df
    
    def _load_exhibitions(self) -> pd.DataFrame:
        """Load exhibition data (cached)."""
        if self._exhibitions_df is None:
            if not os.path.exists(self.exhibitions_path):
                raise FileNotFoundError(f"Exhibition file not found: {self.exhibitions_path}")
            self._exhibitions_df = pd.read_excel(self.exhibitions_path)
            # Check file modification time
            self._last_exhibition_update = datetime.fromtimestamp(
                os.path.getmtime(self.exhibitions_path)
            )
        return self._exhibitions_df
    
    def _load_embeddings(self, required: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load pre-computed embeddings (cached).
        
        Args:
            required: If True, raises FileNotFoundError if embeddings are missing.
                     If False, returns None, None if embeddings are missing.
        
        Returns:
            Tuple of (library_embeddings, exhibition_embeddings) or (None, None) if not found and not required
        """
        if self._library_embeddings is None or self._exhibition_embeddings is None:
            if not os.path.exists(self.library_embeddings_path):
                if required:
                    raise FileNotFoundError(
                        f"Library embeddings not found: {self.library_embeddings_path}\n"
                        f"To generate embeddings, run: python generate_library_embeddings.py"
                    )
                return None, None
            if not os.path.exists(self.exhibition_embeddings_path):
                if required:
                    raise FileNotFoundError(
                        f"Exhibition embeddings not found: {self.exhibition_embeddings_path}\n"
                        f"To generate embeddings, run: python generate_exhibition_embeddings.py"
                    )
                return None, None
            
            self._library_embeddings = np.load(self.library_embeddings_path)
            self._exhibition_embeddings = np.load(self.exhibition_embeddings_path)
        
        return self._library_embeddings, self._exhibition_embeddings
    
    def _is_exhibition_data_stale(self, max_age_days: int = 7) -> bool:
        """Check if exhibition data needs to be refreshed."""
        if self._last_exhibition_update is None:
            if os.path.exists(self.exhibitions_path):
                self._last_exhibition_update = datetime.fromtimestamp(
                    os.path.getmtime(self.exhibitions_path)
                )
            else:
                return True  # File doesn't exist, needs refresh
        
        age = datetime.now() - self._last_exhibition_update
        return age.days > max_age_days
    
    def get_recommendations(
        self,
        territory: str,
        *,
        min_similarity: float = 0.5,
        max_similarity: float = 0.7,
        top_n: int = 5,
        exclude_exact_matches: bool = True,
        exact_match_threshold: float = 0.9
    ) -> List[Dict]:
        """
        Get top recommendations for a territory.
        
        Args:
            territory: Country code (e.g., "US", "UK", "FR", "CA", "MX")
            min_similarity: Minimum similarity score (default: 0.5)
            max_similarity: Maximum similarity score (default: 0.7)
            top_n: Number of recommendations to return (default: 5)
            exclude_exact_matches: Exclude films with similarity > threshold (default: True)
            exact_match_threshold: Threshold for exact matches (default: 0.9)
        
        Returns:
            List of recommendation dictionaries with film info and similarity scores
        """
        # Load data (cached)
        library_df = self._load_library()
        exhibitions_df = self._load_exhibitions()
        lib_embeddings, ex_embeddings = self._load_embeddings(required=True)  # Required for similarity matching
        
        # Filter exhibitions by territory
        territory_exhibitions = exhibitions_df[
            exhibitions_df["country"].str.upper() == territory.upper()
        ].copy()
        
        if len(territory_exhibitions) == 0:
            return []
        
        # Convert to records for matching
        lib_rows = library_df.to_dict(orient="records")
        ex_rows = territory_exhibitions.to_dict(orient="records")
        
        # Get indices of territory exhibitions in full exhibitions array
        territory_indices = territory_exhibitions.index.tolist()
        
        # Vectorized similarity calculation for speed
        territory_ex_embeddings = ex_embeddings[territory_indices]
        
        # Calculate all similarities at once using vectorized operations
        # Shape: (num_territory_exhibitions, embedding_dim) @ (embedding_dim, num_library_films)
        # Result: (num_territory_exhibitions, num_library_films)
        lib_embeddings_norm = lib_embeddings / (np.linalg.norm(lib_embeddings, axis=1, keepdims=True) + 1e-8)
        ex_embeddings_norm = territory_ex_embeddings / (np.linalg.norm(territory_ex_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Matrix multiplication for all similarities at once
        base_similarities = np.dot(ex_embeddings_norm, lib_embeddings_norm.T)  # (ex, lib)
        
        # Calculate enhanced similarities (this still requires per-pair calculation for components)
        # But we can optimize by only calculating for pairs in the target range
        all_matches = []
        
        for ex_idx_in_territory, ex_row in enumerate(ex_rows):
            ex_idx_global = territory_indices[ex_idx_in_territory]
            base_sims_row = base_similarities[ex_idx_in_territory]  # Similarities to all library films
            
            # Pre-filter: only process library films with base similarity in reasonable range
            # This reduces the number of enhanced similarity calculations significantly
            # Use wider range for base similarity to catch candidates that might be boosted
            candidate_indices = np.where(
                (base_sims_row >= min_similarity * 0.7) &  # Lower threshold (wider net)
                (base_sims_row <= max_similarity * 1.3)      # Upper threshold (wider net)
            )[0]
            
            # Limit candidates to top 50 to avoid too many enhanced calculations
            if len(candidate_indices) > 50:
                top_candidate_sims = base_sims_row[candidate_indices]
                top_indices = np.argsort(top_candidate_sims)[-50:][::-1]
                candidate_indices = candidate_indices[top_indices]
            
            for lib_idx in candidate_indices:
                base_sim = float(base_sims_row[lib_idx])
                lib_row = lib_rows[lib_idx]
                
                # Enhanced similarity (only for candidates)
                enhanced_sim = self.matching_agent._calculate_enhanced_similarity(
                    lib_row, ex_row, base_sim
                )
                
                # Filter: exclude exact matches (likely same film)
                if exclude_exact_matches and enhanced_sim > exact_match_threshold:
                    continue  # Skip exact matches
                
                # Focus on the sweet spot: 0.5-0.7 range for unintuitive but logical matches
                if min_similarity <= enhanced_sim <= max_similarity:
                    all_matches.append({
                        "library_film": lib_row,
                        "exhibition_film": ex_row,
                        "similarity": enhanced_sim,
                        "base_similarity": base_sim,
                        "territory": territory,
                        "location": ex_row.get("location", ""),
                        "exhibition_dates": ex_row.get("scheduled_dates", ""),
                    })
        
        # Sort by similarity (highest first) and take top N
        all_matches.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Group by library film and take best match per library film
        # This ensures we get diverse recommendations, not just multiple matches for same film
        seen_library_films = set()
        unique_matches = []
        
        for match in all_matches:
            lib_title = match["library_film"].get("title", "")
            if lib_title not in seen_library_films:
                seen_library_films.add(lib_title)
                unique_matches.append(match)
                if len(unique_matches) >= top_n:
                    break
        
        # Format recommendations with reasoning and poster images
        recommendations = []
        for match in unique_matches[:top_n]:
            lib = match["library_film"]
            ex = match["exhibition_film"]
            
            # Generate reasoning for why this film is recommended
            reasoning = self._generate_recommendation_reasoning(
                lib, ex, match["similarity"]
            )
            
            # Get poster image URL
            poster_url = self._get_poster_url(lib.get("tmdb_id"))
            
            recommendations.append({
                "title": lib.get("title", ""),
                "year": lib.get("release_year", ""),
                "director": lib.get("director", ""),
                "genres": lib.get("genres", ""),
                "similarity": match["similarity"],
                "base_similarity": match["base_similarity"],
                "matched_exhibition": ex.get("title", ""),
                "exhibition_location": match["location"],
                "exhibition_dates": match["exhibition_dates"],
                "territory": territory,
                "themes": lib.get("thematic_descriptors", ""),
                "style": lib.get("stylistic_descriptors", ""),
                "reasoning": reasoning,
                "poster_url": poster_url,
                "tmdb_id": lib.get("tmdb_id"),
            })
        
        return recommendations
    
    def get_dynamic_recommendations(
        self,
        query: str,
        *,
        top_n: int = 5,
        history_prompts: Optional[List[str]] = None
    ) -> Dict:
        """
        Get recommendations based on dynamic query intent parsing.
        
        This method:
        1. Parses user query to extract intent (filters, weights, etc.)
        2. Filters library and exhibition data based on intent
        3. Calculates similarities with dynamic weights
        4. Returns top N recommendations with reasoning
        
        Args:
            query: User query string
            top_n: Maximum number of recommendations (default: 5)
        
        Returns:
            Dictionary with recommendations and query metadata
        """
        # Parse query intent
        intent = self.query_parser.parse(
            query,
            history_prompts=history_prompts,
            previous_intent=self._last_intent,
        )
        # Update last intent for future refinements
        self._last_intent = intent
        
        # Debug: Print parsed intent for troubleshooting
        print(f"[ChatbotAgent] Parsed intent: lead_gender={intent.lead_gender}, time_period={intent.time_period}, "
              f"match_to_specific_film={intent.match_to_specific_film}, director_weight={intent.director_weight}, "
              f"writer_weight={intent.writer_weight}, cast_weight={intent.cast_weight}")
        
        # Load data
        library_df = self._load_library()
        exhibitions_df = self._load_exhibitions()
        lib_embeddings, ex_embeddings = self._load_embeddings(required=True)
        
        # Apply filters to library
        filtered_library_df = self._apply_library_filters(library_df, intent)
        print(f"[ChatbotAgent] After library filters: {len(filtered_library_df)} films (from {len(library_df)} total)")
        
        if len(filtered_library_df) == 0:
            raw = "No library films match the specified filters."
            intent_sum = self._intent_summary(intent)
            lib_genres = self._genre_sample_from_df(library_df)
            explained = self._generate_error_explanation(
                query,
                raw,
                error_type="no_library",
                intent_summary=intent_sum,
                library_total=len(library_df),
                library_after=0,
                library_genres=lib_genres,
            )
            return {
                "error": explained,
                "recommendations": [],
                "intent": intent,
            }
        
        # Apply filters to exhibitions
        filtered_exhibitions_df = self._apply_exhibition_filters(exhibitions_df, intent)
        print(f"[ChatbotAgent] After exhibition filters: {len(filtered_exhibitions_df)} exhibitions (from {len(exhibitions_df)} total)")
        
        if len(filtered_exhibitions_df) == 0:
            raw = "No exhibitions match the specified filters."
            intent_sum = self._intent_summary(intent)
            ex_genres = self._genre_sample_from_df(exhibitions_df)
            lib_genres = self._genre_sample_from_df(filtered_library_df)
            explained = self._generate_error_explanation(
                query,
                raw,
                error_type="no_exhibitions",
                intent_summary=intent_sum,
                library_total=len(library_df),
                library_after=len(filtered_library_df),
                exhibition_total=len(exhibitions_df),
                exhibition_after=0,
                exhibition_genres=ex_genres,
                library_genres=lib_genres,
            )
            return {
                "error": explained,
                "recommendations": [],
                "intent": intent,
            }
        
        # Get indices of filtered data in original arrays
        library_indices = filtered_library_df.index.tolist()
        exhibition_indices = filtered_exhibitions_df.index.tolist()
        
        # Get filtered embeddings
        filtered_lib_embeddings = lib_embeddings[library_indices]
        filtered_ex_embeddings = ex_embeddings[exhibition_indices]
        
        # Convert to records
        lib_rows = filtered_library_df.to_dict(orient="records")
        ex_rows = filtered_exhibitions_df.to_dict(orient="records")

        # ----------------------------
        # Dynamic scoring improvements
        # ----------------------------
        # 1) Exclude exact "same film" matches (e.g. Flash Gordon vs Flash Gordon) from dominating.
        # 2) Add query-to-library relevance (query embedding vs library embeddings) so theme/intent can outrank
        #    simple "same title currently in exhibitions" matches.

        def _normalize_title(t: str) -> str:
            t = (t or "").lower()
            t = re.sub(r"[^a-z0-9\s]", " ", t)
            t = re.sub(r"\s+", " ", t).strip()
            return t

        def _title_signature(t: str) -> str:
            """Coarse grouping to avoid franchise spam (e.g. multiple Flash Gordon titles)."""
            words = _normalize_title(t).split()
            if not words:
                return ""
            if words[0] == "the":
                return " ".join(words[:3])
            return " ".join(words[:2])

        def _is_exact_same_film(lib: Dict, ex: Dict) -> bool:
            lib_id = lib.get("tmdb_id")
            ex_id = ex.get("tmdb_id")
            if lib_id and ex_id and str(lib_id).isdigit() and str(ex_id).isdigit():
                try:
                    return int(lib_id) == int(ex_id)
                except Exception:
                    pass
            # Fallback: exact title + year
            lib_title = _normalize_title(lib.get("title", ""))
            ex_title = _normalize_title(ex.get("title", ""))
            lib_year = str(lib.get("release_year") or "").strip()
            ex_year = str(ex.get("release_year") or "").strip()
            return bool(lib_title and ex_title and lib_title == ex_title and lib_year and ex_year and lib_year == ex_year)

        # Normalize embeddings once for fast cosine similarity
        lib_norm = filtered_lib_embeddings / (np.linalg.norm(filtered_lib_embeddings, axis=1, keepdims=True) + 1e-8)
        ex_norm = filtered_ex_embeddings / (np.linalg.norm(filtered_ex_embeddings, axis=1, keepdims=True) + 1e-8)
        base_matrix = np.dot(ex_norm, lib_norm.T)  # (ex, lib)

        # Query-to-library relevance (0..1)
        query_sims = np.zeros(len(lib_rows), dtype=float)
        if self.openai_client:
            try:
                q_emb_resp = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=query,
                    dimensions=1536,
                )
                q_vec = np.array(q_emb_resp.data[0].embedding, dtype=float)
                q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-8)
                q_cos = np.dot(lib_norm, q_norm)  # [-1, 1]
                query_sims = (q_cos + 1.0) / 2.0  # -> [0, 1]
            except Exception as e:
                print(f"[ChatbotAgent] Warning: query embedding failed: {e}")

        # Weight query relevance higher when prompt is theme/intent driven
        q_lower = query.lower()
        w_query = 0.75
        if "theme" in q_lower or "themes" in q_lower:
            w_query = 0.85
        if "existential" in q_lower:
            w_query = max(w_query, 0.9)
        # If the prompt is explicitly about what's in theaters / current market, give exhibitions more weight
        if any(kw in q_lower for kw in ["in theaters", "in theatres", "current exhibitions", "what's showing", "now showing"]):
            w_query = max(0.6, w_query - 0.15)
        if intent.match_to_specific_film:
            w_query = max(0.6, w_query - 0.15)
        w_ex = 1.0 - w_query

        # For each library film, find its best exhibition match (excluding exact same film), then compute
        # enhanced exhibition similarity + composite relevance score.
        scored = []
        topk = min(8, len(ex_rows)) if len(ex_rows) else 0
        for lib_idx, lib_row in enumerate(lib_rows):
            # Pick best non-exact exhibition candidate
            best_ex_idx = None
            best_base = None
            if topk > 0:
                col = base_matrix[:, lib_idx]
                # Top-K candidates by base similarity
                # np.argpartition expects kth positions; use topk-1 for the top-k set
                cand = np.argpartition(-col, topk - 1)[:topk]
                cand = cand[np.argsort(-col[cand])]
                for ex_i in cand:
                    ex_row = ex_rows[int(ex_i)]
                    if _is_exact_same_film(lib_row, ex_row):
                        continue
                    best_ex_idx = int(ex_i)
                    best_base = float(col[int(ex_i)])
                    break

            if best_ex_idx is None:
                # No valid exhibition match (or all were exact same film)
                continue

            ex_row = ex_rows[best_ex_idx]
            exhibition_similarity = self.matching_agent._calculate_enhanced_similarity(
                lib_row,
                ex_row,
                best_base,
                director_weight=intent.director_weight,
                writer_weight=intent.writer_weight,
                cast_weight=intent.cast_weight,
                thematic_weight=intent.thematic_weight,
                stylistic_weight=intent.stylistic_weight,
            )

            relevance_score = (w_query * float(query_sims[lib_idx])) + (w_ex * float(exhibition_similarity))
            scored.append({
                "library_film": lib_row,
                "exhibition_film": ex_row,
                "relevance_score": relevance_score,
                "exhibition_similarity": float(exhibition_similarity),
                "query_similarity": float(query_sims[lib_idx]),
            })

        scored.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Deduplicate by title signature to avoid multiple franchise entries dominating.
        seen_sig = set()
        unique_matches = []
        for item in scored:
            sig = _title_signature(item["library_film"].get("title", ""))
            if sig and sig in seen_sig:
                continue
            if sig:
                seen_sig.add(sig)
            unique_matches.append(item)
            if len(unique_matches) >= top_n:
                break
        
        # Generate recommendations with reasoning
        recommendations = []
        for match in unique_matches[:top_n]:
            lib = match["library_film"]
            ex = match["exhibition_film"]
            
            # Generate reasoning
            reasoning = self._generate_dynamic_reasoning(
                lib, ex, match["exhibition_similarity"], intent
            )
            
            # Get poster
            poster_url = self._get_poster_url(lib.get("tmdb_id"))
            
            recommendations.append({
                "title": lib.get("title", ""),
                "year": lib.get("release_year", ""),
                "director": lib.get("director", ""),
                "writers": lib.get("writers", ""),
                "cast": lib.get("cast", ""),
                "genres": lib.get("genres", ""),
                # Composite score used for ranking (query relevance + exhibition similarity)
                "relevance_score": match["relevance_score"],
                "exhibition_similarity": match["exhibition_similarity"],
                "query_similarity": match["query_similarity"],
                "similarity": match["relevance_score"],
                "matched_exhibition": ex.get("title", ""),
                "exhibition_location": ex.get("location", ""),
                "exhibition_dates": ex.get("scheduled_dates", ""),
                "territory": intent.territory or "All",
                "themes": lib.get("thematic_descriptors", ""),
                "style": lib.get("stylistic_descriptors", ""),
                "reasoning": reasoning,
                "poster_url": poster_url,
                "tmdb_id": lib.get("tmdb_id"),
            })
        
        return {
            "query_type": "dynamic",
            "intent": intent,
            "recommendations": recommendations,
            "count": len(recommendations),
            "territory": intent.territory,
            "context_mode": "new_search" if intent.is_new_search else "refinement",
        }
    
    def _apply_library_filters(self, library_df: pd.DataFrame, intent: QueryIntent) -> pd.DataFrame:
        """Apply filters to library DataFrame based on query intent."""
        filtered_df = library_df.copy()
        
        # Year filters
        if intent.year_start:
            filtered_df = filtered_df[filtered_df["release_year"] >= intent.year_start]
        if intent.year_end:
            filtered_df = filtered_df[filtered_df["release_year"] <= intent.year_end]
        
        # Genre filters
        if intent.genres:
            genre_mask = filtered_df["genres"].astype(str).str.lower()
            genre_filter = genre_mask.str.contains("|".join(intent.genres), na=False)
            filtered_df = filtered_df[genre_filter]
        
        # Director filters
        if intent.specific_directors:
            director_mask = filtered_df["director"].astype(str).str.lower()
            director_filter = director_mask.str.contains("|".join([d.lower() for d in intent.specific_directors]), na=False)
            filtered_df = filtered_df[director_filter]
        
        # Writer filters
        if intent.specific_writers:
            writer_mask = filtered_df["writers"].astype(str).str.lower()
            writer_filter = writer_mask.str.contains("|".join([w.lower() for w in intent.specific_writers]), na=False)
            filtered_df = filtered_df[writer_filter]
        
        # Lead gender filter (uses pre-computed data from library)
        if intent.lead_gender:
            if "lead_gender" in filtered_df.columns:
                # Filter using stored lead_gender field
                filtered_df = filtered_df[filtered_df["lead_gender"] == intent.lead_gender]
                print(f"[ChatbotAgent] Filtered to {len(filtered_df)} films with {intent.lead_gender} leads")
            else:
                print(f"[ChatbotAgent] Warning: 'lead_gender' column not found in library. Run enrich_with_lead_gender.py first.")
        
        # Note: Gender filters for writers/directors would require additional data
        # (gender information not in current library structure)
        # This could be added via external API or database lookup
        
        return filtered_df
    
    def _filter_by_lead_gender(self, library_df: pd.DataFrame, target_gender: str) -> pd.DataFrame:
        """
        Filter library by lead actor gender using TMDb API lookups.
        
        Args:
            library_df: DataFrame to filter
            target_gender: "female" or "male"
        
        Returns:
            Filtered DataFrame with only films matching the lead gender
        """
        if not self.tmdb_client:
            print("[ChatbotAgent] Warning: TMDb client not available - cannot filter by lead gender")
            return library_df
        
        # TMDb gender codes: 0 = not specified, 1 = female, 2 = male
        target_gender_code = 1 if target_gender.lower() == "female" else 2
        
        matching_indices = []
        
        for idx, row in library_df.iterrows():
            tmdb_id = row.get("tmdb_id")
            if not tmdb_id or pd.isna(tmdb_id):
                continue
            
            try:
                # Get movie details with credits
                details = self.tmdb_client.get_movie_details_with_credits(int(tmdb_id))
                credits = details.get("credits", {})
                cast = credits.get("cast", [])
                
                if not cast:
                    continue
                
                # Get first cast member (the lead)
                lead_actor = cast[0]
                person_id = lead_actor.get("id")
                
                if not person_id:
                    continue
                
                # Get person details to check gender
                person_details = self.tmdb_client.get_person_details(person_id)
                person_gender = person_details.get("gender", 0)
                
                # Match if gender matches target
                if person_gender == target_gender_code:
                    matching_indices.append(idx)
                    
            except Exception as e:
                # Skip films where lookup fails
                print(f"[ChatbotAgent] Warning: Could not check lead gender for {row.get('title', 'unknown')}: {e}")
                continue
        
        if not matching_indices:
            print(f"[ChatbotAgent] No films found with {target_gender} leads after TMDb lookup")
            return library_df.iloc[0:0]  # Return empty DataFrame with same structure
        
        return library_df.loc[matching_indices]
    
    def _apply_exhibition_filters(self, exhibitions_df: pd.DataFrame, intent: QueryIntent) -> pd.DataFrame:
        """Apply filters to exhibition DataFrame based on query intent."""
        filtered_df = exhibitions_df.copy()
        
        # When user specifies ONE film to match against, restrict exhibitions to that film only.
        # All library titles will be matched against this single film (similarity target).
        if intent.match_to_specific_film:
            film_lower = intent.match_to_specific_film.lower().strip()
            title_mask = filtered_df["title"].astype(str).str.lower()
            filtered_df = filtered_df[title_mask.str.contains(film_lower, na=False)]
            if len(filtered_df) == 0:
                return filtered_df
        
        # Territory filter
        if intent.territory:
            filtered_df = filtered_df[
                filtered_df["country"].str.upper() == intent.territory.upper()
            ]
        
        # Time period filter: skip when matching to a specific film. We use all exhibitions
        # of that film; "this month" etc. is about when we promote, not when they screen.
        if intent.time_period and not intent.match_to_specific_film:
            today = datetime.now().date()
            if intent.time_period == "now" or intent.time_period == "week":
                # Filter to exhibitions within the next 7 days
                end_date = today + timedelta(days=7)
            elif intent.time_period == "month":
                # Filter to exhibitions within the next 30 days
                end_date = today + timedelta(days=30)
            else:
                end_date = today + timedelta(days=7)  # Default to week
            
            def _has_date_in_range(dates_str: str) -> bool:
                """Check if any scheduled date falls within the time period."""
                if pd.isna(dates_str) or not dates_str:
                    return False
                for date_str in str(dates_str).split(","):
                    date_str = date_str.strip()
                    try:
                        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                        if today <= date_obj <= end_date:
                            return True
                    except (ValueError, AttributeError):
                        continue
                return False
            
            filtered_df = filtered_df[filtered_df["scheduled_dates"].apply(_has_date_in_range)]
        
        return filtered_df
    
    def _generate_dynamic_reasoning(
        self, library_film: Dict, exhibition_film: Dict, similarity: float, intent: QueryIntent
    ) -> str:
        """Generate reasoning: (1) why recommended from similarity calc, (2) broader market trends."""
        if not self.openai_client:
            return f"Recommended based on similarity score: {similarity:.3f}"
        
        lib_text = self.matching_agent._film_to_text(library_film)
        ex_text = self.matching_agent._film_to_text(exhibition_film)
        
        emphasis_context = []
        if intent.director_weight > 0.5:
            emphasis_context.append("director matching")
        if intent.writer_weight > 0.5:
            emphasis_context.append("writer matching")
        if intent.cast_weight > 0.5:
            emphasis_context.append("cast matching")
        if intent.thematic_weight > 0.5:
            emphasis_context.append("thematic/genre alignment")
        if intent.stylistic_weight > 0.5:
            emphasis_context.append("stylistic similarity")
        emphasis_str = ", ".join(emphasis_context) if emphasis_context else "general similarity"
        
        prompt = (
            f"Library Film:\n{lib_text}\n\n"
            f"Exhibition Film (currently in theaters):\n{ex_text}\n\n"
            f"Similarity score: {similarity:.3f}. Query emphasis: {emphasis_str}.\n\n"
            "Write 2-3 sentences. Structure strictly:\n\n"
            "1. FIRST: Explain why this library film is being recommended based on the similarity calculation. "
            "Be specific: name the exhibition film, the score, and what drove the match (shared personnel, genres, themes, or style). "
            "This match explanation must come first.\n\n"
            "2. THEN: Briefly note how this film fits into broader market trends. "
            "Do not start with or lead with OTT platforms (AVOD, SVOD, FAST); that context is understood."
        )
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a content distribution assistant helping monetize a company's content library. Explain (1) why the film is recommended from the similarity match to the exhibition film, then (2) how it fits broader market trends. Do not preface every response with OTT platforms."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            reasoning = response.choices[0].message.content.strip()
            return reasoning
        except Exception as e:
            print(f"[ChatbotAgent] Error generating dynamic reasoning: {e}")
            return f"Recommended based on similarity score: {similarity:.3f}"
    
    def _intent_summary(self, intent: QueryIntent) -> str:
        """Brief readable summary of active filters from query intent."""
        parts = []
        if intent.genres:
            parts.append(f"genre(s) {', '.join(intent.genres)}")
        if intent.territory:
            parts.append(f"territory {intent.territory}")
        if intent.time_period:
            parts.append(f"time period '{intent.time_period}' (e.g. right now / this week / this month)")
        if intent.year_start or intent.year_end:
            y = [str(x) for x in [intent.year_start, intent.year_end] if x]
            parts.append(f"year range {'-'.join(y)}")
        if intent.lead_gender:
            parts.append(f"lead gender {intent.lead_gender}")
        if intent.specific_directors:
            parts.append(f"director(s) {', '.join(intent.specific_directors)}")
        if intent.specific_writers:
            parts.append(f"writer(s) {', '.join(intent.specific_writers)}")
        if intent.match_to_specific_film:
            parts.append(f"match to film '{intent.match_to_specific_film}'")
        return "; ".join(parts) if parts else "no filters"

    def _genre_sample_from_df(self, df: pd.DataFrame, column: str = "genres", top_n: int = 12) -> List[str]:
        """Unique genres from a DataFrame column (comma-separated), up to top_n."""
        if df is None or len(df) == 0 or column not in df.columns:
            return []
        raw = df[column].astype(str).str.split(",").explode().str.strip()
        raw = raw[raw.notna() & (raw != "") & (raw != "nan")]
        return raw.value_counts().head(top_n).index.tolist()

    def _generate_error_explanation(
        self,
        query: str,
        raw_error: str,
        *,
        error_type: str,
        intent_summary: Optional[str] = None,
        library_total: Optional[int] = None,
        library_after: Optional[int] = None,
        exhibition_total: Optional[int] = None,
        exhibition_after: Optional[int] = None,
        exhibition_genres: Optional[List[str]] = None,
        library_genres: Optional[List[str]] = None,
        exception: Optional[str] = None,
    ) -> str:
        """Generate a conversational, user-friendly explanation of why an error occurred."""
        context_parts = []
        if intent_summary:
            context_parts.append(f"Filters applied from the query: {intent_summary}.")
        if library_total is not None:
            context_parts.append(f"Library: {library_total} total titles, {library_after} matched filters.")
        if exhibition_total is not None:
            context_parts.append(f"Exhibitions: {exhibition_total} total, {exhibition_after} matched filters.")
        if library_genres:
            context_parts.append(f"Library includes genres such as: {', '.join(library_genres[:8])}.")
        if exhibition_genres:
            context_parts.append(f"Exhibition data includes genres such as: {', '.join(exhibition_genres[:8])}.")
        if exception:
            context_parts.append(f"Technical detail: {exception}")
        context_str = " ".join(context_parts) if context_parts else "No extra context."

        if self.openai_client:
            try:
                sys_msg = (
                    "You are a helpful content distribution assistant. When a user's query leads to an error "
                    "(e.g. no recommendations), explain in a brief, conversational way why that happened and what "
                    "they can try. Use the raw error and context provided. Be friendly and avoid jargon. "
                    "Do not mention OTT platforms unless directly relevant. Keep it to 2–4 short sentences."
                )
                user_msg = (
                    f"User asked: \"{query}\"\n\n"
                    f"Our system returned this error: {raw_error}\n\n"
                    f"Context: {context_str}\n\n"
                    "Write a short, user-friendly explanation of why this happened and what they could try instead."
                )
                resp = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=280,
                    temperature=0.5,
                )
                out = resp.choices[0].message.content.strip()
                return out if out else raw_error
            except Exception as e:
                print(f"[ChatbotAgent] Error generating error explanation: {e}")

        # Fallback: structured but still helpful
        if error_type == "no_library":
            fallback = (
                f"We couldn't find any library titles that match your filters. {context_str} "
                "Try relaxing filters (e.g. genre, year, or lead) or rephrasing your query."
            )
        elif error_type == "no_exhibitions":
            fallback = (
                f"We couldn't find any exhibitions that match your filters. {context_str} "
                "For example, \"right now\" or \"this week\" limits us to screenings in the next 7 days—if none of "
                "those match, you'll see zero. Try \"this month\" or dropping the time constraint."
            )
        else:
            fallback = (
                f"Something went wrong while running your query. {context_str} "
                "Try simplifying your question or adjusting filters; if it keeps happening, check that library "
                "and exhibition data are loaded."
            )
        return fallback

    def _extract_territory_from_query(self, query: str) -> Optional[str]:
        """Extract territory/country from user query using keyword matching first, then LLM if needed."""
        query_lower = query.lower()
        # First, try deterministic token/phrase matching (avoid substring matches like "fr" in "from")
        patterns = [
            (r"\b(united states|u\.s\.a\.|u\.s\.|usa)\b", "US"),
            (r"\b(in the|in)\s+us\b", "US"),
            (r"\b(united kingdom|u\.k\.|uk|great britain|britain|england|gb)\b", "UK"),
            (r"\b(france|french|fr)\b", "FR"),
            (r"\b(canada|canadian|ca)\b", "CA"),
            (r"\b(mexico|mexican|mx)\b", "MX"),
        ]
        for pattern, code in patterns:
            if re.search(pattern, query_lower, flags=re.IGNORECASE):
                return code
        
        # If keyword matching didn't work and we have OpenAI, try LLM
        if self.openai_client:
            prompt = f"""Extract the country/territory from this query. Return only the country code (US, UK, FR, CA, MX).

Query: {query}

Return only the 2-letter country code, or "None" if no country is mentioned."""

            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Extract country codes from queries. Return only the code."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=10,
                    temperature=0.1
                )
                territory = response.choices[0].message.content.strip().upper()
                
                # Handle "None" string response
                if territory == "NONE" or territory == "":
                    return None
                
                # Validate territory
                valid_territories = ["US", "UK", "FR", "CA", "MX"]
                if territory in valid_territories:
                    return territory
                
                # If LLM returned something invalid, fall back to None
                return None
            except Exception as e:
                print(f"Error extracting territory with LLM: {e}")
                # Fall through to return None
        
        return None
    
    def _generate_recommendation_reasoning(
        self, library_film: Dict, exhibition_film: Dict, similarity: float
    ) -> str:
        """Generate reasoning: (1) why recommended from similarity calc, (2) broader market trends."""
        if not self.openai_client:
            return f"Recommended based on similarity score: {similarity:.3f}"
        
        lib_text = self.matching_agent._film_to_text(library_film)
        ex_text = self.matching_agent._film_to_text(exhibition_film)
        
        prompt = (
            f"Library Film:\n{lib_text}\n\n"
            f"Exhibition Film (currently in theaters):\n{ex_text}\n\n"
            f"Similarity score: {similarity:.3f}.\n\n"
            "Write 2-3 sentences. Structure strictly:\n\n"
            "1. FIRST: Explain why this library film is being recommended based on the similarity calculation. "
            "Be specific: name the exhibition film, the score, and what drove the match (shared personnel, genres, themes, or style). "
            "This match explanation must come first.\n\n"
            "2. THEN: Briefly note how this film fits into broader market trends. "
            "Do not start with or lead with OTT platforms (AVOD, SVOD, FAST); that context is understood."
        )
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a content distribution assistant helping monetize a company's content library. Explain (1) why the film is recommended from the similarity match to the exhibition film, then (2) how it fits broader market trends. Do not preface every response with OTT platforms."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250,
                temperature=0.7
            )
            reasoning = response.choices[0].message.content.strip()
            return reasoning
        except Exception as e:
            print(f"[ChatbotAgent] Error generating reasoning: {e}")
            return f"Recommended based on similarity score: {similarity:.3f}"
    
    def _get_poster_url(self, tmdb_id: Optional[int]) -> Optional[str]:
        """Get poster image URL from TMDB."""
        if not tmdb_id or not self.tmdb_client:
            return None
        
        try:
            # Get movie details to retrieve poster_path
            details = self.tmdb_client.get_movie_details_with_credits(tmdb_id)
            poster_path = details.get("poster_path")
            
            if poster_path:
                # TMDB poster URL format: https://image.tmdb.org/t/p/w500/{poster_path}
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
            return None
        except Exception as e:
            print(f"[ChatbotAgent] Error fetching poster for TMDB ID {tmdb_id}: {e}")
            return None
    
    def analyze_current_trends(
        self,
        territory: Optional[str] = None,
        *,
        top_genres: int = 5,
        top_themes: int = 5
    ) -> Dict:
        """
        Analyze current trends from exhibition data.
        
        Args:
            territory: Optional country code to filter by (e.g., "US", "UK")
            top_genres: Number of top genres to return
            top_themes: Number of top themes to return
        
        Returns:
            Dictionary with trending genres, themes, and popular films
        """
        exhibitions_df = self._load_exhibitions()
        
        # Filter by territory if specified
        if territory:
            exhibitions_df = exhibitions_df[
                exhibitions_df["country"].str.upper() == territory.upper()
            ].copy()
        
        if len(exhibitions_df) == 0:
            return {
                "trending_genres": [],
                "trending_themes": [],
                "popular_films": [],
                "total_exhibitions": 0
            }
        
        # Analyze trending genres
        trending_genres = []
        if "genres" in exhibitions_df.columns:
            all_genres = exhibitions_df["genres"].astype(str).str.split(",").explode().str.strip()
            all_genres = all_genres[all_genres.notna() & (all_genres != "") & (all_genres != "nan")]
            
            if len(all_genres) > 0:
                genre_counts = all_genres.value_counts()
                total_genre_mentions = len(all_genres)
                
                for genre, count in genre_counts.head(top_genres).items():
                    pct = (count / total_genre_mentions) * 100
                    trending_genres.append({
                        "name": genre,
                        "count": int(count),
                        "percentage": round(pct, 1)
                    })
        
        # Analyze trending themes (from thematic_descriptors if available)
        trending_themes = []
        if "thematic_descriptors" in exhibitions_df.columns:
            # Extract common thematic keywords
            all_themes = exhibitions_df["thematic_descriptors"].astype(str).str.lower()
            # Simple keyword extraction (could be enhanced with NLP)
            theme_keywords = []
            common_themes = [
                "nostalgia", "horror", "thriller", "sci-fi", "drama", "comedy",
                "action", "romance", "coming of age", "family", "war", "crime",
                "fantasy", "supernatural", "psychological", "survival", "revenge"
            ]
            
            for theme in common_themes:
                count = all_themes.str.contains(theme, na=False).sum()
                if count > 0:
                    theme_keywords.append({
                        "theme": theme,
                        "count": int(count),
                        "percentage": round((count / len(exhibitions_df)) * 100, 1)
                    })
            
            trending_themes = sorted(theme_keywords, key=lambda x: x["count"], reverse=True)[:top_themes]
        
        # Get popular films (by venue count)
        popular_films = []
        if "title" in exhibitions_df.columns:
            film_stats = exhibitions_df.groupby("title").agg({
                "location": "nunique",
                "tmdb_id": "first",
                "release_year": "first",
                "genres": "first"
            }).reset_index()
            film_stats.columns = ["title", "venue_count", "tmdb_id", "release_year", "genres"]
            film_stats = film_stats.sort_values("venue_count", ascending=False)
            
            for _, row in film_stats.head(10).iterrows():
                popular_films.append({
                    "title": row["title"],
                    "venue_count": int(row["venue_count"]),
                    "tmdb_id": row.get("tmdb_id"),
                    "release_year": row.get("release_year"),
                    "genres": row.get("genres", "")
                })
        
        return {
            "trending_genres": trending_genres,
            "trending_themes": trending_themes,
            "popular_films": popular_films,
            "total_exhibitions": len(exhibitions_df),
            "territory": territory
        }
    
    def get_trend_based_recommendations(
        self,
        territory: Optional[str] = None,
        *,
        top_n: int = 5,
        min_similarity: float = 0.4
    ) -> List[Dict]:
        """
        Get library film recommendations based on current trends in theaters.
        
        This method:
        1. Analyzes current trends from exhibition data
        2. Matches library films to those trends
        3. Returns films that align with what's popular in theaters
        
        Args:
            territory: Optional country code to filter by
            top_n: Number of recommendations to return
            min_similarity: Minimum similarity threshold
        
        Returns:
            List of recommendation dictionaries
        """
        # Get current trends
        trends = self.analyze_current_trends(territory=territory)
        
        if trends["total_exhibitions"] == 0:
            return []
        
        # Load library data
        library_df = self._load_library()
        exhibitions_df = self._load_exhibitions()
        
        # Filter exhibitions by territory if specified
        if territory:
            exhibitions_df = exhibitions_df[
                exhibitions_df["country"].str.upper() == territory.upper()
            ].copy()
        
        if len(exhibitions_df) == 0:
            return []
        
        # Get trending genre names
        trending_genre_names = [g["name"].lower() for g in trends["trending_genres"]]
        
        # Score library films based on trend alignment
        lib_rows = library_df.to_dict(orient="records")
        scored_films = []
        
        for lib_row in lib_rows:
            score = 0.0
            match_reasons = []
            
            # Check genre alignment
            lib_genres = str(lib_row.get("genres", "")).lower()
            genre_matches = [g for g in trending_genre_names if g in lib_genres]
            if genre_matches:
                # Higher score for matching top genres
                for i, genre in enumerate(trends["trending_genres"]):
                    if genre["name"].lower() in genre_matches:
                        # Top genre gets higher weight
                        weight = (len(trends["trending_genres"]) - i) / len(trends["trending_genres"])
                        score += genre["percentage"] * weight * 0.01
                        match_reasons.append(f"Matches trending genre: {genre['name']} ({genre['percentage']}%)")
            
            # Use semantic similarity to popular films (optional - only if embeddings available)
            avg_similarity = 0.0
            lib_embeddings, ex_embeddings = self._load_embeddings(required=False)  # Optional for trend matching
            
            if lib_embeddings is not None and ex_embeddings is not None:
                lib_idx = lib_rows.index(lib_row)
                
                # Calculate similarity to popular exhibition films
                popular_tmdb_ids = [f.get("tmdb_id") for f in trends["popular_films"][:5] if f.get("tmdb_id")]
                if popular_tmdb_ids:
                    popular_exhibitions = exhibitions_df[
                        exhibitions_df["tmdb_id"].isin(popular_tmdb_ids)
                    ]
                    
                    if len(popular_exhibitions) > 0:
                        popular_indices = popular_exhibitions.index.tolist()
                        popular_ex_embeddings = ex_embeddings[popular_indices]
                        
                        # Calculate average similarity to popular films
                        lib_embedding = lib_embeddings[lib_idx:lib_idx+1]
                        lib_embedding_norm = lib_embedding / (np.linalg.norm(lib_embedding, axis=1, keepdims=True) + 1e-8)
                        popular_ex_norm = popular_ex_embeddings / (np.linalg.norm(popular_ex_embeddings, axis=1, keepdims=True) + 1e-8)
                        
                        similarities = np.dot(lib_embedding_norm, popular_ex_norm.T)[0]
                        avg_similarity = float(np.mean(similarities))
                        
                        if avg_similarity >= min_similarity:
                            score += avg_similarity * 0.5
                            match_reasons.append(f"Similar to popular films in theaters (avg similarity: {avg_similarity:.3f})")
            
            # Only include films with meaningful scores
            if score > 0.1 or match_reasons:
                scored_films.append({
                    "library_film": lib_row,
                    "trend_score": score,
                    "match_reasons": match_reasons,
                    "avg_similarity_to_trends": avg_similarity
                })
        
        # Sort by trend score and take top N
        scored_films.sort(key=lambda x: x["trend_score"], reverse=True)
        
        # Format recommendations
        recommendations = []
        for item in scored_films[:top_n]:
            lib = item["library_film"]
            
            # Generate trend-based reasoning
            reasoning = self._generate_trend_reasoning(
                lib, trends, item["match_reasons"]
            )
            
            # Get poster URL
            poster_url = self._get_poster_url(lib.get("tmdb_id"))
            
            recommendations.append({
                "title": lib.get("title", ""),
                "year": lib.get("release_year", ""),
                "director": lib.get("director", ""),
                "genres": lib.get("genres", ""),
                "trend_score": item["trend_score"],
                "match_reasons": item["match_reasons"],
                "territory": territory or "All",
                "themes": lib.get("thematic_descriptors", ""),
                "style": lib.get("stylistic_descriptors", ""),
                "reasoning": reasoning,
                "poster_url": poster_url,
                "tmdb_id": lib.get("tmdb_id"),
                "trending_genres": [g["name"] for g in trends["trending_genres"][:3]],
            })
        
        return recommendations
    
    def _generate_trend_reasoning(
        self, library_film: Dict, trends: Dict, match_reasons: List[str]
    ) -> str:
        """Generate reasoning: (1) why recommended from trend match, (2) broader market trends."""
        if not self.openai_client:
            return "; ".join(match_reasons) if match_reasons else "Matches current trends."
        
        trend_summary = "Exhibition data (currently in theaters):\n"
        if trends["trending_genres"]:
            trend_summary += f"- Top genres: {', '.join([g['name'] for g in trends['trending_genres'][:3]])}\n"
        if trends["popular_films"]:
            trend_summary += f"- Popular films: {', '.join([f['title'] for f in trends['popular_films'][:3]])}\n"
        
        lib_text = self.matching_agent._film_to_text(library_film)
        
        prompt = (
            f"Exhibition Data (what's in theaters now):\n{trend_summary}\n\n"
            f"Library Film:\n{lib_text}\n\n"
            f"Match reasons: {', '.join(match_reasons)}\n\n"
            "Write 2-3 sentences. Structure strictly:\n\n"
            "1. FIRST: Explain why this library film is being recommended based on how it matches the exhibition/trend data. "
            "Reference specific films in theaters or trending genres, and what drove the match (personnel, genres, themes, styles). "
            "This match explanation must come first.\n\n"
            "2. THEN: Briefly note how this film fits into broader market trends. "
            "Do not start with or lead with OTT platforms (AVOD, SVOD, FAST); that context is understood."
        )
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a content distribution assistant helping monetize a company's content library. Explain (1) why the film is recommended from its match to exhibition/trend data, then (2) how it fits broader market trends. Do not preface every response with OTT platforms."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250,
                temperature=0.7
            )
            reasoning = response.choices[0].message.content.strip()
            return reasoning
        except Exception as e:
            print(f"[ChatbotAgent] Error generating trend reasoning: {e}")
            return "; ".join(match_reasons) if match_reasons else "Matches current trends."
    
    def _is_trend_query(self, query: str) -> bool:
        """Detect if query is asking about trends vs specific recommendations."""
        if not self.openai_client:
            # Fallback: keyword-based detection
            query_lower = query.lower()
            trend_keywords = [
                "trend", "trending", "current", "popular", "what's showing",
                "what is popular", "what are the trends", "based on trends",
                "emphasize based on", "current trends", "running in theaters"
            ]
            return any(keyword in query_lower for keyword in trend_keywords)
        
        prompt = f"""Determine if this query is asking about CURRENT TRENDS vs SPECIFIC RECOMMENDATIONS.

Query: {query}

A TREND query asks about:
- What's popular/trending in theaters
- Films to emphasize based on current trends
- What genres/themes are trending
- Recommendations based on what's currently showing

A SPECIFIC RECOMMENDATION query asks about:
- Specific territory recommendations
- Films similar to specific exhibitions
- Direct matching between library and exhibitions

Respond with only "TREND" or "RECOMMENDATION"."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Classify queries as trend-based or recommendation-based."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )
            result = response.choices[0].message.content.strip().upper()
            return "TREND" in result
        except Exception as e:
            print(f"Error detecting query type: {e}")
            # Fallback to keyword detection
            query_lower = query.lower()
            trend_keywords = ["trend", "trending", "current", "popular", "based on trends"]
            return any(keyword in query_lower for keyword in trend_keywords)
    
    def get_recommendations_for_query(
        self,
        query: str,
        *,
        min_similarity: float = 0.5,
        max_similarity: float = 0.7,
        top_n: int = 5,
        history_prompts: Optional[List[str]] = None
    ) -> Dict:
        """
        Parse user query and get recommendations using dynamic intent parsing.
        
        Now supports:
        - Dynamic queries with intent parsing: "Show me films from the 80's that would be relevant this week"
        - Director/writer/cast-focused queries: "Show me five films directed by the most relevant directors"
        - Trend-based queries: "What films should we emphasize based on current trends?"
        - Specific film matching: "Do we have any library titles relevant to The Housemaid?"
        """
        # Use only dynamic matching (prompt-driven intent, filters, weights).
        # Do not fall back to the old static territory/similarity approach.
        try:
            result = self.get_dynamic_recommendations(query, top_n=top_n, history_prompts=history_prompts)
            if "error" not in result:
                return result
            # Dynamic returned an error (e.g. no matches for filters). Return it; no static fallback.
            return result
        except Exception as e:
            print(f"[ChatbotAgent] Dynamic matching failed: {e}")
            raw = f"Dynamic matching failed: {str(e)}"
            explained = self._generate_error_explanation(
                query,
                raw,
                error_type="dynamic_failed",
                exception=str(e),
            )
            return {
                "error": explained,
                "recommendations": [],
            }
    
    def format_recommendations_for_chat(self, recommendations: List[Dict]) -> str:
        """Format recommendations as a chat-friendly response."""
        if not recommendations:
            return "No recommendations found for the specified criteria."
        
        response = f"**Here are {len(recommendations)} library titles that match your query.**\n\n"
        
        for i, rec in enumerate(recommendations, 1):
            response += f"**{i}. {rec['title']}** ({rec['year']})\n"
            response += f"   - Director: {rec.get('director', 'N/A')}\n"
            if rec.get('writers'):
                response += f"   - Writers: {rec['writers']}\n"
            response += f"   - Genres: {rec.get('genres', 'N/A')}\n"
            
            if "trend_score" in rec:
                response += f"   - Trend Alignment Score: {rec['trend_score']:.3f}\n"
                if rec.get("trending_genres"):
                    response += f"   - Matches Trending Genres: {', '.join(rec['trending_genres'])}\n"
            elif "similarity" in rec:
                response += f"   - Relevance Score: {rec['similarity']:.3f}\n"
                if rec.get('matched_exhibition'):
                    response += f"   - Currently in Theaters: {rec['matched_exhibition']}"
                    if rec.get('exhibition_location'):
                        response += f" at {rec['exhibition_location']}"
                    response += "\n"
                if rec.get('exhibition_dates'):
                    response += f"   - Exhibition Dates: {rec['exhibition_dates']}\n"
            
            if rec.get('reasoning'):
                response += f"   - **Why this film:** {rec['reasoning']}\n"
            response += "\n"
        
        return response
