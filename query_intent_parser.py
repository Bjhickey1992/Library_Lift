"""
Query Intent Parser for Dynamic Film Matching
Extracts intent from user queries to adjust matching weights and filters.
Uses a capable LLM (gpt-4o by default) for accurate intent interpretation.
"""

import re
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from config import get_intent_model
except ImportError:
    def get_intent_model() -> str:
        return "gpt-4o"


@dataclass
class QueryIntent:
    """Structured representation of user query intent."""
    # Weight adjustments (0.0 to 1.0, where 1.0 = 100% weight)
    director_weight: float = 0.2  # Default 20%
    writer_weight: float = 0.15   # Default 15%
    cast_weight: float = 0.15     # Default 15%
    thematic_weight: float = 0.3  # Default 30%
    stylistic_weight: float = 0.2 # Default 20%
    
    # Filters
    year_start: Optional[int] = None
    year_end: Optional[int] = None
    decades: Optional[List[str]] = None  # e.g., ["80s", "90s"]
    genres: Optional[List[str]] = None
    writer_gender: Optional[str] = None  # "female", "male", "women", "men"
    director_gender: Optional[str] = None
    lead_gender: Optional[str] = None  # "female", "male" - for films with female/male leads
    specific_directors: Optional[List[str]] = None
    specific_writers: Optional[List[str]] = None
    specific_actors: Optional[List[str]] = None
    
    # Exhibition matching
    match_to_specific_film: Optional[str] = None  # e.g., "The Housemaid"
    match_to_current_market: bool = True  # Match to current exhibitions
    
    # Time context
    time_period: Optional[str] = None  # "this week", "this month", "now"
    
    # Territory
    territory: Optional[str] = None

    # Conversation context
    # True when the current prompt should start a new search (ignore prior prompts).
    # False when it should refine the ongoing search (carry forward prior constraints unless changed).
    is_new_search: bool = True
    # Optional diagnostics for debugging/tuning
    context_confidence: float = 0.5
    context_reason: Optional[str] = None


class QueryIntentParser:
    """Parse user queries to extract matching intent and filters."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_client = None
        if openai_api_key and OpenAI:
            self.openai_client = OpenAI(api_key=openai_api_key)
        elif OpenAI:
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
    
    def parse(
        self,
        query: str,
        *,
        history_prompts: Optional[List[str]] = None,
        previous_intent: Optional[QueryIntent] = None,
    ) -> QueryIntent:
        """
        Parse user query and extract intent.
        
        Uses a combination of keyword matching and LLM parsing for complex queries.
        """
        query_lower = query.lower()
        intent = QueryIntent()

        # Decide whether this prompt is a new search or a refinement of the prior prompts.
        # This decision is used to control what constraints are allowed to carry forward.
        is_new, conf, reason = self._classify_new_search(query, history_prompts=history_prompts)
        intent.is_new_search = is_new
        intent.context_confidence = conf
        intent.context_reason = reason
        
        # Extract territory (only from explicit mentions; refinements may carry it forward later)
        intent.territory = self._extract_territory(query_lower)
        
        # Extract time period
        intent.time_period = self._extract_time_period(query_lower)
        
        # Extract year/decade filters
        year_start, year_end, decades = self._extract_year_filters(query_lower)
        intent.year_start = year_start
        intent.year_end = year_end
        intent.decades = decades
        
        # Extract genre filters
        intent.genres = self._extract_genres(query_lower)
        
        # Extract gender filters for writers/directors/leads
        writer_gender, director_gender, lead_gender = self._extract_gender_filters(query_lower)
        intent.writer_gender = writer_gender
        intent.director_gender = director_gender
        intent.lead_gender = lead_gender
        
        # Extract specific names (directors, writers, actors)
        intent.specific_directors = self._extract_directors(query_lower)
        intent.specific_writers = self._extract_writers(query_lower)
        intent.specific_actors = self._extract_actors(query_lower)
        
        # Extract specific film to match against
        intent.match_to_specific_film = self._extract_specific_film(query_lower)
        
        # Adjust weights based on query focus
        self._adjust_weights_from_query(intent, query_lower)
        
        # Always use LLM for semantic intent extraction when available.
        # Filters are pulled dynamically from the meaning of the prompt, not just keywords.
        if self.openai_client:
            intent = self._llm_parse_query(query, intent, history_prompts=history_prompts)

        # Territory should only be applied if explicitly mentioned.
        # For refinements, allow territory mentioned in prior prompts to carry forward.
        combined_text = None
        if history_prompts:
            recent = history_prompts[-10:]
            combined_text = "\n".join([*recent, query]).lower()

        if intent.is_new_search:
            intent.territory = self._extract_territory(query_lower)
        else:
            intent.territory = self._extract_territory(combined_text or query_lower)

        # If refining and no explicit match target extracted this turn, allow prior target to carry forward.
        if (not intent.is_new_search) and (not intent.match_to_specific_film) and combined_text:
            intent.match_to_specific_film = self._extract_specific_film(combined_text)

        # Carry-forward for all extractable filters:
        # If we're refining and the LLM/current prompt didn't set a field, try to inherit it from prior prompts.
        # IMPORTANT: only fill missing fields; never override fields explicitly set in the current intent.
        if (not intent.is_new_search) and combined_text:
            # Years / decades
            if intent.year_start is None and intent.year_end is None and intent.decades is None:
                y0, y1, dec = self._extract_year_filters(combined_text)
                if intent.year_start is None:
                    intent.year_start = y0
                if intent.year_end is None:
                    intent.year_end = y1
                if intent.decades is None:
                    intent.decades = dec

            # Genres
            if not intent.genres:
                g = self._extract_genres(combined_text)
                if g:
                    intent.genres = g

            # Time period
            if not intent.time_period:
                tp = self._extract_time_period(combined_text)
                if tp:
                    intent.time_period = tp

            # Gender filters
            if not intent.writer_gender or not intent.director_gender or not intent.lead_gender:
                wg, dg, lg = self._extract_gender_filters(combined_text)
                if not intent.writer_gender and wg:
                    intent.writer_gender = wg
                if not intent.director_gender and dg:
                    intent.director_gender = dg
                if not intent.lead_gender and lg:
                    intent.lead_gender = lg

        # Carry-forward for all remaining fields using the previous parsed intent (most reliable).
        # Only fill fields that are missing/empty in the current intent; never override explicit values.
        if (not intent.is_new_search) and previous_intent:
            # Simple scalars / optionals
            if intent.year_start is None:
                intent.year_start = previous_intent.year_start
            if intent.year_end is None:
                intent.year_end = previous_intent.year_end
            if intent.decades is None:
                intent.decades = previous_intent.decades
            if not intent.genres:
                intent.genres = previous_intent.genres
            if not intent.writer_gender:
                intent.writer_gender = previous_intent.writer_gender
            if not intent.director_gender:
                intent.director_gender = previous_intent.director_gender
            if not intent.lead_gender:
                intent.lead_gender = previous_intent.lead_gender
            if not intent.time_period:
                intent.time_period = previous_intent.time_period
            if not intent.territory:
                intent.territory = previous_intent.territory
            if not intent.match_to_specific_film:
                intent.match_to_specific_film = previous_intent.match_to_specific_film

            # Lists of entities (names) — carry forward if not set this turn
            if not intent.specific_directors:
                intent.specific_directors = previous_intent.specific_directors
            if not intent.specific_writers:
                intent.specific_writers = previous_intent.specific_writers
            if not intent.specific_actors:
                intent.specific_actors = previous_intent.specific_actors

            # Carry forward match_to_current_market flag
            intent.match_to_current_market = previous_intent.match_to_current_market

            # Weights — keep prior weights unless the new prompt explicitly changes focus
            focus_cues = [
                "director", "directed", "written", "writer", "cast", "starring",
                "genre", "genres", "theme", "themes", "style", "stylistic",
                "cinematography", "production design",
            ]
            if not any(cue in query_lower for cue in focus_cues):
                intent.director_weight = previous_intent.director_weight
                intent.writer_weight = previous_intent.writer_weight
                intent.cast_weight = previous_intent.cast_weight
                intent.thematic_weight = previous_intent.thematic_weight
                intent.stylistic_weight = previous_intent.stylistic_weight
        
        return intent

    def _classify_new_search(
        self, query: str, *, history_prompts: Optional[List[str]] = None
    ) -> Tuple[bool, float, str]:
        """
        Classify whether the current query is a new search or a refinement of the prior prompts.
        Uses deterministic cues first, then falls back to a lightweight LLM classifier.
        Returns (is_new_search, confidence_0_to_1, short_reason).
        """
        q = (query or "").strip()
        ql = q.lower()

        if not history_prompts:
            return True, 1.0, "no prior prompts"

        # Explicit reset cues → new search
        reset_phrases = [
            "new search",
            "start over",
            "reset",
            "forget that",
            "ignore that",
            "different question",
            "switch gears",
            "change topics",
        ]
        if any(p in ql for p in reset_phrases):
            return True, 0.95, "explicit reset cue"

        # Explicit refinement cues → refinement
        refine_phrases = [
            "now",
            "also",
            "additionally",
            "instead",
            "actually",
            "only",
            "narrow",
            "broaden",
            "filter",
            "make it",
            "keep",
            "same",
            "still",
            "refine",
            "change to",
        ]
        if any(p in ql for p in refine_phrases):
            # Still allow LLM to override if it strongly believes it’s a new topic,
            # but bias toward refinement.
            heuristic_refine = True
        else:
            heuristic_refine = False

        # If we don't have an LLM, rely on heuristic only.
        if not self.openai_client:
            return (False, 0.6, "heuristic refinement cue") if heuristic_refine else (True, 0.55, "no LLM; default new search")

        # Lightweight LLM classifier
        try:
            recent = history_prompts[-10:]
            hist_lines = "\n".join([f"- {p}" for p in recent if p])
            model = get_intent_model()

            sys_msg = (
                "You decide whether the user's latest message is a NEW SEARCH or a REFINEMENT of the ongoing search. "
                "Use the conversation history. Return JSON only."
            )
            user_msg = f"""Conversation history (up to last 10 user prompts, oldest → newest):
{hist_lines}

Current user prompt:
{q}

Decide:
- is_new_search: true if the user is starting a different, unrelated request (new topic/goal)
- is_new_search: false if the user is continuing/refining the same search (adding/removing filters, changing years/genres, etc.)

Return JSON with:
{{"is_new_search": true|false, "confidence": 0.0-1.0, "reason": "short"}}
"""

            resp = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=120,
            )
            import json

            data = json.loads(resp.choices[0].message.content)
            is_new = bool(data.get("is_new_search", False))
            conf = float(data.get("confidence", 0.7))
            reason = str(data.get("reason", "LLM classified"))

            # If heuristics strongly indicate refinement but LLM is uncertain, keep refinement.
            if heuristic_refine and is_new and conf < 0.75:
                return False, 0.7, f"heuristic refinement overrode low-confidence LLM: {reason}"

            return is_new, max(0.0, min(1.0, conf)), reason
        except Exception as e:
            # Fallback to heuristic
            return (False, 0.6, f"heuristic refinement; LLM classifier failed: {e}") if heuristic_refine else (True, 0.55, f"LLM classifier failed: {e}")
    
    def _extract_territory(self, query_lower: str) -> Optional[str]:
        """Extract territory/country from query."""
        # IMPORTANT: avoid substring matches like "fr" in "from".
        # Only treat territories as present when they appear as standalone tokens/phrases.
        patterns = [
            # US (avoid bare "us" because it's also a pronoun; require context or standard abbreviations)
            (r"\b(united states|u\.s\.a\.|u\.s\.|usa)\b", "US"),
            (r"\b(in the|in)\s+us\b", "US"),
            # UK
            (r"\b(united kingdom|u\.k\.|uk|great britain|britain|england|gb)\b", "UK"),
            # FR
            (r"\b(france|french|fr)\b", "FR"),
            # CA
            (r"\b(canada|canadian|ca)\b", "CA"),
            # MX
            (r"\b(mexico|mexican|mx)\b", "MX"),
        ]
        for pattern, code in patterns:
            if re.search(pattern, query_lower, flags=re.IGNORECASE):
                return code
        return None
    
    def _extract_time_period(self, query_lower: str) -> Optional[str]:
        """Extract time period context."""
        if any(phrase in query_lower for phrase in ["this week", "this month", "right now", "currently", "now"]):
            if "week" in query_lower:
                return "week"
            elif "month" in query_lower:
                return "month"
            else:
                return "now"
        return None
    
    def _extract_year_filters(self, query_lower: str) -> Tuple[Optional[int], Optional[int], Optional[List[str]]]:
        """Extract year range and decade filters."""
        year_start = None
        year_end = None
        decades = []
        
        # Extract decades (80s, 90s, etc.)
        decade_pattern = r'\b(\d{2})s\b'
        decade_matches = re.findall(decade_pattern, query_lower)
        for match in decade_matches:
            decade_num = int(match)
            if 0 <= decade_num <= 99:
                decades.append(match)
                # Convert to year range
                start = 1900 + (decade_num * 10) if decade_num >= 50 else 2000 + (decade_num * 10)
                if year_start is None or start < year_start:
                    year_start = start
                end = start + 9
                if year_end is None or end > year_end:
                    year_end = end
        
        # Extract specific years
        year_pattern = r'\b(19|20)\d{2}\b'
        year_matches = re.findall(year_pattern, query_lower)
        years = [int(match[0] + match[1]) for match in re.finditer(year_pattern, query_lower)]
        
        if years:
            if year_start is None:
                year_start = min(years)
            else:
                year_start = min(year_start, min(years))
            
            if year_end is None:
                year_end = max(years)
            else:
                year_end = max(year_end, max(years))
        
        # Extract year ranges
        range_pattern = r'(\d{4})\s*[-–]\s*(\d{4})'
        range_match = re.search(range_pattern, query_lower)
        if range_match:
            year_start = int(range_match.group(1))
            year_end = int(range_match.group(2))
        
        return year_start, year_end, decades if decades else None
    
    def _extract_genres(self, query_lower: str) -> Optional[List[str]]:
        """Extract genre mentions from query."""
        common_genres = [
            "horror", "thriller", "comedy", "drama", "action", "romance",
            "sci-fi", "science fiction", "fantasy", "crime", "mystery",
            "western", "war", "documentary", "animation", "musical"
        ]
        
        found_genres = []
        for genre in common_genres:
            if genre in query_lower:
                found_genres.append(genre)
        
        return found_genres if found_genres else None
    
    def _extract_gender_filters(self, query_lower: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract gender filters for writers/directors/leads."""
        writer_gender = None
        director_gender = None
        lead_gender = None
        
        # Check for writer gender
        if any(phrase in query_lower for phrase in ["written by women", "women writers", "female writers", "women screenwriters"]):
            writer_gender = "female"
        elif any(phrase in query_lower for phrase in ["written by men", "male writers"]):
            writer_gender = "male"
        
        # Check for director gender
        if any(phrase in query_lower for phrase in ["directed by women", "women directors", "female directors"]):
            director_gender = "female"
        elif any(phrase in query_lower for phrase in ["directed by men", "male directors"]):
            director_gender = "male"
        
        # Check for lead actor gender
        if any(phrase in query_lower for phrase in [
            "female leads", "female lead", "women leads", "women lead",
            "films with female leads", "movies with female leads",
            "female protagonist", "women protagonist", "female-driven",
            "female starring", "women starring"
        ]):
            lead_gender = "female"
        elif any(phrase in query_lower for phrase in [
            "male leads", "male lead", "men leads", "men lead",
            "films with male leads", "movies with male leads",
            "male protagonist", "men protagonist", "male-driven"
        ]):
            lead_gender = "male"
        
        return writer_gender, director_gender, lead_gender
    
    def _extract_directors(self, query_lower: str) -> Optional[List[str]]:
        """Extract specific director names from query."""
        # This is a simplified version - LLM parsing will be more accurate for complex queries
        # For now, return None and let LLM handle it
        return None
    
    def _extract_writers(self, query_lower: str) -> Optional[List[str]]:
        """Extract specific writer names from query."""
        # Simplified - LLM will handle complex extraction
        return None
    
    def _extract_actors(self, query_lower: str) -> Optional[List[str]]:
        """Extract specific actor names from query."""
        # Simplified - LLM will handle complex extraction
        return None
    
    def _extract_specific_film(self, query_lower: str) -> Optional[str]:
        """Extract specific film title mentioned as the similarity reference in query."""
        def clean(s: str) -> str:
            s = s.strip()
            s = re.sub(r'\s+(in theaters|now|currently|,).*$', '', s, flags=re.IGNORECASE)
            return s.strip()

        # Reference-film patterns first (e.g. "The Housemaid is a hit in theaters, do we have...")
        reference_patterns = [
            r'^(.+?)\s+is\s+(?:a\s+)?hit(?:\s+in\s+theaters)?',
            r'^(.+?)\s+is\s+in\s+theaters',
            r'^(.+?)\s+is\s+playing(?:\s+now)?(?:\s+in\s+theaters)?',
            r'^(.+?)\s+is\s+trending(?:\s+in\s+theaters)?',
            r'^(.+?)\s+has\s+been\s+(?:a\s+)?hit',
            r'^(.+?)\s+is\s+(?:big|hot)\s+in\s+theaters',
        ]
        for pattern in reference_patterns:
            m = re.search(pattern, query_lower, re.IGNORECASE)
            if m:
                title = clean(m.group(1))
                if len(title) > 1 and title not in ("the", "a", "an"):
                    return title

        # "Due to / because of / given the success of X" etc. Capture film title until , . or end.
        success_patterns = [
            r'due to the success of\s+([^,\.]+)',
            r'because of the success of\s+([^,\.]+)',
            r'given the success of\s+([^,\.]+)',
            r'promote due to\s+(?:the success of\s+)?([^,\.]+)',
            r'promote because of\s+(?:the success of\s+)?([^,\.]+)',
            r'(?:good to )?promote\s+(?:due to|because of)\s+(?:the success of\s+)?([^,\.]+)',
        ]
        for pattern in success_patterns:
            m = re.search(pattern, query_lower, re.IGNORECASE)
            if m:
                title = clean(m.group(1))
                if len(title) > 1 and title not in ("the", "a", "an"):
                    return title

        # Explicit match patterns: "relevant to X", "similar to X", "like X"
        match_patterns = [
            r'relevant to\s+([^,\.]+)',
            r'similar to\s+([^,\.]+)',
            r'like\s+([^,\.]+)',
            r'match\s+([^,\.]+)',
            r'matched?\s+against\s+([^,\.]+)',
            r'highlight\s+(?:to\s+)?(?:at[- ]?home\s+viewers\s+)?(?:similar\s+to\s+)?([^,\.?]+?)(?:\s+this\s+month|\s*\?|$)',
        ]
        for pattern in match_patterns:
            m = re.search(pattern, query_lower, re.IGNORECASE)
            if m:
                title = clean(m.group(1))
                if len(title) > 1 and title not in ("the", "a", "an"):
                    return title

        return None
    
    def _adjust_weights_from_query(self, intent: QueryIntent, query_lower: str):
        """Adjust similarity weights based on query focus."""
        # Director-focused queries
        if any(phrase in query_lower for phrase in ["directed by", "director", "directors", "most relevant directors"]):
            intent.director_weight = 0.9
            intent.thematic_weight = 0.05
            intent.stylistic_weight = 0.05
            intent.writer_weight = 0.0
            intent.cast_weight = 0.0
        
        # Writer-focused queries
        elif any(phrase in query_lower for phrase in ["written by", "writer", "writers", "screenwriter"]):
            intent.writer_weight = 0.9
            intent.thematic_weight = 0.05
            intent.stylistic_weight = 0.05
            intent.director_weight = 0.0
            intent.cast_weight = 0.0
        
        # Cast-focused queries
        elif any(phrase in query_lower for phrase in ["starring", "actor", "actors", "cast", "featuring"]):
            intent.cast_weight = 0.9
            intent.thematic_weight = 0.05
            intent.stylistic_weight = 0.05
            intent.director_weight = 0.0
            intent.writer_weight = 0.0
        
        # Genre-focused queries
        elif intent.genres:
            intent.thematic_weight = 0.6
            intent.stylistic_weight = 0.3
            intent.director_weight = 0.05
            intent.writer_weight = 0.05
            intent.cast_weight = 0.0
    
    def _needs_llm_parsing(self, query_lower: str) -> bool:
        """Determine if query needs LLM parsing for complex intent."""
        complex_indicators = [
            "most relevant",
            "would be relevant",
            "that would",
            "should we",
            "do we have",
            "are there",
            "are relevant",
            "relevant today",
            "relevant this",
            "is a hit",
            "in theaters",
            "highlight",
            "utilize",
        ]
        return any(indicator in query_lower for indicator in complex_indicators)
    
    def _llm_parse_query(
        self,
        query: str,
        base_intent: QueryIntent,
        *,
        history_prompts: Optional[List[str]] = None,
    ) -> QueryIntent:
        """Use LLM to extract intent from semantic meaning of the prompt."""
        if not self.openai_client:
            return base_intent
        
        model = get_intent_model()
        sys_msg = """You extract structured intent from film-library recommendation queries in a multi-turn conversation.
Pull ALL filters dynamically from the SEMANTIC MEANING of the user's prompts.

CRITICAL RULES:
0. Determine whether the user is starting a NEW search or REFINING the current one.
   - Set is_new_search=true if the user is starting over or changing to a clearly different request.
   - Set is_new_search=false if the prompt is a follow-up/refinement (e.g. "only", "instead", "make it", "change to", "between", "narrow to", "broaden to", "actually", "now show me...").
   - If refining, carry forward prior constraints unless explicitly changed.

1. match_to_specific_film: The ONE film we match library titles AGAINST (similarity target). Set whenever the user refers to a specific film as the reference or success story. Examples:
   - "due to the success of The Housemaid" -> "The Housemaid"
   - "because of The Housemaid", "given The Housemaid", "like The Housemaid", "X is a hit in theaters" -> that film. Extract the EXACT title. Use null ONLY if no specific film is mentioned.

2. year_start, year_end: Extract from meaning. Examples:
   - "between 1970 - 1999", "1970-1999", "from 1970 to 1999" -> year_start: 1970, year_end: 1999
   - "80s films", "from the 80s" -> year_start: 1980, year_end: 1989
   - "older titles", "older films" -> year_end: 2000 or 2010 (infer); year_start: null unless stated
   - "recent" -> year_start: 2020 or similar

3. genres: Infer from context. "thriller and horror", "comedies", "sci-fi" -> list those genres.

4. director_weight, writer_weight, cast_weight, thematic_weight, stylistic_weight: 0.0-1.0. Default ~0.2-0.3 each, thematic ~0.3. Bump the one the query emphasizes (e.g. director-focused -> director_weight 0.9).

5. territory: Only if user explicitly mentions US, UK, FR, CA, MX.

6. time_period: "month"|"week"|"now" only if user says "this month", "right now", "this week".

Return ONLY valid JSON. No markdown, no explanation."""

        history_block = ""
        if history_prompts:
            recent = history_prompts[-10:]
            lines = "\n".join([f"- {p}" for p in recent if p])
            history_block = f"\nConversation history (up to last 10 user prompts, oldest → newest):\n{lines}\n"

        prompt = f"""Extract intent from this film-library query. Consider the conversation history if present.
Return JSON with:
- is_new_search: true|false
- director_weight, writer_weight, cast_weight, thematic_weight, stylistic_weight (0.0-1.0)
- year_start, year_end (int or null), decades (list or null)
- genres (list or null)
- writer_gender, director_gender, lead_gender (or null)
- specific_directors, specific_writers, specific_actors (list or null)
- match_to_specific_film (exact film title or null)
- territory (US/UK/FR/CA/MX or null; only if explicitly mentioned)
- time_period ("month"|"week"|"now" or null)

{history_block}
Query: {query}

Return ONLY valid JSON."""

        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=400,
            )
            import json
            data = json.loads(response.choices[0].message.content)
            
            if "director_weight" in data:
                base_intent.director_weight = float(data["director_weight"])
            if "writer_weight" in data:
                base_intent.writer_weight = float(data["writer_weight"])
            if "cast_weight" in data:
                base_intent.cast_weight = float(data["cast_weight"])
            if "thematic_weight" in data:
                base_intent.thematic_weight = float(data["thematic_weight"])
            if "stylistic_weight" in data:
                base_intent.stylistic_weight = float(data["stylistic_weight"])
            if "year_start" in data and data["year_start"] is not None:
                base_intent.year_start = int(data["year_start"])
            if "year_end" in data and data["year_end"] is not None:
                base_intent.year_end = int(data["year_end"])
            if "decades" in data and data["decades"]:
                base_intent.decades = data["decades"]
            if "genres" in data and data["genres"]:
                base_intent.genres = data["genres"]
            if "writer_gender" in data and data["writer_gender"]:
                base_intent.writer_gender = data["writer_gender"]
            if "director_gender" in data and data["director_gender"]:
                base_intent.director_gender = data["director_gender"]
            if "lead_gender" in data and data["lead_gender"]:
                base_intent.lead_gender = data["lead_gender"]
            if "specific_directors" in data and data["specific_directors"]:
                base_intent.specific_directors = data["specific_directors"]
            if "specific_writers" in data and data["specific_writers"]:
                base_intent.specific_writers = data["specific_writers"]
            if "specific_actors" in data and data["specific_actors"]:
                base_intent.specific_actors = data["specific_actors"]
            if "time_period" in data and data["time_period"]:
                base_intent.time_period = data["time_period"]
            # Only overwrite if LLM returns a non-empty film title; preserve regex extraction otherwise
            llm_film = data.get("match_to_specific_film")
            if isinstance(llm_film, str) and llm_film.strip():
                base_intent.match_to_specific_film = llm_film.strip()
            # else: keep base_intent.match_to_specific_film (e.g. from regex)
            # Territory should only be applied when explicitly mentioned in the prompt.
            # We rely on deterministic token/phrase extraction in _extract_territory to avoid false positives.
            # (Do not let the LLM "infer" a territory.)
            
        except Exception as e:
            print(f"[QueryIntentParser] Error in LLM parsing: {e}")
        
        return base_intent
