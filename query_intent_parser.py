"""
Query Intent Parser for Dynamic Film Matching
Extracts intent from user queries to adjust matching weights and filters.
Uses a capable LLM (gpt-4o-mini by default for latency; override via config) for intent interpretation.
"""

import re
from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, date

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from config import get_intent_model
except ImportError:
    def get_intent_model() -> str:
        return "gpt-4o-mini"


@dataclass
class QueryIntent:
    """Structured representation of user query intent."""
    # Weight adjustments (0.0 to 1.0, where 1.0 = 100% weight)
    director_weight: float = 0.2  # Default 20%
    writer_weight: float = 0.15   # Default 15%
    cast_weight: float = 0.15     # Default 15%
    thematic_weight: float = 0.25  # Default 25% (reduced to allow higher need weight)
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
    # Exhibition date filter: only exhibitions playing on this date or within this range
    exhibition_date_start: Optional[date] = None
    exhibition_date_end: Optional[date] = None

    # Territory
    territory: Optional[str] = None
    # City (when user asks for a specific city, only exhibitions in that city)
    city: Optional[str] = None
    # Venue (when user asks for a specific venue, e.g. "Film Forum", filter exhibitions by location containing this name)
    venue: Optional[str] = None

    # Film descriptor terms: when user describes elements/vibes (e.g. "bridge and lovers", "late-night European art house")
    # without naming a film, match library on these terms in tags, keywords, overview, theme, style, emotional_tone, need.
    film_descriptor_terms: Optional[List[str]] = None

    # Dynamic column filters: any library/exhibition column -> filter value (exact, list-any, or {min, max} for numeric)
    column_filters: Optional[Dict[str, Any]] = None
    # Dynamic column weights for match boosting: column name -> weight 0.0-1.0 (used in similarity calculation)
    column_weights: Optional[Dict[str, float]] = None

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
        # Extract city (when user asks for exhibitions in a specific city)
        intent.city = self._extract_city(query_lower)
        # Extract venue (e.g. "Film Forum", "whatever is doing well at Film Forum")
        intent.venue = self._extract_venue(query_lower)

        # Extract time period
        intent.time_period = self._extract_time_period(query_lower)
        # Exhibition date or date range (e.g. "playing on 2/14/2026", "between 2/1 and 2/14/2026")
        ex_start, ex_end = self._extract_exhibition_date(query_lower)
        intent.exhibition_date_start = ex_start
        intent.exhibition_date_end = ex_end

        # Extract year/decade filters
        year_start, year_end, decades = self._extract_year_filters(query_lower)
        intent.year_start = year_start
        intent.year_end = year_end
        intent.decades = decades
        
        # Extract genre filters
        intent.genres = self._extract_genres(query_lower)
        
        # Extract gender filters for writers/directors/leads (normalize to lowercase for consistent filtering)
        writer_gender, director_gender, lead_gender = self._extract_gender_filters(query_lower)
        intent.writer_gender = (writer_gender or "").strip().lower() or None
        intent.director_gender = (director_gender or "").strip().lower() or None
        intent.lead_gender = (lead_gender or "").strip().lower() or None
        
        # Extract specific names (directors, writers, actors)
        intent.specific_directors = self._extract_directors(query_lower)
        intent.specific_writers = self._extract_writers(query_lower)
        intent.specific_actors = self._extract_actors(query_lower)
        
        # Extract specific film to match against (normalize for partial/catalog matching, e.g. "that zombie movie Bone Temple" -> "Bone Temple")
        raw_film = self._extract_specific_film(query_lower)
        intent.match_to_specific_film = self._normalize_film_title_for_matching(raw_film) if raw_film else None
        
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
            intent.city = self._extract_city(query_lower)
        else:
            intent.territory = self._extract_territory(combined_text or query_lower)
            intent.city = self._extract_city(combined_text or query_lower)

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
            # Exhibition date
            if intent.exhibition_date_start is None and intent.exhibition_date_end is None:
                es, ee = self._extract_exhibition_date(combined_text)
                if es is not None:
                    intent.exhibition_date_start = es
                if ee is not None:
                    intent.exhibition_date_end = ee

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
            if intent.exhibition_date_start is None:
                intent.exhibition_date_start = previous_intent.exhibition_date_start
            if intent.exhibition_date_end is None:
                intent.exhibition_date_end = previous_intent.exhibition_date_end
            if not intent.territory:
                intent.territory = previous_intent.territory
            if not intent.city:
                intent.city = previous_intent.city
            if not intent.venue:
                intent.venue = previous_intent.venue
            if not intent.match_to_specific_film:
                intent.match_to_specific_film = previous_intent.match_to_specific_film
            if not intent.film_descriptor_terms:
                intent.film_descriptor_terms = previous_intent.film_descriptor_terms

            # Lists of entities (names) — carry forward if not set this turn
            if not intent.specific_directors:
                intent.specific_directors = previous_intent.specific_directors
            if not intent.specific_writers:
                intent.specific_writers = previous_intent.specific_writers
            if not intent.specific_actors:
                intent.specific_actors = previous_intent.specific_actors

            # Carry forward match_to_current_market flag
            intent.match_to_current_market = previous_intent.match_to_current_market

            # Dynamic column filters and weights — carry forward if not set this turn
            if not intent.column_filters and previous_intent.column_filters:
                intent.column_filters = dict(previous_intent.column_filters)
            if not intent.column_weights and previous_intent.column_weights:
                intent.column_weights = dict(previous_intent.column_weights)

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

        # Normalize: never treat pronouns or time phrases as film title or venue
        _pronoun_or_time = ("it", "that", "this", "this month", "that month", "this week", "that week")
        if intent.match_to_specific_film and intent.match_to_specific_film.strip().lower() in _pronoun_or_time:
            intent.match_to_specific_film = None
        if intent.venue and intent.venue.strip().lower() in _pronoun_or_time:
            # Restore venue from regex (e.g. "Film Forum" from "at Film Forum this month")
            intent.venue = self._extract_venue(query_lower)

        # Clamp year range to sane values (avoid LLM/carry-forward producing 2800, etc.)
        max_year = datetime.now().year + 2
        if intent.year_start is not None:
            intent.year_start = max(1900, min(intent.year_start, max_year))
        if intent.year_end is not None:
            intent.year_end = max(1900, min(intent.year_end, max_year))
        # Do not use exhibition-date year as library release-year filter (e.g. "playing on 2/14/2026" -> clear 2026 from year_start/year_end)
        if intent.exhibition_date_start or intent.exhibition_date_end:
            ex_years = set()
            if intent.exhibition_date_start:
                ex_years.add(intent.exhibition_date_start.year)
            if intent.exhibition_date_end:
                ex_years.add(intent.exhibition_date_end.year)
            ex_min = min(ex_years) if ex_years else None
            if intent.year_start is not None and intent.year_start in ex_years:
                intent.year_start = None
            if intent.year_end is not None and intent.year_end in ex_years:
                intent.year_end = None
            # If LLM inferred a future range (e.g. 2028) when the only date in the query was exhibition date, clear it
            if ex_min is not None and intent.year_start is not None and intent.year_end is not None:
                if intent.year_start >= ex_min and intent.year_end >= ex_min:
                    intent.year_start = None
                    intent.year_end = None
        
        return intent

    def _classify_new_search(
        self, query: str, *, history_prompts: Optional[List[str]] = None
    ) -> Tuple[bool, float, str]:
        """
        Classify whether the current query is a new search or a refinement of the prior prompts.
        Uses deterministic cues first, then falls back to a lightweight LLM classifier.
        Returns (is_new_search, confidence_0_to_1, short_reason).

        "Right now" handling:
        - At START of message ("right now, only female leads") = refinement lead-in; carry forward prior filters.
        - At END ("show me sci-fi we can emphasize right now") = time context; treat as new search unless other refinement cues.
        """
        q = (query or "").strip()
        ql = q.lower()

        if not history_prompts:
            return True, 1.0, "no prior prompts"

        # Explicit reset cues → new search (user wants to start over; do not carry forward prior filters)
        reset_phrases = [
            "new search",
            "new query",
            "this is a new query",
            "this is a new search",
            "start over",
            "start fresh",
            "reset",
            "forget that",
            "ignore that",
            "different question",
            "switch gears",
            "change topics",
        ]
        if any(p in ql for p in reset_phrases):
            return True, 0.95, "explicit reset cue"

        # Explicit refinement cues → refinement (including adding lead/director/writer gender)
        refine_phrases = [
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
            "female leads",
            "male leads",
            "women leads",
            "men leads",
            "female lead",
            "male lead",
            "with female",
            "with male",
            "with women",
            "with men",
        ]
        # "now" = refinement only when not part of "right now" (e.g. "now show me X" vs "emphasize right now")
        now_is_refinement = "now" in ql and "right now" not in ql
        # "right now" at START = lead-in to refinement; at END = time context (new search)
        right_now_lead_in = ql.strip().startswith("right now")
        if any(p in ql for p in refine_phrases) or now_is_refinement or right_now_lead_in:
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
- is_new_search: true if the user is starting a different, unrelated request (new topic/goal). Example: "show me existential sci-fi we can emphasize right now" = new search (time context at end).
- is_new_search: false if the user is continuing/refining the same search (adding/removing filters, changing years/genres, etc.). Example: "right now, only female leads" or "right now I want thrillers" = refinement (lead-in at start).

"Right now" at the START of the message usually means refining/narrowing the current search. "Right now" at the END (e.g. "emphasize right now") usually means when to promote, i.e. a new request about current timing.

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

            # If heuristics strongly indicate refinement (e.g. "female leads", "only"), treat as refinement so prior filters (genres, etc.) carry forward.
            if heuristic_refine and is_new:
                return False, 0.75, f"refinement cue overrode LLM: {reason}"

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

    def _extract_city(self, query_lower: str) -> Optional[str]:
        """Extract city from query when user asks for exhibitions in a specific city.
        Returns canonical city name for matching against exhibition location strings (e.g. 'Venue (Portland, OR)').
        """
        # Patterns: (regex, canonical name as it appears in exhibition location)
        city_patterns = [
            (r"\bportland\b", "Portland"),
            (r"\b(la\b|los angeles)\b", "Los Angeles"),
            (r"\b(nyc|new york)\b", "New York"),
            (r"\bseattle\b", "Seattle"),
            (r"\b(sf|san francisco)\b", "San Francisco"),
            (r"\batlanta\b", "Atlanta"),
            (r"\bkansas city\b", "Kansas City"),
            (r"\bchicago\b", "Chicago"),
            (r"\baustin\b", "Austin"),
            (r"\btoronto\b", "Toronto"),
            (r"\bvancouver\b", "Vancouver"),
            (r"\blondon\b", "London"),
            (r"\bparis\b", "Paris"),
            (r"\bberlin\b", "Berlin"),
        ]
        for pattern, canonical in city_patterns:
            if re.search(pattern, query_lower):
                return canonical
        return None

    def _extract_venue(self, query_lower: str) -> Optional[str]:
        """Extract venue name when user asks for exhibitions at a specific venue (e.g. 'Film Forum', 'at the Ritz').
        Returns the venue name as it might appear in exhibition location strings."""
        # "at Film Forum", "at the Film Forum", "doing well at Film Forum", "whatever is doing well at X this month"
        at_venue = re.search(r"(?:doing\s+well\s+)?at\s+(?:the\s+)?([a-z0-9\s]+?)(?:\s+this\s+month|\s+this\s+week|,|\?|$)", query_lower)
        if at_venue:
            name = at_venue.group(1).strip()
            # Drop trailing filler
            name = re.sub(r"\s+(month|week|year|now)$", "", name, flags=re.IGNORECASE).strip()
            if len(name) >= 2 and name not in ("the", "a", "us", "uk"):
                return name.title()
        # Known venues (case-insensitive match)
        known = ["film forum", "riff", "metrograph", "bam", "alamo", "landmark", "angelika"]
        for v in known:
            if v in query_lower:
                return v.title() if v != "bam" else "BAM"
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

    def _extract_exhibition_date(self, query_lower: str) -> Tuple[Optional[date], Optional[date]]:
        """Extract exhibition date or date range when user asks for films playing on a specific date.
        Returns (date_start, date_end); for single date both are the same.
        """
        # Range first (so "between 2/1/2026 and 2/14/2026" is not parsed as single 2/1/2026)
        range_pat = re.search(
            r"(?:between|from)\s+(\d{1,2})/(\d{1,2})/(\d{2,4})\s+(?:and|to|-)\s+(\d{1,2})/(\d{1,2})/(\d{2,4})",
            query_lower,
        )
        if range_pat:
            try:
                def _parse(m, d, y):
                    y = int(y)
                    if y < 100:
                        y = 2000 + y if y < 50 else 1900 + y
                    return date(y, int(m), int(d))
                d1 = _parse(range_pat.group(1), range_pat.group(2), range_pat.group(3))
                d2 = _parse(range_pat.group(4), range_pat.group(5), range_pat.group(6))
                return (min(d1, d2), max(d1, d2))
            except (ValueError, TypeError):
                pass
        # ISO: 2026-02-14
        iso = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", query_lower)
        if iso:
            try:
                d = date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
                return d, d
            except ValueError:
                pass
        # US: 2/14/2026, 2/14/26, 02/14/2026
        us = re.search(r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b", query_lower)
        if us:
            try:
                d = date(int(us.group(3)), int(us.group(1)), int(us.group(2)))
                return d, d
            except ValueError:
                pass
        us2 = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{2})\b", query_lower)
        if us2:
            try:
                yy = int(us2.group(3))
                year = 2000 + yy if yy < 50 else 1900 + yy
                d = date(year, int(us2.group(1)), int(us2.group(2)))
                return d, d
            except ValueError:
                pass
        return None, None

    def _extract_year_filters(self, query_lower: str) -> Tuple[Optional[int], Optional[int], Optional[List[str]]]:
        """Extract year range and decade filters."""
        year_start = None
        year_end = None
        decades = []
        
        # Extract decades (80s, 90s, etc.) -> 1980-1989, 1990-1999; 00s, 10s -> 2000-2009, 2010-2019
        decade_pattern = r'\b(\d{2})s\b'
        decade_matches = re.findall(decade_pattern, query_lower)
        for match in decade_matches:
            decade_num = int(match)
            if 0 <= decade_num <= 99:
                decades.append(match)
                # 50-99 -> 1950s-1990s; 00-49 -> 2000s-2040s
                start = 1900 + decade_num if decade_num >= 50 else 2000 + decade_num
                end = start + 9
                if year_start is None or start < year_start:
                    year_start = start
                if year_end is None or end > year_end:
                    year_end = end
        
        # Extract specific years (exclude years that appear only in exhibition-date context, e.g. 2/14/2026)
        year_pattern = r'\b(19|20)\d{2}\b'
        years = [int(match[0] + match[1]) for match in re.finditer(year_pattern, query_lower)]
        years = [
            y for y in years
            if not (re.search(r"\d{1,2}/\d{1,2}/" + str(y), query_lower) or re.search(str(y) + r"-\d{1,2}-\d{1,2}", query_lower))
        ]
        
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
        """Extract genre mentions from query. Handles possessive/plural forms like "sci-fi's", "sci-fis", "thrillers", "comedies"."""
        common_genres = [
            "horror", "thriller", "comedy", "drama", "action", "romance",
            "sci-fi", "science fiction", "fantasy", "crime", "mystery",
            "western", "war", "documentary", "animation", "musical"
        ]
        # Possessive/plural forms that don't match simple genre+"s" (e.g. comedies, sci-fis)
        plural_to_canonical = [
            ("comedies", "comedy"), ("mysteries", "mystery"), ("documentaries", "documentary"),
            ("fantasies", "fantasy"), ("sci-fis", "sci-fi"), ("sci-fi's", "sci-fi"),
        ]
        found_genres: List[str] = []
        seen: set = set()
        for genre in common_genres:
            if genre in query_lower:
                if genre not in seen:
                    seen.add(genre)
                    found_genres.append(genre)
            elif (genre + "'s") in query_lower or (genre + "s") in query_lower:
                if genre not in seen:
                    seen.add(genre)
                    found_genres.append(genre)
        for plural_form, canonical in plural_to_canonical:
            if plural_form in query_lower and canonical not in seen:
                seen.add(canonical)
                found_genres.append(canonical)
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

        def looks_like_date_phrase(s: str) -> bool:
            """Reject if this is a date phrase (e.g. 'with films playing on 2/14/2026')."""
            if not s or len(s) < 4:
                return False
            if re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", s) or re.search(r"20\d{2}-\d{1,2}-\d{1,2}", s):
                return True
            if "playing on" in s.lower() or "on 2/" in s.lower() or "on 1/" in s.lower():
                return True
            return False

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
                if (len(title) > 1 and title.lower() not in ("the", "a", "an", "it", "that", "this")
                        and not looks_like_date_phrase(title)
                        and not self._looks_like_film_elements_not_title(title)):
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
                if (len(title) > 1 and title.lower() not in ("the", "a", "an", "it", "that", "this")
                        and not looks_like_date_phrase(title)):
                    if self._looks_like_film_elements_not_title(title):
                        continue  # try next pattern
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
                if (len(title) > 1 and title.lower() not in ("the", "a", "an", "it", "that", "this")
                        and not looks_like_date_phrase(title)):
                    if self._looks_like_film_elements_not_title(title):
                        return None  # e.g. "the bridge and the lovers" -> use film_descriptor_terms instead
                    return title

        return None

    def _looks_like_film_elements_not_title(self, phrase: str) -> bool:
        """True if phrase looks like described elements/vibes (bridge, lovers) rather than a film title."""
        if not phrase or len(phrase) < 3:
            return False
        pl = phrase.lower()
        # "the X and the Y" or "X and the Y" -> often elements, not a title
        if re.search(r"\b(and\s+the|and\s+a\s+)\w+", pl):
            # Common element words that are rarely a film title by themselves
            element_cues = ("bridge", "lovers", "love", "road", "house", "night", "day", "man", "woman", "boy", "girl", "dog", "car", "train")
            words = set(re.findall(r"[a-z]+", pl))
            if any(c in words for c in element_cues) and not re.search(r"\b(the\s+)?[a-z]+\s+[a-z]+\s+[a-z]+", pl):
                return True
        # Vibe phrases: "late-night European art house", "slow cinema"
        if any(x in pl for x in ("art house", "art-house", "european art", "slow cinema", "late-night", "feel like")):
            return True
        return False

    def _normalize_film_title_for_matching(self, raw_title: Optional[str]) -> Optional[str]:
        """Normalize extracted film phrase to a short, searchable form for catalog/exhibition matching.
        Handles unconventional input: descriptors, filler, punctuation, parentheticals.
        E.g. 'that zombie movie Bone Temple' -> 'Bone Temple'; 'the one with... Bone Temple (2024)' -> 'Bone Temple'.
        """
        if not raw_title or not raw_title.strip():
            return raw_title
        s = raw_title.strip()
        # Remove parentheticals (year, notes, etc.) e.g. "Bone Temple (2024)" or "Title (you know the one)"
        s = re.sub(r"\s*\([^)]*\)\s*", " ", s).strip()
        # Remove trailing filler and optional punctuation (e.g. ", you know?" or " etc.")
        s = re.sub(r"[\s,;]+(?:etc\.?|and stuff|you know\??|right\?|or something)\s*[.?!]*\s*$", "", s, flags=re.IGNORECASE).strip()
        s = s.rstrip(".,;:?!")
        # Strip leading filler / descriptor patterns (order: more specific first)
        for pattern in [
            r"^(?:that|the)\s+.+?\s+(?:movie|film)\s+",
            r"^(?:something\s+like|kind\s+of|sort\s+of|like\s+the\s+one\s+)(?:called\s+)?",
            r"^(?:the\s+one\s+(?:with|about|called)\s*[\.\s…]*)?",
            r"^(?:that|the)\s+",
        ]:
            s = re.sub(pattern, "", s, flags=re.IGNORECASE).strip()
        # Drop leading/trailing articles and single-word noise
        s = re.sub(r"^(?:the|a|an)\s+", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"\s+(?:the|a|an)$", "", s, flags=re.IGNORECASE).strip()
        # Strip leading ellipsis/punctuation (e.g. "... Bone Temple")
        s = re.sub(r"^[\s\.…\-:]+", "", s).strip()
        # Trailing filler words that are not part of a title (e.g. "Bone Temple thing" -> "Bone Temple")
        s = re.sub(r"\s+(?:thing|one|movie|film)$", "", s, flags=re.IGNORECASE).strip()
        # If still empty or too short, use last 2–4 words of original (before parentheticals) as core title
        words = [w for w in re.split(r"\s+", s) if w and len(w) > 0]
        if not words or len(s) < 2:
            raw_words = re.split(r"\s+", re.sub(r"\s*\([^)]*\)", " ", raw_title).strip())
            raw_words = [w for w in raw_words if w and w.lower() not in ("the", "a", "an", "that", "movie", "film")]
            if len(raw_words) >= 2:
                s = " ".join(raw_words[-2:])
            elif len(raw_words) == 1:
                s = raw_words[-1]
            else:
                s = raw_title.strip()[:50].rstrip(".,;")
        elif len(words) > 4:
            # Long phrase: prefer last 2–3 words as distinctive title part (e.g. "28 Years Later: The Bone Temple" -> "Bone Temple")
            s = " ".join(words[-2:]) if len(words) >= 2 else " ".join(words)
        else:
            s = " ".join(words)
        return s.strip() or raw_title.strip()[:50].strip()

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

ROBUSTNESS: Users may phrase things unconventionally—mixing film names with descriptors, filters, or casual wording. Infer intent from context. For each field, extract the clearest value; use null when genuinely unclear. Ignore filler ("that movie", "the one with", "you know") and focus on the actual film title, filters, or preferences.

CRITICAL RULES:
0. Determine whether the user is starting a NEW search or REFINING the current one.
   - Set is_new_search=true if the user is starting over or changing to a clearly different request.
   - Set is_new_search=false if the prompt is a follow-up/refinement (e.g. "only", "instead", "make it", "change to", "between", "narrow to", "broaden to", "actually", "now show me...").
   - If refining, carry forward prior constraints unless explicitly changed.

1. match_to_specific_film: The ONE film we match library titles AGAINST (similarity target). Set whenever the user refers to a specific film, even if phrased oddly. Return ONLY the short, searchable title as it would appear in a catalog (e.g. "Bone Temple", "The Housemaid"). Strip descriptors, filler, and parentheticals so partial matching works (e.g. "28 Years Later: The Bone Temple" matches "Bone Temple"). Examples:
   - "I like that zombie movie Bone Temple" or "the zombie one Bone Temple" -> "Bone Temple"
   - "due to the success of The Housemaid" -> "The Housemaid"
   - "something like Bone Temple (the sequel)" or "that Bone Temple thing" -> "Bone Temple"
   Use null ONLY if no specific film is mentioned.

3. year_start, year_end: Extract from meaning. Examples:
   - "between 1970 - 1999", "1970-1999", "from 1970 to 1999" -> year_start: 1970, year_end: 1999
   - "80s films", "from the 80s" -> year_start: 1980, year_end: 1989
   - "older titles", "older films" -> year_end: 2000 or 2010 (infer); year_start: null unless stated
   - "recent" -> year_start: 2020 or similar

3. genres: Infer from context. Include the canonical genre whenever the user mentions a genre in possessive or plural form: "sci-fi's", "sci-fis", "thrillers", "comedies", "horrors", etc. mean films with that genre. Use canonical names: sci-fi, thriller, comedy, horror, drama, action, romance, fantasy, crime, mystery, western, war, documentary, animation, musical.

5. director_weight, writer_weight, cast_weight, thematic_weight, stylistic_weight: 0.0-1.0. Default ~0.2-0.3 each, thematic ~0.3. Bump the one the query emphasizes (e.g. director-focused -> director_weight 0.9).

6. territory: Only if user explicitly mentions US, UK, FR, CA, MX.

7. venue: When user asks for a specific venue (e.g. "Film Forum", "whatever is doing well at Film Forum"), set to the venue name string. Otherwise null.

8. time_period: "month"|"week"|"now" only if user says "this month", "right now", "this week".

7. exhibition_date_start, exhibition_date_end: When the user asks for films playing on a specific date or date range (e.g. "playing on 2/14/2026", "between 2/1 and 2/14/2026"), set these to ISO date strings "YYYY-MM-DD". For a single date use the same for both. Use null if no exhibition date is mentioned.

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
- genres (list or null): use canonical names; "(genre)'s" or "(genre)s" (e.g. sci-fi's, sci-fis, thrillers, comedies) = that genre
- writer_gender, director_gender, lead_gender (or null)
- specific_directors, specific_writers, specific_actors (list or null)
- match_to_specific_film (short searchable film title only, or null; use null when user describes elements/vibes and set film_descriptor_terms instead)
- film_descriptor_terms (list or null): key terms when user describes elements/vibes (e.g. ["bridge", "lovers"], ["late-night", "european", "art house"]); null when a specific film is named
- venue (string or null): venue name when user asks for a specific venue (e.g. "Film Forum")
- territory (US/UK/FR/CA/MX or null; only if explicitly mentioned)
- time_period ("month"|"week"|"now" or null)
- exhibition_date_start, exhibition_date_end (ISO "YYYY-MM-DD" or null; when user says "playing on 2/14/2026" or "between 2/1 and 2/14/2026")
- column_filters (object or null): optional filters on ANY library/exhibition column. Infer from context even if phrased loosely. Keys = column names (e.g. release_year, genres, director, lead_gender). Values: string/number (exact or substring), list of strings (match any), or {{"min": n, "max": n}} for numeric range.
- column_weights (object or null): optional boost weights for match calculation. Infer when user emphasizes something (e.g. "really care about genre"). Keys = column names (e.g. genres, release_year). Values: 0.0-1.0.

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
                max_tokens=600,
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
            # Preserve deterministic gender extractions: do not overwrite when regex already found one (e.g. "female leads")
            if "writer_gender" in data and data["writer_gender"]:
                if base_intent.writer_gender is None:
                    base_intent.writer_gender = (data["writer_gender"] or "").strip().lower() or None
            if "director_gender" in data and data["director_gender"]:
                if base_intent.director_gender is None:
                    base_intent.director_gender = (data["director_gender"] or "").strip().lower() or None
            if "lead_gender" in data and data["lead_gender"]:
                llm_lead = (data["lead_gender"] or "").strip().lower()
                if llm_lead in ("female", "male"):
                    # Only set from LLM when regex did not already set lead_gender (refinement: keep user's "female leads")
                    if base_intent.lead_gender is None:
                        base_intent.lead_gender = llm_lead
                    # If regex already set it, do not overwrite (prevents LLM from returning null/male and breaking refinement)
            if "specific_directors" in data and data["specific_directors"]:
                base_intent.specific_directors = data["specific_directors"]
            if "specific_writers" in data and data["specific_writers"]:
                base_intent.specific_writers = data["specific_writers"]
            if "specific_actors" in data and data["specific_actors"]:
                base_intent.specific_actors = data["specific_actors"]
            if "time_period" in data and data["time_period"]:
                base_intent.time_period = data["time_period"]
            # Exhibition date or range (LLM may return ISO "YYYY-MM-DD")
            for key, attr in [("exhibition_date_start", "exhibition_date_start"), ("exhibition_date_end", "exhibition_date_end")]:
                val = data.get(key)
                if isinstance(val, str) and val.strip():
                    try:
                        setattr(base_intent, attr, datetime.strptime(val.strip()[:10], "%Y-%m-%d").date())
                    except ValueError:
                        pass
            # Only overwrite if LLM returns a non-empty film title; preserve regex extraction otherwise. Normalize for catalog matching.
            llm_film = data.get("match_to_specific_film")
            if isinstance(llm_film, str) and llm_film.strip():
                s = llm_film.strip()
                # Reject pronouns and time phrases that the LLM sometimes wrongly returns as film titles
                if s.lower() in ("it", "that", "this", "this month", "that month", "this week", "that week"):
                    base_intent.match_to_specific_film = None
                # Do not treat date phrases as film titles (e.g. "with films playing on 2/14/2026")
                elif not re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", s) and not re.search(r"20\d{2}-\d{1,2}-\d{1,2}", s) and "playing on" not in s.lower():
                    base_intent.match_to_specific_film = self._normalize_film_title_for_matching(s) or s
            # else: keep base_intent.match_to_specific_film (e.g. from regex)
            # Territory should only be applied when explicitly mentioned in the prompt.
            # We rely on deterministic token/phrase extraction in _extract_territory to avoid false positives.
            # (Do not let the LLM "infer" a territory.)

            # Dynamic column_filters and column_weights from LLM
            if "column_filters" in data and isinstance(data["column_filters"], dict) and data["column_filters"]:
                base_intent.column_filters = data["column_filters"]
            if "column_weights" in data and isinstance(data["column_weights"], dict) and data["column_weights"]:
                base_intent.column_weights = {k: float(v) for k, v in data["column_weights"].items() if isinstance(v, (int, float))}
            if "film_descriptor_terms" in data and isinstance(data["film_descriptor_terms"], list) and data["film_descriptor_terms"]:
                base_intent.film_descriptor_terms = [str(t).strip() for t in data["film_descriptor_terms"] if t and str(t).strip()]
            if "venue" in data and isinstance(data["venue"], str) and data["venue"].strip():
                venue_val = data["venue"].strip()
                # Don't let LLM overwrite with a time phrase (e.g. "this month" from "at Film Forum this month")
                if venue_val.lower() not in ("this month", "that month", "this week", "that week", "now"):
                    base_intent.venue = venue_val
            
        except Exception as e:
            print(f"[QueryIntentParser] Error in LLM parsing: {e}")
        
        return base_intent
