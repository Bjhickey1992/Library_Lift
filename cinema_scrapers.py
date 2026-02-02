import datetime as dt
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

try:
    from openai import OpenAI
except ImportError:  # soft dependency; LLM scraping is optional
    OpenAI = None


# Rotate user agents to avoid detection
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
]

DEFAULT_HEADERS = {
    "User-Agent": USER_AGENTS[0],
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


@dataclass
class CinemaSource:
    id: str
    name: str
    country: str  # ISO 2-letter
    programme_url: str
    city: Optional[str] = None
    type: Optional[str] = None  # chain | independent | nonprofit
    enabled: bool = True
    scraper: str = "auto"  # auto | jsonld_events


@dataclass
class Screening:
    title: str
    start_dt: dt.datetime
    venue_name: str
    country: str
    city: Optional[str]
    source_id: str
    source_url: str
    raw: Optional[Dict[str, Any]] = None


def fetch_html(url: str, *, session: Optional[requests.Session] = None, timeout: int = 45, retries: int = 3) -> str:
    """Fetch HTML with retry logic for rate-limited sites and user-agent rotation."""
    import random
    import time
    s = session or requests.Session()
    
    for attempt in range(retries):
        try:
            # Rotate user agent on each attempt
            headers = DEFAULT_HEADERS.copy()
            headers["User-Agent"] = random.choice(USER_AGENTS)
            
            resp = s.get(url, headers=headers, timeout=timeout)
            
            if resp.status_code == 429:  # Rate limited
                if attempt < retries - 1:
                    wait_time = (attempt + 1) * 5  # Exponential backoff: 5s, 10s, 15s
                    print(f"  Rate limited (429), waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
            
            if resp.status_code == 403:  # Forbidden - try different user agent
                if attempt < retries - 1:
                    wait_time = (attempt + 1) * 3
                    print(f"  Forbidden (403), trying different user agent in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            
            resp.raise_for_status()
            return resp.text
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                raise
            wait_time = (attempt + 1) * 2
            print(f"  Request failed, retrying in {wait_time}s...")
            time.sleep(wait_time)
    raise requests.exceptions.RequestException("Failed after retries")


def _get_openai_client() -> Optional["OpenAI"]:
    """
    Return an OpenAI client if the library and API key are available.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


def _iter_jsonld_blocks(soup: BeautifulSoup) -> Iterable[Dict[str, Any]]:
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        if not tag.string:
            continue
        txt = tag.string.strip()
        if not txt:
            continue
        try:
            data = json.loads(txt)
        except json.JSONDecodeError:
            # Some sites embed multiple JSON objects without wrapping; skip for now.
            continue
        yield data


def _flatten_jsonld(obj: Any) -> Iterable[Dict[str, Any]]:
    """
    Yield JSON-LD nodes (dicts) from common containers:
    - dict with @graph
    - list of dicts
    - a single dict
    """
    if isinstance(obj, dict):
        if "@graph" in obj and isinstance(obj["@graph"], list):
            for item in obj["@graph"]:
                if isinstance(item, dict):
                    yield item
        else:
            yield obj
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item


def _is_event_type(t: Any) -> bool:
    if not t:
        return False
    if isinstance(t, str):
        return t.lower().endswith("event")
    if isinstance(t, list):
        return any(isinstance(x, str) and x.lower().endswith("event") for x in t)
    return False


def _parse_event_datetime(value: Any) -> Optional[dt.datetime]:
    if not value:
        return None
    if isinstance(value, str):
        try:
            return dateparser.parse(value)
        except (ValueError, TypeError, OverflowError):
            return None
    return None


def scrape_jsonld_events(
    source: CinemaSource,
    *,
    session: Optional[requests.Session] = None,
) -> List[Screening]:
    """
    Scrape screenings from JSON-LD Event nodes on the programme page.

    Many venues publish showtimes as schema.org Event data. This is the most
    robust "generic" scraper we can do without site-specific parsers.
    """
    html = fetch_html(source.programme_url, session=session)
    soup = BeautifulSoup(html, "html.parser")

    screenings: List[Screening] = []

    for block in _iter_jsonld_blocks(soup):
        for node in _flatten_jsonld(block):
            if not _is_event_type(node.get("@type")):
                continue

            name = node.get("name") or node.get("headline") or node.get("description")
            if not name or not isinstance(name, str):
                continue

            start_dt = _parse_event_datetime(node.get("startDate"))
            if start_dt is None:
                continue

            # Prefer venue name from config; fall back to node.location
            venue_name = source.name
            loc = node.get("location")
            if isinstance(loc, dict):
                loc_name = loc.get("name")
                if isinstance(loc_name, str) and loc_name.strip():
                    venue_name = loc_name.strip()

            screenings.append(
                Screening(
                    title=name.strip(),
                    start_dt=start_dt,
                    venue_name=venue_name,
                    country=source.country.upper(),
                    city=source.city,
                    source_id=source.id,
                    source_url=source.programme_url,
                    raw=node,
                )
            )

    return screenings


def scrape_programme_via_llm(
    source: CinemaSource,
    *,
    session: Optional[requests.Session] = None,
) -> List[Screening]:
    """
    Fallback scraper that asks an LLM (OpenAI) to extract film screenings
    from the raw HTML when JSON-LD `Event` data is missing or insufficient.

    This expects the model to return a JSON object of the form:
    {
      "screenings": [
        {"title": "...", "start_datetime": "YYYY-MM-DDTHH:MM", "url": "optional"}
      ]
    }
    """
    client = _get_openai_client()
    if client is None:
        return []

    html = fetch_html(source.programme_url, session=session, retries=3)
    soup = BeautifulSoup(html, "html.parser")
    # Use text-only to reduce markup noise; truncate to stay within context limits.
    text = soup.get_text(separator="\n", strip=True)
    max_chars = 18000  # Increased to capture more content
    if len(text) > max_chars:
        # Try to keep the most relevant part (often schedules are in the middle/end)
        # Take first 6000 (header/nav) and last 12000 (schedule content)
        text = text[:6000] + "\n...\n" + text[-12000:]

    today = dt.datetime.now().strftime("%Y-%m-%d")
    four_weeks_later = (dt.datetime.now() + dt.timedelta(weeks=4)).strftime("%Y-%m-%d")
    
    system_prompt = (
        "You are a precise data extraction assistant for cinema showtimes.\n"
        f"Today's date is {today}. Extract ALL film screenings scheduled between {today} and {four_weeks_later}.\n"
        "Given the text content of a cinema's programme webpage, "
        "extract EVERY individual film screening as structured JSON.\n"
        "IMPORTANT: Be thorough and extract ALL films mentioned, even if dates are unclear.\n"
        "Focus only on feature film screenings (ignore ads, gift cards, generic news, past events).\n"
        "Extract EVERY film that appears on the schedule, including:\n"
        "- Films marked as 'now playing', 'coming soon', 'this week', 'next week', 'upcoming'\n"
        "- Films with specific dates in the next 4 weeks\n"
        "- Films in current/upcoming series or retrospectives\n"
        "- Films listed in calendars, schedules, or programme listings\n"
        "- Films mentioned in any context that suggests they will be shown\n"
        "For each screening, provide:\n"
        "- title: the film title (string, exact as shown, remove any 'Q&A', '35mm', etc. annotations)\n"
        "- start_datetime: an ISO 8601 datetime string (YYYY-MM-DDTHH:MM) or date-only (YYYY-MM-DD).\n"
        f"  - If 'now playing' or no date shown, use today's date ({today})\n"
        "  - If 'this week', use a date within the next 7 days\n"
        "  - If 'next week', use a date 7-14 days from today\n"
        "  - If only a date range is given, use the start date\n"
        "  - If multiple showtimes for same film, return one entry per distinct date\n"
        "  - If date is completely unclear but film is listed, use today's date\n"
        "CRITICAL: If you see ANY film titles mentioned anywhere on the page, extract them. "
        "It's better to extract too many than to miss films.\n"
        "Look for film titles in: lists, calendars, schedules, programme descriptions, "
        "film series names, retrospective titles, 'now showing', 'coming soon' sections, "
        "and anywhere else films might be mentioned.\n"
        "Return ONLY valid JSON with this exact top-level shape:\n"
        "{ \"screenings\": [ { \"title\": \"Film Title\", \"start_datetime\": \"2024-01-15\" }, ... ] }"
    )

    user_prompt = (
        f"Cinema: {source.name}\n"
        f"City: {source.city or ''}\n"
        f"Country: {source.country}\n"
        f"Programme URL: {source.programme_url}\n\n"
        f"Page text:\n{text}"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
    except Exception as e:
        print(f"OpenAI API error for {source.name}: {e}")
        return []

    try:
        content = resp.choices[0].message.content
        data = json.loads(content)
    except Exception as e:
        print(f"JSON parsing error for {source.name}: {e}")
        return []

    items = data.get("screenings") or []
    screenings: List[Screening] = []
    today_dt = dt.datetime.now().replace(hour=19, minute=0, second=0, microsecond=0)  # Default 7pm
    
    for item in items:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        if not title:
            continue
        dt_raw = item.get("start_datetime") or item.get("date")
        start_dt = _parse_event_datetime(dt_raw)
        # Fallback: if date parsing fails but we have a title, use today as default
        # (assumes "now playing" if no date was extractable)
        if start_dt is None:
            start_dt = today_dt
        # If date is more than 30 days in the past, assume it's "now playing" and use today
        # (some sites show past dates for currently playing films)
        if start_dt < (dt.datetime.now() - dt.timedelta(days=30)):
            start_dt = today_dt
        screenings.append(
            Screening(
                title=title,
                start_dt=start_dt,
                venue_name=source.name,
                country=source.country.upper(),
                city=source.city,
                source_id=source.id,
                source_url=source.programme_url,
                raw=item,
            )
        )
    return screenings


def scrape_programme_via_llm_aggressive(
    source: CinemaSource,
    *,
    session: Optional[requests.Session] = None,
) -> List[Screening]:
    """
    More aggressive LLM extraction that tries harder to find ANY film titles on the page.
    Used as a fallback when regular LLM extraction returns 0 screenings.
    """
    client = _get_openai_client()
    if client is None:
        return []

    html = fetch_html(source.programme_url, session=session, retries=3)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    max_chars = 20000  # Even more content for aggressive extraction
    if len(text) > max_chars:
        text = text[:8000] + "\n...\n" + text[-12000:]

    today = dt.datetime.now().strftime("%Y-%m-%d")
    four_weeks_later = (dt.datetime.now() + dt.timedelta(weeks=4)).strftime("%Y-%m-%d")
    
    system_prompt = (
        "You are an EXTREMELY thorough data extraction assistant for cinema showtimes.\n"
        f"Today's date is {today}. Extract ALL film screenings scheduled between {today} and {four_weeks_later}.\n"
        "CRITICAL: Extract EVERY film title you see on this page, even if:\n"
        "- The date is unclear or missing\n"
        "- It's only mentioned in passing\n"
        "- It's in a list, calendar, or schedule\n"
        "- It's part of a series or retrospective name\n"
        "- It appears in 'now playing', 'coming soon', 'upcoming', or any other section\n"
        "Look in EVERY part of the page: headers, navigation, main content, sidebars, footers, calendars, lists.\n"
        "If you see a film title ANYWHERE, extract it with today's date if no date is given.\n"
        "Return ONLY valid JSON with this exact top-level shape:\n"
        "{ \"screenings\": [ { \"title\": \"Film Title\", \"start_datetime\": \"2024-01-15\" }, ... ] }"
    )

    user_prompt = (
        f"Cinema: {source.name}\n"
        f"City: {source.city or ''}\n"
        f"Country: {source.country}\n"
        f"Programme URL: {source.programme_url}\n\n"
        f"Extract ALL film titles from this page text:\n{text}"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,  # Lower temperature for more consistent extraction
        )
    except Exception as e:
        return []

    try:
        content = resp.choices[0].message.content
        data = json.loads(content)
    except Exception:
        return []

    items = data.get("screenings") or []
    screenings: List[Screening] = []
    today_dt = dt.datetime.now().replace(hour=19, minute=0, second=0, microsecond=0)
    
    for item in items:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        if not title:
            continue
        dt_raw = item.get("start_datetime") or item.get("date")
        start_dt = _parse_event_datetime(dt_raw)
        if start_dt is None:
            start_dt = today_dt
        # If date is more than 30 days in the past, assume it's "now playing" and use today
        # (some sites show past dates for currently playing films)
        if start_dt < (dt.datetime.now() - dt.timedelta(days=30)):
            start_dt = today_dt
        screenings.append(
            Screening(
                title=title,
                start_dt=start_dt,
                venue_name=source.name,
                country=source.country.upper(),
                city=source.city,
                source_id=source.id,
                source_url=source.programme_url,
                raw=item,
            )
        )
    return screenings


def scrape_programme(
    source: CinemaSource,
    *,
    session: Optional[requests.Session] = None,
    ) -> List[Screening]:
    """
    Scrape a programme page using the configured or automatic strategy.

    Order of operations:
    1. Try JSON-LD `Event` extraction.
    2. If no screenings are found, and OPENAI_API_KEY is set, fall back to
       LLM-based text extraction.
    """
    strategy = (source.scraper or "auto").lower()
    screenings: List[Screening] = []

    if strategy in ("auto", "jsonld_events"):
        try:
            screenings = scrape_jsonld_events(source, session=session)
            if screenings:
                print(f"[OK] {source.name}: Found {len(screenings)} screenings via JSON-LD")
        except Exception as e:
            print(f"[ERROR] {source.name}: JSON-LD extraction failed: {e}")
            screenings = []

    if not screenings:
        llm_screenings = []
        try:
            print(f"  -> {source.name}: Trying LLM extraction...")
            llm_screenings = scrape_programme_via_llm(source, session=session)
            if llm_screenings:
                print(f"[OK] {source.name}: Found {len(llm_screenings)} screenings via LLM")
            else:
                print(f"[WARN] {source.name}: LLM extraction returned 0 screenings")
                # Try one more time with a more aggressive prompt if first attempt failed
                if len(llm_screenings) == 0:
                    print(f"  -> {source.name}: Retrying LLM extraction with more aggressive extraction...")
                    try:
                        # Retry with a more aggressive approach - extract ANY film titles seen
                        llm_screenings_retry = scrape_programme_via_llm_aggressive(source, session=session)
                        if llm_screenings_retry:
                            print(f"[OK] {source.name}: Found {len(llm_screenings_retry)} screenings via LLM (aggressive retry)")
                            llm_screenings = llm_screenings_retry
                    except Exception as e2:
                        print(f"[ERROR] {source.name}: Aggressive retry also failed: {e2}")
        except Exception as e:
            print(f"[ERROR] {source.name}: LLM extraction failed: {e}")
            # For rate limiting, wait and retry once
            if "429" in str(e) or "rate limit" in str(e).lower():
                print(f"  -> {source.name}: Rate limited, waiting 10s before retry...")
                import time
                time.sleep(10)
                try:
                    llm_screenings = scrape_programme_via_llm(source, session=session)
                    if llm_screenings:
                        print(f"[OK] {source.name}: Found {len(llm_screenings)} screenings via LLM (after retry)")
                except Exception as e2:
                    print(f"[ERROR] {source.name}: LLM retry also failed: {e2}")
                    llm_screenings = []
            else:
                llm_screenings = []
        screenings = llm_screenings

    if not screenings and strategy not in ("auto", "jsonld_events"):
        raise ValueError(f"Unknown scraper strategy: {source.scraper}")

    return screenings


def filter_screenings_to_window(
    screenings: List[Screening],
    *,
    weeks_ahead: int = 4,
    now: Optional[dt.datetime] = None,
) -> List[Screening]:
    """
    Filter screenings to those within the next N weeks.
    Also includes screenings from up to 14 days ago (in case of "now playing" with past dates).
    More lenient to catch films that are currently playing.
    """
    now = now or dt.datetime.now()
    start = now - dt.timedelta(days=14)  # Include films that started up to 14 days ago (more lenient)
    end = now + dt.timedelta(weeks=weeks_ahead)
    out = []
    for s in screenings:
        if start <= s.start_dt <= end:
            out.append(s)
    return out


def normalize_title_for_lookup(title: str) -> str:
    """
    Lightweight normalizer for matching showtime titles to movie titles.
    Strips common suffixes like "Q&A", "35mm", etc.
    """
    t = (title or "").strip()
    # Remove common separators and trailing annotations
    for sep in (" - ", " — ", " – ", ": "):
        if sep in t:
            t = t.split(sep, 1)[0].strip()
    # Remove bracketed / parenthetical suffixes at end
    for opener, closer in (("(", ")"), ("[", "]")):
        if t.endswith(closer) and opener in t:
            t = t[: t.rfind(opener)].strip()
    return t

