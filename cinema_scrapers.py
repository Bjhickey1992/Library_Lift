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
    Used only for cinema programme extraction (LLM fallback). Anthropic/Claude
    is never used here; it is only used for 'need' field generation elsewhere.
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
    """Parse date/datetime from LLM or JSON-LD. Handles ISO and calendar dates. Does NOT treat
    'today' or 'now playing' as a date (caller must use date_context for that, with a per-venue cap).
    """
    if not value:
        return None
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    lower = value.lower()
    # Do not treat these as valid dates — too many films get dumped with "today"
    if lower in ("today", "now", "now playing", "currently showing"):
        return None
    now = dt.datetime.now()
    if lower == "tomorrow":
        d = now + dt.timedelta(days=1)
        return d.replace(hour=19, minute=0, second=0, microsecond=0)
    try:
        parsed = dateparser.parse(value)
        if parsed is not None:
            return parsed
    except (ValueError, TypeError, OverflowError):
        pass
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
    Fallback scraper that asks an LLM (OpenAI only; never Anthropic) to extract
    film screenings from the raw HTML when JSON-LD Event data is missing.

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
    # Use text-only to reduce markup noise. Include enough content to capture full calendars.
    text = soup.get_text(separator="\n", strip=True)
    max_chars = 28000  # Larger window so multi-week calendars are not cut off
    if len(text) > max_chars:
        # Keep head, middle, and tail so we don't drop calendar blocks (often in middle of page)
        n = len(text)
        head_len = 7000
        tail_len = 7000
        mid_len = max_chars - head_len - tail_len - 100
        mid_start = (n - mid_len) // 2
        text = text[:head_len] + "\n...[calendar/schedule section]...\n" + text[mid_start:mid_start + mid_len] + "\n...\n" + text[-tail_len:]

    today = dt.datetime.now().strftime("%Y-%m-%d")
    four_weeks_later = (dt.datetime.now() + dt.timedelta(weeks=4)).strftime("%Y-%m-%d")
    
    system_prompt = (
        "You are a thorough extraction assistant for cinema programming calendars.\n"
        f"Today is {today}. The venue has a programming calendar. Extract EVERY film screening that has a date between {today} and {four_weeks_later}.\n\n"
        "WHERE TO FIND DATES: Search the ENTIRE page for:\n"
        "- Calendar grids, schedule tables, or date columns next to film titles\n"
        "- Date headers like 'Feb 15', 'March 1', 'Fri Feb 28', 'Week of Feb 17', 'Feb 20–22'\n"
        "- Phrases like 'opening Friday', 'through March 2', 'starts Feb 15', 'runs Feb 20–27'\n"
        "- Sections titled 'Coming Soon', 'Next Week', 'February', 'March', or by date range\n"
        "- Each film may be listed with a specific date, a range (use the start date), or 'now playing' (use today + date_context \"now_playing\")\n\n"
        "RULES:\n"
        "1. For each film that has a specific date on the page (or start of a range), set start_datetime to that date (YYYY-MM-DD). Do NOT set date_context.\n"
        "2. Only when the page explicitly says a film is 'now playing' or 'showing today', set start_datetime to " + today + " AND set date_context to \"now_playing\".\n"
        "3. If a film appears under a date header (e.g. 'February 15' then a list of films), use that date for those films.\n"
        "4. Include every film you can match to a date in the " + today + " to " + four_weeks_later + " window. Be thorough—venues often list 2–4 weeks of programming.\n"
        "5. Never use today's date without date_context \"now_playing\". Omit films that have no date on the page.\n\n"
        "Return ONLY valid JSON: { \"screenings\": [ { \"title\": \"Film Title\", \"start_datetime\": \"YYYY-MM-DD\" }, ... ] } Add \"date_context\": \"now_playing\" only when the page says now playing."
    )

    user_prompt = (
        f"Cinema: {source.name}\n"
        f"Programme URL: {source.programme_url}\n\n"
        "Extract the FULL programming calendar (all films with dates in the next 4 weeks) from the page text below. "
        "Look for calendar blocks, schedule tables, and date headers; output every film with its actual date.\n\n"
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
        dt_raw = item.get("start_datetime") or item.get("date") or item.get("start_date")
        start_dt = _parse_event_datetime(dt_raw)
        date_context = (item.get("date_context") or item.get("date_source") or "").lower()
        # Only allow "today" when the model explicitly indicated now playing / this week
        if start_dt is None:
            if date_context in ("now_playing", "this_week", "currently_showing"):
                start_dt = today_dt
            else:
                continue  # Skip films with no parseable date; do not default to today
        # If the parsed date is today, only accept when model set date_context (avoid "all today" venue dumps)
        elif start_dt.date() == dt.datetime.now().date():
            if date_context not in ("now_playing", "this_week", "currently_showing"):
                continue  # Model returned today's date but did not label as now playing — skip
        # If date is far in the past, treat as "now playing" only if model said so
        if start_dt < (dt.datetime.now() - dt.timedelta(days=30)):
            if date_context in ("now_playing", "this_week", "currently_showing", ""):
                start_dt = today_dt
            else:
                continue  # Skip past-dated unless it's clearly current
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
    # Cap how many screenings per venue can have "today" — avoids calendar dumps all dated today
    screenings = _cap_today_screenings_per_venue(screenings)
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
    max_chars = 28000  # Same as main LLM; use head+middle+tail so calendar blocks are not dropped
    if len(text) > max_chars:
        n = len(text)
        head_len = 7000
        tail_len = 7000
        mid_len = max_chars - head_len - tail_len - 100
        mid_start = (n - mid_len) // 2
        text = text[:head_len] + "\n...[middle]...\n" + text[mid_start:mid_start + mid_len] + "\n...\n" + text[-tail_len:]

    today = dt.datetime.now().strftime("%Y-%m-%d")
    four_weeks_later = (dt.datetime.now() + dt.timedelta(weeks=4)).strftime("%Y-%m-%d")
    
    system_prompt = (
        "You are an aggressive data extraction assistant for cinema showtimes.\n"
        f"Today's date is {today}. Extract EVERY film title you can find on this page.\n"
        f"For each film: set start_datetime to a specific date (YYYY-MM-DD) if the page gives one; "
        f"otherwise use {today} and set date_context to \"now_playing\". Include films in 'now playing', "
        "'coming soon', calendar lists, or any programme listing even if the date is unclear (use today + date_context \"now_playing\").\n"
        "Return ONLY valid JSON: { \"screenings\": [ { \"title\": \"...\", \"start_datetime\": \"YYYY-MM-DD\", \"date_context\": \"now_playing\" if no date }, ... ] }"
    )

    user_prompt = (
        f"Cinema: {source.name}\n"
        f"Programme URL: {source.programme_url}\n\n"
        "Extract ALL film titles from this page text. Include every movie/screening you see. Use today's date and date_context \"now_playing\" when no date is given.\n\n"
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
        dt_raw = item.get("start_datetime") or item.get("date") or item.get("start_date")
        start_dt = _parse_event_datetime(dt_raw)
        date_context = (item.get("date_context") or item.get("date_source") or "").lower()
        if start_dt is None:
            if date_context in ("now_playing", "this_week", "currently_showing"):
                start_dt = today_dt
            else:
                continue
        elif start_dt.date() == dt.datetime.now().date():
            if date_context not in ("now_playing", "this_week", "currently_showing"):
                continue
        if start_dt < (dt.datetime.now() - dt.timedelta(days=30)):
            if date_context in ("now_playing", "this_week", "currently_showing", ""):
                start_dt = today_dt
            else:
                continue
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
    screenings = _cap_today_screenings_per_venue(screenings)
    return screenings


def _cap_today_screenings_per_venue(screenings: List[Screening], max_today_per_venue: int = 8) -> List[Screening]:
    """If a venue has more than max_today_per_venue screenings with today's date, keep only the first that many."""
    if not screenings:
        return screenings
    today = dt.datetime.now().date()
    today_list: List[Screening] = []
    other_list: List[Screening] = []
    for s in screenings:
        if s.start_dt.date() == today:
            today_list.append(s)
        else:
            other_list.append(s)
    if len(today_list) <= max_today_per_venue:
        return screenings
    return other_list + today_list[:max_today_per_venue]


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
        # Fallback 1: standard LLM-based extraction.
        llm_screenings = []
        try:
            print(f"  -> {source.name}: Trying LLM extraction...")
            llm_screenings = scrape_programme_via_llm(source, session=session)
            if llm_screenings:
                print(f"[OK] {source.name}: Found {len(llm_screenings)} screenings via LLM")
            else:
                print(f"[WARN] {source.name}: LLM extraction returned 0 screenings")
        except Exception as e:
            print(f"[ERROR] {source.name}: LLM extraction failed: {e}")
            # For rate limiting, wait and retry once with the same precise prompt.
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

    if not screenings:
        # Fallback 2: more aggressive LLM extraction for stubborn venues.
        aggressive_screenings: List[Screening] = []
        try:
            print(f"  -> {source.name}: Trying aggressive LLM extraction...")
            aggressive_screenings = scrape_programme_via_llm_aggressive(source, session=session)
            if aggressive_screenings:
                print(f"[OK] {source.name}: Found {len(aggressive_screenings)} screenings via aggressive LLM")
            else:
                print(f"[WARN] {source.name}: Aggressive LLM extraction returned 0 screenings")
        except Exception as e:
            print(f"[ERROR] {source.name}: Aggressive LLM extraction failed: {e}")
            aggressive_screenings = []
        screenings = aggressive_screenings

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
    Filter screenings to those within the date window.

    Date range limits:
      - start: now - 14 days (include films that started up to 14 days ago, "now playing")
      - end:   now + weeks_ahead (default 4 weeks)

    So the window is [now-14d, now+4w]. Screenings with start_dt outside this range are dropped.
    Note: When scrapers (especially LLM) don't return a date, start_dt defaults to "today", so
    those screenings always pass the filter. That can make per-venue counts much higher than
    a real 4-week calendar; the caller may apply a per-venue cap to avoid full calendar dumps.
    """
    def _as_naive(d: dt.datetime) -> dt.datetime:
        """Normalize both aware/naive datetimes into naive UTC-like values for safe comparison."""
        if d.tzinfo is not None and d.utcoffset() is not None:
            return d.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return d

    now = _as_naive(now or dt.datetime.now())
    start = now - dt.timedelta(days=14)  # Include films that started up to 14 days ago (more lenient)
    end = now + dt.timedelta(weeks=weeks_ahead)
    out = []
    for s in screenings:
        s_start = _as_naive(s.start_dt)
        if start <= s_start <= end:
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

