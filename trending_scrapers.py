"""
Scrapers for trending/popular content from IMDB and Nielsen.
"""

import requests
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time


def scrape_imdb_trending(top_n: int = 50) -> List[Dict]:
    """
    Scrape IMDB's top trending films and TV shows.
    
    Args:
        top_n: Number of top items to scrape (default: 50)
    
    Returns:
        List of dicts with keys: title, year, imdb_id, type ('movie' or 'tv'), rank
    """
    results = []
    
    # IMDB trending pages
    urls = [
        ("https://www.imdb.com/chart/moviemeter/", "movie"),
        ("https://www.imdb.com/chart/tvmeter/", "tv"),
    ]
    
    for url, content_type in urls:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, "html.parser")
            
            # IMDB uses a different structure - find all links to titles
            # Strategy 1: Look for h3 elements with ipc-title__text class (most reliable)
            title_links = []
            h3_elements = soup.find_all("h3", class_="ipc-title__text")
            for h3 in h3_elements:
                link = h3.find("a", href=lambda x: x and "/title/tt" in str(x))
                if link and link.get("href"):
                    title_links.append(link)
            
            # Strategy 2: If that doesn't work, look for links with href containing "/title/tt" and chart reference
            if not title_links:
                title_links = soup.find_all("a", href=lambda x: x and "/title/tt" in str(x) and ("chtmvm" in str(x) or "chtvtv" in str(x)))
            
            # Strategy 3: Find all /title/tt links and filter by context (exclude navigation)
            if not title_links:
                all_links = soup.find_all("a", href=lambda x: x and "/title/tt" in str(x))
                # Filter out navigation/footer links
                for link in all_links:
                    href = link.get("href", "")
                    # Skip if it's in footer, header, or navigation
                    parent = link.find_parent(["footer", "header", "nav"])
                    if parent:
                        continue
                    # Only include if it has a chart reference or is in main content
                    if "chtmvm" in href or "chtvtv" in href:
                        title_links.append(link)
                    elif link.find_parent(["li", "div"]):
                        # Check if parent looks like a chart item
                        parent = link.find_parent(["li", "div"])
                        parent_text = parent.get_text().lower() if parent else ""
                        if any(word in parent_text for word in ["#", "rank", "rating", "year"]):
                            title_links.append(link)
            
            if title_links:
                for idx, title_link in enumerate(title_links[:top_n], 1):
                    try:
                        title = title_link.get_text(strip=True)
                        href = title_link.get("href", "")
                        
                        # Extract IMDB ID
                        imdb_id = None
                        if "/title/" in href:
                            parts = href.split("/title/")
                            if len(parts) > 1:
                                imdb_id = parts[1].split("/")[0]
                        
                        # Try to find year - look in parent elements
                        year = None
                        parent = title_link.find_parent(["li", "div", "td"])
                        if parent:
                            # Look for year in spans or text
                            year_text = parent.get_text()
                            # Try to extract 4-digit year
                            import re
                            year_match = re.search(r'\b(19|20)\d{2}\b', year_text)
                            if year_match:
                                try:
                                    year = int(year_match.group(0))
                                except ValueError:
                                    pass
                        
                        if title and imdb_id:
                            results.append({
                                "title": title,
                                "year": year,
                                "imdb_id": imdb_id,
                                "type": content_type,
                                "rank": idx,
                                "source": "IMDB Trending"
                            })
                    except Exception as e:
                        continue
                
                print(f"  Scraped {len([r for r in results if r['type'] == content_type])} {content_type} titles from IMDB")
                time.sleep(2)
                continue
            else:
                print(f"  Warning: Could not find any titles for {content_type}")
                continue
            
            for idx, row in enumerate(title_rows[:top_n], 1):
                try:
                    # Extract title and year
                    title_link = row.find("a")
                    if not title_link:
                        continue
                    
                    title = title_link.text.strip()
                    href = title_link.get("href", "")
                    
                    # Extract IMDB ID from href (e.g., /title/tt1234567/)
                    imdb_id = None
                    if "/title/" in href:
                        parts = href.split("/title/")
                        if len(parts) > 1:
                            imdb_id = parts[1].split("/")[0]
                    
                    # Extract year
                    year_span = row.find("span", class_="secondaryInfo")
                    year = None
                    if year_span:
                        year_text = year_span.text.strip().strip("()")
                        try:
                            year = int(year_text)
                        except ValueError:
                            pass
                    
                    results.append({
                        "title": title,
                        "year": year,
                        "imdb_id": imdb_id,
                        "type": content_type,
                        "rank": idx,
                        "source": "IMDB Trending"
                    })
                except Exception as e:
                    print(f"  Warning: Error parsing row {idx} for {content_type}: {e}")
                    continue
            
            print(f"  Scraped {len([r for r in results if r['type'] == content_type])} {content_type} titles from IMDB")
            time.sleep(2)  # Be respectful with requests
            
        except Exception as e:
            print(f"  Error scraping IMDB {content_type} trending: {e}")
            continue
    
    return results[:top_n]


def scrape_nielsen_streaming_top10() -> List[Dict]:
    """
    Scrape Nielsen's top 10 streaming data (streaming only, exclude linear TV).
    
    Note: Nielsen's website structure may change. This scraper attempts to find
    streaming-specific content and may need adjustment if the site structure changes.
    
    Returns:
        List of dicts with keys: title, platform, rank, viewership (if available)
    """
    results = []
    
    try:
        url = "https://www.nielsen.com/data-center/top-ten/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Strategy 1: Look for streaming-specific sections
        # Nielsen often has sections labeled "Streaming Originals" or similar
        streaming_keywords = ["streaming", "original", "acquired", "netflix", "disney", "hulu", "prime", "max", "paramount"]
        linear_keywords = ["broadcast", "cable", "network tv", "linear", "abc", "cbs", "nbc", "fox"]
        
        # Find all text content that might be titles
        all_text_elements = soup.find_all(["div", "span", "p", "li", "td"], string=True)
        
        potential_titles = []
        for elem in all_text_elements:
            text = elem.get_text(strip=True)
            if not text or len(text) < 3:
                continue
            
            # Skip navigation and common page elements
            skip_words = ["home", "about", "contact", "privacy", "cookie", "menu", "search", "login", "sign up"]
            if any(word in text.lower() for word in skip_words):
                continue
            
            # Skip if it's clearly linear TV
            if any(keyword in text.lower() for keyword in linear_keywords):
                continue
            
            # Look for context that suggests streaming
            parent_text = ""
            parent = elem.find_parent(["div", "section", "article"])
            if parent:
                parent_text = parent.get_text().lower()
            
            # If parent mentions streaming keywords, it's likely streaming content
            is_streaming = any(keyword in parent_text for keyword in streaming_keywords)
            
            # Also check if text looks like a title (not too long, not a sentence)
            looks_like_title = (
                len(text) < 100 and 
                not text.endswith(".") and
                text[0].isupper() if text else False
            )
            
            if (is_streaming or looks_like_title) and text not in [t["title"] for t in potential_titles]:
                # Try to extract platform from context
                platform = None
                for keyword in ["netflix", "disney", "hulu", "prime", "paramount", "max", "peacock", "apple"]:
                    if keyword in parent_text:
                        platform = keyword.title()
                        break
                
                potential_titles.append({
                    "title": text,
                    "platform": platform,
                    "rank": len(potential_titles) + 1
                })
        
        # Take top 10 unique titles
        seen_titles = set()
        for item in potential_titles[:20]:  # Check more to filter duplicates
            title_lower = item["title"].lower()
            if title_lower not in seen_titles and len(results) < 10:
                seen_titles.add(title_lower)
                results.append({
                    "title": item["title"],
                    "platform": item["platform"] or "Unknown",
                    "rank": len(results) + 1,
                    "source": "Nielsen Streaming Top 10"
                })
        
        # If we didn't find enough, try a simpler approach: look for numbered lists
        if len(results) < 5:
            numbered_items = soup.find_all(["ol", "ul"])
            for list_elem in numbered_items:
                items = list_elem.find_all("li")
                for idx, item in enumerate(items[:10], 1):
                    text = item.get_text(strip=True)
                    # Clean up text (remove numbers, bullets, etc.)
                    text = text.lstrip("0123456789. ").strip()
                    
                    if len(text) > 3 and len(text) < 100:
                        # Skip if already found or if it's navigation
                        if text.lower() not in seen_titles and not any(skip in text.lower() for skip in skip_words):
                            seen_titles.add(text.lower())
                            results.append({
                                "title": text,
                                "platform": None,
                                "rank": len(results) + 1,
                                "source": "Nielsen Streaming Top 10"
                            })
                            if len(results) >= 10:
                                break
                if len(results) >= 10:
                    break
        
        print(f"  Scraped {len(results)} titles from Nielsen streaming data")
        
        # Filter out generic navigation/service terms that aren't actual titles
        filtered_results = []
        skip_terms = [
            "audience measurement", "audiences", "cross-media", "digital measurement",
            "tv & streaming", "audio measurement", "media planning", "audience segmentation",
            "market intelligence", "scenario planning", "home", "about", "contact",
            "privacy", "cookie", "menu", "search", "login", "sign up", "nielsen"
        ]
        
        for item in results:
            title_lower = item["title"].lower()
            # Skip if it's clearly not a film/TV title
            if any(term in title_lower for term in skip_terms):
                continue
            # Skip if it's too short or looks like a service description
            if len(item["title"]) < 3 or len(item["title"]) > 100:
                continue
            filtered_results.append(item)
        
        results = filtered_results[:10]
        
        # If still no results, warn the user
        if len(results) == 0:
            print(f"  WARNING: Could not scrape Nielsen data. Website structure may have changed.")
            print(f"  You may need to manually update the scraper or check the Nielsen website.")
        
    except Exception as e:
        print(f"  Error scraping Nielsen streaming data: {e}")
        print(f"  This may be due to website changes or network issues.")
        return []
    
    return results[:10]  # Top 10 only


def convert_imdb_to_film_record(imdb_item: Dict, tmdb_client) -> Optional[Dict]:
    """
    Convert an IMDB trending item to a FilmRecord-like dict using TMDb lookup.
    
    Args:
        imdb_item: Dict with title, year, imdb_id, type, rank
        tmdb_client: TMDbClient instance
    
    Returns:
        Dict compatible with FilmRecord structure, or None if not found
    """
    try:
        # Import here to avoid circular imports
        from film_agent import film_record_from_tmdb_details
        
        # Try to find in TMDb using title and year
        title = imdb_item.get("title", "")
        year = imdb_item.get("year")
        
        if not title:
            return None
        
        # Search TMDb
        candidates = tmdb_client.search_movie(title, limit=5)
        if not candidates:
            return None
        
        # Try to match by year if available
        best_match = None
        for candidate in candidates:
            release_date = candidate.get("release_date", "")
            candidate_year = None
            if release_date:
                try:
                    candidate_year = int(release_date[:4])
                except ValueError:
                    pass
            
            if year and candidate_year and abs(candidate_year - year) <= 1:
                best_match = candidate
                break
        
        if not best_match:
            best_match = candidates[0]
        
        movie_id = best_match.get("id")
        if not movie_id:
            return None
        
        # Get full details
        details = tmdb_client.get_movie_details_with_credits(movie_id)
        
        # Convert to FilmRecord-like dict
        record = film_record_from_tmdb_details(
            details,
            country="US",  # Default to US for trending data
            location=f"IMDB Trending ({imdb_item.get('type', 'movie').upper()})",
            scheduled_dates=datetime.now().date().isoformat(),  # Current date
            programme_url=f"https://www.imdb.com/title/{imdb_item.get('imdb_id', '')}/"
        )
        
        # Add source metadata
        record_dict = {
            "title": record.title,
            "release_year": record.release_year,
            "director": record.director,
            "writers": record.writers,
            "producers": record.producers,
            "cinematographers": record.cinematographers,
            "production_designers": record.production_designers,
            "cast": record.cast,
            "genres": record.genres,
            "tmdb_id": record.tmdb_id,
            "overview": record.overview,
            "keywords": record.keywords,
            "tagline": record.tagline,
            "thematic_descriptors": record.thematic_descriptors,
            "stylistic_descriptors": record.stylistic_descriptors,
            "emotional_tone": record.emotional_tone,
            "country": record.country,
            "location": record.location,
            "start_date": record.start_date,
            "end_date": record.end_date,
            "scheduled_dates": record.scheduled_dates,
            "programme_url": record.programme_url,
            "lead_gender": record.lead_gender,
            "source_type": "IMDB Trending",
            "trending_rank": imdb_item.get("rank")
        }
        
        return record_dict
        
    except Exception as e:
        print(f"  Warning: Could not convert IMDB item '{imdb_item.get('title', 'unknown')}' to film record: {e}")
        return None


def convert_nielsen_to_film_record(nielsen_item: Dict, tmdb_client) -> Optional[Dict]:
    """
    Convert a Nielsen streaming item to a FilmRecord-like dict using TMDb lookup.
    
    Args:
        nielsen_item: Dict with title, platform, rank
        tmdb_client: TMDbClient instance
    
    Returns:
        Dict compatible with FilmRecord structure, or None if not found
    """
    try:
        # Import here to avoid circular imports
        from film_agent import film_record_from_tmdb_details
        
        title = nielsen_item.get("title", "")
        if not title:
            return None
        
        # Search TMDb
        candidates = tmdb_client.search_movie(title, limit=5)
        if not candidates:
            return None
        
        best_match = candidates[0]
        movie_id = best_match.get("id")
        if not movie_id:
            return None
        
        # Get full details
        details = tmdb_client.get_movie_details_with_credits(movie_id)
        
        # Convert to FilmRecord-like dict
        record = film_record_from_tmdb_details(
            details,
            country="US",
            location=f"Nielsen Streaming ({nielsen_item.get('platform', 'Unknown Platform')})",
            scheduled_dates=datetime.now().date().isoformat(),
            programme_url="https://www.nielsen.com/data-center/top-ten/"
        )
        
        # Add source metadata
        record_dict = {
            "title": record.title,
            "release_year": record.release_year,
            "director": record.director,
            "writers": record.writers,
            "producers": record.producers,
            "cinematographers": record.cinematographers,
            "production_designers": record.production_designers,
            "cast": record.cast,
            "genres": record.genres,
            "tmdb_id": record.tmdb_id,
            "overview": record.overview,
            "keywords": record.keywords,
            "tagline": record.tagline,
            "thematic_descriptors": record.thematic_descriptors,
            "stylistic_descriptors": record.stylistic_descriptors,
            "emotional_tone": record.emotional_tone,
            "country": record.country,
            "location": record.location,
            "start_date": record.start_date,
            "end_date": record.end_date,
            "scheduled_dates": record.scheduled_dates,
            "programme_url": record.programme_url,
            "lead_gender": record.lead_gender,
            "source_type": "Nielsen Streaming",
            "streaming_rank": nielsen_item.get("rank"),
            "streaming_platform": nielsen_item.get("platform")
        }
        
        return record_dict
        
    except Exception as e:
        print(f"  Warning: Could not convert Nielsen item '{nielsen_item.get('title', 'unknown')}' to film record: {e}")
        return None
