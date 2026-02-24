#!/usr/bin/env python
"""
Fix the Safe exhibition entry: replace incorrect data (Boaz Yakin 2012) with
correct data from TMDB (Todd Haynes 1995), populate need, then re-embed.
"""
import json
import re
from pathlib import Path

import pandas as pd
import numpy as np
from anthropic import Anthropic
from openai import OpenAI

from config import get_anthropic_api_key, get_openai_api_key, get_tmdb_api_key
from film_agent import TMDbClient, film_record_from_tmdb_details

# TMDB ID for Safe (1995), Todd Haynes
SAFE_1995_TMDB_ID = 32646

# Exhibition columns we update from TMDB (preserve location, country, scheduled_dates, etc.)
TMDB_UPDATE_COLS = [
    "tmdb_id", "title", "release_year", "director", "writers", "producers",
    "cinematographers", "production_designers", "cast", "genres", "overview",
    "keywords", "tagline",
]

NEED_PROMPT = "What viewer desires or needs are met by the film {title}? Reply with a short summary only; no follow-up questions or prompts."
LAST_SENTENCE_PROMPT_PATTERNS = re.compile(
    r"\b(would you like|let me know|want to|would you|can i |shall i|need (more|another)|anything else|any other)\b",
    re.IGNORECASE,
)


def _strip_trailing_prompt_sentence(text: str) -> str:
    if not text or not isinstance(text, str):
        return (text or "").strip()
    text = text.strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    if not parts:
        return text
    last = parts[-1].strip()
    if last.endswith("?") or LAST_SENTENCE_PROMPT_PATTERNS.search(last):
        parts = parts[:-1]
    if not parts:
        return text
    return " ".join(parts).strip()


def main():
    ex_path = Path("upcoming_exhibitions.xlsx")
    if not ex_path.exists():
        raise FileNotFoundError(f"Exhibitions file not found: {ex_path}")

    print("=" * 70)
    print("FIX SAFE EXHIBITION ENTRY (Todd Haynes 1995)")
    print("=" * 70)

    tmdb = TMDbClient(tmdb_api_key=get_tmdb_api_key())
    ex_df = pd.read_excel(ex_path)

    # Find Safe row (Northwest Film Forum)
    mask = (
        ex_df["title"].astype(str).str.strip().str.lower() == "safe"
    ) & (
        ex_df["location"].astype(str).str.contains("Northwest Film Forum", case=False, na=False)
    )
    safe_idx = ex_df[mask].index
    if len(safe_idx) == 0:
        # Fallback: any Safe
        safe_idx = ex_df[ex_df["title"].astype(str).str.strip().str.lower() == "safe"].index
    if len(safe_idx) == 0:
        raise ValueError("No 'Safe' row found in exhibition file")
    row_idx = safe_idx[0]

    print(f"\n1. Fetching Safe (1995) from TMDB (id={SAFE_1995_TMDB_ID})...")
    details = tmdb.get_movie_details_with_credits(SAFE_1995_TMDB_ID)
    record = film_record_from_tmdb_details(
        details,
        country=ex_df.at[row_idx, "country"],
        location=ex_df.at[row_idx, "location"],
        scheduled_dates=ex_df.at[row_idx, "scheduled_dates"],
        programme_url=ex_df.at[row_idx, "programme_url"],
    )

    # Map FilmRecord to DataFrame row
    for col in TMDB_UPDATE_COLS:
        val = getattr(record, col, None)
        if val is not None:
            ex_df.at[row_idx, col] = val

    print(f"   Title: {record.title} ({record.release_year})")
    print(f"   Director: {record.director}")
    print(f"   Genres: {record.genres}")

    # 2. Generate thematic, stylistic, emotional_tone via LLM
    print("\n2. Generating thematic/stylistic/emotional descriptors...")
    openai_client = OpenAI(api_key=get_openai_api_key())
    film_desc_parts = [
        f"Title: {record.title}",
        f"Year: {record.release_year}",
        f"Director: {record.director}",
        f"Genres: {record.genres}",
        f"Plot: {record.overview or ''}",
    ]
    if record.keywords:
        film_desc_parts.append(f"Keywords: {record.keywords}")
    if record.tagline:
        film_desc_parts.append(f"Tagline: {record.tagline}")
    film_description = "\n".join(film_desc_parts)

    prompt = f"""Analyze this film and provide concise descriptors:

{film_description}

Provide a JSON response with:
1. "thematic_descriptors": 3-5 key themes (e.g., "alienation, urban isolation, existential crisis, betrayal, psychological complexity")
2. "stylistic_descriptors": 1-2 sentences describing cinematic style (e.g., "minimalist cinematography, slow-paced, contemplative, non-linear narrative")
3. "emotional_tone": 1 sentence describing emotional atmosphere (e.g., "melancholic, introspective, existential, darkly humorous")

Return ONLY valid JSON: {{"thematic_descriptors": "...", "stylistic_descriptors": "...", "emotional_tone": "..."}}"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a film analysis expert. Provide concise, accurate descriptors."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=300,
        )
        data = json.loads(response.choices[0].message.content)
        ex_df.at[row_idx, "thematic_descriptors"] = data.get("thematic_descriptors", "")
        ex_df.at[row_idx, "stylistic_descriptors"] = data.get("stylistic_descriptors", "")
        ex_df.at[row_idx, "emotional_tone"] = data.get("emotional_tone", "")
        print(f"   Themes: {data.get('thematic_descriptors', '')[:80]}...")
    except Exception as e:
        print(f"   [WARN] LLM enrichment failed: {e}")

    # 3. Generate need via Claude
    print("\n3. Generating need (viewer desires) via Claude...")
    claude = Anthropic(api_key=get_anthropic_api_key())
    need_prompt = NEED_PROMPT.format(title=record.title)
    try:
        msg = claude.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=256,
            messages=[{"role": "user", "content": need_prompt}],
        )
        raw = ""
        for block in msg.content:
            if hasattr(block, "text"):
                raw += block.text
        need = _strip_trailing_prompt_sentence(raw) if raw else ""
        ex_df.at[row_idx, "need"] = need
        print(f"   Need: {need[:100]}...")
    except Exception as e:
        print(f"   [WARN] Claude need failed: {e}")

    # 4. Lead gender (from TMDB)
    print("\n4. Fetching lead gender from TMDB...")
    try:
        credits = details.get("credits") or {}
        cast = credits.get("cast") or []
        if cast:
            lead_id = cast[0].get("id")
            if lead_id:
                person = tmdb.get_person_details(lead_id)
                gender = person.get("gender")
                if gender == 1:
                    ex_df.at[row_idx, "lead_gender"] = "female"
                elif gender == 2:
                    ex_df.at[row_idx, "lead_gender"] = "male"
                else:
                    ex_df.at[row_idx, "lead_gender"] = None
                print(f"   Lead: {cast[0].get('name')} ({ex_df.at[row_idx, 'lead_gender'] or 'unknown'})")
    except Exception as e:
        print(f"   [WARN] Lead gender: {e}")

    # 5. Save Excel
    print("\n5. Saving updated exhibitions Excel...")
    ex_df.to_excel(ex_path, index=False)
    print(f"   Saved to {ex_path}")

    # 6. Regenerate embeddings
    print("\n6. Regenerating exhibition embeddings...")
    from generate_exhibition_embeddings import generate_exhibition_embeddings

    generate_exhibition_embeddings(
        exhibitions_path=str(ex_path),
        npy_path="upcoming_exhibitions_embeddings.npy",
        metadata_xlsx_path="upcoming_exhibitions_embeddings.xlsx",
    )
    print("   [OK] Embeddings regenerated")

    print("\n" + "=" * 70)
    print("[SUCCESS] Safe exhibition fixed and re-embedded")
    print("=" * 70)


if __name__ == "__main__":
    main()
