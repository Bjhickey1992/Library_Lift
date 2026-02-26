"""
Diagnostic script: run "What thrillers from the 2000s are relevant right now?"
and trace exhibition filter steps to find where count goes to 0.
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, date

# Ensure project root
_app_root = Path(__file__).resolve().parent
os.chdir(_app_root)
sys.path.insert(0, str(_app_root))

import pandas as pd
from query_intent_parser import QueryIntentParser, QueryIntent
from chatbot_agent import ChatbotAgent, TERRITORY_COUNTRY_ALIASES
from dataclasses import replace
import re

QUERY = "What thrillers from the 2000s are relevant right now?"


def main():
    print("=" * 70)
    print(f"DIAGNOSTIC: Query = {QUERY!r}")
    print("=" * 70)

    # Load agent and data
    print("\n[1] Loading ChatbotAgent and data...")
    try:
        agent = ChatbotAgent(studio_name="Lionsgate", app_root=_app_root)
        library_df = agent._load_library()
        exhibitions_df = agent._load_exhibitions()
        print(f"    Library: {len(library_df)} rows")
        print(f"    Exhibitions: {len(exhibitions_df)} rows")
    except Exception as e:
        print(f"    ERROR loading: {e}")
        return

    # Parse intent (no history)
    print("\n[2] Parsing intent...")
    intent = agent.query_parser.parse(QUERY, history_prompts=[], previous_intent=None)

    print("\n    --- Parsed intent (relevant fields) ---")
    print(f"    year_start: {intent.year_start}")
    print(f"    year_end: {intent.year_end}")
    print(f"    time_period: {intent.time_period!r}")
    print(f"    territory: {intent.territory!r}")
    print(f"    city: {intent.city!r}")
    print(f"    venue: {intent.venue!r}")
    print(f"    match_to_specific_film: {intent.match_to_specific_film!r}")
    print(f"    exhibition_date_start: {intent.exhibition_date_start}")
    print(f"    exhibition_date_end: {intent.exhibition_date_end}")
    print(f"    column_filters: {intent.column_filters}")
    print(f"    apply_time_period_to_exhibitions: {getattr(intent, 'apply_time_period_to_exhibitions', 'ATTR_MISSING')}")
    print(f"    filter_year_to_exhibition: {getattr(intent, 'filter_year_to_exhibition', 'ATTR_MISSING')}")

    # Step-by-step exhibition filter trace
    print("\n[3] Exhibition filter step-by-step trace")
    print("-" * 50)
    filtered_df = exhibitions_df.copy()
    initial_count = len(filtered_df)
    print(f"    Start: {initial_count} exhibitions")

    # Territory
    territory_mode = getattr(intent, "territory_mode", "hard")
    prefs = getattr(intent, "territory_preferences", None) or ([intent.territory] if intent.territory else None)
    if territory_mode == "hard" and prefs and "country" in filtered_df.columns:
        country_upper = filtered_df["country"].astype(str).str.strip().str.upper()
        mask = pd.Series(False, index=filtered_df.index)
        for ter in prefs:
            t = (ter or "").upper().strip()
            if not t:
                continue
            accepted = TERRITORY_COUNTRY_ALIASES.get(t, (t,))
            mask = mask | country_upper.isin(accepted)
        filtered_df = filtered_df[mask]
        print(f"    After TERRITORY (prefs={prefs}): {len(filtered_df)} (removed {initial_count - len(filtered_df)})")
    else:
        print(f"    After TERRITORY: skipped (prefs={prefs})")

    # City
    if intent.city and "location" in filtered_df.columns:
        before = len(filtered_df)
        loc_col = filtered_df["location"].astype(str).str
        city_lower = intent.city.lower()
        filtered_df = filtered_df[loc_col.lower().str.contains(re.escape(city_lower), na=False)]
        print(f"    After CITY ({intent.city}): {len(filtered_df)} (removed {before - len(filtered_df)})")
    else:
        print(f"    After CITY: skipped")

    # match_to_specific_film
    if intent.match_to_specific_film:
        before = len(filtered_df)
        # simplified - would need full logic
        print(f"    After MATCH_TO_FILM ({intent.match_to_specific_film}): would filter (skipping full logic)")
    else:
        print(f"    After MATCH_TO_FILM: skipped")

    # Venue
    venue_prefs = getattr(intent, "venue_preferences", None) or ([intent.venue] if intent.venue else None)
    if venue_prefs and "location" in filtered_df.columns:
        before = len(filtered_df)
        loc_col = filtered_df["location"].astype(str).str.lower()
        mask = pd.Series(False, index=filtered_df.index)
        for v in venue_prefs:
            if v:
                mask = mask | loc_col.str.contains(re.escape((v or "").strip().lower()), na=False)
        filtered_df = filtered_df[mask]
        print(f"    After VENUE ({venue_prefs}): {len(filtered_df)} (removed {before - len(filtered_df)})")
    else:
        print(f"    After VENUE: skipped")

    # exhibition_film_type
    ex_type = getattr(intent, "exhibition_film_type", None)
    if ex_type:
        before = len(filtered_df)
        # simplified
        print(f"    After EXHIBITION_FILM_TYPE ({ex_type}): would filter")
    else:
        print(f"    After EXHIBITION_FILM_TYPE: skipped")

    # filter_year_to_exhibition
    if getattr(intent, "filter_year_to_exhibition", False) and intent.year_start is not None:
        before = len(filtered_df)
        if "release_year" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["release_year"].notna() & (filtered_df["release_year"] >= intent.year_start)]
        if intent.year_end is not None and "release_year" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["release_year"].notna() & (filtered_df["release_year"] <= intent.year_end)]
        print(f"    After FILTER_YEAR_TO_EXHIBITION ({intent.year_start}-{intent.year_end}): {len(filtered_df)}")
    else:
        print(f"    After FILTER_YEAR_TO_EXHIBITION: skipped (filter_year_to_exhibition={getattr(intent, 'filter_year_to_exhibition', False)})")

    # TIME PERIOD - the key suspect
    apply_time = getattr(intent, "apply_time_period_to_exhibitions", True)
    time_would_run = apply_time and intent.time_period and not intent.match_to_specific_film and not (intent.exhibition_date_start or intent.exhibition_date_end)

    print(f"\n    --- TIME PERIOD BLOCK ---")
    print(f"    apply_time_period_to_exhibitions: {apply_time}")
    print(f"    intent.time_period: {intent.time_period!r}")
    print(f"    Block WOULD run: {time_would_run}")

    if time_would_run:
        before = len(filtered_df)
        today = datetime.now().date()
        end_date = today + timedelta(days=7)  # "now"

        def _has_date_in_range(dates_str, start_date, end_date_val):
            if pd.isna(dates_str) or not dates_str:
                return False
            for date_str in str(dates_str).split(","):
                date_str = date_str.strip()
                try:
                    date_obj = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
                    if start_date <= date_obj <= end_date_val:
                        return True
                except (ValueError, AttributeError):
                    continue
            return False

        filtered_df = filtered_df[
            filtered_df["scheduled_dates"].apply(lambda s: _has_date_in_range(s, today, end_date))
        ]
        print(f"    After TIME_PERIOD (now = {today}, +7d = {end_date}): {len(filtered_df)} (removed {before - len(filtered_df)})")
        if len(filtered_df) == 0 and before > 0:
            # Sample exhibition dates
            sample_dates = exhibitions_df["scheduled_dates"].dropna().head(3).tolist()
            print(f"    >>> EXHIBITION DATES IN DATA (sample): {sample_dates}")
            print(f"    >>> Today is {today}; exhibitions are likely in 2026 -> 0 match 'now' window")
    else:
        print(f"    TIME_PERIOD block SKIPPED (apply_time={apply_time})")

    # column_filters
    ex_column_filters = intent.column_filters
    if ex_column_filters and not getattr(intent, "apply_time_period_to_exhibitions", True):
        ex_column_filters = {k: v for k, v in ex_column_filters.items() if k.lower() != "release_year"}
    print(f"\n    --- COLUMN_FILTERS ---")
    print(f"    intent.column_filters: {intent.column_filters}")
    print(f"    ex_column_filters (after stripping release_year when lib-focused): {ex_column_filters}")

    if ex_column_filters:
        before = len(filtered_df)
        after_cf = agent._apply_column_filters_to_df(filtered_df, ex_column_filters)
        filtered_df = after_cf
        print(f"    After COLUMN_FILTERS: {len(filtered_df)} (removed {before - len(filtered_df)})")
    else:
        print(f"    COLUMN_FILTERS: skipped (none)")

    # Final: run actual _apply_exhibition_filters for comparison
    print("\n[4] Actual _apply_exhibition_filters result")
    actual = agent._apply_exhibition_filters(exhibitions_df, intent)
    print(f"    Count: {len(actual)} exhibitions")

    # Library filter for comparison
    print("\n[5] Library filter result")
    filtered_lib = agent._apply_library_filters(library_df, intent)
    print(f"    Count: {len(filtered_lib)} library films")
    if len(filtered_lib) > 0:
        print(f"    Sample: {filtered_lib['title'].iloc[0]} ({filtered_lib['release_year'].iloc[0]})")

    print("\n" + "=" * 70)
    if len(actual) == 0:
        print("CONCLUSION: Exhibition filters return 0 -> triggers 'No exhibitions matched' fallback")
    else:
        print("CONCLUSION: Exhibition filters return >0 -> no fallback message expected")
    print("=" * 70)


if __name__ == "__main__":
    main()
