#!/usr/bin/env python
"""Deduplicate exhibition metadata: keep one row per (title, venue). Only removes rows where BOTH title AND venue match."""

import pandas as pd
from pathlib import Path

EXHIBITIONS_PATH = "upcoming_exhibitions.xlsx"


def dedupe_exhibition_by_title_venue(
    path: str = EXHIBITIONS_PATH,
    *,
    title_col: str = "title",
    venue_col: str = "programme_url",
    keep: str = "first",
    inplace: bool = True,
) -> int:
    """
    Remove duplicate rows only when the same film title AND the same venue both match.
    Keeps one row per (title, venue). Same title at different venues is kept.

    Uses programme_url as venue by default (one row per title per programme_url).
    Only rows that are true duplicates (same title + same venue id) are removed.

    Args:
        path: Path to the exhibition Excel file.
        title_col: Column name for film title.
        venue_col: Column that uniquely identifies the venue (default "programme_url").
        keep: Which duplicate to keep ('first', 'last').
        inplace: If True, overwrite the file.

    Returns:
        Number of rows after deduplication.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Exhibition file not found: {path}")
    df = pd.read_excel(path)
    if title_col not in df.columns or venue_col not in df.columns:
        raise ValueError(
            f"Columns {title_col!r} and/or {venue_col!r} not found. "
            f"Available: {list(df.columns)}"
        )
    before = len(df)
    df_deduped = df.drop_duplicates(subset=[title_col, venue_col], keep=keep)
    after = len(df_deduped)
    if inplace:
        df_deduped.to_excel(path, index=False)
    return after


if __name__ == "__main__":
    path = EXHIBITIONS_PATH
    before = len(pd.read_excel(path))
    after = dedupe_exhibition_by_title_venue(path, inplace=True)
    print(f"Deduplicated by (title, programme_url): {before} -> {after} rows.")
    print(f"Removed {before - after} duplicate rows.")
