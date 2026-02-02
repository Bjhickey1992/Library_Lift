#!/usr/bin/env python
"""
Add 'need' field to Lionsgate library and exhibitions: for each title, call Claude
with "What viewer desires or needs are met by the film [title]?", strip any trailing
conversational/question sentence from the response, and save. Then regenerate
embeddings (including need in the embedded text) for both datasets.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

from config import get_anthropic_api_key, get_openai_api_key

NEED_PROMPT = "What viewer desires or needs are met by the film {title}? Reply with a short summary only; no follow-up questions or prompts."
CLAUDE_MODEL = "claude-sonnet-4-5"
# Patterns that suggest the last sentence is a user-facing prompt to continue the conversation
LAST_SENTENCE_PROMPT_PATTERNS = re.compile(
    r"\b(would you like|let me know|want to|would you|can i |shall i|need (more|another)|anything else|any other)\b",
    re.IGNORECASE,
)


def _strip_trailing_prompt_sentence(text: str) -> str:
    """Remove the last sentence if it is a question or conversational prompt."""
    if not text or not isinstance(text, str):
        return (text or "").strip()
    text = text.strip()
    # Split on sentence boundaries (period, question mark, exclamation + space or end)
    parts = re.split(r"(?<=[.!?])\s+", text)
    if not parts:
        return text
    last = parts[-1].strip()
    # Drop last if it ends with ? or looks like a prompt
    if last.endswith("?") or LAST_SENTENCE_PROMPT_PATTERNS.search(last):
        parts = parts[:-1]
    if not parts:
        return text
    return " ".join(parts).strip()


def _get_need_for_title(client, title: str) -> str:
    """Call Claude and return the need summary for the given film title."""
    prompt = NEED_PROMPT.format(title=title or "this film")
    try:
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = ""
        for block in message.content:
            if hasattr(block, "text"):
                raw += block.text
        return _strip_trailing_prompt_sentence(raw) if raw else ""
    except Exception as e:
        print(f"  [Claude error for '{title}']: {e}")
        return ""


def add_need_to_dataframe(df: pd.DataFrame, title_col: str, client, label: str) -> pd.DataFrame:
    """Add or fill 'need' column using Claude. title_col is the column name for the film title."""
    if "need" not in df.columns:
        df["need"] = ""
    n = len(df)
    for i, (idx, row) in enumerate(df.iterrows()):
        if pd.notna(row.get("need")) and str(row["need"]).strip():
            continue
        title = row.get(title_col)
        if pd.isna(title) or not str(title).strip():
            df.at[idx, "need"] = ""
            continue
        title = str(title).strip()
        print(f"  [{label}] {i+1}/{n}: {title[:50]}...", flush=True)
        need = _get_need_for_title(client, title)
        df.at[idx, "need"] = need
    return df


def main():
    from anthropic import Anthropic
    from openai import OpenAI

    print("=" * 80, flush=True)
    print("ADD 'NEED' FIELD (Lionsgate library + exhibitions) THEN REGENERATE EMBEDDINGS", flush=True)
    print("=" * 80, flush=True)

    # 1. Claude client and add need to both files
    api_key = get_anthropic_api_key()
    claude = Anthropic(api_key=api_key)

    lib_path = Path("lionsgate_library.xlsx")
    ex_path = Path("upcoming_exhibitions.xlsx")
    if not lib_path.exists():
        raise FileNotFoundError(f"Library not found: {lib_path}")
    if not ex_path.exists():
        raise FileNotFoundError(f"Exhibitions not found: {ex_path}")

    print("\n1. Adding 'need' to Lionsgate library...", flush=True)
    lib_df = pd.read_excel(lib_path)
    lib_df = add_need_to_dataframe(lib_df, "title", claude, "library")
    lib_df.to_excel(lib_path, index=False)
    print(f"   Saved {lib_path}", flush=True)

    print("\n2. Adding 'need' to exhibitions...", flush=True)
    ex_df = pd.read_excel(ex_path)
    ex_df = add_need_to_dataframe(ex_df, "title", claude, "exhibitions")
    ex_df.to_excel(ex_path, index=False)
    print(f"   Saved {ex_path}", flush=True)

    # 2. Regenerate embeddings (include need in text)
    openai_key = get_openai_api_key()
    openai_client = OpenAI(api_key=openai_key)

    def _library_text(row):
        parts = []
        if pd.notna(row.get("title")):
            parts.append(f"Film: {row['title']}")
        if pd.notna(row.get("release_year")):
            parts.append(f"Released: {int(row['release_year'])}")
        if pd.notna(row.get("tagline")):
            parts.append(f"Tagline: {row['tagline']}")
        if pd.notna(row.get("director")):
            parts.append(f"Directed by: {row['director']}")
        if pd.notna(row.get("writers")):
            parts.append(f"Written by: {row['writers']}")
        if pd.notna(row.get("cinematographers")):
            parts.append(f"Cinematography: {row['cinematographers']}")
        if pd.notna(row.get("production_designers")):
            parts.append(f"Production Design: {row['production_designers']}")
        if pd.notna(row.get("cast")):
            parts.append(f"Starring: {row['cast']}")
        if pd.notna(row.get("genres")):
            parts.append(f"Genres: {row['genres']}")
        if pd.notna(row.get("keywords")):
            parts.append(f"Keywords: {row['keywords']}")
        if pd.notna(row.get("overview")):
            parts.append(f"Plot: {row['overview']}")
        if pd.notna(row.get("thematic_descriptors")):
            parts.append(f"Themes: {row['thematic_descriptors']}")
        if pd.notna(row.get("stylistic_descriptors")):
            parts.append(f"Style: {row['stylistic_descriptors']}")
        if pd.notna(row.get("emotional_tone")):
            parts.append(f"Tone: {row['emotional_tone']}")
        if pd.notna(row.get("need")):
            parts.append(f"Viewer needs: {row['need']}")
        return "\n".join(parts)

    def _exhibition_text(row):
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
        if pd.notna(row.get("thematic_descriptors")):
            parts.append(f"Themes: {row['thematic_descriptors']}")
        if pd.notna(row.get("stylistic_descriptors")):
            parts.append(f"Style: {row['stylistic_descriptors']}")
        if pd.notna(row.get("emotional_tone")):
            parts.append(f"Tone: {row['emotional_tone']}")
        if pd.notna(row.get("need")):
            parts.append(f"Viewer needs: {row['need']}")
        return "\n".join(parts)

    def embed_batches(texts, batch_size=100):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                r = openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch,
                    dimensions=1536,
                )
                embeddings.extend([item.embedding for item in r.data])
            except Exception as e:
                print(f"  Embedding batch error: {e}")
                embeddings.extend([[0.0] * 1536 for _ in batch])
        return np.asarray(embeddings)

    print("\n3. Regenerating Lionsgate library embeddings (with need)...", flush=True)
    lib_texts = [_library_text(row) for _, row in lib_df.iterrows()]
    lib_emb = embed_batches(lib_texts)
    lib_meta = lib_df[["tmdb_id", "title", "release_year"]].copy()
    lib_meta.to_excel("lionsgate_library_embeddings.xlsx", index=False)
    np.save("lionsgate_library_embeddings.npy", lib_emb)
    print(f"   Saved lionsgate_library_embeddings.npy and .xlsx ({lib_emb.shape[0]} films)", flush=True)

    print("\n4. Regenerating exhibition embeddings (with need)...", flush=True)
    ex_texts = [_exhibition_text(row) for _, row in ex_df.iterrows()]
    ex_emb = embed_batches(ex_texts)
    ex_meta_cols = ["tmdb_id", "title", "release_year", "country", "location"]
    ex_meta = ex_df[[c for c in ex_meta_cols if c in ex_df.columns]].copy()
    ex_meta.to_excel("upcoming_exhibitions_embeddings.xlsx", index=False)
    np.save("upcoming_exhibitions_embeddings.npy", ex_emb)
    print(f"   Saved upcoming_exhibitions_embeddings.npy and .xlsx ({ex_emb.shape[0]} films)", flush=True)

    print("\n[SUCCESS] Need field added and embeddings updated.", flush=True)


if __name__ == "__main__":
    main()
