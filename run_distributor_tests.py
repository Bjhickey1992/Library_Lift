"""
Run 10 test prompts with increasing complexity from a Lionsgate content distributor perspective.
Someone trying to place library films in the market — specific, challenging queries.
"""
import sys
import io
from pathlib import Path

_app_root = Path(__file__).resolve().parent

# 10 prompts: increasing complexity, distributor-focused
DISTRIBUTOR_PROMPTS = [
    # 1. Simple territory + genre
    "We're pushing thrillers in the UK market this quarter. What library titles should we emphasize?",
    # 2. Venue-specific programming
    "Film Forum's audience skews art-house. Which of our titles would fit their programming best?",
    # 3. Thematic + territory
    "Show me library films that are the best thematic matches for what's playing in US theaters right now.",
    # 4. Match to specific exhibition title + territory
    "We have The Housemaid in theaters. What library titles should we pair with it for double features in France?",
    # 5. Multi-filter: genre + decade + territory + lead gender
    "Female-led dramas from the 2010s that would play well in Canadian arthouse venues.",
    # 6. Exhibition date window + genre
    "Romance or drama titles that align with exhibitions between February 15 and March 5, 2026.",
    # 7. Soft preference + thematic
    "Prioritize US exhibitions, but show library titles with strong thematic overlap to what's trending anywhere.",
    # 8. City + venue type + genre
    "Thrillers and crime films that could work at Alamo Drafthouse or similar venues in Austin.",
    # 9. Director-style match + territory
    "Library films that match the aesthetic of Sicario—tense, morally complex—for the US market.",
    # 10. Complex: time period + exhibition vibe + need-based
    "What library titles meet the need for psychological complexity and match the tone of what's playing at Northwest Film Forum in Seattle this month?",
]


def run_one(prompt: str, agent) -> dict:
    """Run one prompt, return result dict with recommendations and metadata."""
    result = agent.get_recommendations_for_query(prompt, top_n=5)
    return result


def main():
    from chatbot_agent import ChatbotAgent

    agent = ChatbotAgent(studio_name="Lionsgate", app_root=_app_root)
    results = []

    print("=" * 70, flush=True)
    print("LIONSGATE DISTRIBUTOR TEST PROMPTS — 10 INCREASING COMPLEXITY", flush=True)
    print("=" * 70, flush=True)

    for i, prompt in enumerate(DISTRIBUTOR_PROMPTS, 1):
        print(f"\n--- [{i}/10] {prompt[:60]}{'...' if len(prompt) > 60 else ''}", flush=True)
        result = run_one(prompt, agent)
        results.append({"prompt": prompt, "result": result})

        if "error" in result and result["error"]:
            print(f"  ERROR: {result['error'][:150]}", flush=True)
        else:
            recs = result.get("recommendations", [])
            print(f"  OK: {len(recs)} recs | territory={result.get('territory', 'N/A')}", flush=True)
            for j, r in enumerate(recs[:3], 1):
                ex = r.get("matched_exhibition", "")
                loc = r.get("exhibition_location", "")
                print(f"    {j}. {r.get('title', '')} ({r.get('year', '')}) -> {ex} @ {loc}", flush=True)
            if len(recs) > 3:
                print(f"    ... +{len(recs)-3} more", flush=True)

    # --- Full report ---
    report_lines = []
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("FULL RESULTS REPORT — 10 LIONSGATE DISTRIBUTOR PROMPTS")
    report_lines.append("=" * 80)

    for i, item in enumerate(results, 1):
        prompt = item["prompt"]
        result = item["result"]
        report_lines.append("")
        report_lines.append("-" * 80)
        report_lines.append(f"[{i}] {prompt}")
        report_lines.append("-" * 80)

        if "error" in result and result["error"]:
            report_lines.append(f"ERROR: {result['error']}")
        else:
            report_lines.append(f"Territory: {result.get('territory', 'N/A')} | Count: {result.get('count', 0)}")
            for note in ["territory_fallback_note", "venue_fallback_note", "exhibition_unstructured_note",
                         "genre_fallback_note", "unstructured_fallback_note", "fallback_summary"]:
                if result.get(note):
                    report_lines.append(f"Note: {result[note]}")
            report_lines.append("")
            for j, r in enumerate(result.get("recommendations", []), 1):
                report_lines.append(f"  {j}. {r.get('title', '')} ({r.get('year', '')})")
                report_lines.append(f"     Director: {r.get('director', '')} | Genres: {r.get('genres', '')}")
                report_lines.append(f"     Relevance: {r.get('relevance_score', 0):.3f}")
                report_lines.append(f"     Exhibition: {r.get('matched_exhibition', '')} at {r.get('exhibition_location', '')}")
                if r.get("reasoning"):
                    report_lines.append(f"     Why: {r['reasoning'][:120]}...")
                report_lines.append("")

    report_lines.append("=" * 80)

    # Write report
    out_path = _app_root / "distributor_test_report.txt"
    out_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n[Report written to: {out_path}]", flush=True)

    # Print summary table
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY TABLE", flush=True)
    print("=" * 70, flush=True)
    for i, item in enumerate(results, 1):
        r = item["result"]
        status = "FAIL" if r.get("error") else "OK"
        count = len(r.get("recommendations", [])) if not r.get("error") else 0
        terr = r.get("territory", "N/A")
        print(f"  {i:2}. [{status}] {count} recs | {terr} | {item['prompt'][:45]}...", flush=True)
    print("=" * 70, flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
