"""
Test 10 randomly varied new-search prompts; record time and success/failure per prompt.
"""
import time
import json
from chatbot_agent import ChatbotAgent

# 10 varied new-search prompts (diverse: genre, territory, date, match-to-film, vibe, etc.)
PROMPTS = [
    "Show me thrillers we can push in the UK right now.",
    "I want romance films that match what's playing on Valentine's Day 2026.",
    "Edgy art-house movies are hitting this time of year, how can we capitalize on that?",
    "New search: sci-fi with female leads for the US market.",
    "What library titles pair well with Mauvais Sang?",
    "Best action movies from the 2010s to emphasize in Los Angeles.",
    "Show me dramas with strong thematic depth for current exhibitions.",
    "Comedy films that fit the vibe of what's playing in February 2026.",
    "I need something like 28 Years Later: The Bone Temple—do we have anything?",
    "Recommend horror or suspense we can highlight for Doc Films in Chicago.",
]

def run_test():
    agent = ChatbotAgent()
    results = []
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n[{i}/10] {prompt[:60]}...")
        start = time.perf_counter()
        try:
            out = agent.get_dynamic_recommendations(prompt, top_n=5, history_prompts=[])
            elapsed = time.perf_counter() - start
            failed = "error" in out and out.get("error")
            rec_count = len(out.get("recommendations", []))
            results.append({
                "prompt": prompt,
                "prompt_num": i,
                "elapsed_seconds": round(elapsed, 2),
                "success": not failed,
                "recommendation_count": rec_count if not failed else 0,
                "error": out.get("error", None) if failed else None,
                "fallback_notes": {
                    "territory": out.get("territory_fallback_note"),
                    "genre": out.get("genre_fallback_note"),
                    "unstructured": out.get("unstructured_fallback_note"),
                },
            })
            status = "FAIL" if failed else f"OK ({rec_count} recs)"
            print(f"    Time: {elapsed:.2f}s  Status: {status}")
            if failed and out.get("error"):
                print(f"    Error: {out['error'][:200]}...")
        except Exception as e:
            elapsed = time.perf_counter() - start
            results.append({
                "prompt": prompt,
                "prompt_num": i,
                "elapsed_seconds": round(elapsed, 2),
                "success": False,
                "recommendation_count": 0,
                "error": str(e),
                "fallback_notes": None,
            })
            print(f"    Time: {elapsed:.2f}s  Status: EXCEPTION - {e}")
    return results

if __name__ == "__main__":
    print("Testing 10 new-search prompts (time + success/failure)...")
    results = run_test()
    total_time = sum(r["elapsed_seconds"] for r in results)
    success_count = sum(1 for r in results if r["success"])
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total time: {total_time:.2f}s  |  Success: {success_count}/10  |  Failed: {10 - success_count}/10")
    print("\nDETAILED RESULTS (for copy/paste):")
    print(json.dumps(results, indent=2, default=str))
