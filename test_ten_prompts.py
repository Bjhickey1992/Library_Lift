"""
Test 10 randomly varied new-search prompts; record time and success/failure per prompt.
"""
import time
import json
from chatbot_agent import ChatbotAgent

# 10 new varied new-search prompts (diverse: genre, territory, date, match-to-film, vibe, etc.)
PROMPTS = [
    "What do we have that pairs with Eternal Sunshine of the Spotless Mind?",
    "Crime dramas from the 90s for the US.",
    "Romantic comedies playing in Austin this February.",
    "New search: something dark and existential for current exhibitions.",
    "I want family-friendly titles we can push in Paris.",
    "Best library matches for what's showing at American Cinematheque.",
    "Mystery or thriller from the 2000s to emphasize right now.",
    "Films with strong female leads that fit the current market.",
    "What can we recommend for the Love Film Festival?",
    "Indie or drama that matches the vibe of Punch-Drunk Love.",
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
