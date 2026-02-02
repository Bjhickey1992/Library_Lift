"""
Test 10 unusual / limit-testing prompts; record time, success/failure, and error reason.
Uses get_recommendations_for_query (Lionsgate, same as app).
"""
import time
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chatbot_agent import ChatbotAgent

# 10 unusual prompts designed to stress different parts of the system
PROMPTS = [
    "That film with the bridge and the lovers—do we have anything like it?",
    "Movies for people who want to feel devastated but in a beautiful way.",
    "New search: library titles that match something playing on Valentine's Day 2026.",
    "Show me spy movies to emphasize in the US.",
    "What can we push in Berlin that feels like late-night European art house?",
    "New search: only films from the 1980s with female directors.",
    "We have Bad Blood playing—anything in our library to pair?",
    "Audiences who love slow cinema and long takes—what do we have?",
    "Romance or drama from the 70s or 80s, playing in London, bittersweet tone.",
    "Films similar to whatever is doing well at Film Forum this month.",
]


def run_test():
    agent = ChatbotAgent(studio_name="Lionsgate")
    results = []
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n[{i}/10] {prompt[:70]}{'...' if len(prompt) > 70 else ''}")
        start = time.perf_counter()
        try:
            out = agent.get_recommendations_for_query(prompt, top_n=5)
            elapsed = time.perf_counter() - start
            failed = "error" in out and out.get("error")
            rec_count = len(out.get("recommendations", []))
            why_failed = None
            if failed:
                why_failed = (out.get("error") or "").strip()
                if out.get("suggestion"):
                    why_failed += " [Suggestion: " + str(out.get("suggestion")) + "]"
            results.append({
                "prompt_num": i,
                "prompt": prompt,
                "elapsed_seconds": round(elapsed, 2),
                "success": not failed,
                "recommendation_count": rec_count,
                "error": why_failed if failed else None,
            })
            status = "FAIL" if failed else f"OK ({rec_count} recs)"
            print(f"    Time: {elapsed:.2f}s  Status: {status}")
            if failed and why_failed:
                print(f"    Why: {why_failed[:250]}")
        except Exception as e:
            elapsed = time.perf_counter() - start
            results.append({
                "prompt_num": i,
                "prompt": prompt,
                "elapsed_seconds": round(elapsed, 2),
                "success": False,
                "recommendation_count": 0,
                "error": f"Exception: {str(e)}",
            })
            print(f"    Time: {elapsed:.2f}s  Status: EXCEPTION")
            print(f"    Why: {e}")
    return results


if __name__ == "__main__":
    print("Testing 10 unusual/limit-testing prompts...")
    results = run_test()
    total_time = sum(r["elapsed_seconds"] for r in results)
    success_count = sum(1 for r in results if r["success"])
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total time: {total_time:.2f}s  |  Success: {success_count}/10  |  Failed: {10 - success_count}/10")
    print("\nDETAILED RESULTS:")
    print(json.dumps(results, indent=2, default=str))
