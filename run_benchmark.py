"""
Run the Mauvais Sang test prompt and save results as benchmark JSON.
Usage: python run_benchmark.py [legacy|deep]  (default: legacy)
Output: benchmark_mauvais_sang_legacy.json or benchmark_mauvais_sang_deep.json
"""
import json
import os
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

TEST_PROMPT = "I see Mauvais Sang is playing, do we have anything we can push based on that?"


def main():
    strategy = (sys.argv[1] if len(sys.argv) > 1 else "legacy").lower()
    if strategy not in ("legacy", "deep"):
        strategy = "legacy"
    out_file = f"benchmark_mauvais_sang_{strategy}.json"

    from chatbot_agent import ChatbotAgent

    agent = ChatbotAgent(studio_name="Lionsgate")
    agent._scoring_strategy = "deep_gate_tie_nudge" if strategy == "deep" else "legacy"

    print(f"Running test prompt with strategy={strategy}...")
    result = agent.get_recommendations_for_query(TEST_PROMPT, top_n=5)

    # Serialize for benchmark (strip non-JSON fields if any)
    recs = []
    for r in result.get("recommendations", []):
        recs.append({
            "title": r.get("title"),
            "year": r.get("year"),
            "director": r.get("director"),
            "relevance_score": r.get("relevance_score"),
            "exhibition_similarity": r.get("exhibition_similarity"),
            "query_similarity": r.get("query_similarity"),
            "matched_exhibition": r.get("matched_exhibition"),
            "reasoning": r.get("reasoning", "")[:500],
        })
    payload = {
        "strategy": strategy,
        "query": TEST_PROMPT,
        "error": result.get("error"),
        "count": result.get("count", 0),
        "recommendations": recs,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
