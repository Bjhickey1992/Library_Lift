"""One-off test: run a prompt and print top 5 recommendations."""
import sys
from pathlib import Path

# Project root for path resolution (same as streamlit app)
_app_root = Path(__file__).resolve().parent

from chatbot_agent import ChatbotAgent

def main():
    query = "show me comedies to highlight in france"
    agent = ChatbotAgent(studio_name="Lionsgate", app_root=_app_root)
    result = agent.get_recommendations_for_query(query, top_n=5)
    if "error" in result:
        print("ERROR:", result["error"], file=sys.stderr)
        sys.exit(1)
    recs = result.get("recommendations", [])
    print(f"Query: {query}")
    print(f"Territory: {result.get('territory', 'N/A')}")
    print(f"Results: {len(recs)}\n")
    for i, r in enumerate(recs, 1):
        print(f"{i}. {r.get('title', '')} ({r.get('year', '')})")
        print(f"   Director: {r.get('director', '')}")
        print(f"   Genres: {r.get('genres', '')}")
        print(f"   Relevance: {r.get('relevance_score', 0):.3f}")
        print(f"   Matched exhibition: {r.get('matched_exhibition', '')} at {r.get('exhibition_location', '')}")
        print(f"   Dates: {r.get('exhibition_dates', '')}")
        print()
    return 0

if __name__ == "__main__":
    sys.exit(main())
