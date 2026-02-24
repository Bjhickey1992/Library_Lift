"""One-off test: run prompts and print recommendations."""
import sys
from pathlib import Path

# Project root for path resolution (same as streamlit app)
_app_root = Path(__file__).resolve().parent

from chatbot_agent import ChatbotAgent

PROMPTS = [
    "female-led action films in the US",
    "male-led comedies for the UK",
    "give me 5 thrillers with female leads",
]

def main():
    agent = ChatbotAgent(studio_name="Lionsgate", app_root=_app_root)
    for i, query in enumerate(PROMPTS, 1):
        result = agent.get_recommendations_for_query(query, top_n=5)
        if "error" in result:
            print(f"\n[{i}] ERROR: {result['error']}", file=sys.stderr)
            continue
        recs = result.get("recommendations", [])
        print(f"\n{'='*60}")
        print(f"[{i}] Query: {query}")
        print(f"    Territory: {result.get('territory', 'N/A')} | Results: {len(recs)}")
        print("="*60)
        for j, r in enumerate(recs, 1):
            print(f"  {j}. {r.get('title', '')} ({r.get('year', '')})")
            print(f"     Director: {r.get('director', '')} | Genres: {r.get('genres', '')}")
            print(f"     Relevance: {r.get('relevance_score', 0):.3f}")
            print(f"     Exhibition: {r.get('matched_exhibition', '')} at {r.get('exhibition_location', '')}")
        print()
    return 0

if __name__ == "__main__":
    sys.exit(main())
