"""Run Mauvais Sang test and save current results for comparison."""
import json
from chatbot_agent import ChatbotAgent

TEST_PROMPT = "I see Mauvais Sang is playing, do we have anything we can push based on that?"

agent = ChatbotAgent()
result = agent.get_dynamic_recommendations(TEST_PROMPT, top_n=5, history_prompts=[])
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
        "reasoning": (r.get("reasoning") or "")[:400],
    })
current = {
    "strategy": "deep_gate_tie_nudge",
    "query": TEST_PROMPT,
    "count": len(recs),
    "recommendations": recs,
}
with open("benchmark_mauvais_sang_current_run.json", "w", encoding="utf-8") as f:
    json.dump(current, f, indent=2, ensure_ascii=False)
print("Saved benchmark_mauvais_sang_current_run.json")
for i, r in enumerate(recs, 1):
    print(f"  {i}. {r['title']} ({r['year']})  rel={r['relevance_score']:.4f}  ex_sim={r['exhibition_similarity']:.4f}  q_sim={r['query_similarity']:.4f}")
