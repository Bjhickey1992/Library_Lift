"""
Run 10 random test prompts and check if "I, Frankenstein" appears in any recommendations.
Set N_PROMPTS=3 (etc.) to run fewer prompts.
"""
import os
import sys
from chatbot_agent import ChatbotAgent

TARGET_TITLE = "I, Frankenstein"
TEST_PROMPTS = [
    "What action or thriller titles should we emphasize in the US?",
    "Give me 5 horror films that match current theater trends.",
    "Recommend library titles similar to what's playing now.",
    "What are the best films to emphasize this month?",
    "Thrillers with strong male leads.",
    "Give me 7 titles that match current exhibitions.",
    "What's trending in theaters right now?",
    "Sci-fi or fantasy library titles for the US market.",
    "Dark action films we should push.",
    "Recommendations for Canada based on current trends.",
]


def main():
    print("Loading ChatbotAgent (Lionsgate)...")
    try:
        agent = ChatbotAgent(studio_name="Lionsgate")
    except Exception as e:
        print(f"Failed to init agent: {e}")
        sys.exit(1)
    n_prompts = int(os.environ.get("N_PROMPTS", "10"))
    prompts_to_run = TEST_PROMPTS[:n_prompts]
    print(f"Running {len(prompts_to_run)} test prompts...\n")
    found_in = []
    for i, prompt in enumerate(prompts_to_run, 1):
        print(f"  {i}. {prompt[:60]}...")
        try:
            result = agent.get_recommendations_for_query(
                prompt,
                top_n=5,
                min_similarity=0.5,
                max_similarity=0.7,
            )
        except Exception as e:
            print(f"     Error: {e}")
            continue
        recs = result.get("recommendations") or []
        titles = [r.get("title") or "" for r in recs]
        if any(TARGET_TITLE.lower() in t.lower() or t.lower() in TARGET_TITLE.lower() for t in titles):
            found_in.append((prompt, titles))
            print(f"     --> FOUND '{TARGET_TITLE}' in results: {titles}")
        else:
            print(f"     --> Top titles: {titles[:3]}")
    print("\n" + "=" * 60)
    if found_in:
        print(f"'{TARGET_TITLE}' appeared in {len(found_in)} of {len(prompts_to_run)} prompts:")
        for p, titles in found_in:
            print(f"  - {p[:50]}...")
    else:
        print(f"'{TARGET_TITLE}' did not appear in any of the {len(prompts_to_run)} test prompts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
