"""
Run 40 test prompts (10 original + 30 more) with varied metadata and mixes of fields.
Captures fallbacks and failures and produces an aggregated report.
"""
import sys
import io
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Any

_app_root = Path(__file__).resolve().parent

# 40 prompts: different data fields and mixes (territory, city, venue, genre, time, date, film match, etc.)
TEST_PROMPTS = [
    # --- Original 10 ---
    "show me comedies to highlight in france",
    "thrillers in Paris this month",
    "give me 5 library titles that match what's showing at Film Forum",
    "romance films in the UK",
    "documentaries playing in Canada",
    "female-led action films in the US",
    "what's trending in theaters right now in Austin",
    "films similar to Sicario for the US market",
    "comedies and dramas playing on 2026-03-15",
    "edgy art-house movies in Berlin",
    # --- Territory + genre / territory only ---
    "what library titles should we emphasize in the US this month?",
    "give me 7 thrillers for the UK",
    "comedies to play in Mexico",
    "drama recommendations for Canada",
    "action films in France",
    # --- City / city + genre / city + time ---
    "what's showing in London?",
    "romance films in New York",
    "comedies and dramas in Seattle this week",
    "documentaries in San Francisco",
    "thrillers in Chicago",
    # --- Venue ---
    "titles that match what's doing well at Metrograph",
    "library films that fit the programming at BAM",
    "recommendations for Alamo Drafthouse",
    # --- Time + territory/city ---
    "what's trending in theaters in the US right now",
    "thrillers playing this week in the UK",
    "comedies in Austin this month",
    # --- Exhibition date ---
    "romance and drama playing on 2026-02-28",
    "action films with exhibitions on 2026-03-10",
    "give me 10 titles playing between 2026-02-15 and 2026-02-25",
    # --- Match to specific film ---
    "films similar to Inception for current exhibitions",
    "library titles relevant to The Housemaid",
    "something like Bone Temple in the US",
    # --- Genre combos / single ---
    "horror and thriller in the UK",
    "sci-fi and drama for France",
    "just comedies, 5 titles",
    # --- Lead gender / year ---
    "male-led action in the US",
    "female director films in the UK",
    "comedies from the 2010s for Canada",
    # --- Soft preference / refinement ---
    "prioritize US exhibitions but show others too",
    "narrow to Paris only",
    "only documentaries",
]


@dataclass
class TestRun:
    prompt: str
    error: Optional[str] = None
    count: int = 0
    territory: Optional[str] = None
    territory_fallback_note: Optional[str] = None
    venue_fallback_note: Optional[str] = None
    exhibition_unstructured_note: Optional[str] = None
    genre_fallback_note: Optional[str] = None
    unstructured_fallback_note: Optional[str] = None
    fallback_summary: Optional[str] = None
    log_snippet: str = ""
    intent_snapshot: str = ""
    fallbacks: List[str] = field(default_factory=list)
    failure_reason: Optional[str] = None


def run_one(prompt: str, agent, real_stdout: io.TextIOBase) -> TestRun:
    """Run one prompt, capture stdout and result, return TestRun."""
    run = TestRun(prompt=prompt)
    buf = io.StringIO()
    class Tee:
        def __init__(self, cap: io.StringIO, out: io.TextIOBase):
            self.cap = cap
            self.out = out
        def write(self, s: str):
            self.cap.write(s)
            self.out.write(s)
        def flush(self):
            self.cap.flush()
            self.out.flush()
    old_stdout = sys.stdout
    sys.stdout = Tee(buf, real_stdout)
    try:
        result = agent.get_recommendations_for_query(prompt, top_n=5)
    finally:
        sys.stdout = old_stdout
    log = buf.getvalue()

    run.log_snippet = log
    if "Parsed intent:" in log:
        for line in log.splitlines():
            if "Parsed intent:" in line:
                run.intent_snapshot = line.strip()
                break
    if "After library filters:" in log:
        for line in log.splitlines():
            if "After library filters:" in line:
                run.log_snippet = line.strip() + "; "
                break
    if "After exhibition filters:" in log:
        for line in log.splitlines():
            if "After exhibition filters:" in line:
                run.log_snippet += line.strip()
                break

    if "error" in result and result["error"]:
        run.error = result["error"]
        run.failure_reason = "Agent returned error"
        return run

    run.count = result.get("count", 0) or len(result.get("recommendations", []))
    run.territory = getattr(result.get("intent"), "territory", None) or result.get("territory")

    run.territory_fallback_note = result.get("territory_fallback_note")
    run.venue_fallback_note = result.get("venue_fallback_note")
    run.exhibition_unstructured_note = result.get("exhibition_unstructured_note")
    run.genre_fallback_note = result.get("genre_fallback_note")
    run.unstructured_fallback_note = result.get("unstructured_fallback_note")
    run.fallback_summary = result.get("fallback_summary")

    if run.territory_fallback_note:
        run.fallbacks.append("territory_fallback")
    if run.venue_fallback_note:
        run.fallbacks.append("venue_fallback")
    if run.exhibition_unstructured_note:
        run.fallbacks.append("exhibition_unstructured")
    if run.genre_fallback_note:
        run.fallbacks.append("genre_fallback")
    if run.unstructured_fallback_note:
        run.fallbacks.append("unstructured_fallback")

    # Detect fallback from log when structured filters yielded 0 then we used fallback
    if "Exhibition unstructured fallback:" in log and "0 exhibitions" in log:
        run.fallbacks.append("exhibition_structured_yielded_0_then_unstructured")
    if "Genre fallback:" in log:
        run.fallbacks.append("genre_exact_0_then_text_terms")

    return run


def main():
    from chatbot_agent import ChatbotAgent
    agent = ChatbotAgent(studio_name="Lionsgate", app_root=_app_root)
    runs: List[TestRun] = []
    n = len(TEST_PROMPTS)
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n--- Test {i}/{n}: {prompt[:55]}{'...' if len(prompt) > 55 else ''}", flush=True)
        run = run_one(prompt, agent, sys.stdout)
        runs.append(run)
        if run.error:
            print(f"  FAIL: {run.error[:200]}", flush=True)
        else:
            print(f"  OK: {run.count} recs, territory={run.territory}, fallbacks={run.fallbacks}", flush=True)

    # --- Aggregated report ---
    failed = [r for r in runs if r.error]
    with_fallbacks = [r for r in runs if not r.error and r.fallbacks]
    clean = [r for r in runs if not r.error and not r.fallbacks]

    report = []
    report.append("=" * 70)
    report.append(f"AGGREGATED REPORT: {len(runs)} PROMPT TESTS")
    report.append("=" * 70)
    report.append("")
    report.append(f"Total: {len(runs)} | Clean (no error, no fallback): {len(clean)} | With fallbacks: {len(with_fallbacks)} | Failures: {len(failed)}")
    report.append("")

    if failed:
        report.append("--- FAILURES ---")
        for r in failed:
            report.append(f"  Prompt: \"{r.prompt}\"")
            report.append(f"  Reason: {r.failure_reason}")
            report.append(f"  Error: {(r.error or '')[:300]}")
            report.append("")
        report.append("")

    if with_fallbacks:
        report.append("--- FALLBACKS / LLM DEFAULTS TO BASELINE ---")
        for r in with_fallbacks:
            report.append(f"  Prompt: \"{r.prompt}\"")
            report.append(f"  Fallbacks: {r.fallbacks}")
            if r.territory_fallback_note:
                report.append(f"    Territory: {r.territory_fallback_note[:200]}")
            if r.venue_fallback_note:
                report.append(f"    Venue: {r.venue_fallback_note[:200]}")
            if r.exhibition_unstructured_note:
                report.append(f"    Exhibition unstructured: {r.exhibition_unstructured_note[:200]}")
            if r.genre_fallback_note:
                report.append(f"    Genre: {r.genre_fallback_note[:200]}")
            if r.unstructured_fallback_note:
                report.append(f"    Unstructured: {r.unstructured_fallback_note[:200]}")
            report.append(f"    Log: {r.log_snippet[:180]}")
            report.append("")
        report.append("")

    report.append("--- WHY FALLBACKS OCCURRED (summary) ---")
    reasons = []
    for r in with_fallbacks + failed:
        if r.territory_fallback_note:
            reasons.append("Territory: no exhibitions in requested region → expanded to all exhibitions")
        if r.venue_fallback_note:
            reasons.append("Venue: no exhibitions at requested venue → removed venue filter")
        if r.exhibition_unstructured_note:
            reasons.append("Exhibition: structured filters (territory/city/date) matched 0 → matched query terms in location/title/description")
        if r.genre_fallback_note:
            reasons.append("Genre: no exact genre match in library → matched query terms in plot/keywords/themes")
        if r.unstructured_fallback_note:
            reasons.append("Library: no films matched structured filters → matched wording in plot/themes/need")
        if r.error and "no library" in (r.error or "").lower():
            reasons.append("Failure: no library films matched filters (and unstructured fallback had no candidates)")
        if r.error and "no exhibitions" in (r.error or "").lower():
            reasons.append("Failure: no exhibitions matched filters after all fallbacks")
    for reason in sorted(set(reasons)):
        report.append(f"  • {reason}")
    report.append("")
    report.append("--- PROMPT LIST ---")
    for i, p in enumerate(TEST_PROMPTS, 1):
        r = runs[i - 1]
        status = "FAIL" if r.error else ("FALLBACK" if r.fallbacks else "OK")
        report.append(f"  {i}. [{status}] {p}")
    report.append("")
    report.append("=" * 70)

    def safe(s: str) -> str:
        return s.replace("\u2265", ">=").replace("\u2014", "-").replace("\u2019", "'") if s else s
    report_safe = [safe(line) for line in report]
    text = "\n".join(report_safe)
    out_path = _app_root / "prompt_test_report.txt"
    out_path.write_text(text, encoding="utf-8")
    # Print summary only (full report may contain Unicode that breaks Windows console)
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY: " + f"Clean={len(clean)}, With fallbacks={len(with_fallbacks)}, Failures={len(failed)}", flush=True)
    print("Report written to: " + str(out_path), flush=True)
    print("=" * 70, flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
