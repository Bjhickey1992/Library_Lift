"""Quick test of filter fixes: venue, time-period, date parsing, territory."""
from pathlib import Path

from chatbot_agent import ChatbotAgent
from query_intent_parser import QueryIntentParser

agent = ChatbotAgent(studio_name="Lionsgate", app_root=Path(__file__).resolve().parent)
parser = QueryIntentParser()
df = agent._load_exhibitions()

# Test 1: Film Forum (venue vs film - match_to_specific_film should be None, venue set)
p1 = "Film Forum's audience skews art-house. Which of our titles would fit their programming best?"
intent1 = parser.parse(p1)
print("1. Film Forum:")
print("   venue:", intent1.venue, "| match_to_specific_film:", intent1.match_to_specific_film)
f1 = agent._apply_exhibition_filters(df, intent1)
print("   exhibitions:", len(f1), "(expected 8 with venue+column_filter safeguard)")

# Test 2: Date parsing
p2 = "exhibitions between February 15 and March 5, 2026"
intent2 = parser.parse(p2)
print("\n2. Date range:")
print("   ex_start:", intent2.exhibition_date_start, "| ex_end:", intent2.exhibition_date_end)
f2 = agent._apply_exhibition_filters(df, intent2)
print("   exhibitions:", len(f2), "(expected 100+)")

# Test 3: UK + quarter
p3 = "thrillers in the UK market this quarter"
intent3 = parser.parse(p3)
print("\n3. UK + quarter:")
print("   territory:", intent3.territory, "| time_period:", intent3.time_period)
f3 = agent._apply_exhibition_filters(df, intent3)
print("   exhibitions:", len(f3), "(expected 8+ with fallback)")

# Test 4: Germany (territory only - no thriller so no genre column_filter on exhibitions)
p4a = "films in Germany"  # No genre - territory only
intent4a = parser.parse(p4a)
print("\n4. Germany (territory only):")
print("   territory:", intent4a.territory, "| column_filters:", getattr(intent4a, "column_filters", None))
f4a = agent._apply_exhibition_filters(df, intent4a)
print("   exhibitions:", len(f4a), "(expected 13)")

print("\nDone.")
