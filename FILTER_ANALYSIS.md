# Territory, Venue, and Date Filtering — Root Causes & Improvements

## Summary

Exhibition filtering often yields 0 results because of:

1. **Venue vs. film confusion** — "Film Forum" can be parsed as `match_to_specific_film` and used to filter by exhibition *title*, which returns 0.
2. **Time-period vs. future exhibition dates** — "this quarter/week/month" is interpreted from today, while exhibition data is in 2026, so nothing matches.
3. **UK vs. country codes** — UK is handled via the GB alias, but the time filter can still eliminate matches.
4. **Date parsing gaps** — "February 15 and March 5, 2026" is not parsed deterministically; it relies on the LLM, which can fail.

---

## 1. Venue Mapping Failure

### Root Cause

The LLM can set `match_to_specific_film = "Film Forum"` when the user mentions Film Forum as a *venue*. The exhibition filter runs in this order:

1. Territory  
2. City  
3. **match_to_specific_film** ← filters by exhibition *title*  
4. Venue  

If `match_to_specific_film = "Film Forum"`, the system filters exhibitions whose *title* contains "film forum". No exhibition titles match, so the filter returns 0 rows and exits before the venue filter runs.

### Evidence

- Query: "Film Forum's audience skews art-house…"
- `venue = "Film Forum"` (correct)
- `match_to_specific_film = "Film Forum"` (incorrect; venue misparsed as film)
- Result: 0 exhibitions

### Recommended Fixes

1. **Venue blocklist:** If `match_to_specific_film` matches a known venue (e.g. Film Forum, Metrograph, BAM, Alamo), clear it and keep/use `venue` instead.
2. **Conflict handling:** When both `venue` and `match_to_specific_film` are set, and they are identical or very similar, treat the value as venue, not film.
3. **Filter order:** Run venue (and location) filters before `match_to_specific_film` so venue-only queries still work.

---

## 2. Territory Mapping

### Root Cause

Territory logic itself works: UK → (UK, GB), and exhibition data uses `country = "GB"`. The problem is that other filters (especially **time_period**) run after territory and reduce the set to 0.

### Data

- Unique countries: US, CA, GB, FR, DE  
- Aliases: US→(US,USA), UK→(UK,GB), FR→(FR,FRANCE), CA→(CA,CANADA), MX→(MX,MEXICO)

### Gaps

- **DE (Germany)** is in the data but not in `TERRITORY_COUNTRY_ALIASES`, so queries for "Berlin" or "Germany" do not map correctly.
- **Venue names as locations:** "Film Forum (New York, NY)" is in `location`, not `country`; venue filtering uses `location`, which is correct.

### Recommended Fixes

1. Add DE and other relevant codes to `TERRITORY_COUNTRY_ALIASES`, e.g. `"DE": ("DE", "GERMANY")`.
2. Ensure territory extraction runs after LLM parsing so deterministic extraction is not overwritten by incorrect LLM output.

---

## 3. Time-Period Mapping Failure

### Root Cause

1. **"this quarter"** — Not handled; it falls through to the default of 7 days from today.
2. **Exhibition dates are in the future** — Data range: 2026-02-13 to 2026-03-23. Today is 2025-02-02.
3. **Result:** For "this week" (7 days) or "this month" (30 days), the filter keeps exhibitions in 2025-02-02 to 2025-03-04. All exhibition dates are in 2026, so the time filter yields 0.

### Code Reference

```python
# chatbot_agent.py _apply_exhibition_filters
if intent.time_period == "now" or intent.time_period == "week":
    end_date = today + timedelta(days=7)
elif intent.time_period == "month":
    end_date = today + timedelta(days=30)
else:
    end_date = today + timedelta(days=7)  # "quarter" falls here!
```

### Recommended Fixes

1. Add explicit handling for `"quarter"` (e.g. 90 days from today).
2. Support a configurable reference date so tests or forward-looking data can use a different "today".
3. When the time filter returns 0 and all exhibition dates are in the future, widen the window to the next 90 days from the earliest exhibition date in the dataset.
4. Optionally treat "this quarter" and similar phrases as soft filters (rank by proximity) rather than hard filters, when the user intent is about planning.

---

## 4. Exhibition Date Mapping Failure

### Root Cause

1. **Deterministic parsing** only supports numeric formats (`2/14/2026`, `2026-02-14`, `between 2/1/2026 and 2/14/2026`). It does not handle "February 15 and March 5, 2026".
2. **LLM parsing** can extract these dates, but a parsing error in the LLM step (e.g. `float() argument must be a string or a real number`) can break the merge, and the dates are lost.

### Evidence

- "exhibitions between February 15 and March 5, 2026"  
  - LLM can return `exhibition_date_start: "2026-02-15"`, `exhibition_date_end: "2026-03-05"`.  
  - If the LLM step errors, these are never set and no date filter is applied.

### Recommended Fixes

1. Extend the deterministic extractor to handle month names, e.g. "February 15, 2026" or "between February 15 and March 5, 2026", using a month-name map.
2. Make the LLM merge more robust so partial success is possible (e.g. keep exhibition dates even if another field fails).
3. Add a fallback: if the LLM returns ISO date strings but the merge fails, parse them in a try/except and set the intent fields separately.

---

## 5. Venue Data Gaps

### Root Cause

- **Alamo Drafthouse:** 0 exhibitions with "Alamo" in the location. The dataset does not include Alamo Drafthouse venues.
- **Film Forum:** 8 exhibitions contain "Film Forum" in location. The failure in testing is from the venue vs. film confusion, not missing data.

### Recommended Fixes

1. Add more venues to the known-venue list when they appear in the data.
2. Document or expose which venues are present in the exhibition file so expectations are clear.

---

## Implementation Priority

| Priority | Issue                         | Fix Effort | Impact |
|----------|-------------------------------|------------|--------|
| High     | Venue vs. film confusion      | Low        | High   |
| High     | Time-period vs. future dates  | Medium     | High   |
| Medium   | Date parsing (month names)    | Low        | Medium |
| Medium   | Add DE to territory aliases   | Trivial    | Low    |
| Low      | LLM merge robustness          | Medium     | Medium |
