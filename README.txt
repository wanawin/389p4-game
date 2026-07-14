NS FAST WF V1

Purpose
- Re-run Northern Star per-core walk-forward without the slow nested Pandas implementation.
- Export the complete daily rank ledger.
- Test fixed row cutoffs.
- Compare how easily competing cores separate.
- Compare current, diverse-grid, mixed, or custom core sets.

The scoring contract preserved:
NSScore =
  HitsPerWeek
  + capped DaysSinceLastHit soft boost
  + position-percentile soft boost
  + optional seed-trait score
  + optional cadence score

Important
- Positive/negative seed-trait CSVs are optional. If omitted, seed-trait contribution is zero.
- Every core has exactly three boxed AABC-family members.
- This program ranks stream+core rows. It does not score permutations.
- Use the same history, date range, weights, and trait files when comparing core sets.

Recommended first tests
1. Current 12, 180 days, 2026-03-19 through 2026-06-17.
2. Diverse grid 9 with identical settings.
3. Mixed 12 with identical settings.
Compare DayHitPct, WinningCoreTop1Pct, AverageWinningCoreRank, and winner margin.
