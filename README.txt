NORTHERN STAR V51.10 — 12-CORE WALK-FORWARD REPORT BUILD

Changes only:
- Added missing selectable cores 138 and 256.
- Added one-click intended 12-core selection:
  016,027,028,067,138,145,256,389,457,458,567,679
- Preserved the current Northern Star scoring and backtest logic.
- Added a complete walk-forward candidate ledger for every:
  test date × selected core × ranked stream.
- Added always-visible CSV downloads:
  WF_WINNERS.csv
  WF_ALL_RANKS.csv
  WF_CUTOFFS.csv
- Added WF_REPORTS.zip with config, winner ledger, all daily ranks,
  fixed-cutoff audit, and day summary.
- Clarified that member prediction outputs Top1/Top2 only. It does not
  create a third predicted rank. Every core still has exactly three boxed members.

Recommended first full run:
- Walk-forward (no cheating)
- Intended 12 cores
- 180-day window first
- Full chosen test date range
- Evaluate only hit days OFF for real calendar-day performance
- Track member accuracy ON
- Per-core (all streams) first
- Include rare OFF
