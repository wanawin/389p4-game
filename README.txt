NORTHERN STAR V51.10 FAST EXACT BACKTEST PATCH

What changed
- Added missing cores 138 and 256.
- Replaced only the walk-forward backtest renderer.
- Daily app scoring, playlists, caches, and other tabs are unchanged.
- Uses Polars for full date x core x stream ledger generation.
- Preserves the old backtest definition: Top BaseScore bucket + Due bucket.
- Uses the original app member predictor for Top1/Top2 winner rows.
- Adds full ZIP reports and fixed rank cutoffs.

Deploy
- Replace the existing Streamlit entrypoint with app.py.
- Add ns_fast_backtest.py beside it.
- Add polars and pyarrow to requirements.txt (or merge these lines into the existing requirements).

First run
- Select intended 12 cores.
- 180-day history window.
- Full date range.
- Evaluate only hit days OFF.
- Track members ON.
