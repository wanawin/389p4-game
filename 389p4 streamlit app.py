
from __future__ import annotations
from pathlib import Path
import io, zipfile, datetime as dt, time
import polars as pl
import streamlit as st

from ns_fast_engine import (
    BUILD, Settings, CURRENT12, DIVERSE9, MIXED12,
    normalize_history, load_trait_lookup, run_walk_forward,
    core_geometry, pair_overlap,
)

st.set_page_config(page_title="Northern Star Fast WF", layout="wide")
st.title("Northern Star Fast Walk-Forward")
st.caption(f"{BUILD} — Polars/DuckDB-ready, full daily ledger, fixed-budget and core-separation audits")

up=st.file_uploader("History CSV/TXT",type=["csv","txt","tsv"])
preset=st.selectbox("Core set",["Current 12","Diverse grid 9","Mixed 12","Custom"])
if preset=="Current 12":
    default=",".join(CURRENT12)
elif preset=="Diverse grid 9":
    default=",".join(DIVERSE9)
elif preset=="Mixed 12":
    default=",".join(MIXED12)
else:
    default=""
core_text=st.text_input("Cores (comma-separated)",value=default)
cores=[x.strip().zfill(3) for x in core_text.split(",") if x.strip()]

c1,c2,c3=st.columns(3)
window=c1.selectbox("History window",[180,365],index=0)
start=c2.date_input("Start date",value=dt.date(2026,3,19))
end=c3.date_input("End date",value=dt.date(2026,6,17))

with st.expander("Scoring settings",expanded=False):
    due=st.number_input("Due weight",value=0.20,step=0.05)
    pos=st.number_input("Position-percentile weight",value=0.25,step=0.05)
    seedw=st.number_input("Seed-trait weight",value=0.35,step=0.05)
    cadw=st.number_input("Cadence weight",value=0.25,step=0.05)
    use_seed=st.checkbox("Use seed traits when lookup files are supplied",value=True)
    use_cad=st.checkbox("Use cadence",value=True)

pos_file=st.file_uploader("Positive seed-trait CSV (optional)",type=["csv"],key="pos")
neg_file=st.file_uploader("Negative seed-trait CSV (optional)",type=["csv"],key="neg")

if cores:
    st.subheader("Core-set geometry")
    st.dataframe(core_geometry(cores),use_container_width=True,hide_index=True)
    ov=pair_overlap(cores)
    if ov.height:
        st.write({
            "Average shared digits per core pair":round(float(ov.get_column("SharedDigits").mean()),3),
            "Pairs sharing 0 digits":int(ov.filter(pl.col("SharedDigits")==0).height),
            "Pairs sharing 2+ digits":int(ov.filter(pl.col("SharedDigits")>=2).height),
        })

if st.button("Run optimized walk-forward",type="primary",disabled=(up is None or not cores)):
    try:
        raw=up.getvalue()
        hist=normalize_history(raw)
        st.info(f"Loaded {hist.height:,} rows, {hist.get_column('Stream').n_unique()} streams, through {hist.get_column('Date').max()}.")
        pos_l=load_trait_lookup(pos_file.getvalue()) if pos_file else {}
        neg_l=load_trait_lookup(neg_file.getvalue()) if neg_file else {}
        settings=Settings(window_days=int(window),due_weight=float(due),pos_weight=float(pos),
                          seed_weight=float(seedw),cadence_weight=float(cadw),
                          enable_seed_traits=bool(use_seed),enable_cadence=bool(use_cad))
        bar=st.progress(0.0,text="Starting…")
        t0=time.time()
        def prog(frac,msg):
            bar.progress(min(1.0,float(frac)),text=msg)
        all_df,wins,cuts,sep=run_walk_forward(
            hist,cores,start,end,settings,pos_l,neg_l,progress=prog
        )
        elapsed=time.time()-t0
        bar.progress(1.0,text=f"Complete in {elapsed:.1f} seconds")
        st.success(f"Completed {all_df.height:,} candidate rows in {elapsed:.1f} seconds.")

        st.subheader("Fixed-cutoff results")
        st.dataframe(cuts,use_container_width=True,hide_index=True)

        st.subheader("Core-separation results")
        if sep.height:
            summary=sep.select([
                pl.len().alias("WinnerEvents"),
                pl.col("WinningCoreRankAmongSet").mean().alias("AverageWinningCoreRank"),
                (pl.col("WinningCoreRankAmongSet")==1).mean().mul(100).alias("WinningCoreTop1Pct"),
                pl.col("WinnerMarginVsBestCompetitor").mean().alias("AverageWinnerMargin"),
            ])
            st.dataframe(summary,use_container_width=True,hide_index=True)
            st.dataframe(sep.head(200),use_container_width=True,hide_index=True)
        else:
            st.info("No selected-core winner events occurred in the chosen range.")

        cfg=pl.DataFrame([{
            "BUILD":BUILD,"StartDate":str(start),"EndDate":str(end),"WindowDays":window,
            "Cores":",".join(cores),"CoreCount":len(cores),"ElapsedSeconds":round(elapsed,3),
            "SeedTraitsLoadedPositive":len(pos_l),"SeedTraitsLoadedNegative":len(neg_l),
        }])
        bio=io.BytesIO()
        with zipfile.ZipFile(bio,"w",zipfile.ZIP_DEFLATED) as z:
            z.writestr("00_CONFIG.csv",cfg.write_csv())
            z.writestr("01_ALL_DAILY_RANKS.csv",all_df.write_csv())
            z.writestr("02_WINNER_ROWS.csv",wins.write_csv())
            z.writestr("03_FIXED_CUTOFFS.csv",cuts.write_csv())
            z.writestr("04_CORE_SEPARATION.csv",sep.write_csv() if sep.height else "")
            z.writestr("05_CORE_GEOMETRY.csv",core_geometry(cores).write_csv())
            z.writestr("06_CORE_PAIR_OVERLAP.csv",pair_overlap(cores).write_csv())
        st.download_button("Download all reports (ZIP)",bio.getvalue(),"NS_FAST_WF_REPORTS.zip","application/zip")
    except Exception as e:
        st.exception(e)
