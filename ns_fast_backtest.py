from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import datetime as dt
import io
import time
import zipfile

import pandas as pd
import polars as pl
import streamlit as st

BUILD = "NS_FAST_EXACT_BT_V1"
WF12 = ["016","027","028","067","138","145","256","389","457","458","567","679"]


def _to_polars(df: pd.DataFrame) -> pl.DataFrame:
    need=["Date","Stream","Result"]
    miss=[c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Backtest requires columns {need}; missing {miss}")
    x=df[need].copy()
    x["Date"]=pd.to_datetime(x["Date"],errors="coerce").dt.normalize()
    x=x.dropna(subset=["Date","Stream","Result"])
    x["Stream"]=x["Stream"].astype(str).str.strip()
    x["Result"]=x["Result"].astype(str).str.replace(r"\D","",regex=True).str.zfill(4).str[-4:]
    x=x[x["Result"].str.fullmatch(r"\d{4}",na=False)]
    return pl.from_pandas(x).with_columns([
        pl.col("Date").cast(pl.Date),
        pl.col("Result").str.split("").list.sort().list.join("").alias("Member"),
    ]).unique(["Date","Stream"],keep="last").sort(["Date","Stream"])


def _member_map(cores, members_from_core, box_key, include_rare=False):
    d={}
    for core in cores:
        members=list(members_from_core(core,"AABC"))
        if include_rare:
            for struct in ("AAAB","AABB","AAAA"):
                members.extend(members_from_core(core,struct))
        for m in members:
            d.setdefault(box_key(m),[]).append(core)
    return d


def _exact_stats_for_date(hist: pl.DataFrame, test_date: dt.date, cores: list[str], window_days: int,
                          members_from_core, box_key, progress=None):
    train=hist.filter(pl.col("Date") < pl.lit(test_date))
    if train.is_empty():
        return pl.DataFrame()
    max_date=train.get_column("Date").max()
    cutoff=max_date-dt.timedelta(days=int(window_days))
    win=train.filter(pl.col("Date") >= pl.lit(cutoff))
    streams=train.get_column("Stream").unique().sort()
    draws=win.group_by("Stream").len().rename({"len":"DrawsWindow"})
    frames=[]
    for i,core in enumerate(cores,1):
        mems=[box_key(m) for m in members_from_core(core,"AABC")]
        hits_win=win.filter(pl.col("Member").is_in(mems))
        hits_all=train.filter(pl.col("Member").is_in(mems))
        hc=hits_win.group_by("Stream").len().rename({"len":"HitsWindow"})
        lh=hits_all.group_by("Stream").agg(pl.col("Date").max().alias("LastHitDate"))
        s=(pl.DataFrame({"Stream":streams})
           .join(draws,on="Stream",how="left")
           .join(hc,on="Stream",how="left")
           .join(lh,on="Stream",how="left")
           .with_columns([
              pl.col("DrawsWindow").fill_null(0).cast(pl.Int64),
              pl.col("HitsWindow").fill_null(0).cast(pl.Int64),
           ])
           .with_columns([
              (pl.col("HitsWindow")/(float(window_days)/7.0)).alias("HitsPerWeek"),
              (pl.lit(max_date)-pl.col("LastHitDate")).dt.total_days().fill_null(0).cast(pl.Int64).alias("DaysSinceLastHit"),
           ])
           # Matches pandas sort_values([HitsPerWeek,HitsWindow], descending). Stable Stream tie-break makes runs deterministic.
           .sort(["HitsPerWeek","HitsWindow","Stream"],descending=[True,True,False])
           .with_row_index("RankPos",offset=1)
           .with_columns([
              pl.col("RankPos").alias("BaseScoreRank"),
              pl.col("HitsPerWeek").alias("BaseScore"),
              pl.col("DaysSinceLastHit").alias("DueIndex"),
              pl.col("DaysSinceLastHit").rank("dense",descending=True).cast(pl.Int64).alias("DueIndexRank"),
              pl.lit(core).alias("Core"),pl.lit(test_date).alias("Date"),pl.lit(max_date).alias("AsOfMaxDate")
           ]))
        frames.append(s)
        if progress: progress(i/len(cores),f"{test_date}: core {core} ({i}/{len(cores)})")
    return pl.concat(frames,how="vertical")


def _add_buckets(full: pl.DataFrame, cfg) -> pl.DataFrame:
    top_n=int(getattr(cfg,"top_base",12)); due_n=int(getattr(cfg,"top_due",8))
    lo=int(getattr(cfg,"due_from_rank",13)); hi=int(getattr(cfg,"due_to_rank",60))
    base=full.filter(pl.col("BaseScoreRank")<=top_n).select(["Date","Core","Stream"]).with_columns(pl.lit(True).alias("InBase"))
    due=(full.filter(pl.col("BaseScoreRank").is_between(lo,hi,closed="both"))
         .sort(["Date","Core","DueIndexRank","RankPos","Stream"])
         .with_columns(pl.int_range(1,pl.len()+1).over(["Date","Core"]).alias("DueTakeRank"))
         .filter(pl.col("DueTakeRank")<=due_n)
         .select(["Date","Core","Stream"]).with_columns(pl.lit(True).alias("InDue")))
    return (full.join(base,on=["Date","Core","Stream"],how="left")
            .join(due,on=["Date","Core","Stream"],how="left")
            .with_columns([pl.col("InBase").fill_null(False),pl.col("InDue").fill_null(False)])
            .with_columns([
              (pl.col("InBase")|pl.col("InDue")).alias("Predicted"),
              pl.when(pl.col("InBase")&pl.col("InDue")).then(pl.lit("Both"))
               .when(pl.col("InBase")).then(pl.lit("BaseScore"))
               .when(pl.col("InDue")).then(pl.lit("Due8")).otherwise(pl.lit("None")).alias("Bucket")
            ]))


def _fixed_cutoffs(full: pl.DataFrame, dates: list[dt.date], cores: list[str]) -> pl.DataFrame:
    rows=[]
    for n in [1,2,3,4,5,6,7,8,9,10,12,15,20,30,40,50,60,78,80]:
        sub=full.filter(pl.col("RankPos")<=n)
        hit=sub.filter(pl.col("ExactStreamCoreHit"))
        days=hit.get_column("Date").n_unique() if hit.height else 0
        rows.append({"RowsPerCore":n,"CoreCount":len(cores),"StreamCoreRowsPerDay":n*len(cores),
                     "All3MemberPlaysPerDay":n*len(cores)*3,"DaysTested":len(dates),
                     "DaysWithAtLeast1ExactStreamCoreHit":days,"DayHitPct":round(days/max(1,len(dates))*100,2),
                     "WinnerEventsCaptured":hit.height})
    return pl.DataFrame(rows)


def render_fast_walk_forward_backtest(*,df_all,cfg,cores_for_cache,core_presets,members_from_core,box_key,
                                      core_member_label,predict_core_member):
    st.markdown("### Fast exact walk-forward backtest")
    st.caption("Only this backtest is optimized. The app's daily Northern Star/Northern Lights calculations are unchanged.")
    if df_all is None or df_all.empty:
        st.warning("No data loaded."); return

    all_cores=sorted(set(str(c).zfill(3) for c in core_presets))
    c1,c2,c3=st.columns([1.1,1.1,1])
    use_all=c1.checkbox("Test ALL listed cores",False,key="fbt_all")
    include_rare=c2.checkbox("Include rare members",False,key="fbt_rare")
    max_dates=int(c3.number_input("Max test dates",1,3650,120,10,key="fbt_max"))
    b1,b2=st.columns(2)
    if b1.button("Select intended 12 cores",key="fbt_sel12"):
        st.session_state["fbt_cores"]=[c for c in WF12 if c in all_cores]; st.rerun()
    if "fbt_cores" not in st.session_state:
        st.session_state["fbt_cores"]=[c for c in (cores_for_cache or WF12) if c in all_cores]
    if b2.button("Clear",key="fbt_clear"):
        st.session_state["fbt_cores"]=[]; st.rerun()
    cores=all_cores if use_all else st.multiselect("Cores to test",all_cores,key="fbt_cores")
    if not cores: st.info("Select at least one core."); return

    d=pd.to_datetime(df_all["Date"],errors="coerce"); dmin=d.min(); dmax=d.max()
    default_start=(dmax-pd.Timedelta(days=90)).date() if (dmax-dmin).days>120 else dmin.date()
    dr=st.date_input("Test date range",(default_start,dmax.date()),min_value=dmin.date(),max_value=dmax.date(),key="fbt_dates")
    if not isinstance(dr,(tuple,list)) or len(dr)!=2: st.info("Choose a start and end date."); return
    start,end=dr
    only_hit=st.checkbox("Evaluate only selected-core hit days",False,key="fbt_hits_only")
    st.markdown("##### Member-pick tracking")
    m1,m2=st.columns(2)
    track=m1.checkbox("Track member Top1/Top2",True,key="fbt_member")
    basis_label=m2.selectbox("Member basis",["Per-core (all streams)","Per-core + stream"],key="fbt_basis")
    basis="core_stream" if basis_label.startswith("Per-core +") else "core"

    if not st.button("Run fast walk-forward",type="primary",key="fbt_run"): return
    t0=time.time(); hist=_to_polars(df_all)
    all_dates=[x for x in hist.get_column("Date").unique().sort().to_list() if start<=x<=end]
    member_to_cores=_member_map(cores,members_from_core,box_key,include_rare)
    day_hist=hist.filter(pl.col("Date").is_in(all_dates))
    if only_hit:
        hit_dates=[]
        for r in day_hist.select(["Date","Member"]).iter_rows(named=True):
            if r["Member"] in member_to_cores: hit_dates.append(r["Date"])
        all_dates=sorted(set(hit_dates))
    if len(all_dates)>max_dates: all_dates=all_dates[-max_dates:]
    if not all_dates: st.warning("No test dates."); return

    bar=st.progress(0.0,text="Starting")
    frames=[]
    for ix,date in enumerate(all_dates,1):
        def inner(frac,msg): bar.progress(min(.99,(ix-1+frac)/len(all_dates)),text=msg)
        frames.append(_exact_stats_for_date(hist,date,cores,int(cfg.window_days),members_from_core,box_key,inner))
    full=_add_buckets(pl.concat(frames,how="vertical"),cfg)

    # Attach exact winner labels.
    actual=[]
    for dte,stream,member,result in day_hist.filter(pl.col("Date").is_in(all_dates)).select(["Date","Stream","Member","Result"]).iter_rows():
        for core in member_to_cores.get(member,[]): actual.append({"Date":dte,"Core":core,"Stream":stream,"Winner":result,"WinnerMember":member})
    act=pl.DataFrame(actual) if actual else pl.DataFrame(schema={"Date":pl.Date,"Core":pl.Utf8,"Stream":pl.Utf8,"Winner":pl.Utf8,"WinnerMember":pl.Utf8})
    full=(full.join(act,on=["Date","Core","Stream"],how="left")
          .with_columns(pl.col("Winner").is_not_null().alias("ExactStreamCoreHit")))
    winners=full.filter(pl.col("ExactStreamCoreHit"))

    # Member prediction remains the original exact app function and is only run on winner rows.
    member_rows=[]
    if track and winners.height:
        cache={}
        for r in winners.iter_rows(named=True):
            core,stream,date,winner=r["Core"],r["Stream"],pd.Timestamp(r["Date"]),r["Winner"]
            key=(core,date,int(cfg.window_days),basis,stream if basis=="core_stream" else None)
            if key not in cache:
                cache[key]=predict_core_member(df_all,core,date,int(cfg.window_days),basis=basis,
                                                stream=stream if basis=="core_stream" else None,include_rare=False)
            mp=cache[key]; actual_label=core_member_label(core,winner,include_rare=False)
            t1=mp.get("top1"); t2=mp.get("top2"); counts=mp.get("counts") or {}
            member_rows.append({"Date":r["Date"],"Core":core,"Stream":stream,"Winner":winner,
              "ActualMember":actual_label,"PredMemberTop1":t1,"PredMemberTop2":t2,
              "MemberHitTop1":bool(actual_label and t1 and actual_label==t1),
              "MemberHitTop2":bool(actual_label and t1 and (actual_label==t1 or actual_label==t2)),
              "MemberTrainN":int(mp.get("n") or 0),"TrainCnt_AABC":int(counts.get("AABC") or 0),
              "TrainCnt_ABBC":int(counts.get("ABBC") or 0),"TrainCnt_ABCC":int(counts.get("ABCC") or 0)})
    members=pl.DataFrame(member_rows) if member_rows else pl.DataFrame()
    cuts=_fixed_cutoffs(full,all_dates,cores)
    bucket_summary=pl.DataFrame([{
      "DaysTested":len(all_dates),"WinnerEvents":winners.height,
      "BucketWinnerEventsCaptured":winners.filter(pl.col("Predicted")).height,
      "DaysWithBucketHit":winners.filter(pl.col("Predicted")).get_column("Date").n_unique() if winners.height else 0,
    }]).with_columns((pl.col("DaysWithBucketHit")/pl.col("DaysTested")*100).round(2).alias("BucketDayHitPct"))
    bar.progress(1.0,text=f"Complete in {time.time()-t0:.1f} seconds")
    st.success(f"Complete: {full.height:,} candidate rows, {winners.height} winner rows, {time.time()-t0:.1f}s")
    st.dataframe(bucket_summary.to_pandas(),use_container_width=True,hide_index=True)
    st.markdown("#### Fixed row cutoffs")
    st.dataframe(cuts.to_pandas(),use_container_width=True,hide_index=True)
    if members.height:
        st.markdown("#### Member Top1/Top2")
        ms=members.group_by("Core").agg([pl.len().alias("Events"),pl.col("MemberHitTop1").mean().mul(100).round(2).alias("Top1Pct"),pl.col("MemberHitTop2").mean().mul(100).round(2).alias("Top2Pct")]).sort("Core")
        st.dataframe(ms.to_pandas(),use_container_width=True,hide_index=True)

    cfgdf=pl.DataFrame([{"BUILD":BUILD,"Start":str(start),"End":str(end),"WindowDays":int(cfg.window_days),
                        "Cores":",".join(cores),"CoreCount":len(cores),"OnlyHitDays":only_hit,
                        "TrackMembers":track,"MemberBasis":basis,"ElapsedSeconds":round(time.time()-t0,3)}])
    bio=io.BytesIO()
    with zipfile.ZipFile(bio,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("00_CONFIG.csv",cfgdf.write_csv())
        z.writestr("01_ALL_DAILY_RANKS.csv",full.write_csv())
        z.writestr("02_WINNER_ROWS.csv",winners.write_csv())
        z.writestr("03_BUCKET_SUMMARY.csv",bucket_summary.write_csv())
        z.writestr("04_FIXED_CUTOFFS.csv",cuts.write_csv())
        z.writestr("05_MEMBER_RESULTS.csv",members.write_csv() if members.height else "")
    st.download_button("Download all backtest reports (ZIP)",bio.getvalue(),"NS_FAST_EXACT_BT_REPORTS.zip","application/zip",key="fbt_zip")
