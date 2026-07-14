
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import io
import math
import re
import time
import zipfile

import polars as pl

BUILD = "NS_FAST_WF_V1"

CURRENT12 = ["016","027","028","067","138","145","256","389","457","458","567","679"]
DIVERSE9 = ["123","456","789","147","258","369","159","249","357"]
MIXED12 = ["016","027","067","138","145","256","389","457","567","123","258","369"]

@dataclass
class Settings:
    window_days: int = 180
    due_weight: float = 0.20
    pos_weight: float = 0.25
    seed_weight: float = 0.35
    cadence_weight: float = 0.25
    enable_seed_traits: bool = True
    enable_cadence: bool = True
    exclude_maryland: bool = True

def core_members(core: str) -> list[str]:
    c = ''.join(sorted(re.sub(r"\D", "", str(core)).zfill(3)[-3:]))
    if len(c) != 3 or len(set(c)) != 3:
        raise ValueError(f"Core must have exactly 3 distinct digits: {core}")
    a,b,d = c
    return [''.join(sorted(x)) for x in (a+a+b+d, a+b+b+d, a+b+d+d)]

def normalize_history(path_or_bytes) -> pl.DataFrame:
    if isinstance(path_or_bytes, (str, Path)):
        p = Path(path_or_bytes)
        raw = p.read_bytes()
        name = p.name.lower()
    else:
        raw = path_or_bytes
        name = ""
    # Try comma then tab.
    last_err = None
    for sep in [",", "\t"]:
        try:
            df = pl.read_csv(io.BytesIO(raw), separator=sep, infer_schema_length=5000,
                             ignore_errors=True, truncate_ragged_lines=True)
            if df.width >= 4:
                break
        except Exception as e:
            last_err = e
    else:
        raise ValueError(f"Could not read history: {last_err}")

    cols = {c.lower().strip(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    date_c = pick("date")
    state_c = pick("state")
    game_c = pick("game")
    result4_c = pick("result4", "pick4")
    result_c = pick("result", "results", "winning numbers", "winning_numbers")
    stream_c = pick("stream", "streamkey")

    if date_c is None:
        raise ValueError("History must contain Date.")
    if stream_c is None and (state_c is None or game_c is None):
        raise ValueError("History needs Stream/StreamKey or State + Game.")
    if result4_c is None and result_c is None:
        raise ValueError("History needs Result4/Pick4 or Result/Results.")

    if result4_c:
        r_expr = pl.col(result4_c).cast(pl.Utf8, strict=False).str.replace_all(r"\D","").str.zfill(4).str.tail(4)
    else:
        r_expr = pl.col(result_c).cast(pl.Utf8, strict=False).str.extract(r"(\d)\D*(\d)\D*(\d)\D*(\d)", 0)
        # extract(0) returns full match; strip non-digits.
        r_expr = r_expr.str.replace_all(r"\D","").str.zfill(4).str.tail(4)

    if stream_c:
        s_expr = pl.col(stream_c).cast(pl.Utf8, strict=False).str.strip_chars().str.replace_all(r"\s+"," ")
    else:
        s_expr = (
            pl.col(state_c).cast(pl.Utf8, strict=False).str.strip_chars()
            + pl.lit(" | ")
            + pl.col(game_c).cast(pl.Utf8, strict=False).str.strip_chars()
        )

    out = df.select([
        pl.col(date_c).cast(pl.Utf8, strict=False).str.to_date(strict=False).alias("Date"),
        s_expr.alias("Stream"),
        r_expr.alias("Result4"),
    ]).drop_nulls(["Date","Stream","Result4"])

    out = out.filter(pl.col("Result4").str.contains(r"^\d{4}$"))
    out = out.with_columns([
        pl.col("Result4").str.split("").list.sort().list.join("").alias("Member"),
        pl.col("Result4").str.split("").list.unique().list.sort().list.join("").alias("Core"),
    ])
    # Keep one draw per date/stream. Latest supplied row wins.
    out = out.unique(["Date","Stream"], keep="last").sort(["Date","Stream"])
    return out

def load_trait_lookup(path_or_bytes) -> dict[tuple[str,str,str], float]:
    if path_or_bytes is None:
        return {}
    try:
        raw = Path(path_or_bytes).read_bytes() if isinstance(path_or_bytes,(str,Path)) else path_or_bytes
        df = pl.read_csv(io.BytesIO(raw), infer_schema_length=5000, ignore_errors=True)
    except Exception:
        return {}
    cols = {c.lower(): c for c in df.columns}
    req = [cols.get("core_family"), cols.get("trait"), cols.get("value"), cols.get("lift")]
    if any(x is None for x in req):
        return {}
    cf,tr,va,li=req
    d={}
    for r in df.select([cf,tr,va,li]).iter_rows():
        try:
            core=str(int(r[0])).zfill(3)
            d[(core,str(r[1]).strip(),str(r[2]).strip())]=float(r[3])
        except Exception:
            pass
    return d

def seed_features(seed: str, core: str, last5_union: set[str] | None = None) -> dict[str,list[str]]:
    s = re.sub(r"\D","",str(seed)).zfill(4)[-4:]
    c = ''.join(sorted(str(core).zfill(3)))
    digs=[int(x) for x in s]
    sm=sum(digs); spread=max(digs)-min(digs)
    ov=len(set(s)&set(c))
    overlap_vals = ["3",">=2"] if ov==3 else ([">=2"] if ov==2 else [str(ov)])
    grid=len((last5_union or set())&set(c))
    grid_vals=["3",">=2"] if grid==3 else ([">=2"] if grid==2 else [str(grid)])
    # Adjacent core pair, matching current app behavior.
    pairs={c[i]+c[j] for i in range(3) for j in range(3) if i!=j}
    adj=[s[i:i+2] for i in range(3)]
    structure = {"1111":"AAAA"}
    counts=sorted([s.count(x) for x in set(s)], reverse=True)
    struct="AAAA" if counts==[4] else ("AAAB" if counts==[3,1] else ("AABB" if counts==[2,2] else ("AABC" if counts==[2,1,1] else "ABCD")))
    ranges=[]
    for start in (sm,sm-1,sm-2,sm-3):
        if 0 <= start and start+3 <= 36:
            ranges.append(f"{start}-{start+3}")
    return {
        "seed_structure":[struct],
        "seed_even_count":[str(sum(d%2==0 for d in digs))],
        "seed_high_count":[str(sum(d>=5 for d in digs))],
        "seed_spread":["<=2" if spread<=2 else (">=6" if spread>=6 else "3-5")],
        "seed_sum_mod2":[str(sm%2)],
        "seed_sum_mod3":[str(sm%3)],
        "seed_sum_range4_best":ranges,
        "seed_sum_range4_worst":ranges,
        "overlap_unique":overlap_vals,
        "seed_contains_core_pair":["yes" if any(a in pairs for a in adj) else "no"],
        "seed_first_in_core":["yes" if s[0] in set(c) else "no"],
        "seed_last_in_core":["yes" if s[-1] in set(c) else "no"],
        "grid_last5_core_digits":grid_vals,
    }

def trait_score(core, seed, stream, pos, neg, last5, cap=2.0):
    if not seed:
        return 0.0
    feats=seed_features(seed,core,last5.get(stream,set()))
    score=0.0
    for trait,vals in feats.items():
        for val in vals:
            if (core,trait,val) in pos:
                score += pos[(core,trait,val)] - 1.0
            if (core,trait,val) in neg:
                score -= neg[(core,trait,val)] - 1.0
    return max(-cap,min(cap,float(score)))

def cadence_score(days_since, mean_gap):
    if mean_gap <= 0:
        return 0.0
    return max(0.0,min(1.0,(float(days_since)/float(mean_gap)-1.0)/2.0))

def _history_maps(hist: pl.DataFrame):
    # Small Python maps are used only for as-of seed/last-five retrieval.
    by_stream={}
    for d,s,r in hist.select(["Date","Stream","Result4"]).iter_rows():
        by_stream.setdefault(s,[]).append((d,r))
    return by_stream

def run_walk_forward(
    hist: pl.DataFrame,
    cores: list[str],
    start_date,
    end_date,
    settings: Settings,
    pos_lookup=None,
    neg_lookup=None,
    progress=None,
):
    pos_lookup=pos_lookup or {}
    neg_lookup=neg_lookup or {}
    cores=[''.join(sorted(str(c).zfill(3))) for c in cores]
    streams=hist.get_column("Stream").unique().sort().to_list()
    dates=hist.get_column("Date").unique().sort()
    dates=[d for d in dates if d>=start_date and d<=end_date]
    by_stream=_history_maps(hist)
    total=max(1,len(dates)*len(cores))
    done=0
    all_rows=[]
    winner_rows=[]

    # Precompute member -> core for selected cores.
    member_core={}
    for c in cores:
        for m in core_members(c):
            member_core[m]=c

    for test_date in dates:
        train=hist.filter(pl.col("Date") < pl.lit(test_date))
        day=hist.filter(pl.col("Date") == pl.lit(test_date))
        if train.is_empty():
            continue
        cutoff=test_date - __import__("datetime").timedelta(days=settings.window_days)
        win=train.filter(pl.col("Date") >= pl.lit(cutoff))
        max_train=train.get_column("Date").max()

        # Latest seed and last-five union by stream as of date.
        seeds={}
        last5={}
        for s, vals in by_stream.items():
            past=[(d,r) for d,r in vals if d<test_date]
            if past:
                seeds[s]=past[-1][1]
                last5[s]=set(''.join(r for _,r in past[-5:]))

        # Actual selected-core winners on this date.
        actual={}
        for s,m,r in day.select(["Stream","Member","Result4"]).iter_rows():
            c=member_core.get(m)
            if c:
                actual[(c,s)]=(m,r)

        # Draw counts are same for every core.
        draws=(win.group_by("Stream").len().rename({"len":"DrawsWindow"}))
        for core in cores:
            members=core_members(core)
            hits=win.filter(pl.col("Member").is_in(members))
            hit_counts=hits.group_by("Stream").len().rename({"len":"HitsWindow"})
            last_hits=train.filter(pl.col("Member").is_in(members)).group_by("Stream").agg(pl.col("Date").max().alias("LastHitDate"))

            stats=(
                pl.DataFrame({"Stream":streams})
                .join(draws,on="Stream",how="left")
                .join(hit_counts,on="Stream",how="left")
                .join(last_hits,on="Stream",how="left")
                .with_columns([
                    pl.col("DrawsWindow").fill_null(0).cast(pl.Int64),
                    pl.col("HitsWindow").fill_null(0).cast(pl.Int64),
                ])
                .with_columns([
                    (pl.col("HitsWindow")/(settings.window_days/7.0)).alias("HitsPerWeek"),
                    (pl.lit(max_train)-pl.col("LastHitDate")).dt.total_days().fill_null(0).alias("DaysSinceLastHit"),
                ])
                .sort(["HitsPerWeek","HitsWindow","Stream"], descending=[True,True,False])
                .with_row_index("RankPos", offset=1)
            )

            # Current app's position percentile strength: percentile of HitsWindow by rank position.
            stats=stats.with_columns(
                (pl.col("HitsWindow").rank(method="average") / pl.len() * 100.0).alias("PosPctStrength")
            )
            total_hits=float(stats.get_column("HitsWindow").sum())
            mean_gap=settings.window_days/total_hits if total_hits>0 else 0.0

            pyrows=[]
            for row in stats.iter_rows(named=True):
                s=row["Stream"]
                seed=seeds.get(s)
                ss=trait_score(core,seed,s,pos_lookup,neg_lookup,last5) if settings.enable_seed_traits else 0.0
                cad=cadence_score(row["DaysSinceLastHit"],mean_gap) if settings.enable_cadence else 0.0
                ns=(
                    float(row["HitsPerWeek"])
                    + min(float(row["DaysSinceLastHit"]),50.0)*0.01*settings.due_weight
                    + float(row["PosPctStrength"])*0.01*settings.pos_weight
                    + ss*settings.seed_weight
                    + cad*settings.cadence_weight
                )
                member=result=""
                is_hit=False
                if (core,s) in actual:
                    member,result=actual[(core,s)]
                    is_hit=True
                pyrows.append({
                    "Date":test_date,"HistoryThrough":max_train,"Core":core,"Stream":s,
                    "Seed":seed or "","RankPos":int(row["RankPos"]),
                    "HitsWindow":int(row["HitsWindow"]),"DrawsWindow":int(row["DrawsWindow"]),
                    "HitsPerWeek":float(row["HitsPerWeek"]),
                    "DaysSinceLastHit":int(row["DaysSinceLastHit"]),
                    "PosPctStrength":float(row["PosPctStrength"]),
                    "SeedTraitsScore":float(ss),"CadenceScore":float(cad),"NSScore":float(ns),
                    "ExactStreamCoreHit":is_hit,"WinnerMember":member,"WinnerResult":result,
                })

            scored=pl.DataFrame(pyrows).sort(["NSScore","HitsPerWeek","Stream"],descending=[True,True,False]).with_row_index("NSRank",offset=1)
            all_rows.extend(scored.to_dicts())
            winner_rows.extend(scored.filter(pl.col("ExactStreamCoreHit")).to_dicts())
            done+=1
            if progress:
                progress(done/total, f"{test_date} core {core} ({done}/{total})")

    all_df=pl.DataFrame(all_rows) if all_rows else pl.DataFrame()
    wins_df=pl.DataFrame(winner_rows) if winner_rows else pl.DataFrame()

    # Fixed budget results for both base RankPos and final NSRank.
    cut_rows=[]
    n_days=len(dates)
    for rank_col in ["RankPos","NSRank"]:
        if all_df.is_empty():
            continue
        for n in [1,2,3,4,5,10,12,20,30,40,50]:
            sub=all_df.filter(pl.col(rank_col)<=n)
            hit=sub.filter(pl.col("ExactStreamCoreHit"))
            days=hit.get_column("Date").n_unique() if hit.height else 0
            cut_rows.append({
                "RankSystem":rank_col,"RowsPerCore":n,"CoreCount":len(cores),
                "StreamCoreRowsPerDay":n*len(cores),
                "All3MemberPlaysPerDay":n*len(cores)*3,
                "DaysTested":n_days,"DaysWithAtLeast1Hit":days,
                "DayHitPct":round(days/max(1,n_days)*100,2),
                "WinnerEventsCaptured":hit.height,
            })
    cuts=pl.DataFrame(cut_rows)

    # Core separation: compare selected-core NS scores inside each date+stream.
    sep_rows=[]
    if not all_df.is_empty():
        ranked=all_df.with_columns(
            pl.col("NSScore").rank("dense",descending=True).over(["Date","Stream"]).alias("CoreRankWithinStream")
        )
        for r in ranked.filter(pl.col("ExactStreamCoreHit")).iter_rows(named=True):
            competitors=ranked.filter((pl.col("Date")==r["Date"])&(pl.col("Stream")==r["Stream"])&(pl.col("Core")!=r["Core"]))
            best_comp=competitors.get_column("NSScore").max() if competitors.height else None
            sep_rows.append({
                "Date":r["Date"],"Stream":r["Stream"],"WinningCore":r["Core"],
                "WinningCoreNSScore":r["NSScore"],"WinningCoreRankAmongSet":int(r["CoreRankWithinStream"]),
                "BestCompetingScore":best_comp,
                "WinnerMarginVsBestCompetitor":(r["NSScore"]-best_comp) if best_comp is not None else None,
            })
    separation=pl.DataFrame(sep_rows) if sep_rows else pl.DataFrame()

    return all_df,wins_df,cuts,separation

def core_geometry(cores: list[str]) -> pl.DataFrame:
    rows=[]
    for a in cores:
        sa=set(str(a).zfill(3))
        digs=[int(x) for x in str(a).zfill(3)]
        rows.append({"Core":str(a).zfill(3),"DigitSpread":max(digs)-min(digs),"DigitSum":sum(digs)})
    return pl.DataFrame(rows)

def pair_overlap(cores: list[str]) -> pl.DataFrame:
    rows=[]
    for i,a in enumerate(cores):
        for b in cores[i+1:]:
            rows.append({"CoreA":a,"CoreB":b,"SharedDigits":len(set(a)&set(b))})
    return pl.DataFrame(rows)
