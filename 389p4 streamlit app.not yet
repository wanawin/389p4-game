
# pk4_northern_star_app_2026-02-01_v3.py
# Streamlit app: Pick 4 "Northern Star" core stream ranking + Rare/Ultra-Rare engines (AAAB+AABB, AAAA)
# Notes:
# - Designed to work with LotteryPost-style exports (tab .txt or .csv) that include Date, State, Game, Results.
# - Ignores Wild Ball / Fireball / multipliers by extracting the first 4 digits like "1-2-3-4".
# - Excludes Maryland by default (toggle in sidebar).

from __future__ import annotations

import re
import math
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import streamlit as st
import hashlib
import datetime



# -------------------------
# Parsing + helpers
# -------------------------

# -------------------------
# Disk baseline cache (optional, keeps runs fast)
# -------------------------
from pathlib import Path as _Path

DISK_CACHE_DIR = _Path("pk4_baseline_cache")
DISK_CACHE_DIR.mkdir(exist_ok=True)

def _cache_key(max_date: pd.Timestamp, rows: int, streams: int, exclude_md: bool, window_days: int, cores: List[str]) -> str:
    # small, stable key so your cache survives restarts
    core_sig = "-".join(cores)
    base = f"{max_date.date()}|{rows}|{streams}|md={int(exclude_md)}|w={window_days}|{core_sig}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]

def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return True
        except Exception:
            return False

def _safe_write_table(df: pd.DataFrame, path: _Path) -> Tuple[bool, str]:
    """Write as parquet if possible, else as CSV (human readable)."""
    try:
        if _parquet_available():
            df.to_parquet(path.with_suffix(".parquet"), index=False)
            return True, str(path.with_suffix(".parquet"))
        df.to_csv(path.with_suffix(".csv"), index=False)
        return True, str(path.with_suffix(".csv"))
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def _safe_read_table(path: _Path) -> Optional[pd.DataFrame]:
    p_parq = path.with_suffix(".parquet")
    p_csv = path.with_suffix(".csv")
    try:
        if p_parq.exists():
            return pd.read_parquet(p_parq)
        if p_csv.exists():
            return pd.read_csv(p_csv)
    except Exception:
        return None
    return None

def _read_meta(path: _Path) -> Dict[str, Any]:
    p = path.with_suffix(".json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def _write_meta(path: _Path, meta: Dict[str, Any]) -> None:
    p = path.with_suffix(".json")
    p.write_text(json.dumps(meta, indent=2, default=str))

FOUR_DIGITS_RE = re.compile(r"(\d)\s*-\s*(\d)\s*-\s*(\d)\s*-\s*(\d)")


def _bytes_of_upload(uploaded) -> bytes:
    if uploaded is None:
        return b""
    try:
        return uploaded.getvalue()
    except Exception:
        try:
            return uploaded.read()
        except Exception:
            return b""

def file_fingerprint(uploaded) -> str:
    """Stable fingerprint for an uploaded file (used to auto-recompute when data changes)."""
    data = _bytes_of_upload(uploaded)
    if not data:
        return ""
    return hashlib.sha1(data).hexdigest()

def most_recent_date(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    if df is None or df.empty or "Date" not in df.columns:
        return None
    try:
        return pd.to_datetime(df["Date"]).max()
    except Exception:
        return None
def extract_pick4_digits(results: str) -> Optional[str]:
    """Return 4-digit string from LotteryPost 'Results' cell, else None."""
    if results is None or (isinstance(results, float) and np.isnan(results)):
        return None
    m = FOUR_DIGITS_RE.search(str(results))
    if not m:
        # Sometimes results can be plain "1234"
        m2 = re.search(r"\b(\d{4})\b", str(results))
        if m2:
            return m2.group(1)
        return None
    return "".join(m.groups())

def box_key(s: str) -> str:
    return "".join(sorted(s))

def structure_of_4(d4: str) -> str:
    """Return AABC / AAAB / AABB / AAAA / ABCD based on counts."""
    from collections import Counter
    c = Counter(d4)
    counts = sorted(c.values(), reverse=True)
    if counts == [4]:
        return "AAAA"
    if counts == [3,1]:
        return "AAAB"
    if counts == [2,2]:
        return "AABB"
    if counts == [2,1,1]:
        return "AABC"
    return "ABCD"

def canonical_core_key(core: str) -> str:
    core = re.sub(r"\D", "", str(core))
    if len(core) == 3:
        return "".join(sorted(core))
    raise ValueError("Core must be 3 digits like 389")

def members_from_core(core_key: str, structure: str) -> List[str]:
    """Return box members (4-digit strings) for a 3-digit core, for a given structure."""
    core_key = canonical_core_key(core_key)
    a, b, c = list(core_key)
    if structure == "AABC":
        # Doubles for a 3-digit core: repeat one digit, include the other two once
        return [a+a+b+c, b+b+a+c, c+c+a+b]
    if structure == "AAAB":
        # Triples: one digit x3 + one other digit x1 => 6 members
        return [a+a+a+b, a+a+a+c, b+b+b+a, b+b+b+c, c+c+c+a, c+c+c+b]
    if structure == "AABB":
        # Double-doubles: choose two digits x2 => 3 members
        return [a+a+b+b, a+a+c+c, b+b+c+c]
    if structure == "AAAA":
        return [a*4, b*4, c*4]
    raise ValueError("Unsupported structure for core members: " + structure)

def try_read_tablelike(uploaded) -> pd.DataFrame:
    """
    Accept .csv or LotteryPost tab .txt.
    Expected columns (any case): Date, State, Game, Results (or Result/Winning Numbers).
    """
    if uploaded is None:
        return pd.DataFrame()

    name = getattr(uploaded, "name", "") or ""
    # try csv first
    try:
        df = pd.read_csv(uploaded)
        if df.shape[1] == 1:
            raise ValueError("Looks like 1-column; try tab.")
    except Exception:
        uploaded.seek(0)
        df = pd.read_csv(uploaded, sep="\t", header=None)
        # try to name columns if 4+ cols
        if df.shape[1] >= 4:
            df = df.iloc[:, :4]
            df.columns = ["Date", "State", "Game", "Results"]
        else:
            # fallback
            df.columns = [f"col_{i}" for i in range(df.shape[1])]

    # normalize column names
    colmap = {c.lower().strip(): c for c in df.columns}
    def pick(*cands):
        for c in cands:
            if c in colmap:
                return colmap[c]
        return None

    date_col = pick("date")
    state_col = pick("state")
    game_col = pick("game")
    results_col = pick("results", "result", "winning numbers", "winning_numbers", "winningnumbers")

    if date_col is None or state_col is None or game_col is None or results_col is None:
        # best-effort: if there are exactly 4 columns, assume those
        if df.shape[1] >= 4:
            df = df.iloc[:, :4].copy()
            df.columns = ["Date", "State", "Game", "Results"]
        else:
            raise ValueError("Could not detect Date/State/Game/Results columns.")
    else:
        df = df.rename(columns={
            date_col: "Date",
            state_col: "State",
            game_col: "Game",
            results_col: "Results",
        })

    # parse date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].notna()].copy()

    # parse results
    df["Pick4"] = df["Results"].map(extract_pick4_digits)
    df = df[df["Pick4"].notna()].copy()

    df["Structure"] = df["Pick4"].map(structure_of_4)
    df["Box"] = df["Pick4"].map(box_key)
    df["Stream"] = df["State"].astype(str).str.strip() + " | " + df["Game"].astype(str).str.strip()
    return df


# -------------------------
# Stats + ranking
# -------------------------

@dataclass
class RankConfig:
    window_days: int = 180
    top_base: int = 12
    due_from_rank: int = 13
    due_to_rank: int = 60
    top_due: int = 8

def within_last_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty:
        return df
    max_date = df["Date"].max()
    cutoff = max_date - pd.Timedelta(days=days)
    return df[df["Date"] >= cutoff].copy()

def compute_core_hits(df: pd.DataFrame, core: str, structures: Iterable[str]) -> pd.DataFrame:
    """
    Return df subset containing only rows that are hits for the core, for the chosen structures.
    We match by Box membership, so order doesn't matter.
    """
    core = canonical_core_key(core)
    boxes = set()
    for s in structures:
        for mem in members_from_core(core, s):
            boxes.add(box_key(mem))
    return df[df["Box"].isin(boxes)].copy()

def stream_summary(df_all: pd.DataFrame, df_hits: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """
    For each stream, compute:
    - draws_window, hits_window, hits_per_week_window
    - days_since_last_hit_window (based on df_hits within full history)
    """
    if df_all.empty:
        return pd.DataFrame(columns=[
            "Stream","DrawsWindow","HitsWindow","HitsPerWeek","LastHitDate","DaysSinceLastHit"
        ])

    dfw = within_last_days(df_all, window_days)
    max_date = df_all["Date"].max()

    draws = dfw.groupby("Stream").size().rename("DrawsWindow")
    hitsw = within_last_days(df_hits, window_days).groupby("Stream").size().rename("HitsWindow")

    # last hit date from full history (not just window) for "due"
    last_hit = df_hits.groupby("Stream")["Date"].max().rename("LastHitDate")

    out = pd.concat([draws, hitsw, last_hit], axis=1).fillna({"HitsWindow":0})
    out["HitsWindow"] = out["HitsWindow"].astype(int)
    out["DrawsWindow"] = out["DrawsWindow"].astype(int)

    weeks = max(window_days / 7.0, 1e-9)
    out["HitsPerWeek"] = out["HitsWindow"] / weeks

    out["DaysSinceLastHit"] = (max_date - out["LastHitDate"]).dt.days
    out.loc[out["LastHitDate"].isna(), "DaysSinceLastHit"] = np.nan

    out = out.reset_index().sort_values(["HitsPerWeek","HitsWindow"], ascending=False)
    out["RankPos"] = np.arange(1, len(out)+1)
    return out

def position_percentile_map(stream_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Create a rank-position map (RankPos -> HitCount) + percentiles and cumulative share.
    This is what you wanted for Northern Star: positions are individual, not ranges.
    """
    if stream_stats.empty:
        return pd.DataFrame(columns=["RankPos","HitCount","HitShare","CumHitShare","HitCountPctile"])

    total_hits = stream_stats["HitsWindow"].sum()
    pos = stream_stats[["RankPos","HitsWindow"]].rename(columns={"HitsWindow":"HitCount"}).copy()
    pos["HitShare"] = (pos["HitCount"] / total_hits) if total_hits > 0 else 0.0
    pos = pos.sort_values("RankPos")
    pos["CumHitShare"] = pos["HitShare"].cumsum()

    # percentile by HitCount (higher hitcount -> higher percentile)
    pos["HitCountPctile"] = pos["HitCount"].rank(pct=True) * 100.0
    pos["HitCountPctile"] = pos["HitCountPctile"].round(1)
    pos["HitShare"] = (pos["HitShare"] * 100.0).round(2)
    pos["CumHitShare"] = (pos["CumHitShare"] * 100.0).round(2)

    return pos


def get_position_percentiles_cached(core: str, window_days: int, stream_stats: pd.DataFrame) -> pd.DataFrame:
    """Cache position percentile maps per core/window so UI tweaks don't constantly recompute.
    Cache is automatically cleared when input data changes or when the user clicks 'Recompute percentile maps now'.
    """
    cache: Dict[str, pd.DataFrame] = st.session_state.get("pos_map_cache", {})
    data_hash = st.session_state.get("data_hash_all", "")
    key = f"{core}|{window_days}|{data_hash}"

    if key in cache:
        return cache[key]

    pos_map = position_percentile_map(stream_stats)
    cache[key] = pos_map
    st.session_state["pos_map_cache"] = cache

    if not st.session_state.get("recompute_token"):
        st.session_state["recompute_token"] = datetime.datetime.now().isoformat(timespec="seconds")

    return pos_map

def bucket_recommendations(stream_stats: pd.DataFrame, cfg: RankConfig) -> Dict[str, List[str]]:
    """
    Bucket method:
    - Take Top cfg.top_base by BaseScore (HitsPerWeek).
    - From ranks cfg.due_from_rank..cfg.due_to_rank, take Top cfg.top_due by DueIndex (DaysSinceLastHit).
    """
    if stream_stats.empty:
        return {"base_top":[], "due_top":[], "combined":[]}

    s = stream_stats.copy()

    # Base top
    base = s.nsmallest(cfg.top_base, "RankPos")["Stream"].tolist()

    # Due bucket from rank range
    due_pool = s[(s["RankPos"] >= cfg.due_from_rank) & (s["RankPos"] <= cfg.due_to_rank)].copy()
    # When last-hit is NaN (never hit), treat as very due
    due_pool["DueIndex"] = due_pool["DaysSinceLastHit"].fillna(due_pool["DaysSinceLastHit"].max() if due_pool["DaysSinceLastHit"].notna().any() else 0) + 0.01
    due = due_pool.sort_values("DueIndex", ascending=False).head(cfg.top_due)["Stream"].tolist()

    combined = []
    for x in base + due:
        if x not in combined:
            combined.append(x)

    return {"base_top": base, "due_top": due, "combined": combined}

def top_dense_positions(pos_map: pd.DataFrame, top_k_positions: int = 10) -> List[int]:
    """
    Your definition: "Top30" == the Top-10 positions with most winners (HitCount),
    regardless of where those positions occur.
    """
    if pos_map.empty:
        return []
    tmp = pos_map.sort_values(["HitCount","RankPos"], ascending=[False, True]).head(top_k_positions)
    return sorted(tmp["RankPos"].tolist())

def engine_cluster_positions(df_24h_hits: pd.DataFrame, stream_stats: pd.DataFrame, top_n: int) -> List[int]:
    """
    Map 24h hits to rank positions (based on baseline rank positions in stream_stats),
    count by RankPos, return the top_n positions by 24h frequency.
    """
    if df_24h_hits.empty or stream_stats.empty:
        return []
    pos_map = dict(zip(stream_stats["Stream"], stream_stats["RankPos"]))
    df = df_24h_hits.copy()
    df["RankPos"] = df["Stream"].map(pos_map)
    df = df[df["RankPos"].notna()].copy()
    if df.empty:
        return []
    counts = df["RankPos"].value_counts().sort_values(ascending=False)
    top = counts.head(top_n).index.tolist()
    return sorted([int(x) for x in top])


# -------------------------
# Rare / Ultra-Rare engines
# -------------------------

def evaluate_rare_engine(
    df_all: pd.DataFrame,
    core: str,
    df_24h: pd.DataFrame,
    enable_r1: bool,
    enable_r2: bool,
    enable_r3: bool,
    enable_r4: bool,
    window_days_recent: int = 180,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Rare Engine checks (AAAB + AABB together):
      R1: stream is in top 20% for combined AAAB+AABB baseline rate (full history).
      R2: stream is in top 20% for combined AAAB+AABB rate in last 180 days.
      R3: the last 24h map contains ≥3 AAAB/AABB hits across ≥3 distinct streams (global condition).
      R4: last 24h AAAB/AABB hits cluster into Top-10 RankPos, and stream RankPos is in that set.
    Trigger: at least 3 of enabled checks True.

    Returns:
      - per-stream table with booleans and trigger
      - summary dict with thresholds and cluster sets
    """
    if df_all.empty:
        return pd.DataFrame(), {"error":"No history loaded."}

    core = canonical_core_key(core)
    df_hits_all = compute_core_hits(df_all, core, structures=["AAAB","AABB"])

    # Baseline stream stats
    base_stats = stream_summary(df_all, df_hits_all, window_days=min(365*5, int((df_all["Date"].max()-df_all["Date"].min()).days) or 365))
    # But we want baseline based on full history span; use hits per week in that span:
    span_days = max(int((df_all["Date"].max()-df_all["Date"].min()).days), 1)
    base_stats["HitsPerWeek_full"] = base_stats["HitsWindow"] / (span_days/7.0)
    base_stats = base_stats.sort_values(["HitsPerWeek_full","HitsWindow"], ascending=False).reset_index(drop=True)
    base_stats["RankPos_full"] = np.arange(1, len(base_stats)+1)

    # Recent stats (180d)
    recent_stats = stream_summary(df_all, df_hits_all, window_days=window_days_recent)

    # Thresholds
    def pct_threshold(series: pd.Series, pct: float) -> float:
        vals = series.dropna().values
        if len(vals)==0:
            return float("nan")
        return float(np.quantile(vals, pct))

    # R1 top 20% based on HitsPerWeek_full
    thr_r1 = pct_threshold(base_stats["HitsPerWeek_full"], 0.80)

    # R2 top 20% based on recent HitsPerWeek
    thr_r2 = pct_threshold(recent_stats["HitsPerWeek"], 0.80)

    # R3 global condition from 24h file: ≥3 hits across ≥3 distinct streams
    df_24h_core_hits = pd.DataFrame()
    top10_cluster = []
    r3_global = False
    if df_24h is not None and not df_24h.empty:
        df_24h_core_hits = compute_core_hits(df_24h, core, structures=["AAAB","AABB"])
        n_hits_24h = int(len(df_24h_core_hits))
        n_streams_24h = int(df_24h_core_hits["Stream"].nunique()) if n_hits_24h else 0
        r3_global = (n_hits_24h >= 3) and (n_streams_24h >= 3)
        # R4 cluster based on 24h file (Top-10 RankPos positions by 24h frequency)
        top10_cluster = engine_cluster_positions(
            df_24h_core_hits,
            base_stats.rename(columns={"RankPos_full":"RankPos"}).assign(RankPos=base_stats["RankPos_full"]),
            top_n=10,
        )

    # Merge per stream
    out = pd.DataFrame({"Stream": base_stats["Stream"]})
    out = out.merge(base_stats[["Stream","HitsPerWeek_full","RankPos_full"]], on="Stream", how="left")
    out = out.merge(recent_stats[["Stream","HitsPerWeek","DaysSinceLastHit","RankPos"]].rename(columns={"RankPos":"RankPos_recent"}), on="Stream", how="left")

    out["R1_top20_baseline"] = out["HitsPerWeek_full"] >= thr_r1 if enable_r1 else False
    out["R2_top20_recent"] = out["HitsPerWeek"] >= thr_r2 if enable_r2 else False
    out["R3_24h_has_3plus_across_3streams"] = r3_global if enable_r3 else False
    out["R4_24h_cluster_top10pos"] = out["RankPos_full"].isin(top10_cluster) if enable_r4 else False

    enabled_cols = [c for c, en in [
        ("R1_top20_baseline", enable_r1),
        ("R2_top20_recent", enable_r2),
        ("R3_24h_has_3plus_across_3streams", enable_r3),
        ("R4_24h_cluster_top10pos", enable_r4),
    ] if en]

    out["ChecksTrue"] = out[enabled_cols].sum(axis=1) if enabled_cols else 0
    out["RareEngine_TRIG"] = out["ChecksTrue"] >= 3 if enabled_cols else False

    out = out.sort_values(["RareEngine_TRIG","ChecksTrue","HitsPerWeek_full"], ascending=[False, False, False]).reset_index(drop=True)

    summary = {
        "thr_r1": thr_r1,
        "thr_r2": thr_r2,
        "r3_global": r3_global,
        "top10_cluster_positions": top10_cluster,
        "n_24h_core_hits": int(len(df_24h_core_hits)) if df_24h is not None else 0,
        "n_24h_core_streams": int(df_24h_core_hits["Stream"].nunique()) if df_24h is not None and not df_24h_core_hits.empty else 0,
        "span_days_full": span_days,
        "enabled_checks": enabled_cols,
    }
    return out, summary

def evaluate_ultra_rare_engine(
    df_all: pd.DataFrame,
    core: str,
    df_24h: pd.DataFrame,
    enable_q1: bool,
    enable_q2: bool,
    enable_q3: bool,
    enable_q4: bool,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Ultra-Rare Engine checks (AAAA quads for the core's digits):
      Q1: stream in top 10% for quad baseline rate (full history)
      Q2: days since last quad >= 90th percentile across streams
      Q3: last 24h has at least 1 quad anywhere (for any digit in core)
      Q4: 24h quad hits (for core) cluster into Top-5 RankPos; stream position is in that set
    Trigger: at least 2 of enabled checks True.
    """
    if df_all.empty:
        return pd.DataFrame(), {"error":"No history loaded."}

    core = canonical_core_key(core)
    df_hits_all = compute_core_hits(df_all, core, structures=["AAAA"])

    span_days = max(int((df_all["Date"].max()-df_all["Date"].min()).days), 1)
    base_stats = stream_summary(df_all, df_hits_all, window_days=min(365*5, span_days))
    base_stats["HitsPerWeek_full"] = base_stats["HitsWindow"] / (span_days/7.0)
    base_stats = base_stats.sort_values(["HitsPerWeek_full","HitsWindow"], ascending=False).reset_index(drop=True)
    base_stats["RankPos_full"] = np.arange(1, len(base_stats)+1)

    # thresholds
    def pct_threshold(series: pd.Series, pct: float) -> float:
        vals = series.dropna().values
        if len(vals)==0:
            return float("nan")
        return float(np.quantile(vals, pct))

    thr_q1 = pct_threshold(base_stats["HitsPerWeek_full"], 0.90)
    thr_q2 = pct_threshold(base_stats["DaysSinceLastHit"], 0.90)  # DaysSinceLastHit computed from last quad date

    # Q3 global 24h quad exists for any core digit
    q3_global = False
    df_24h_core_hits = pd.DataFrame()
    top5_cluster = []
    if df_24h is not None and not df_24h.empty:
        # any quad in 24h that uses one of core digits
        core_digits = set(core)
        df_24h_quads = df_24h[df_24h["Structure"]=="AAAA"].copy()
        df_24h_quads["quad_digit"] = df_24h_quads["Pick4"].str[0]
        q3_global = df_24h_quads["quad_digit"].isin(core_digits).any()
        df_24h_core_hits = compute_core_hits(df_24h, core, structures=["AAAA"])
        top5_cluster = engine_cluster_positions(df_24h_core_hits, base_stats.rename(columns={"RankPos_full":"RankPos"}).assign(RankPos=base_stats["RankPos_full"]), top_n=5)

    out = pd.DataFrame({"Stream": base_stats["Stream"]})
    out = out.merge(base_stats[["Stream","HitsPerWeek_full","DaysSinceLastHit","RankPos_full"]], on="Stream", how="left")

    out["Q1_top10_baseline"] = out["HitsPerWeek_full"] >= thr_q1 if enable_q1 else False
    out["Q2_due_p90"] = out["DaysSinceLastHit"] >= thr_q2 if enable_q2 else False
    out["Q3_24h_quad_exists"] = q3_global if enable_q3 else False
    out["Q4_24h_cluster_top5pos"] = out["RankPos_full"].isin(top5_cluster) if enable_q4 else False

    enabled_cols = [c for c, en in [
        ("Q1_top10_baseline", enable_q1),
        ("Q2_due_p90", enable_q2),
        ("Q3_24h_quad_exists", enable_q3),
        ("Q4_24h_cluster_top5pos", enable_q4),
    ] if en]

    out["ChecksTrue"] = out[enabled_cols].sum(axis=1) if enabled_cols else 0
    out["UltraRare_TRIG"] = out["ChecksTrue"] >= 2 if enabled_cols else False

    out = out.sort_values(["UltraRare_TRIG","ChecksTrue","HitsPerWeek_full"], ascending=[False, False, False]).reset_index(drop=True)

    summary = {
        "thr_q1": thr_q1,
        "thr_q2": thr_q2,
        "q3_global": q3_global,
        "top5_cluster_positions": top5_cluster,
        "n_24h_core_quad_hits": int(len(df_24h_core_hits)) if df_24h is not None else 0,
        "enabled_checks": enabled_cols,
        "span_days_full": span_days,
    }
    return out, summary


# -------------------------
# UI
# -------------------------

st.set_page_config(page_title="Pick 4 Northern Star", layout="wide")

st.title("Pick 4 — Northern Star + Rare Engine (AAAB+AABB) + Ultra‑Rare (AAAA)")

# Safe init for sidebar footer (values are filled after parsing uploads)
last_all = None
last_24 = None
df_all = None
df_24 = None


with st.sidebar:
    st.header("Data")
    master_file = st.file_uploader("All‑states history file (.csv or LotteryPost .txt)", type=["csv","txt"])
    map24_file = st.file_uploader("24h map file (optional, same format)", type=["csv","txt"])

    exclude_md = st.checkbox("Exclude Maryland streams", value=True)

    # Percentile tools (updateable)
    if "pos_map_cache" not in st.session_state:
        st.session_state["pos_map_cache"] = {}
    if "recompute_token" not in st.session_state:
        st.session_state["recompute_token"] = ""
    if st.button("Recompute percentile maps now"):
        st.session_state["pos_map_cache"] = {}
        st.session_state["recompute_token"] = datetime.datetime.now().isoformat(timespec="seconds")
    if st.session_state.get("recompute_token"):
        st.caption(f"Percentiles last recomputed: {st.session_state['recompute_token']}")
    st.divider()

    st.header("Core selection")
    # Pick a single core to view (dropdown), and optionally a larger set for cache building.
    CORE_OPTIONS = ['016', '017', '018', '019', '023', '024', '025', '027', '028', '029', '038', '046', '048', '056', '059', '067', '068', '078', '129', '135', '145', '146', '149', '167', '168', '169', '179', '236', '238', '239', '245', '246', '249', '257', '258', '278', '279', '345', '348', '357', '359', '378', '379', '389', '457', '459', '489', '567', '579', '589', '679', '689', '789']
    view_core = st.selectbox("View core (dropdown)", options=CORE_OPTIONS, index=CORE_OPTIONS.index("389") if "389" in CORE_OPTIONS else 0)
    cores_for_cache = st.multiselect("Cores to include for cache building / batch tools", options=CORE_OPTIONS, default=[view_core])
    show_tabs = st.checkbox("Show tabs for all selected cores (can be slower)", value=False)
    cores = sorted(list(dict.fromkeys((cores_for_cache if show_tabs else [view_core]))))

    st.divider()
    st.header("Northern Star window")
    window_days = st.radio("Window (days)", options=[180, 365], index=0, horizontal=True)
    cfg = RankConfig(window_days=window_days)

    st.caption("Bucket method: Top 12 BaseScore (Hits/week) + Top 8 DueIndex from ranks 13–60.")

    st.divider()
    st.header("Rare Engine trigger — AAAB + AABB")
    r1 = st.checkbox("R1: Top‑20% baseline AAAB+AABB", value=True)
    r2 = st.checkbox("R2: Top‑20% recent (last window)", value=True)
    r3 = st.checkbox("R3: 24h has ≥3 AAAB/AABB hits across ≥3 streams", value=True)
    r4 = st.checkbox("R4: 24h cluster ∈ Top‑10 positions", value=True)

    st.divider()
    st.header("Ultra‑Rare trigger — AAAA")
    q1 = st.checkbox("Q1: Top‑10% quad baseline", value=True)
    q2 = st.checkbox("Q2: Due pressure ≥ P90", value=True)
    q3 = st.checkbox("Q3: 24h quad exists (core digits)", value=True)
    q4 = st.checkbox("Q4: 24h cluster ∈ Top‑5 positions", value=True)

    st.divider()
    straights_opt = st.checkbox("Generate straights shortlist (optional last)", value=False)


# Load data
df_all = try_read_tablelike(master_file) if master_file else pd.DataFrame()
df_24h = try_read_tablelike(map24_file) if map24_file else pd.DataFrame()

if exclude_md and not df_all.empty:
    df_all = df_all[df_all["State"].astype(str).str.strip().str.lower() != "maryland"].copy()
if exclude_md and not df_24h.empty:
    df_24h = df_24h[df_24h["State"].astype(str).str.strip().str.lower() != "maryland"].copy()


# Auto-clear cached percentile maps when input data changes
all_hash = file_fingerprint(master_file)
map_hash = file_fingerprint(map24_file)
if "data_hash_all" not in st.session_state:
    st.session_state["data_hash_all"] = ""
if "data_hash_24h" not in st.session_state:
    st.session_state["data_hash_24h"] = ""

if all_hash and all_hash != st.session_state["data_hash_all"]:
    st.session_state["pos_map_cache"] = {}
    st.session_state["data_hash_all"] = all_hash
    st.session_state["recompute_token"] = ""  # will refresh on next compute
if map_hash != st.session_state["data_hash_24h"]:
    st.session_state["pos_map_cache"] = {}
    st.session_state["data_hash_24h"] = map_hash
    st.session_state["recompute_token"] = ""

# Show data freshness (so the instructions never need updating)
last_all = most_recent_date(df_all)
last_24 = most_recent_date(df_24h)

today = datetime.date.today()
def _age_days(ts: Optional[pd.Timestamp]) -> Optional[int]:
    if ts is None or pd.isna(ts):
        return None
    try:
        d = pd.to_datetime(ts).date()
        return (today - d).days
    except Exception:
        return None

st.sidebar.markdown("### Data freshness")
if last_all is not None and not pd.isna(last_all):
    st.sidebar.caption(f"All‑states history most recent date: {pd.to_datetime(last_all).date()}  (age: {_age_days(last_all)} days)")
else:
    st.sidebar.caption("All‑states history most recent date: (not found)")

if last_24 is not None and not pd.isna(last_24):
    st.sidebar.caption(f"24h map most recent date: {pd.to_datetime(last_24).date()}  (age: {_age_days(last_24)} days)")
else:
    st.sidebar.caption("24h map most recent date: (not uploaded)")

st.sidebar.caption(f"Tip: if ages are >1–2 days, your files are probably behind.")

if master_file is None:
    st.info("Upload your all‑states history file to start.")
    st.stop()

if df_all.empty:
    st.error("Could not parse your history file. Make sure it contains Date, State, Game, Results.")
    st.stop()

# One place to show dataset info
colA, colB, colC = st.columns(3)
with colA:
    st.metric("Rows (draws)", f"{len(df_all):,}")
with colB:
    st.metric("Streams", f"{df_all['Stream'].nunique():,}")
with colC:
    st.markdown(f"<div style='font-size:10px; line-height:1.1;'><b>Date span</b><br>{df_all['Date'].min().date()} → {df_all['Date'].max().date()}</div>", unsafe_allow_html=True)


st.divider()
# ----------------------------
# Baseline disk cache (optional)
# ----------------------------
def _baseline_paths(core: str, window_days: int):
    core = str(core).zfill(3)
    base = DISK_CACHE_DIR / f"baseline_{window_days}d_{core}"
    return {
        "stream": base.with_suffix(".stream.parquet"),
        "pos": base.with_suffix(".pos.parquet"),
        "meta": base.with_suffix(".meta.json"),
    }

def _load_baseline_from_disk(core: str, window_days: int, expected_last_date: str | None):
    p = _baseline_paths(core, window_days)
    meta = _read_meta(p["meta"])
    if not meta:
        return None, None, None
    if expected_last_date and meta.get("last_date") != expected_last_date:
        return None, None, meta
    stream_df = _safe_read_table(p["stream"])
    pos_df = _safe_read_table(p["pos"])
    if stream_df is None or pos_df is None:
        return None, None, meta
    return stream_df, pos_df, meta

def _save_baseline_to_disk(core: str, window_days: int, stream_df, pos_df, last_date: str | None):
    p = _baseline_paths(core, window_days)
    _safe_write_table(stream_df, p["stream"])
    _safe_write_table(pos_df, p["pos"])
    _write_meta(p["meta"], {
        "core": str(core).zfill(3),
        "window_days": int(window_days),
        "last_date": last_date,
        "built_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

def get_stream_stats_cached(core: str, window_days: int, df_all, last_date: str | None):
    key = f"stream_stats::{window_days}::{str(core).zfill(3)}"
    if key in st.session_state:
        return st.session_state[key]
    # Try disk cache
    stream_df, pos_df, meta = _load_baseline_from_disk(core, window_days, expected_last_date=last_date)
    if stream_df is not None:
        st.session_state[key] = stream_df
        # also stash pos if present
        if pos_df is not None:
            st.session_state[f"pos_map::{window_days}::{str(core).zfill(3)}"] = pos_df
        return stream_df
    # Compute fresh
    hits = compute_core_hits(df_all, core, structures=("AABC",))
    stream_df = stream_summary(df_all, hits, window_days=window_days)
    st.session_state[key] = stream_df
    return stream_df

def get_pos_map_cached(core: str, window_days: int, stream_stats_df, last_date: str | None):
    k = f"pos_map::{window_days}::{str(core).zfill(3)}"
    if k in st.session_state:
        return st.session_state[k]
    # If stream_stats was loaded from disk, pos may already be in session_state
    pos_df = position_percentile_map(stream_stats_df)
    st.session_state[k] = pos_df
    # Save to disk alongside stream stats (best effort)
    try:
        _save_baseline_to_disk(core, window_days, stream_stats_df, pos_df, last_date)
    except Exception:
        pass
    return pos_df

# Baseline cache builder UI (runs only after history is loaded)
st.subheader("Baseline cache builder")
st.caption(
    "Build/refresh an on-disk cache for the selected cores so daily runs stay fast. "
    "Cache auto-invalidates when your history file’s latest date changes."
)

build_both = st.checkbox("Build cache for both windows (180 & 365)", value=False)
if st.button("Build baseline cache now"):
    if df_all is None or df_all.empty:
        st.warning("Upload a history file first.")
    else:
        build_windows = [180, 365] if build_both else [window_days]
        built = 0
        for w in build_windows:
            for c in cores_for_cache:
                stream_df = get_stream_stats_cached(c, w, df_all=df_all, last_date=last_all)
                pos_df = get_pos_map_cached(c, w, stream_df, last_date=last_all)
                try:
                    _save_baseline_to_disk(c, w, stream_df, pos_df, last_all)
                    built += 1
                except Exception:
                    pass
        st.success(f"Cache built for {built} core-window combinations. Latest history date: {last_all}")


tab_labels = ["Northern Lights (Master playlist)"] + [f"Core {c}" for c in cores]
tabs = st.tabs(tab_labels)

# --- Northern Lights master playlist (best -> worst across streams/cores) ---
with tabs[0]:
    st.subheader("Northern Lights master playlist")
    st.caption(
        "This aggregates the Northern Star bucket picks across your selected cores and ranks them using a universal score "
        "(recent hits/week + due + position-percentile strength)."
    )

    rows = []
    for core in cores:
        stats = get_stream_stats_cached(core, cfg.window_days, df_all=df_all, last_date=last_all)
        pos_map = get_pos_map_cached(core, cfg.window_days, stats, last_date=last_all)

        # quick lookup: rankpos -> HitCountPctile
        pos_lookup = {int(r["RankPos"]): float(r["HitCountPctile"]) for _, r in pos_map.iterrows()}

        # build combined list + label bucket origin
        buckets = bucket_recommendations(stats, cfg)
        base_set = set(buckets["base_top"])
        due_set = set(buckets["due_top"])
        for stream in buckets["combined"]:
            r = stats[stats["Stream"] == stream].head(1)
            if r.empty:
                continue
            r0 = r.iloc[0]
            bucket = "base_top" if stream in base_set else ("due_top" if stream in due_set else "other")
            rankpos = int(r0["RankPos"])
            rows.append({
                "Core": core,
                "Stream": stream,
                "Bucket": bucket,
                "RankPos": rankpos,
                "HitsWindow": float(r0["HitsWindow"]),
                "HitsPerWeek": float(r0["HitsPerWeek"]),
                "DaysSinceLastHit": float(r0["DaysSinceLastHit"]) if pd.notna(r0["DaysSinceLastHit"]) else np.nan,
                "PosHitCountPctile": pos_lookup.get(rankpos, np.nan),
            })

    df_master = pd.DataFrame(rows)
    if df_master.empty:
        st.info("No master playlist yet (check your cores and file inputs).")
    else:
        # universal score components across the entire master list
        df_master = df_master.copy()
        df_master["HitsPerWeekPct"] = df_master["HitsPerWeek"].rank(pct=True) * 100.0
        df_master["DuePct"] = df_master["DaysSinceLastHit"].rank(pct=True) * 100.0
        df_master["DuePct"] = df_master["DuePct"].fillna(100.0)  # never-hit -> maximally due

        # score weights (kept simple + testable)
        w_hits, w_due, w_pos = 0.50, 0.30, 0.20
        df_master["MasterScore"] = (
            w_hits * df_master["HitsPerWeekPct"] +
            w_due  * df_master["DuePct"] +
            w_pos  * df_master["PosHitCountPctile"].fillna(df_master["PosHitCountPctile"].median())
        ).round(2)

        df_master = df_master.sort_values(["MasterScore","HitsPerWeek","DuePct"], ascending=False).reset_index(drop=True)
        df_master.insert(0, "MasterRank", np.arange(1, len(df_master)+1))

        st.dataframe(df_master, use_container_width=True, hide_index=True)

        csv_bytes = df_master.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download master playlist CSV",
            data=csv_bytes,
            file_name="pk4_northern_lights_master_playlist.csv",
            mime="text/csv",
        )

        st.caption(
            "Interpretation: higher MasterScore = stronger overall signal (recent strength + due pressure + position map support). "
            "If you want to make this more conservative, we can weight HitsPerWeek higher and Due lower."
        )

# --- Per-core tabs ---
for core, tab in zip(cores, tabs[1:]):
    with tab:
        st.subheader(f"Core {core}")

        # Base (AABC doubles) core hits + stream ranking
        stats = get_stream_stats_cached(core, cfg.window_days, df_all=df_all, last_date=last_all)
        pos_map = get_pos_map_cached(core, cfg.window_days, stats, last_date=last_all)
        top30_positions = top_dense_positions(pos_map, top_k_positions=10)
        capture = None
        if not pos_map.empty and pos_map["HitCount"].sum() > 0:
            capture = pos_map[pos_map["RankPos"].isin(top30_positions)]["HitCount"].sum() / pos_map["HitCount"].sum() * 100.0

        # Bucket recommendations
        buckets = bucket_recommendations(stats, cfg)

        left, right = st.columns([1.3, 1.0])

        with left:
            st.markdown("### Stream ranking (AABC doubles)")
            show = stats[["RankPos","Stream","HitsWindow","HitsPerWeek","DaysSinceLastHit"]].copy()
            show = show.rename(columns={
                "HitsWindow":"Hits",
                "HitsPerWeek":"Hits/week",
                "DaysSinceLastHit":"Days since last hit",
            })
            st.dataframe(show, use_container_width=True, hide_index=True)

        with right:
            st.markdown("### Northern Star buckets")
            st.write("**Top 12 (BaseScore)**")
            st.write("\n".join([f"- {s}" for s in buckets["base_top"]]) if buckets["base_top"] else "_None_")

            st.write("**Top 8 DueIndex (Ranks 13–60)**")
            st.write("\n".join([f"- {s}" for s in buckets["due_top"]]) if buckets["due_top"] else "_None_")

            st.write("**Combined list**")
            st.write("\n".join([f"- {s}" for s in buckets["combined"]]) if buckets["combined"] else "_None_")

            if straights_opt:
                st.markdown("### Straights shortlist (optional)")
                members = members_from_core(core, "AABC")
                st.caption("All permutations are shown (not ranked yet).")
                for m in members:
                    perms = sorted(set("".join(p) for p in itertools.permutations(m)))
                    st.write(f"**Box {box_key(m)}** → {', '.join(perms[:24])}" + (" …" if len(perms) > 24 else ""))

        st.divider()

        with st.expander("Northern Star percentile stats (rank‑position map 1..N)", expanded=False):
            st.caption("Each position is labeled individually. 'Top30' is the Top‑10 positions by winner count in this window.")
            if pos_map.empty:
                st.info("No data for this core in the selected window.")
            else:
                st.write(f"**Top30 positions (Top‑10 by HitCount):** {top30_positions}")
                if capture is not None:
                    st.write(f"**Top30 capture rate:** {capture:.1f}% of hits in the window")
                st.dataframe(pos_map, use_container_width=True, hide_index=True)

        st.divider()

        # Rare Engine panel
        st.markdown("## Rare Engine — AAAB + AABB")
        rare_tbl, rare_summary = evaluate_rare_engine(
            df_all=df_all,
            core=core,
            df_24h=df_24h,
            enable_r1=r1, enable_r2=r2, enable_r3=r3, enable_r4=r4,
            window_days_recent=cfg.window_days
        )

        c1, c2 = st.columns(2)
        with c1:
            st.caption("Trigger rule: at least **3** enabled checks must be TRUE (per stream).")
            if "error" in rare_summary:
                st.error(rare_summary["error"])
            else:
                st.write(f"R1 threshold (top‑20% baseline): **{rare_summary['thr_r1']:.4f} hits/week**")
                st.write(f"R2 threshold (top‑20% recent): **{rare_summary['thr_r2']:.4f} hits/week**")
                st.write(f"R3 global (24h ≥3 hits across ≥3 streams): **{rare_summary['r3_global']}**")
                st.write(f"R4 Top‑10 cluster positions (24h): **{rare_summary['top10_cluster_positions']}**")
                st.write(f"24h core hits: **{rare_summary['n_24h_core_hits']}** across **{rare_summary['n_24h_core_streams']}** streams")

        with c2:
            # Quick readout: which streams fire?
            if not rare_tbl.empty:
                trig_streams = rare_tbl.loc[rare_tbl["RareEngine_TRIG"], "Stream"].tolist()
                st.write("**Streams where Rare Engine fires:**")
                st.write("\n".join([f"- {s}" for s in trig_streams]) if trig_streams else "_None_")

                st.write("**When it fires, add these boxes:**")
                members = members_from_core(core, "AAAB") + members_from_core(core, "AABB")
                boxes = sorted(set(box_key(m) for m in members))
                st.write(", ".join(boxes))
            else:
                st.info("No rare-engine table yet.")

        if not rare_tbl.empty:
            cols = ["Stream","RankPos_full","HitsPerWeek_full","HitsPerWeek","DaysSinceLastHit",
                    "R1_top20_baseline","R2_top20_recent","R3_24h_has_3plus_across_3streams","R4_24h_cluster_top10pos",
                    "ChecksTrue","RareEngine_TRIG"]
            view = rare_tbl[cols].copy()
            view = view.rename(columns={
                "RankPos_full":"RankPos (baseline)",
                "HitsPerWeek_full":"Baseline hits/week",
                "HitsPerWeek":"Recent hits/week",
                "DaysSinceLastHit":"Days since last rare hit",
                "R1_top20_baseline":"R1",
                "R2_top20_recent":"R2",
                "R3_24h_has_3plus_across_3streams":"R3 (24h ≥3 hits/≥3 streams)",
                "R4_24h_cluster_top10pos":"R4",
                "ChecksTrue":"# checks TRUE",
                "RareEngine_TRIG":"RARE ENGINE",
            })
            st.dataframe(view, use_container_width=True, hide_index=True)

        st.divider()

        # Ultra-rare engine panel
        st.markdown("## Ultra‑Rare — Quads (AAAA)")
        ultra_tbl, ultra_summary = evaluate_ultra_rare_engine(
            df_all=df_all,
            core=core,
            df_24h=df_24h,
            enable_q1=q1, enable_q2=q2, enable_q3=q3, enable_q4=q4
        )

        u1, u2 = st.columns(2)
        with u1:
            st.caption("Trigger rule: at least **2** enabled checks must be TRUE (per stream).")
            if "error" in ultra_summary:
                st.error(ultra_summary["error"])
            else:
                st.write(f"Q1 threshold (top‑10% baseline): **{ultra_summary['thr_q1']:.4f} hits/week**")
                st.write(f"Q2 threshold (P90 due): **{ultra_summary['thr_q2']:.0f} days**")
                st.write(f"Q3 24h quad exists (core digits): **{ultra_summary['q3_global']}**")
                st.write(f"Q4 Top‑5 cluster positions (24h): **{ultra_summary['top5_cluster_positions']}**")
                st.write(f"24h core quad hits: **{ultra_summary['n_24h_core_quad_hits']}**")

        with u2:
            if not ultra_tbl.empty:
                trig_streams = ultra_tbl.loc[ultra_tbl["UltraRare_TRIG"], "Stream"].tolist()
                st.write("**Streams where Ultra‑Rare fires:**")
                st.write("\n".join([f"- {s}" for s in trig_streams]) if trig_streams else "_None_")
                st.write("**When it fires, add these quad boxes:**")
                q_members = members_from_core(core, "AAAA")
                st.write(", ".join(sorted(set(box_key(m) for m in q_members))))
            else:
                st.info("No ultra-rare table yet.")

        if not ultra_tbl.empty:
            cols = ["Stream","RankPos_full","HitsPerWeek_full","DaysSinceLastHit",
                    "Q1_top10_baseline","Q2_due_p90","Q3_24h_quad_exists","Q4_24h_cluster_top5pos",
                    "ChecksTrue","UltraRare_TRIG"]
            view = ultra_tbl[cols].copy()
            view = view.rename(columns={
                "RankPos_full":"RankPos (baseline)",
                "HitsPerWeek_full":"Baseline hits/week",
                "DaysSinceLastHit":"Days since last quad",
                "ChecksTrue":"# checks TRUE",
                "UltraRare_TRIG":"ULTRA‑RARE",
            })
            st.dataframe(view, use_container_width=True, hide_index=True)

        st.caption("Tip: Quads are extremely sparse. Use Ultra‑Rare as a *bonus watch*—not a daily expectation.")
