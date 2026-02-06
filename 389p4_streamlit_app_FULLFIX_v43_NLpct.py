from __future__ import annotations
# pk4_northern_star_app_2026-02-04_v41.py
# Streamlit app: Pick 4 "Northern Star" core stream ranking + Rare/Ultra-Rare engines (AAAB+AABB, AAAA)
# Notes:
# - Designed to work with LotteryPost-style exports (tab .txt or .csv) that include Date, State, Game, Results.
# - Ignores Wild Ball / Fireball / multipliers by extracting the first 4 digits like "1-2-3-4".
# - Excludes Maryland by default (toggle in sidebar).


APP_VERSION = "v42+NLpct"

# Core presets (family IDs) shown in the UI. Keep this list additive.
# These are the cores you and I have explicitly worked on so far.
CORE_PRESETS = [
    "016",
    "129",
    "278",
    "359",
    "389",
    "457",
]

import re

def _safe_int(x):
    """Convert x to int if possible; returns None if not."""
    if x is None:
        return None
    if isinstance(x, int):
        return int(x)
    try:
        import numpy as _np
        if isinstance(x, (_np.integer,)):
            return int(x)
    except Exception:
        pass
    if isinstance(x, str):
        m = re.search(r"\d+", x)
        if not m:
            return None
        try:
            return int(m.group(0))
        except Exception:
            return None
    try:
        return int(x)
    except Exception:
        return None

import math
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable, Any

import numpy as np
import pandas as pd
import streamlit as st
import hashlib
import datetime
import json
from functools import lru_cache

# ---- Safe defaults to prevent NameError during first render ----
cfg = None  # set later after window selection
exclude_md = True  # default behavior: exclude Maryland unless user toggles off
map_file = None  # backward-compatible alias set in sidebar





# -------------------------
# Parsing + helpers
# -------------------------

# -------------------------
# Disk baseline cache (optional, keeps runs fast)
# -------------------------
from pathlib import Path as _Path

DISK_CACHE_DIR = _Path("pk4_baseline_cache")
DISK_CACHE_DIR.mkdir(exist_ok=True)

# -------------------------
# Rolling baseline store (optional)
# - Lets the app "self-maintain" a rolling ~3-year history by appending from the 24h file
# - Purges rows older than ~3 years from the newest date in the store
# - Stored on disk as parquet (preferred) or CSV (fallback), plus a small JSON meta file
# -------------------------
BASELINE_STORE_DIR = _Path("pk4_baseline_store")
BASELINE_STORE_DIR.mkdir(exist_ok=True)
BASELINE_STORE_BASE = BASELINE_STORE_DIR / "pk4_allstates_rolling_3y"


def _ensure_list(x):
    """Return x as a list suitable for pandas .isin()."""
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    # pandas Series / Index
    try:
        import pandas as _pd
        if isinstance(x, (_pd.Series, _pd.Index)):
            return x.tolist()
    except Exception:
        pass
    # scalar -> list
    return [x]

def _coerce_store_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    # Ensure Date is datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df[df["Date"].notna()].copy()
    # Ensure required columns exist (best-effort)
    for c in ["State","Game","Results","Pick4","Structure","Box","Stream"]:
        if c not in df.columns:
            df[c] = None
    return df

def load_baseline_store() -> pd.DataFrame:
    df = _safe_read_table(BASELINE_STORE_BASE)
    if df is None:
        return pd.DataFrame()
    df = _coerce_store_df(df)
    # Recompute derived fields if store was CSV without them
    if not df.empty:
        if "Pick4" not in df.columns or df["Pick4"].isna().all():
            df["Pick4"] = df.get("Results", pd.Series([None]*len(df))).map(extract_pick4_digits)
        if "Structure" not in df.columns or df["Structure"].isna().all():
            df["Structure"] = df["Pick4"].map(structure_of_4)
        if "Box" not in df.columns or df["Box"].isna().all():
            df["Box"] = df["Pick4"].map(box_key)
        if "Stream" not in df.columns or df["Stream"].isna().all():
            df["Stream"] = df["State"].astype(str).str.strip() + " | " + df["Game"].astype(str).str.strip()
        df = df[df["Pick4"].notna()].copy()
    return df

def write_baseline_store(df: pd.DataFrame, note: str = "") -> Tuple[bool, str]:
    df = _coerce_store_df(df)
    ok, path_written = _safe_write_table(df, BASELINE_STORE_BASE)
    meta = _read_meta(BASELINE_STORE_BASE)
    meta.update({
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "note": note,
        "rows": int(df.shape[0]) if df is not None else 0,
        "max_date": str(df["Date"].max()) if df is not None and not df.empty else "",
    })
    _write_meta(BASELINE_STORE_BASE, meta)
    return ok, path_written

def purge_to_rolling_3y(df: pd.DataFrame, years: int = 3) -> pd.DataFrame:
    df = _coerce_store_df(df)
    if df.empty:
        return df
    max_date = df["Date"].max()
    # 3-year rolling window; add a small buffer for leap years
    cutoff = pd.Timestamp(max_date) - pd.Timedelta(days=(365*years + 7))
    df2 = df[df["Date"] >= cutoff].copy()
    return df2

def append_from_24h(df_store: pd.DataFrame, df_24h: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    df_store = _coerce_store_df(df_store)
    df_24h = _coerce_store_df(df_24h)
    if df_24h.empty:
        return df_store, 0
    # Only keep rows with the needed fields
    df_24h = df_24h[["Date","State","Game","Results","Pick4","Structure","Box","Stream"]].copy()
    df_store = df_store[["Date","State","Game","Results","Pick4","Structure","Box","Stream"]].copy() if not df_store.empty else df_store

    # Dedup key: Date + State + Game (unique stream-day draw)
    def _key(df):
        return (
            df["Date"].dt.strftime("%Y-%m-%d").astype(str)
            + "|" + df["State"].astype(str).str.strip().str.lower()
            + "|" + df["Game"].astype(str).str.strip().str.lower()
        )
    if df_store.empty:
        out = df_24h.copy()
        return out, int(out.shape[0])

    store_keys = set(_key(df_store).tolist())
    df_24h["_k"] = _key(df_24h)
    new_rows = df_24h[~df_24h["_k"].isin(store_keys)].drop(columns=["_k"]).copy()
    if new_rows.empty:
        return df_store, 0
    out = pd.concat([df_store, new_rows], ignore_index=True)
    # Final dedup safety
    out["_k"] = _key(out)
    out = out.drop_duplicates(subset=["_k"]).drop(columns=["_k"]).copy()
    return out, int(new_rows.shape[0])

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
    # Accept either a "base" path (no suffix) or a fully-qualified .parquet/.csv path.
    try:
        if path.suffix.lower() == ".parquet" and path.exists():
            return pd.read_parquet(path)
        if path.suffix.lower() == ".csv" and path.exists():
            return pd.read_csv(path)

        p_parq = path.with_suffix(".parquet")
        p_csv = path.with_suffix(".csv")
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


from itertools import permutations

def extract_4digit(x: Any) -> Optional[str]:
    """Best-effort normalize to a 4-digit string (used for straight permutation generation)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    # If already exactly 4 digits
    m = re.search(r"(?<!\d)(\d{4})(?!\d)", s)
    if m:
        return m.group(1)
    # Try LotteryPost hyphenated format
    return extract_pick4_digits(s)


@lru_cache(maxsize=10000)
def unique_straights_for_box(box4: str) -> tuple[str, ...]:
    """Return unique 4-digit straight permutations for a 4-digit string (digits may repeat).

    Cached because the same box patterns repeat across streams/days.
    """
    box4 = extract_4digit(box4)
    if not box4:
        return tuple()
    return tuple(sorted({"".join(p) for p in permutations(box4, 4)}))

def _value_counts_result(df: pd.DataFrame) -> pd.Series:
    """Safe value_counts for Result column."""
    if df is None or df.empty:
        return pd.Series(dtype=int)
    if "Result" not in df.columns:
        return pd.Series(dtype=int)
    return df["Result"].astype(str).value_counts()

def _get_stream_result_counts(df_all: pd.DataFrame, stream: str) -> pd.Series:
    """Value counts of Result within a single stream."""
    if df_all is None or df_all.empty:
        return pd.Series(dtype=int)
    if "Stream" not in df_all.columns:
        return pd.Series(dtype=int)
    sub = df_all[df_all["Stream"].astype(str) == str(stream)]
    return _value_counts_result(sub)


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
        return [box_key(x) for x in [a+a+b+c, b+b+a+c, c+c+a+b]]
    if structure == "AAAB":
        # Triples: one digit x3 + one other digit x1 => 6 members
        return [box_key(x) for x in [a+a+a+b, a+a+a+c, b+b+b+a, b+b+b+c, c+c+c+a, c+c+c+b]]
    if structure == "AABB":
        # Double-doubles: choose two digits x2 => 3 members
        return [box_key(x) for x in [a+a+b+b, a+a+c+c, b+b+c+c]]
    if structure == "AAAA":
        return [box_key(x) for x in [a*4, b*4, c*4]]
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
    # compatibility aliases used in other modules
    df["Result"] = df["Pick4"]

    df["Structure"] = df["Pick4"].map(structure_of_4)
    df["Box"] = df["Pick4"].map(box_key)
    df["BoxKey4"] = df["Box"]
    df["Stream"] = df["State"].astype(str).str.strip() + " | " + df["Game"].astype(str).str.strip()
    return df



def try_read_picklist(uploaded) -> pd.DataFrame:
    """
    Accept a simple list of Pick4 numbers (previous-day file) in .txt or .csv form.
    Extracts 4-digit sequences anywhere in the file.
    Returns a dataframe with: Result, Pick4, Box, BoxKey4, Structure.
    """
    if uploaded is None:
        return pd.DataFrame()

    try:
        raw = uploaded.read()
    except Exception:
        raw = None

    # Reset pointer for possible re-reads by caller
    try:
        uploaded.seek(0)
    except Exception:
        pass

    if raw is None:
        return pd.DataFrame()

    if isinstance(raw, bytes):
        try:
            s = raw.decode("utf-8", errors="ignore")
        except Exception:
            s = str(raw)
    else:
        s = str(raw)

    # Common case: one number per line, but we accept any separators
    nums = re.findall(r"(?<!\d)(\d{4})(?!\d)", s)
    if not nums:
        # try to read as a one-column csv
        try:
            uploaded.seek(0)
        except Exception:
            pass
        try:
            df1 = pd.read_csv(uploaded, header=None)
            flat = []
            for v in df1.iloc[:, 0].astype(str).tolist():
                flat += re.findall(r"(?<!\d)(\d{4})(?!\d)", v)
            nums = flat
        except Exception:
            nums = []

    nums = [str(n).zfill(4) for n in nums if n is not None]
    # de-dupe while preserving order
    seen = set()
    out = []
    for n in nums:
        if n not in seen:
            seen.add(n)
            out.append(n)

    df = pd.DataFrame({"Result": out})
    if df.empty:
        return df
    df["Pick4"] = df["Result"]
    df["Structure"] = df["Pick4"].map(structure_of_4)
    df["Box"] = df["Pick4"].map(box_key)
    df["BoxKey4"] = df["Box"]
    return df


# -------------------------
# Stats + ranking
# -------------------------

@dataclass
class RankConfig:
    # History window used for rank stats (default 180, switchable to 365 in UI)
    window_days: int = 180

    # Bucket method:
    # - Top 'top_base' by BaseScore (HitsPerWeek)
    # - From base ranks due_from_rank..due_to_rank, take Top 'top_due' by DueIndex (DaysSinceLastHit)
    top_base: int = 12
    due_from_rank: int = 13
    due_to_rank: int = 60
    top_due: int = 8

    # Display / scoring knobs (kept as soft signals)
    max_master_rows: int = 120
    max_final_rows: int = 300
    include_24h_signals: bool = True
    pos_strength_weight: float = 0.25
    seed_core_key: str = "core"  # reserved for future compatibility

    # Back-compat aliases (older UI keys)
    @property
    def top12(self) -> int:
        return int(self.top_base)

    @property
    def due_ranks(self):
        return (int(self.due_from_rank), int(self.due_to_rank))

@property
def top_n(self) -> int:
    """Legacy alias: some older builds referenced RankConfig.top_n."""
    return int(self.top_base)

@property
def base_top_n(self) -> int:
    """Legacy alias for the Top bucket size."""
    return int(self.top_base)

@property
def due_top_n(self) -> int:
    """Legacy alias for the Due bucket size."""
    return int(self.top_due)


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
    out.loc[out["LastHitDate"].isna(), "DaysSinceLastHit"] = 0

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

def top_dense_positions(pos_map, top_n: int = 10, top_k_positions: int | None = None):
    """Return the top-N rank positions (1..76) that collectively hold the most winners."""
    if pos_map is None or pos_map.empty:
        return []
    tmp = pos_map.copy()
    # Defensive: sometimes RankPos can come back as strings; coerce to numeric
    tmp["RankPos_num"] = pd.to_numeric(tmp["RankPos"], errors="coerce")
    tmp = tmp.dropna(subset=["RankPos_num"])
    if tmp.empty:
        return []
    tmp["RankPos_num"] = tmp["RankPos_num"].astype(int)
    counts = tmp.groupby("RankPos_num")["HitCount"].sum().sort_values(ascending=False).head(int(top_n))
    return [int(x) for x in counts.index.tolist()]

def engine_cluster_positions(
    df_24h_core_hits: pd.DataFrame,
    base_stats: pd.DataFrame,
    top_n: int = 10,
    use_rank_col: str = "RankPos",
) -> list[int]:
    """Return the *clustered* top-N rank positions for a 24h core-hit sample.

    Robust against empty inputs, NaNs, and callers accidentally passing Series/dicts.
    Always returns a Python list (possibly empty).
    """
    if df_24h_core_hits is None or len(df_24h_core_hits) == 0:
        return []

    if base_stats is None:
        return []

    # Normalize base_stats to a DataFrame with a numeric rank column.
    if not isinstance(base_stats, pd.DataFrame):
        try:
            base_stats = pd.DataFrame(base_stats)
        except Exception:
            return []

    if use_rank_col not in base_stats.columns:
        return []

    rank_input = base_stats[use_rank_col]
    if isinstance(rank_input, pd.DataFrame):
        # if a list-like column selector was passed, take first column
        rank_input = rank_input.iloc[:, 0]
    rank_series = pd.to_numeric(rank_input, errors="coerce")
    rank_map = pd.DataFrame({"RankPos": rank_series}).dropna().sort_values("RankPos")
    if rank_map.empty:
        return []

    # Candidate rank positions actually observed in the 24h hit set
    try:
        hit_ranks = pd.to_numeric(df_24h_core_hits.get("RankPos"), errors="coerce").dropna().astype(int).tolist()
    except Exception:
        hit_ranks = []

    if not hit_ranks:
        return []

    # Keep only ranks that exist in base_stats map
    rank_set = set(rank_map["RankPos"].astype(int).tolist())
    hit_ranks = [int(r) for r in hit_ranks if int(r) in rank_set]
    if not hit_ranks:
        return []

    # Find a "dense cluster" around the most common local neighborhood.
    hit_ranks_sorted = sorted(hit_ranks)

    best_window = None
    best_score = -1
    span = 12  # neighborhood width

    for anchor in hit_ranks_sorted:
        lo = anchor
        hi = anchor + span
        members = [r for r in hit_ranks_sorted if lo <= r <= hi]
        score = len(members)
        if score > best_score:
            best_score = score
            best_window = (lo, hi)

    if best_window is None:
        return []

    lo, hi = best_window
    clustered = [r for r in rank_map["RankPos"].astype(int).tolist() if lo <= r <= hi]
    # Return up to top_n, but keep as list[int]
    return clustered[: max(1, int(top_n))]


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
            base_stats.rename(columns={"RankPos_full":"RankPos"}).assign(RankPos=lambda d: d["RankPos"]),
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
        top5_cluster = engine_cluster_positions(df_24h_core_hits, base_stats.rename(columns={"RankPos_full":"RankPos"}).assign(RankPos=lambda d: d["RankPos"]), top_n=5)

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
st.caption(APP_VERSION)

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
    map_file = map24_file  # backward-compatible alias

    exclude_md = st.checkbox("Exclude Maryland (MD)", value=True, help="Optional global exclusion. When enabled (default), MD rows are removed from both the baseline and 24h files before ranking.")
    st.session_state["exclude_md"] = exclude_md



    st.divider()
    st.subheader("Self-update rolling baseline (optional)")
    use_store = st.checkbox("Use local rolling ~3-year baseline store", value=False, help="Keeps a local rolling baseline by appending new rows from the 24h file and purging rows older than ~3 years. This improves speed and keeps your baseline fresh without you manually editing the all-states file.")
    st.session_state["use_store"] = use_store

    if use_store:
        store_df_preview = load_baseline_store()
        store_meta = _read_meta(BASELINE_STORE_BASE)
        store_rows = int(store_meta.get("rows", store_df_preview.shape[0] if store_df_preview is not None else 0) or 0)
        store_max = store_meta.get("max_date", "") or (str(store_df_preview["Date"].max()) if store_df_preview is not None and not store_df_preview.empty else "")
        st.caption(f"Store status: {store_rows:,} rows | newest date: {store_max if store_max else '—'}")

        colA, colB = st.columns(2)
        with colA:
            if st.button("Initialize/overwrite store from uploaded all-states file"):
                if master_file is None:
                    st.warning("Upload the all-states history file first.")
                else:
                    try:
                        master_file.seek(0)
                    except Exception:
                        pass
                    try:
                        df_init = try_read_tablelike(master_file)
                        df_init = purge_to_rolling_3y(df_init, years=3)
                        ok, wrote = write_baseline_store(df_init, note="Initialized from uploaded all-states file (rolling 3y).")
                        st.success(f"Baseline store saved: {wrote}")
                    except Exception as e:
                        st.error(f"Could not initialize store: {e}")
                    try:
                        master_file.seek(0)
                    except Exception:
                        pass
        with colB:
            if st.button("Append 24h file into store (and purge)"):
                if map24_file is None:
                    st.warning("Upload the 24h file first.")
                else:
                    try:
                        map24_file.seek(0)
                    except Exception:
                        pass
                    try:
                        df_new = try_read_tablelike(map24_file)
                        df_store = load_baseline_store()
                        merged, added = append_from_24h(df_store, df_new)
                        merged = purge_to_rolling_3y(merged, years=3)
                        ok, wrote = write_baseline_store(merged, note=f"Appended from 24h file (+{added} new rows), then purged to rolling 3y.")
                        st.success(f"Updated store: +{added} new rows. Saved: {wrote}")
                    except Exception as e:
                        st.error(f"Could not append 24h into store: {e}")
# ---------- Core selection ----------
st.header("Core selection")

# Start from the curated preset list, then union with any cores detected from data (if present)
_detected_cores: list[str] = []
try:
    if isinstance(df_all, pd.DataFrame) and "Core" in df_all.columns:
        _det = df_all["Core"].dropna().astype(str).str.extract(r"(\d{1,3})", expand=False).dropna().unique().tolist()
        _det = [str(x).zfill(3) for x in _det]
        _det = [c for c in _det if c.isdigit()]
        _detected_cores = sorted(set(_det))
except Exception:
    _detected_cores = []

available_cores = sorted(set([str(c).zfill(3) for c in CORE_PRESETS] + _detected_cores))
cores = available_cores  # alias used throughout UI

# Ensure default core exists
default_core = str(getattr(cfg, "default_core", "389")).zfill(3)
if default_core not in available_cores:
    available_cores = [default_core] + available_cores

# Persist selection
if "cores_for_cache" not in st.session_state:
    st.session_state.cores_for_cache = [default_core]

# Core view dropdown (single core)
# View core (dropdown)
vc_default = str(default_core).zfill(3) if str(default_core).zfill(3) in cores else (str(cores[0]).zfill(3) if cores else '000')
vc = vc_default
try:
    vc = str(st.session_state.get('view_core', vc_default)).zfill(3)
except Exception:
    vc = vc_default
if vc not in cores:
    vc = vc_default
view_core = st.selectbox("View core (dropdown)", cores, index=cores.index(vc) if vc in cores else 0, key="view_core")
core_for_view = view_core  # backward-compatible variable name

# Multi-core selection for cache build / batch tools
# Keep a stable widget key, and initialize it only once to avoid Streamlit warnings.
if 'cores_for_cache_ms' not in st.session_state:
    st.session_state.cores_for_cache_ms = list(st.session_state.get('cores_for_cache', [view_core]))
if view_core not in [str(c).zfill(3) for c in st.session_state.cores_for_cache_ms]:
    st.session_state.cores_for_cache_ms = [*st.session_state.cores_for_cache_ms, view_core]
cores_for_cache_ms = st.multiselect(
    "Cores to include for cache building / batch tools",
    cores,
    key="cores_for_cache_ms",
    help="Select one or more 3-digit cores. Cache building and batch tools will use this list.",
)
st.session_state.cores_for_cache = [str(c).zfill(3) for c in cores_for_cache_ms]
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
use_store = bool(st.session_state.get("use_store", False))

if use_store:
    # Prefer the on-disk rolling store if present
    df_all = load_baseline_store()
    # If the store is empty but the user uploaded a master file, auto-initialize (rolling 3y)
    if (df_all is None or df_all.empty) and master_file is not None:
        try:
            master_file.seek(0)
        except Exception:
            pass
        df_all = try_read_tablelike(master_file)
        df_all = purge_to_rolling_3y(df_all, years=3)
        try:
            write_baseline_store(df_all, note="Auto-initialized store from uploaded all-states file (rolling 3y).")
        except Exception:
            pass
        try:
            master_file.seek(0)
        except Exception:
            pass
else:
    df_all = try_read_tablelike(master_file) if master_file else pd.DataFrame()

prev_picklist = pd.DataFrame()
df_24h = pd.DataFrame()
if map24_file:
    try:
        df_24h = try_read_tablelike(map24_file)
    except Exception:
        # Many users upload a simple pick-list here (one 4-digit number per line).
        # Accept it without crashing; it will NOT be used for 24h engines or baseline self-update.
        try:
            map24_file.seek(0)
        except Exception:
            pass
        try:
            prev_picklist = try_read_picklist(map24_file)
            if not prev_picklist.empty:
                st.info(
                    "Optional 24h/previous-day file detected as a pick-list (not a LotteryPost history export). "
                    "It will be used only for annotation/downranking where applicable."
                )
        except Exception as e:
            st.warning(f"Could not parse optional 24h/previous-day file: {e}")


if exclude_md and not df_all.empty:
    df_all = df_all[df_all["State"].astype(str).str.strip().str.lower() != "maryland"].copy()
if exclude_md and not df_24h.empty:
    df_24h = df_24h[df_24h["State"].astype(str).str.strip().str.lower() != "maryland"].copy()

# Back-compat alias used by the Northern Lights block
df_24 = df_24h


# Auto-clear cached percentile maps when input data changes
if use_store:
    _m = _read_meta(BASELINE_STORE_BASE)
    all_hash = f"store|{_m.get('max_date','')}|{_m.get('rows','')}"
else:
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
    min_d = df_all["Date"].min().date().isoformat()
    max_d = df_all["Date"].max().date().isoformat()
    st.caption(f"Date span: {min_d} → {max_d}")


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
        "built_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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


def compute_stream_stats(df_all: pd.DataFrame, core: str, window_days: int, exclude_md: bool = False) -> pd.DataFrame:
    """Back-compat wrapper used by the Northern Lights block."""
    # exclude_md is already applied upstream; kept only for compatibility
    last_date = most_recent_date(df_all)
    last_s = None
    if last_date is not None and not pd.isna(last_date):
        try:
            last_s = str(pd.to_datetime(last_date).date())
        except Exception:
            last_s = None
    return get_stream_stats_cached(core=str(core), window_days=int(window_days), df_all=df_all, last_date=last_s)


def build_northern_star_buckets(
    stats_df: pd.DataFrame,
    stream: str,
    top_n: int,
    due_ranks: Tuple[int, int],
    seed_core_key: str,
    include_24h: bool,
    df_24: Optional[pd.DataFrame],
    core: str,
) -> Dict[str, object]:
    """
    Back-compat bucket logic for the Northern Lights table:
    - Base bucket: Top N streams by HitsPerWeek
    - Due bucket: from base ranks [due_from..due_to], take Top cfg.top_due by DaysSinceLastHit
    Returns fields expected by the Northern Lights renderer.
    """
    if stats_df is None or stats_df.empty:
        return {}

    s = stats_df.copy()
    s = s.sort_values(["HitsPerWeek", "HitsWindow"], ascending=[False, False]).reset_index(drop=True)
    s["BaseRank"] = s.index + 1

    # Base top
    base_top_streams = set(s.head(int(top_n))["Stream"].astype(str).tolist())

    # Due candidates are chosen from the *base-ranked* band
    d1, d2 = int(due_ranks[0]), int(due_ranks[1])
    band = s[(s["BaseRank"] >= d1) & (s["BaseRank"] <= d2)].copy()
    if band.empty:
        due_top_streams = set()
    else:
        band = band.sort_values(["DaysSinceLastHit", "HitsPerWeek"], ascending=[False, False])
        due_top_streams = set(band.head(int(getattr(st.session_state.get("_cfg", RankConfig()), "top_due", 8)))["Stream"].astype(str).tolist())

    in_base = str(stream) in base_top_streams
    in_due = str(stream) in due_top_streams

    # Pull row for this stream
    row = s[s["Stream"].astype(str) == str(stream)]
    if row.empty:
        return {}
    r = row.iloc[0]
    hits = _safe_int(r.get("HitsWindow", 0)) or 0
    hpw = float(r.get("HitsPerWeek", 0.0) or 0.0)
    dslh = (_safe_int(r.get("DaysSinceLastHit", 0)) or 0)

    # Due pressure as soft signal
    due_pressure = 0.0
    if in_due:
        due_pressure = float(dslh)

    # Optional 24h soft signal: add a small nudge if this core hit this stream in df_24
    if include_24h and df_24 is not None and not df_24.empty:
        try:
            cache_key = f"_nl_24h_corehits_{core}"
            if cache_key not in st.session_state:
                st.session_state[cache_key] = compute_core_hits(df_24, str(core), structures=["AABC"])
            df_24h_core_hits = st.session_state.get(cache_key, pd.DataFrame())
            if not df_24h_core_hits.empty and (df_24h_core_hits["Stream"].astype(str) == str(stream)).any():
                due_pressure += 1.0
        except Exception:
            pass

    seed_key = canonical_core_key(str(core))

    bucket_label = "Top12" if in_base else ("Due" if in_due else "")
    bucket_pick = "BASE" if in_base else ("DUE" if in_due else "")

    return {
        "Top12": bucket_label,
        "BucketPick": bucket_pick,
        "SeedKey": seed_key,
        "DuePressure": due_pressure,
        "HitsPerWeek": hpw,
        "Hits": hits,
        "DaysSinceLastHit": dslh,
    }




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
st.caption("This aggregates the Northern Star bucket picks across your selected cores and ranks them using a universal score (recent strength + due pressure + position-percentile strength).")

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




# ===============================
# Main tabs
# ===============================

tab_labels = ["Northern Lights (Master playlist)", "Core view", "Backtest (optional)"]
_t_nl, _t_core, _t_bt = st.tabs(tab_labels)

# --- Northern Lights master playlist (best -> worst across streams/cores) ---
with _t_nl:
    st.subheader("Northern Lights master playlist")
    st.caption("Aggregated bucket picks across your selected cores. Use this as your universal stream playlist.")

    # Ensure cores_for_cache is always defined (selected cores for cache building / views)
    cores_for_cache = list(st.session_state.get('cores_for_cache') or st.session_state.get('selected_cores') or [])
    if not cores_for_cache:
        cores_for_cache = [core_for_view] if 'core_for_view' in locals() else (cores[:1] if 'cores' in locals() and cores else ['000'])
    cores_for_cache = [str(c).zfill(3) for c in cores_for_cache]

    if not cores_for_cache:
        st.info("Select one or more cores above to populate the playlist.")
    else:
        cfg = st.session_state.get("_cfg", RankConfig())
        include_24h = bool(st.session_state.get("include_24h", True))

        # Build a master list: (core, stream) -> universal score
        rows = []
        for core in cores_for_cache:
            try:
                stats_df = compute_stream_stats(df_all, core, window_days=window_days, exclude_md=False)
            except Exception:
                stats_df = pd.DataFrame()

            if stats_df is None or stats_df.empty:
                continue

            for stream in stats_df["Stream"].astype(str).tolist():
                meta = build_northern_star_buckets(
                    stats_df=stats_df,
                    stream=stream,
                    top_n=cfg.top_base,
                    due_ranks=(cfg.due_from_rank, cfg.due_to_rank),
                    seed_core_key=canonical_core_key(str(core)),
                    include_24h=include_24h,
                    df_24=df_24h,
                    core=str(core),
                )
                if not meta:
                    continue

                # UniversalScore is what we rank by in the playlist
                # (recent strength + due pressure + position-percentile strength)
                # Position-percentile strength comes from the cached pos map
                try:
                    last_s = None
                    if last_all is not None and not pd.isna(last_all):
                        last_s = str(pd.to_datetime(last_all).date())
                    pos_df = get_pos_map_cached(str(core), int(window_days), stats_df, last_date=last_s)
                    p = 0.0
                    if pos_df is not None and not pos_df.empty:
                        m = pos_df[pos_df["Stream"].astype(str) == str(stream)]
                        if not m.empty:
                            p = float(m.iloc[0].get("PctStrength", 0.0) or 0.0)
                except Exception:
                    p = 0.0

                universal = float(meta.get("HitsPerWeek", 0.0) or 0.0) + (float(meta.get("DuePressure", 0.0) or 0.0) * 0.01) + (p * 0.01)

                rows.append({
                    "Core": str(core).zfill(3),
                    "Stream": str(stream),
                    "BucketPick": meta.get("BucketPick", ""),
                    "HitsPerWeek": float(meta.get("HitsPerWeek", 0.0) or 0.0),
                    "DuePressure": float(meta.get("DuePressure", 0.0) or 0.0),
                    "PctStrength": float(p or 0.0),
                    "UniversalScore": float(universal),
                    "Hits": int(meta.get("Hits", 0) or 0),
                    "DaysSinceLastHit": int(meta.get("DaysSinceLastHit", 0) or 0),
                })

        if not rows:
            st.warning("No playlist rows were produced. Double-check that your history file contains your selected cores in AABC structure.")
        else:
            nl_df = pd.DataFrame(rows)
            nl_df = nl_df.sort_values(["UniversalScore", "HitsPerWeek", "DuePressure"], ascending=[False, False, False]).reset_index(drop=True)
            nl_df.insert(0, "Rank", nl_df.index + 1)

            st.dataframe(nl_df, use_container_width=True, height=520)
            
            # Percentile map(s) for selected core(s) (tie-breaker visibility in Northern Lights view)
            with st.expander("Core ranking percentile map (tie-breaker)"):
                if cores_for_cache:
                    _tabs = st.tabs([f"Core {c}" for c in cores_for_cache]) if len(cores_for_cache) > 1 else [st.container()]
                    for _tab, _c in zip(_tabs, cores_for_cache):
                        with _tab:
                            _pm = get_position_percentiles_cached(str(_c).zfill(3))
                            st.dataframe(_pm, use_container_width=True, height=240)
                else:
                    st.info("Select one or more cores above to view percentile maps.")


            # Optional: straights shortlist (keep existing feature)
            if st.session_state.get("do_straights", False):
                st.divider()
                st.subheader("Generate straights shortlist (optional last)")
                st.caption("This feature is unchanged; it runs only after the master playlist is built.")
                try:
                    render_straights_shortlist(nl_df)
                except Exception as e:
                    st.error(f"Straights shortlist failed: {e}")


# --- Core view (single core or tabbed multi-core) ---
with _t_core:
    st.subheader("Core view")

    if df_all is None or df_all.empty:
        st.info("Upload your history file first.")
        st.stop()

    if not cores_for_cache:
        st.info("Select one or more cores above to view core stats.")
        st.stop()

    show_tabs = st.checkbox(
        "Show tabs for all selected cores (optional)",
        value=False,
        key="show_tabs_for_all_selected_cores",
        help="If ON, you'll get a separate Core tab for each selected core. If OFF, you only see the core chosen in 'View core'.",
    )

    cfg = st.session_state.get("_cfg", RankConfig())

    def _render_one_core(core_id: str):
        core_id = str(core_id).zfill(3)
        st.markdown(f"### Core {core_id}")

        # Compute the AABC stream stats
        stats_df = compute_stream_stats(df_all, core_id, window_days=window_days, exclude_md=False)
        if stats_df is None or stats_df.empty:
            st.warning(f"No AABC stream stats found for core {core_id}.")
            return

        stats_df = stats_df.copy()
        st.subheader("Stream ranking (AABC doubles)")
        st.dataframe(stats_df, use_container_width=True, height=420)

        # Buckets
        try:
            top_bucket, due_bucket, combined_bucket = build_northern_star_buckets(stats_df, cfg)
        except Exception:
            # fallback for older signature
            top_bucket, due_bucket, combined_bucket = build_northern_star_buckets(
                stats_df,
                stream="",
                top_n=cfg.top_base,
                due_ranks=(cfg.due_from_rank, cfg.due_to_rank),
                seed_core_key=canonical_core_key(core_id),
                include_24h=bool(st.session_state.get("include_24h", True)),
                df_24=df_24h,
                core=core_id,
            ), [], []

        st.subheader("Northern Star buckets")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("Top 12 (BaseScore)")
            st.write(top_bucket)
        with c2:
            st.caption(f"Due {getattr(cfg, 'top_due', 8)} (DueIndex)")
            st.write(due_bucket)
        with c3:
            st.caption("Combined (Top+Due)")
            st.write(combined_bucket)

        # Core percentile map expander (tie-breaker)
        with st.expander("Core ranking percentile map (tie-breaker)", expanded=False):
            try:
                pos_map = get_position_percentiles_cached(core_id, window_days, stats_df)
                if pos_map is None or pos_map.empty:
                    st.info("No percentile map available for this core.")
                else:
                    st.dataframe(pos_map, use_container_width=True, height=420)
                    st.caption("Tip: use PctStrength as a soft tie-breaker when streams are close.")
            except Exception as e:
                st.error(f"Could not compute percentile map: {e}")

    if show_tabs:
        # Always render *all* selected cores in their own tabs
        core_tabs = st.tabs([str(c).zfill(3) for c in cores_for_cache])
        for c, t in zip(cores_for_cache, core_tabs):
            with t:
                _render_one_core(str(c))
    else:
        # Render only the currently selected view core
        _render_one_core(str(core_for_view))


# --- Backtest (optional) ---
with _t_bt:
    st.subheader("Backtest (optional)")
    st.caption("Optional diagnostics. This does not change your core ranking output.")

    if df_all is None or df_all.empty:
        st.info("Upload your history file first.")
        st.stop()

    if not cores_for_cache:
        st.info("Select one or more cores above to backtest.")
        st.stop()

    try:
        render_backtest(df_all=df_all, cores=cores_for_cache, window_days=window_days)
    except NameError:
        st.warning("Backtest utility is not available in this build.")
    except Exception as e:
        st.error(f"Backtest failed: {e}")
