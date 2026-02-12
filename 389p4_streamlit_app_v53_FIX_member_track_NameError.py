from __future__ import annotations
# pk4_northern_star_app_2026-02-04_v41.py
# Streamlit app: Pick 4 "Northern Star" core stream ranking + Rare/Ultra-Rare engines (AAAB+AABB, AAAA)
# Notes:
# - Designed to work with LotteryPost-style exports (tab .txt or .csv) that include Date, State, Game, Results.
# - Ignores Wild Ball / Fireball / multipliers by extracting the first 4 digits like "1-2-3-4".
# - Excludes Maryland by default (toggle in sidebar).


APP_VERSION = "v51.09 (Backtest Walk-Forward Fixed)"

CHANGE_LOG_V51 = """v51 — SeedTraits + Cadence + AllCores Cache-Only (built from v50, NO regressions)

✅ Added:
- Seed Traits (positive + negative) autoload + optional upload; soft scoring applied to:
  - Northern Lights UniversalScore
  - Core scoring (Northern Star + Core View helper)
- Cadence scoring (windowed 180/365) as a soft, configurable boost (no hard filters)
- Northern Star tab (restores Rare Engine + Ultra-Rare engine outputs in UI)
- Global all-cores RankPos percentile map (cache-only) + per-core maps remain distinct
- “Select all cores” button for multi-core selection (cache building / batch tools)

✅ Fixed (signature/return + robustness):
- Added PctStrength alias to percentile map output (back-compat)
- Northern Lights: position strength now resolves via RankPos (not Stream) to avoid empty maps
- Bucket recommendations now include back-compat metadata keys (top_n / due_ranks / etc)
- All Cores mode in Northern Lights is now STRICT cache-only; refuses live compute if any core cache missing

Notes:
- No sections removed or disabled. New functionality is additive and defaults are conservative.
"""


# Core presets (family IDs) shown in the UI. Keep this list additive.
# These are the cores you and I have explicitly worked on so far.
CORE_PRESETS = [
    "012",
    "013",
    "016",
    "017",
    "018",
    "019",
    "023",
    "024",
    "025",
    "027",
    "028",
    "029",
    "035",
    "038",
    "046",
    "048",
    "056",
    "059",
    "067",
    "068",
    "078",
    "129",
    "134",
    "135",
    "145",
    "146",
    "149",
    "167",
    "168",
    "169",
    "178",
    "179",
    "236",
    "238",
    "239",
    "245",
    "246",
    "249",
    "257",
    "258",
    "278",
    "279",
    "345",
    "348",
    "357",
    "358",
    "359",
    "378",
    "379",
    "389",
    "456",
    "457",
    "458",
    "459",
    "468",
    "479",
    "489",
    "567",
    "568",
    "579",
    "589",
    "679",
    "689",
    "789",
]


# Compatibility: the legacy 'old app' core set (kept for quick selection)
OLD_APP_CORE_SET = ['016', '017', '018', '019', '023', '024', '025', '027', '028', '029', '038', '046', '048', '056', '059', '067', '068', '078', '129', '135', '145', '146', '149', '167', '168', '169', '179', '236', '238', '239', '245', '246', '249', '257', '258', '278', '279', '345', '348', '357', '359', '378', '379', '389', '457', '459', '489', '567', '579', '589', '679', '689', '789']
# --- Optional: "Trigger Map" weighting for a fixed 39-play list (soft boost, never an elimination) ---
# This is intentionally conservative: it only adds a small score nudge to prioritize certain plays per-stream
# based on the previous winner in that same stream.
TRIGGER_PLAYLIST_39 = [
    "3389","3889","3899",
    "0013","0113","0133","0019","0119","0199",
    "1145","1445","1455","1147","1447","1477","1149","1499","1449",
    "1136","1336","1366",
    "1667","1167","1677","1169","1669","1699",
    "3356","3566","3556","3367","3667","3677",
    "5567","5667","5677","6679","6779","6799",
]

# Override trigger: previous winner contains >=3 digits from {7,8,9,0}
_TRIGGER_OVERRIDE_SET = set("7890")

# Default decision tree: bucket by last digit of previous winner
TRIGGER_BY_PREV_LAST = {
    "0": ["3556","3899","1677","5677","3677"],
    "1": ["5567","0113","1499","1699","1167"],
    "2": ["1699","1677","3566","1667","1149"],
    "3": ["6679","1167","0019","3566","1669"],
    "4": ["1366","1667","3356","1455"],
    "5": ["0133","1147","1136","1445","1145"],
    "6": ["1449","0199","3356","3367","3556"],
    "7": ["3677","1145","0013","1447","1169"],
    "8": ["1366","1149","3389","1669","5667"],
    "9": ["1445","1149","6679","1669","1169"],
}

TRIGGER_OVERRIDE_EMPHASIS = ["3889","3677","1169","3899","3556","6679"]

def trigger_map_emphasis(prev_result_4d: str) -> list[str]:
    """Return a (possibly empty) ordered emphasis list for the Trigger Map."""
    s = re.sub(r"\D", "", str(prev_result_4d or ""))[:4]
    if len(s) != 4:
        return []
    # Override: >=3 digits from 7/8/9/0
    if sum(1 for ch in s if ch in _TRIGGER_OVERRIDE_SET) >= 3:
        return list(TRIGGER_OVERRIDE_EMPHASIS)
    return list(TRIGGER_BY_PREV_LAST.get(s[-1], []))

def trigger_map_boost(play_4d: str, prev_result_4d: str, *, boost_points: float = 2.0) -> float:
    """Soft boost points for a play given the previous result in the stream."""
    if not play_4d:
        return 0.0
    p = re.sub(r"\D", "", str(play_4d)).zfill(4)[-4:]
    emph = trigger_map_emphasis(prev_result_4d)
    if not emph:
        return 0.0
    return float(boost_points) if p in set(emph) else 0.0


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

def _to_dataframe(obj) -> pd.DataFrame:
    """Best-effort conversion for Streamlit display; prevents 'dict has no dtype' crashes."""
    if isinstance(obj, pd.DataFrame):
        return obj
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, dict):
        # Prefer a single-row DF for dicts
        return pd.DataFrame([obj])
    try:
        return pd.DataFrame(obj)
    except Exception:
        return pd.DataFrame({"value": [str(obj)]})

import streamlit as st

# --- Safety init (prevents NameError when UI blocks are skipped) ---
member_track: bool = False

import hashlib
import datetime
import json
from functools import lru_cache

# ---- Safe defaults to prevent NameError during first render ----
cfg = None  # set later after window selection
exclude_md = True  # default behavior: exclude Maryland unless user toggles off
map_file = None  # backward-compatible alias set in sidebar

# Tab containers (assigned after tabs() is created; keep placeholders to avoid NameError)
_t_nl = None
_t_ns = None
_t_core = None
_t_bt = None

# ---- Rerun helper (must be defined early; used by core-selection buttons) ----
def _rerun() -> None:
    """Compatibility rerun across Streamlit versions."""
    try:
        # Streamlit >= 1.30
        st.rerun()
        return
    except Exception:
        pass
    try:
        # Older Streamlit
        st.experimental_rerun()
        return
    except Exception:
        pass
    # Last resort: no-op (should not happen on Streamlit Cloud)
    return






# -------------------------
# Parsing + helpers
# -------------------------

# -------------------------
# Disk baseline cache (optional, keeps runs fast)
# -------------------------
from pathlib import Path as _Path

DISK_CACHE_DIR = _Path("pk4_baseline_cache")

DISK_PCT_DIR = DISK_CACHE_DIR / "pctmaps"
DISK_PCT_DIR.mkdir(parents=True, exist_ok=True)

def _pctmap_path(core: str, window_days: int) -> Path:
    safe_core = core.replace("/", "_")
    return DISK_PCT_DIR / f"rankpos_pct_{safe_core}_{window_days}d.csv"

def _save_pctmap_to_disk(core: str, window_days: int, pct_df: pd.DataFrame, asof_last_date: str) -> None:
    if pct_df is None or pct_df.empty:
        return
    out = pct_df.copy()
    out.insert(0, "core", core)
    out.insert(1, "window_days", int(window_days))
    out.insert(2, "asof_last_date", asof_last_date)
    out.to_csv(_pctmap_path(core, window_days), index=False)

def _load_pctmap_from_disk(core: str, window_days: int, expected_last_date: Optional[str] = None) -> Optional[pd.DataFrame]:
    p = _pctmap_path(core, window_days)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    if expected_last_date is not None and "asof_last_date" in df.columns:
        # Keep only if matches current history last date (prevents stale/inaccurate maps)
        if str(df["asof_last_date"].iloc[0]) != str(expected_last_date):
            return None
    return df

def build_allcores_rankpos_pctmap(
    cores_list: List[str],
    window_days: int,
    expected_last_date: Optional[str],
    cache_only: bool = True,
    df_all: Optional[pd.DataFrame] = None,
    stream_rankings_df: Optional[pd.DataFrame] = None,
    family_counts_df: Optional[pd.DataFrame] = None,
    struct_counts_df: Optional[pd.DataFrame] = None,
    cfg: Optional["RankConfig"] = None,
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Aggregate RankPos->HitCount across many cores and return a percentile map.

    If cache_only=True, requires baseline cache for every core; if any is missing/outdated,
    returns None (caller decides how to handle).
    """
    frames = []
    missing = []
    for core in cores_list:
        if cache_only:
            ss, _pos_df, _meta = _load_baseline_from_disk(core, window_days, expected_last_date=expected_last_date)
            if ss is None or ss.empty:
                missing.append(str(core).zfill(3))
                continue
        else:
            if cfg is None:
                cfg = RankConfig()
                cfg.window_days = window_days
            ss = compute_stream_stats(df_all, core, window_days=window_days, exclude_md=False)
        if ss is None or ss.empty or "RankPos" not in ss.columns:
            continue
        # Keep minimal columns
        if "HitsWindow" in ss.columns:
            frames.append(ss[["RankPos", "HitsWindow"]].copy())
        elif "HitCount" in ss.columns:
            tmp = ss[["RankPos", "HitCount"]].copy()
            tmp = tmp.rename(columns={"HitCount": "HitsWindow"})
            frames.append(tmp)

    if missing:
        return None, missing
    if not frames:
        return pd.DataFrame(), []

    comb = pd.concat(frames, ignore_index=True)
    # Aggregate by RankPos; position_percentile_map will also group, but we keep it clean
    comb = comb.groupby("RankPos", as_index=False)["HitsWindow"].sum()
    pct, _ = position_percentile_map(comb)
    return pct, []
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

def members_from_core(core_key: str, structure: str | None = None, **kwargs) -> List[str]:
    """Return 4-digit members for a 3-digit core.

    Two calling patterns are supported (backwards-compatible):
      1) members_from_core(core, "AABC") -> returns [AABC, ABBC, ABCC]
      2) members_from_core(core, include_family=True, include_aaab=True, include_aabb=True, include_aaaa=False)
         -> returns a combined, de-duplicated list of requested structures.
    """
    core_key = canonical_core_key(core_key)
    a, b, c = core_key[0], core_key[1], core_key[2]

    def _one(struct: str) -> List[str]:
        if struct == "AABC":
            x, y, z = f"{a}{a}{b}{c}", f"{a}{b}{b}{c}", f"{a}{b}{c}{c}"
        elif struct == "AAAB":
            x, y, z = f"{a}{a}{a}{b}", f"{a}{a}{a}{c}", f"{a}{b}{c}{c}"  # third is ABCC (rare engine uses it)
        elif struct == "AABB":
            x, y, z = f"{a}{a}{b}{b}", f"{a}{a}{c}{c}", f"{b}{b}{c}{c}"
        elif struct == "AAAA":
            x, y, z = f"{a}{a}{a}{a}", f"{b}{b}{b}{b}", f"{c}{c}{c}{c}"
        else:
            raise ValueError(f"Unknown structure: {struct}")
        return [box_key(x), box_key(y), box_key(z)]

    if structure is not None:
        return _one(structure)

    # Legacy / combined-call form
    include_family = bool(kwargs.get("include_family", True))
    include_aaab = bool(kwargs.get("include_aaab", False))
    include_aabb = bool(kwargs.get("include_aabb", False))
    include_aaaa = bool(kwargs.get("include_aaaa", False))

    out: List[str] = []
    if include_family:
        out.extend(_one("AABC"))
    if include_aaab:
        out.extend(_one("AAAB"))
    if include_aabb:
        out.extend(_one("AABB"))
    if include_aaaa:
        out.extend(_one("AAAA"))

    # De-duplicate while preserving order
    seen = set()
    res: List[str] = []
    for m in out:
        if m in seen:
            continue
        seen.add(m)
        res.append(m)
    return res



# ------------------------------------------------------------
# Core member labeling + member-pick prediction (walk-forward)
# ------------------------------------------------------------

@lru_cache(maxsize=4096)
def _core_member_label_map(core_key: str, include_rare: bool = False) -> dict[str, str]:
    """Map a core's member box-keys to human-readable member labels.

    Family (doubles) labels:
      - AABC = double of A (the first digit in sorted core)
      - ABBC = double of B
      - ABCC = double of C

    Rare labels (optional):
      - AAAB, AAAC (triple A with B/C)
      - AABB, AACC, BBCC
      - AAAA_A, AAAA_B, AAAA_C
    """
    core_key = canonical_core_key(core_key)
    a, b, c = core_key[0], core_key[1], core_key[2]

    m: dict[str, str] = {}

    # Family first (priority)
    fam_boxes = members_from_core(core_key, "AABC")
    for bk, lab in zip(fam_boxes, ["AABC", "ABBC", "ABCC"]):
        m.setdefault(str(bk), lab)

    if include_rare:
        # AAAB engine (note: third entry in members_from_core("AAAB") may overlap ABCC)
        aaab_boxes = members_from_core(core_key, "AAAB")
        for bk, lab in zip(aaab_boxes, ["AAAB", "AAAC", "ABCC"]):
            m.setdefault(str(bk), lab)

        aabb_boxes = members_from_core(core_key, "AABB")
        for bk, lab in zip(aabb_boxes, ["AABB", "AACC", "BBCC"]):
            m.setdefault(str(bk), lab)

        aaaa_boxes = members_from_core(core_key, "AAAA")
        for bk, lab in zip(aaaa_boxes, ["AAAA_A", "AAAA_B", "AAAA_C"]):
            m.setdefault(str(bk), lab)

    return m


def core_member_label(core_key: str, winner_4d: str, include_rare: bool = False) -> Optional[str]:
    """Return the member label for a core given a 4-digit winner (string).

    Uses box-key lookup first (fast and stable), then falls back to structure_of_4.
    """
    try:
        w = extract_4digit(winner_4d) or str(winner_4d).strip()
    except Exception:
        w = str(winner_4d).strip()
    if not w:
        return None
    bk = box_key(w)
    m = _core_member_label_map(core_key, include_rare=bool(include_rare))
    if bk in m:
        return m[bk]
    # Fallback (should be rare if winner is a member)
    try:
        return structure_of_4(w)
    except Exception:
        return None


def predict_core_member(
    df_all: pd.DataFrame,
    core_key: str,
    test_date: pd.Timestamp,
    window_days: int,
    *,
    basis: str = "core",
    stream: str | None = None,
    include_rare: bool = False,
) -> dict[str, Any]:
    """Predict which member label is most likely for this core (walk-forward safe).

    Prediction is based ONLY on rows with Date < test_date, restricted to the last `window_days`.

    basis:
      - "core": use all streams (global member distribution for that core)
      - "core_stream": use only that one stream (per-core-per-stream distribution)
    """
    if df_all is None or df_all.empty:
        return {"top1": None, "top2": None, "n": 0, "counts": {}}

    # Window slice: [test_date - window_days, test_date)
    try:
        td = pd.to_datetime(test_date).normalize()
    except Exception:
        td = pd.Timestamp(test_date).normalize()
    start = td - pd.Timedelta(days=int(window_days))

    sub = df_all
    try:
        if "Date" in sub.columns:
            sub = sub[sub["Date"].notna()]
            sub = sub[(sub["Date"] >= start) & (sub["Date"] < td)]
    except Exception:
        pass

    if basis == "core_stream" and stream is not None and "Stream" in sub.columns:
        try:
            sub = sub[sub["Stream"].astype(str) == str(stream)]
        except Exception:
            pass

    label_map = _core_member_label_map(core_key, include_rare=bool(include_rare))
    member_boxes = set(label_map.keys())
    box_col = "BoxKey4" if "BoxKey4" in sub.columns else ("Box" if "Box" in sub.columns else None)

    if box_col is None:
        # As a last resort, compute box keys on the fly
        try:
            tmp = sub.copy()
            tmp["_bk4"] = tmp["Result"].astype(str).map(box_key)
            box_col = "_bk4"
            sub = tmp
        except Exception:
            return {"top1": None, "top2": None, "n": 0, "counts": {}}

    try:
        hit_rows = sub[sub[box_col].astype(str).isin(member_boxes)]
    except Exception:
        hit_rows = pd.DataFrame()

    if hit_rows is None or hit_rows.empty:
        return {"top1": None, "top2": None, "n": 0, "counts": {}}

    # Map box keys -> labels (fast), then count
    labs = hit_rows[box_col].astype(str).map(label_map)
    counts = labs.value_counts(dropna=True)
    if counts.empty:
        return {"top1": None, "top2": None, "n": 0, "counts": {}}

    top = counts.index.tolist()
    top1 = top[0] if len(top) >= 1 else None
    top2 = top[1] if len(top) >= 2 else None
    return {
        "top1": top1,
        "top2": top2,
        "n": int(counts.sum()),
        "counts": counts.to_dict(),
    }



def _member_last_label(
    df_all: pd.DataFrame,
    core_key: str,
    test_date: pd.Timestamp,
    window_days: int,
    *,
    stream: str | None = None,
) -> tuple[Optional[str], int]:
    """Return LAST observed family-member label (AABC/ABBC/ABCC) for a core in the lookback window.
    Returns (label, n_hits_in_window). Walk-forward safe (uses Date < test_date only).
    """
    if df_all is None or df_all.empty:
        return (None, 0)
    td = pd.to_datetime(test_date).normalize()
    start = td - pd.Timedelta(days=int(window_days))
    sub = df_all
    if "Date" in sub.columns:
        sub = sub[sub["Date"].notna()]
        sub = sub[(sub["Date"] >= start) & (sub["Date"] < td)]
    if stream is not None and "Stream" in sub.columns:
        sub = sub[sub["Stream"].astype(str) == str(stream)]
    label_map = _core_member_label_map(core_key, include_rare=False)
    member_boxes = set(label_map.keys())
    box_col = "BoxKey4" if "BoxKey4" in sub.columns else ("Box" if "Box" in sub.columns else None)
    if box_col is None:
        tmp = sub.copy()
        tmp["_bk4"] = tmp["Result"].astype(str).map(box_key)
        box_col = "_bk4"
        sub = tmp
    hit_rows = sub[sub[box_col].astype(str).isin(member_boxes)].copy()
    if hit_rows.empty:
        return (None, 0)
    hit_rows = hit_rows.sort_values("Date")
    last_bk = hit_rows.iloc[-1][box_col]
    lab = label_map.get(str(last_bk))
    return (lab if lab in ("AABC", "ABBC", "ABCC") else None, int(len(hit_rows)))


def _seed_for_stream_asof(df_all: pd.DataFrame, stream: str, asof_date: pd.Timestamp) -> Optional[str]:
    """Most recent 4-digit result for a stream strictly before asof_date."""
    if df_all is None or df_all.empty:
        return None
    td = pd.to_datetime(asof_date).normalize()
    sub = df_all[(df_all["Date"] < td) & (df_all["Stream"].astype(str) == str(stream))].copy()
    if sub.empty:
        return None
    sub = sub.sort_values("Date")
    return str(sub.iloc[-1].get("Result", "")).strip() or None


def _seed_traits_for_core_stream(
    df_all: pd.DataFrame,
    core_key: str,
    stream: str,
    asof_date: pd.Timestamp,
) -> dict[str, str]:
    """Compute the standard seed-trait fields used in the seed-traits CSVs, for rulecards."""
    seed = _seed_for_stream_asof(df_all, stream, asof_date)
    if not seed:
        return {}
    seed = extract_4digit(seed) or seed
    digs = [int(ch) for ch in str(seed).zfill(4) if ch.isdigit()]
    if len(digs) != 4:
        return {}
    core_digs = set(str(core_key).zfill(3))
    ssum = sum(digs)
    spread = max(digs) - min(digs)
    even_cnt = sum(1 for d in digs if d % 2 == 0)
    high_cnt = sum(1 for d in digs if d >= 5)
    traits = {
        "seed_structure": structure_of_4(str(seed).zfill(4)),
        "seed_spread": ("<=2" if spread <= 2 else ("3-5" if spread <= 5 else ">=6")),
        "seed_even_count": str(even_cnt),
        "seed_high_count": str(high_cnt),
        "seed_sum_mod2": str(ssum % 2),
        "seed_sum_mod3": str(ssum % 3),
        # Sliding 4-sum band: (sum-1) to (sum+2) matches labels like 3-6, 11-14, etc.
        "seed_sum_range4_best": f"{ssum-1}-{ssum+2}",
        "seed_first_in_core": ("yes" if str(seed).zfill(4)[0] in core_digs else "no"),
        "seed_last_in_core": ("yes" if str(seed).zfill(4)[-1] in core_digs else "no"),
        "overlap_unique": str(len(set(str(seed).zfill(4)) & core_digs)),
        "seed_contains_core_pair": ("yes" if len(set(str(seed).zfill(4)) & core_digs) >= 2 else "no"),
    }
    # grid_last5_core_digits: how many of the core digits appear in the last-5 union digits for this stream (as of asof_date)
    try:
        sub = df_all[(df_all["Stream"].astype(str) == str(stream)) & (df_all["Date"] < pd.to_datetime(asof_date).normalize())].copy()
        sub = sub.sort_values("Date").tail(5)
        union = set("".join(sub["Result"].astype(str).tolist()))
        cnt = len(core_digs & union)
        if cnt == 3:
            traits["grid_last5_core_digits"] = "3"
        elif cnt >= 2:
            traits["grid_last5_core_digits"] = ">=2"
        else:
            traits["grid_last5_core_digits"] = str(cnt)
    except Exception:
        pass
    return traits


def _pick_best_seed_trait_rule(traits_pos_df: pd.DataFrame, core_key: str) -> Optional[tuple[str, str]]:
    """From the positive seed-traits CSV, pick the single highest-lift (trait, value) for this core."""
    if traits_pos_df is None or traits_pos_df.empty:
        return None
    ck = int(str(core_key).zfill(3))
    sub = traits_pos_df.copy()
    # core_family column in these CSVs is numeric (e.g., 12 for 012)
    sub = sub[sub["core_family"].astype(int) == ck]
    if sub.empty:
        return None
    sub = sub.sort_values(["lift", "trait_hits"], ascending=[False, False])
    r = sub.iloc[0]
    return (str(r.get("trait","")).strip(), str(r.get("value","")).strip())


def _member_mode_from_trait(
    df_all: pd.DataFrame,
    core_key: str,
    test_date: pd.Timestamp,
    window_days: int,
    trait_name: str,
    trait_value: str,
    *,
    stream: str | None = None,
) -> Optional[str]:
    """Within the walk-forward window, among hits where (trait==value) at the seed, return the MODE member label."""
    if not trait_name or trait_value is None:
        return None
    td = pd.to_datetime(test_date).normalize()
    start = td - pd.Timedelta(days=int(window_days))
    # Build stream-day transitions: seed = last result before day, winner = day's result.
    sub = df_all
    if "Date" not in sub.columns:
        return None
    sub = sub[sub["Date"].notna()].copy()
    sub = sub[(sub["Date"] >= start) & (sub["Date"] < td)]
    if stream is not None:
        sub = sub[sub["Stream"].astype(str) == str(stream)]
    if sub.empty:
        return None
    label_map = _core_member_label_map(core_key, include_rare=False)
    member_boxes = set(label_map.keys())
    # Keep only rows where the RESULT is a member of this core
    box_col = "BoxKey4" if "BoxKey4" in sub.columns else ("Box" if "Box" in sub.columns else None)
    if box_col is None:
        sub = sub.copy()
        sub["_bk4"] = sub["Result"].astype(str).map(box_key)
        box_col = "_bk4"
    hit = sub[sub[box_col].astype(str).isin(member_boxes)].copy()
    if hit.empty:
        return None
    # Compute trait per row based on the *seed* for that stream at that date (as-of that date)
    vals = []
    for _, r in hit.iterrows():
        s = str(r.get("Stream",""))
        d = pd.to_datetime(r.get("Date")).normalize()
        t = _seed_traits_for_core_stream(df_all, core_key, s, d).get(trait_name)
        vals.append(t)
    hit["_trait_val"] = vals
    hit = hit[hit["_trait_val"].astype(str) == str(trait_value)]
    if hit.empty:
        return None
    labs = hit[box_col].astype(str).map(label_map)
    vc = labs.value_counts()
    if vc.empty:
        return None
    top = vc.index.tolist()[0]
    return top if top in ("AABC","ABBC","ABCC") else None


def _member_prediction_variants(
    df_all: pd.DataFrame,
    traits_pos_df: pd.DataFrame,
    core_key: str,
    test_date: pd.Timestamp,
    window_days: int,
    *,
    stream: str,
    basis: str,
    min_stream_hits_for_last: int = 3,
) -> dict[str, Optional[str]]:
    """Compute member Top1 predictions under multiple strategies, walk-forward safe."""
    # MODE (same as existing predictor)
    mp_mode = predict_core_member(df_all, core_key, test_date, window_days, basis=("core_stream" if basis=="core_stream" else "core"), stream=(stream if basis=="core_stream" else None), include_rare=False)
    pred_mode = mp_mode.get("top1")
    # LAST(global)
    last_g, _ = _member_last_label(df_all, core_key, test_date, window_days, stream=None)
    # LAST(stream)
    last_s, n_s = _member_last_label(df_all, core_key, test_date, window_days, stream=stream)
    # Hierarchical LAST: use LAST(stream) if enough samples, else LAST(global), else MODE
    pred_last_h = None
    if last_s is not None and n_s >= int(min_stream_hits_for_last):
        pred_last_h = last_s
    elif last_g is not None:
        pred_last_h = last_g
    else:
        pred_last_h = pred_mode
    # Seed-structure override
    pred_seed_ovr = pred_last_h
    try:
        seed = _seed_for_stream_asof(df_all, stream, test_date)
        if seed:
            sstruct = structure_of_4(extract_4digit(seed) or str(seed).zfill(4))
            override = {"AAAB": "AABC", "AABB": "ABBC", "AAAA": "ABCC"}.get(str(sstruct))
            if override in ("AABC","ABBC","ABCC"):
                pred_seed_ovr = override
    except Exception:
        pass
    # Trait-lift override: use best (trait,value) for this core, then MODE among past hits where that trait fires
    pred_trait_ovr = pred_last_h
    best = _pick_best_seed_trait_rule(traits_pos_df, core_key)
    if best is not None:
        tname, tval = best
        # if the current seed's trait matches, apply the member-mode for that trait value
        cur_traits = _seed_traits_for_core_stream(df_all, core_key, stream, test_date)
        if cur_traits.get(tname) == tval:
            m = _member_mode_from_trait(df_all, core_key, test_date, window_days, tname, tval, stream=(stream if basis=="core_stream" else None))
            if m in ("AABC","ABBC","ABCC"):
                pred_trait_ovr = m
    return {
        "MODE": pred_mode,
        "LAST_GLOBAL": last_g,
        "LAST_HIER": pred_last_h,
        "SEED_OVERRIDE": pred_seed_ovr,
        "TRAIT_OVERRIDE": pred_trait_ovr,
    }

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
    
    # Derived ranking columns (for bucket picks + backtest)
    # BaseScoreRank: same as RankPos (1 = strongest recent strength)
    if "RankPos" in out.columns and "BaseScoreRank" not in out.columns:
        out["BaseScoreRank"] = out["RankPos"]
    # BaseScore: a simple continuous strength proxy (used for sorting/UX)
    if "BaseScore" not in out.columns:
        out["BaseScore"] = out.get("HitsPerWeek", 0.0)
    # DueIndex: "how due" a stream is (days since last hit)
    if "DueIndex" not in out.columns:
        out["DueIndex"] = out.get("DaysSinceLastHit", 0)
    # DueIndexRank: 1 = most due (largest DueIndex)
    if "DueIndexRank" not in out.columns:
        try:
            _di = pd.to_numeric(out["DueIndex"], errors="coerce").fillna(-1)
            out["DueIndexRank"] = (-_di).rank(method="dense", ascending=True).astype(int)
        except Exception:
            out["DueIndexRank"] = out["BaseScoreRank"] if "BaseScoreRank" in out.columns else range(1, len(out) + 1)

    return out

def position_percentile_map(df_rankpos: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create a percentile map over RankPos (1..78) using hit counts.

    Input must have at least:
      - RankPos (int)
      - HitsWindow (int/float) OR HitCount (int/float)

    Returns:
      (pos_df, meta)

    pos_df columns include:
      - RankPos
      - HitCount
      - HitCountPctile (0-100)
      - PctStrength (alias of HitCountPctile; back-compat)
      - HitSharePct (percent of total hits at this RankPos)
      - CumuHitSharePct (cumulative share across RankPos ascending)

    Notes:
      - If multiple rows share the same RankPos (e.g., aggregating many cores),
        they are summed first.
    """
    empty_cols = [
        "RankPos", "HitCount", "HitCountPctile", "PctStrength", "HitSharePct", "CumuHitSharePct",
        "HitShare", "CumHitShare",
    ]
    if df_rankpos is None or df_rankpos.empty:
        return pd.DataFrame(columns=empty_cols), {"total_hits": 0.0, "rows": 0}

    pos = df_rankpos.copy()

    # Normalize column name
    if "HitCount" not in pos.columns and "HitsWindow" in pos.columns:
        pos = pos.rename(columns={"HitsWindow": "HitCount"})
    if "HitCount" not in pos.columns:
        # Best effort: try common alternatives
        for alt in ["Hits", "Count", "Hit_Count"]:
            if alt in pos.columns:
                pos = pos.rename(columns={alt: "HitCount"})
                break

    if "RankPos" not in pos.columns or "HitCount" not in pos.columns:
        return pd.DataFrame(columns=empty_cols), {"total_hits": 0.0, "rows": 0}

    # Aggregate by RankPos (important for ALL-CORES maps)
    pos = pos.groupby("RankPos", as_index=False)["HitCount"].sum()

    # Sort by RankPos for consistent cumulative share
    pos["RankPos"] = pos["RankPos"].astype(int)
    pos = pos.sort_values("RankPos").reset_index(drop=True)

    # Percentile rank by hit count (ties handled by average rank)
    pos["HitCountPctile"] = pos["HitCount"].rank(pct=True) * 100.0

    total_hits = float(pos["HitCount"].sum())
    denom = total_hits if total_hits != 0.0 else 1.0
    pos["HitSharePct"] = (pos["HitCount"] / denom) * 100.0
    pos["CumuHitSharePct"] = pos["HitSharePct"].cumsum()

    # Back-compat aliases used in older UI text + newer tie-break key
    pos["HitShare"] = pos["HitSharePct"]
    pos["CumHitShare"] = pos["CumuHitSharePct"]
    pos["PctStrength"] = pos["HitCountPctile"]

    meta = {
        "total_hits": total_hits,
        "rows": int(pos.shape[0]),
    }
    return pos, meta


# -------------------------
# Seed Traits (positive/negative) + Cadence (v51)
# -------------------------
def _read_local_or_uploaded_csv(uploaded_file, local_path: str) -> pd.DataFrame:
    """Read CSV from uploaded file-like or a local repo path. Returns empty df on failure."""
    try:
        if uploaded_file is not None:
            try:
                uploaded_file.seek(0)
            except Exception:
                pass
            return pd.read_csv(uploaded_file)
    except Exception:
        pass
    try:
        if local_path and os.path.exists(local_path):
            return pd.read_csv(local_path)
    except Exception:
        pass
    return pd.DataFrame()

def _read_local_or_uploaded_text(uploaded_file, local_path: str) -> str:
    try:
        if uploaded_file is not None:
            try:
                uploaded_file.seek(0)
            except Exception:
                pass
            return uploaded_file.read().decode('utf-8', errors='ignore') if hasattr(uploaded_file, 'read') else str(uploaded_file)
    except Exception:
        pass
    try:
        if local_path and os.path.exists(local_path):
            return open(local_path, 'r', encoding='utf-8', errors='ignore').read()
    except Exception:
        pass
    return ""

def _build_traits_lookup(df: pd.DataFrame) -> Dict[int, Dict[str, Dict[str, float]]]:
    """lookup[core_family][trait][value] -> lift"""
    lookup: Dict[int, Dict[str, Dict[str, float]]] = {}
    if df is None or df.empty:
        return lookup
    # normalize columns
    cols = {c.lower(): c for c in df.columns}
    core_col = cols.get("core_family", None)
    trait_col = cols.get("trait", None)
    val_col = cols.get("value", None)
    lift_col = cols.get("lift", None)
    if not (core_col and trait_col and val_col and lift_col):
        return lookup
    for _, r in df.iterrows():
        try:
            core = int(r[core_col])
        except Exception:
            continue
        trait = str(r[trait_col]).strip()
        val = str(r[val_col]).strip()
        try:
            lift = float(r[lift_col])
        except Exception:
            lift = 1.0
        if not trait:
            continue
        lookup.setdefault(core, {}).setdefault(trait, {})[val] = lift
    return lookup

def _seed_sum_range4_labels(seed_sum: int) -> List[str]:
    labels = []
    for start in (seed_sum, seed_sum-1, seed_sum-2, seed_sum-3):
        if start < 0:
            continue
        end = start + 3
        if end > 36:  # pick4 max sum
            continue
        labels.append(f"{start}-{end}")
    return labels

def _seed_spread_bucket(spread: int) -> str:
    if spread <= 2:
        return "<=2"
    if spread >= 6:
        return ">=6"
    return "3-5"

def _count_high_digits(digits: List[int]) -> int:
    # Pick-4 convention in this app: high digits are 5–9
    return sum(1 for d in digits if d >= 5)

def _seed_contains_core_pair(seed: str, core: str) -> str:
    seed = str(seed)
    core = canonical_core_key(core)
    core_digits = list(core)
    pairs = set()
    for i in range(len(core_digits)):
        for j in range(len(core_digits)):
            if i == j:
                continue
            pairs.add(core_digits[i] + core_digits[j])
    adj = [seed[i:i+2] for i in range(len(seed)-1)]
    return "yes" if any(a in pairs for a in adj) else "no"

def _feature_values_for_seed(seed: str, core: str, last5_union_digits: Optional[set] = None) -> Dict[str, List[str]]:
    seed = str(seed).zfill(4)
    core = canonical_core_key(core)
    digits = [int(ch) for ch in seed]
    core_set = set(core)

    seed_sum = sum(digits)
    even_ct = sum(1 for d in digits if d % 2 == 0)
    high_ct = _count_high_digits(digits)
    spread = max(digits) - min(digits) if digits else 0

    overlap = len(set(seed) & core_set)  # unique overlap
    overlap_vals: List[str]
    if overlap == 3:
        overlap_vals = ["3", ">=2"]
    elif overlap == 2:
        overlap_vals = [">=2"]
    elif overlap == 0:
        overlap_vals = ["0"]
    else:
        overlap_vals = [str(overlap)]

    grid_overlap = 0
    if last5_union_digits:
        try:
            grid_overlap = len(set(last5_union_digits) & core_set)
        except Exception:
            grid_overlap = 0
    if grid_overlap == 3:
        grid_vals = ["3", ">=2"]
    elif grid_overlap == 2:
        grid_vals = [">=2"]
    else:
        grid_vals = [str(grid_overlap)]

    feats: Dict[str, List[str]] = {
        "seed_structure": [structure_of_4(seed)],
        "seed_even_count": [str(even_ct)],
        "seed_high_count": [str(high_ct)],
        "seed_spread": [_seed_spread_bucket(spread)],
        "seed_sum_mod2": [str(seed_sum % 2)],
        "seed_sum_mod3": [str(seed_sum % 3)],
        "seed_sum_range4_best": _seed_sum_range4_labels(seed_sum),
        "seed_sum_range4_worst": _seed_sum_range4_labels(seed_sum),
        "overlap_unique": overlap_vals,
        "seed_contains_core_pair": [_seed_contains_core_pair(seed, core)],
        "seed_first_in_core": ["yes" if seed[0] in core_set else "no"],
        "seed_last_in_core": ["yes" if seed[-1] in core_set else "no"],
        "grid_last5_core_digits": grid_vals,
    }
    return feats

def compute_seed_traits_score(
    core: str,
    seed: Optional[str],
    stream: Optional[str],
    *,
    pos_lookup: Dict[int, Dict[str, Dict[str, float]]],
    neg_lookup: Dict[int, Dict[str, Dict[str, float]]],
    last5_union_digits_by_stream: Optional[Dict[str, set]] = None,
    cap: float = 2.0,
) -> Tuple[float, List[Tuple[str, str, float, str]]]:
    """Return (net_score, matches). net_score is sum((lift-1) pos) - sum((lift-1) neg), capped."""
    if seed is None:
        return 0.0, []
    core = canonical_core_key(core)
    try:
        core_num = int(core)
    except Exception:
        core_num = int(core.lstrip('0') or 0)
    last5_union = None
    if stream and last5_union_digits_by_stream and stream in last5_union_digits_by_stream:
        last5_union = last5_union_digits_by_stream.get(stream)
    feats = _feature_values_for_seed(str(seed), core, last5_union_digits=last5_union)

    matches: List[Tuple[str, str, float, str]] = []
    score = 0.0
    for trait, vals in feats.items():
        for val in vals:
            # positive
            liftp = pos_lookup.get(core_num, {}).get(trait, {}).get(val)
            if liftp is not None:
                delta = float(liftp) - 1.0
                score += delta
                matches.append((trait, val, float(liftp), "+"))
            # negative
            liftn = neg_lookup.get(core_num, {}).get(trait, {}).get(val)
            if liftn is not None:
                delta = float(liftn) - 1.0
                score -= delta
                matches.append((trait, val, float(liftn), "-"))
    # Cap for safety
    score = max(-cap, min(cap, score))
    return float(score), matches

def compute_cadence_score(days_since_last_hit: float, mean_gap_days: float) -> float:
    """Soft cadence score in [0,1]. 0 = not due vs cadence, 1 = very due."""
    try:
        d = float(days_since_last_hit)
    except Exception:
        return 0.0
    try:
        g = float(mean_gap_days)
    except Exception:
        g = 0.0
    if g <= 0:
        return 0.0
    ratio = d / g
    # Map ratio: 1.0 -> 0, 3.0 -> 1 (cap)
    val = (ratio - 1.0) / 2.0
    if val < 0:
        return 0.0
    if val > 1:
        return 1.0
    return float(val)


def get_position_percentiles_cached(core: str, window_days: int, stream_stats: pd.DataFrame) -> pd.DataFrame:
    """Cache position percentile maps per core/window so UI tweaks don't constantly recompute.
    Cache is automatically cleared when input data changes or when the user clicks 'Recompute percentile maps now'.
    """
    cache: Dict[str, pd.DataFrame] = st.session_state.get("pos_map_cache", {})
    data_hash = st.session_state.get("data_hash_all", "")
    key = f"{core}|{window_days}|{data_hash}"

    if key in cache:
        return cache[key]

    pos_map, _ = position_percentile_map(stream_stats)
    cache[key] = pos_map
    st.session_state["pos_map_cache"] = cache

    if not st.session_state.get("recompute_token"):
        st.session_state["recompute_token"] = datetime.datetime.now().isoformat(timespec="seconds")

    return pos_map

def bucket_recommendations(
    stream_stats: pd.DataFrame,
    cfg: Optional[RankConfig] = None,
    *,
    top_n: Optional[int] = None,
    due_n: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Build Northern Star buckets from a stream_stats table.

    Returns a dict with **multiple key aliases** for compatibility:
      - Top12BaseScore / Top12  -> top base-score bucket
      - Due8                   -> due bucket
      - Combined               -> merged bucket
      - base_top / due_top / combined -> lists of stream labels (for meta)

    `top_n` / `due_n` are accepted as legacy keyword overrides.
    """
    if cfg is None:
        cfg = RankConfig()

    base_n = int(top_n) if top_n is not None else int(getattr(cfg, "top_base", 12))
    due_take = int(due_n) if due_n is not None else int(getattr(cfg, "top_due", 8))

    # Normalize expected columns so this helper works with:
    #  - stream_summary() output (RankPos, HitsPerWeek, DaysSinceLastHit, ...)
    #  - legacy bucket tables (BaseRank/DueRank)
    #  - future member-level tables (Pick/Member columns)
    df = stream_stats.copy()

    # Base rank
    if "BaseScoreRank" not in df.columns:
        if "BaseRank" in df.columns:
            df["BaseScoreRank"] = df["BaseRank"]
        elif "RankPos" in df.columns:
            df["BaseScoreRank"] = df["RankPos"]
        elif "HitsPerWeek" in df.columns:
            _hpw = pd.to_numeric(df["HitsPerWeek"], errors="coerce").fillna(0.0)
            df["BaseScoreRank"] = (-_hpw).rank(method="dense", ascending=True).astype(int)
        else:
            df["BaseScoreRank"] = range(1, len(df) + 1)

    # RankPos (for display ordering)
    if "RankPos" not in df.columns:
        df["RankPos"] = df["BaseScoreRank"]

    # Due index
    if "DueIndex" not in df.columns:
        if "DaysSinceLastHit" in df.columns:
            df["DueIndex"] = pd.to_numeric(df["DaysSinceLastHit"], errors="coerce")
        else:
            df["DueIndex"] = 0

    # Due rank
    if "DueIndexRank" not in df.columns:
        if "DueRank" in df.columns:
            df["DueIndexRank"] = df["DueRank"]
        else:
            _di = pd.to_numeric(df["DueIndex"], errors="coerce").fillna(-1)
            # 1 = most due (largest DueIndex)
            df["DueIndexRank"] = (-_di).rank(method="dense", ascending=True).astype(int)

    # Ensure Stream exists for downstream logic
    if "Stream" not in df.columns and "stream" in df.columns:
        df["Stream"] = df["stream"]
    due_lo = int(getattr(cfg, "due_from_rank", 13))
    due_hi = int(getattr(cfg, "due_to_rank", 60))

    # Defensive: tolerate missing columns; caller should validate upstream.
    if stream_stats is None or len(df) == 0:
        empty = pd.DataFrame()
        return {
            "Top12BaseScore": empty,
            "Top12": empty,
            "Due8": empty,
            "Combined": empty,
            "base_top": [],
            "due_top": [],
            "combined": [],
        }

    base_df = df.sort_values("BaseScoreRank", ascending=True).head(base_n)
    due_pool = df[
        (df["BaseScoreRank"] >= due_lo) & (df["BaseScoreRank"] <= due_hi)
    ].sort_values("DueIndexRank", ascending=True)
    due_df = due_pool.head(due_take)

    combined_df = pd.concat([base_df, due_df], ignore_index=True).drop_duplicates(subset=["Stream"], keep="first")
    combined_df = combined_df.sort_values("RankPos", ascending=True)

    base_streams = base_df["Stream"].tolist() if "Stream" in base_df.columns else []
    due_streams = due_df["Stream"].tolist() if "Stream" in due_df.columns else []
    combined_streams = combined_df["Stream"].tolist() if "Stream" in combined_df.columns else []

    return {
        "Top12BaseScore": base_df,
        "Top12": base_df,
        "Due8": due_df,
        "Combined": combined_df,
        "base_top": base_streams,
        "due_top": due_streams,
        "combined": combined_streams,
    }

def build_northern_star_bucket_meta(
    stream_stats: pd.DataFrame,
    cfg: RankConfig,
    *,
    seed_core_key: str = "",
    include_24h: bool = False,
    df_24: pd.DataFrame | None = None,
    core: str = "",
) -> List[Dict[str, Any]]:
    """Compatibility helper used by some older app revisions.

    Returns a list of per-stream bucket metadata rows (one dict per stream).
    This intentionally mirrors the data shape consumed by the Northern Lights master playlist.
    """
    if stream_stats is None or not isinstance(stream_stats, pd.DataFrame) or stream_stats.empty:
        return []

    # Pre-compute which streams are in which bucket for this core.
    rec = bucket_recommendations(stream_stats, cfg)
    base_streams = set(rec.get("base_top", []))
    due_streams = set(rec.get("due_top", []))

    rows: List[Dict[str, Any]] = []
    for stream in stream_stats["Stream"].tolist():
        try:
            rows.append(
                build_northern_star_buckets(
                    stats_df=stream_stats,
                    stream=stream,
                    top_n=cfg.top_base,
                    due_ranks=(cfg.due_from_rank, cfg.due_to_rank),
                    seed_core_key=seed_core_key,
                    include_24h=include_24h,
                    df_24=df_24,
                    core=core,
                    base_streams=base_streams,
                    due_streams=due_streams,
                )
            )
        except TypeError:
            # Oldest signature (no precomputed sets)
            rows.append(
                build_northern_star_buckets(
                    stats_df=stream_stats,
                    stream=stream,
                    top_n=cfg.top_base,
                    due_ranks=(cfg.due_from_rank, cfg.due_to_rank),
                    seed_core_key=seed_core_key,
                    include_24h=include_24h,
                    df_24=df_24,
                    core=core,
                )
            )
    return rows

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
    df_24h: pd.DataFrame | None = None,
    enable_r1: bool = True,
    enable_r2: bool = True,
    enable_r3: bool = True,
    enable_r4: bool = True,
    window_days_recent: int = 180,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if df_24h is None:
        df_24h = pd.DataFrame()
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
    df_24h: pd.DataFrame | None = None,
    enable_q1: bool = True,
    enable_q2: bool = True,
    enable_q3: bool = True,
    enable_q4: bool = True,
    # Some UI call-sites pass this (mirroring the rare engine). We accept it for
    # compatibility. The ultra-rare engine is primarily computed on the full
    # history; when provided, we use it only for optional recent-window fields.
    window_days_recent: int | None = None,
    # Forward-compat: ignore any extra kwargs passed from older/newer UIs.
    **_ignored_kwargs,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Ultra-Rare Engine checks (AAAA quads for the core's digits):
      Q1: stream in top 10% for quad baseline rate (full history)
      Q2: days since last quad >= 90th percentile across streams
      Q3: last 24h has at least 1 quad anywhere (for any digit in core)
      Q4: 24h quad hits (for core) cluster into Top-5 RankPos; stream position is in that set
    Trigger: at least 2 of enabled checks True.
    """
    if df_24h is None:
        df_24h = pd.DataFrame()

    if df_all.empty:
        return pd.DataFrame(), {"error":"No history loaded."}

    core = canonical_core_key(core)
    df_hits_all = compute_core_hits(df_all, core, structures=["AAAA"])

    span_days = max(int((df_all["Date"].max()-df_all["Date"].min()).days), 1)
    base_stats = stream_summary(df_all, df_hits_all, window_days=min(365*5, span_days))

    # Optional recent window (used by some UI displays). This is non-breaking: if
    # absent, downstream behavior matches historical-only mode.
    base_stats_recent = None
    if window_days_recent is not None and not df_all.empty and "Date" in df_all.columns:
        try:
            cutoff = df_all["Date"].max() - pd.Timedelta(days=int(window_days_recent))
            df_recent = df_all[df_all["Date"] >= cutoff].copy()
            df_hits_recent = compute_core_hits(df_recent, core, structures=["AAAA"])
            base_stats_recent = stream_summary(df_recent, df_hits_recent, window_days=int(window_days_recent))
        except Exception:
            base_stats_recent = None
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


def render_backtest(
    df_all: pd.DataFrame,
    cfg: "RankConfig | None" = None,
    cores_for_cache: "list[str] | None" = None,
    df_24h: "pd.DataFrame | None" = None,
    # backwards-compatible aliases (older call sites)
    cores: "list[str] | None" = None,
    window_days: "int | None" = None,
):
    """Backtest / diagnostics.

    **Walk-forward mode (no-cheat):**
    For each test_date, builds rankings/traps using ONLY rows with Date < test_date, then scores the
    winner(s) that occurred on test_date.

    **Playlist diagnostic mode:**
    Uses the current Northern Lights playlist in-session (helpful for quick validation, but can
    include future leakage if the playlist was built using the full dataset).
    """

    # Normalize args (avoid brittle keyword mismatches across revisions)
    if cfg is None:
        cfg = RankConfig(window_days=int(window_days or 180))
    if cores_for_cache is None:
        cores_for_cache = list(cores or [])

    st.subheader("Backtest (optional)")
    st.caption("Optional diagnostics. Walk-forward mode avoids future leakage by training only on Date < test_date.")

    if df_all is None or getattr(df_all, "empty", True):
        st.warning("Upload an all-states history file to use Backtest.")
        return

    # Ensure Date dtype
    if "Date" in df_all.columns:
        try:
            if not pd.api.types.is_datetime64_any_dtype(df_all["Date"]):
                df_all = df_all.copy()
                df_all["Date"] = pd.to_datetime(df_all["Date"], errors="coerce")
        except Exception:
            pass

    mode = st.radio(
        "Backtest mode",
        ["Walk-forward (no cheating)", "Playlist diagnostic (uses current playlist)"],
        horizontal=True,
        key="bt_mode_v51",
    )

    if mode.startswith("Walk-forward"):
        _render_backtest_walk_forward(df_all=df_all, cfg=cfg, cores_for_cache=cores_for_cache)
        return

    # Playlist diagnostic (legacy / quick)
    """Optional diagnostics backtest.

    This evaluates how often the *current* Northern Lights master playlist (streams+cores)
    would have caught a matching family member in those streams over a selected historical range.
    It does **not** change any scoring or ranking output.
    """

    if df_all is None or df_all.empty:
        st.warning("Upload an all-states history file first.")
        return

    # Ensure required columns exist
    if "Date" not in df_all.columns or "Stream" not in df_all.columns:
        st.error("History file is missing required columns (Date, Stream).")
        return

    # Date bounds
    try:
        dmin_ts = pd.to_datetime(df_all["Date"]).min()
        dmax_ts = pd.to_datetime(df_all["Date"]).max()
    except Exception:
        st.error("Could not read Date values from the history file.")
        return

    if pd.isna(dmin_ts) or pd.isna(dmax_ts):
        st.error("History file has no valid dates.")
        return

    dmin = dmin_ts.date()
    dmax = dmax_ts.date()
    default_start = max(dmin, (dmax_ts - pd.Timedelta(days=min(180, max(7, int((dmax_ts - dmin_ts).days * 0.25))))).date())
    date_range = st.date_input(
        "Backtest date range (inclusive)",
        value=(default_start, dmax),
        min_value=dmin,
        max_value=dmax,
        help="This checks historical draws in the selected range against your current playlist picks."
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = date_range, date_range

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    # Structures used to define a 'hit' for a core in a stream
    structure_mode = st.selectbox(
        "Match mode (what counts as a hit for a core)",
        options=[
            "AABC only (single-member focus)",
            "Family mode (AABC + ABBC + ABCC)",
            "Rare engine mode (AAAB + AABB)",
            "Ultra-rare mode (AAAA only)",
        ],
        index=1,
        help="This affects only the backtest match check (not your ranking)."
    )
    if structure_mode.startswith("AABC only"):
        structures = ["AABC"]
    elif structure_mode.startswith("Family mode"):
        structures = ["AABC", "ABBC", "ABCC"]
    elif structure_mode.startswith("Rare engine mode"):
        structures = ["AAAB", "AABB"]
    else:
        structures = ["AAAA"]

    # Playlist source
    nl_df = st.session_state.get("nl_df_current")
    if not isinstance(nl_df, pd.DataFrame) or nl_df.empty or "Core" not in nl_df.columns or "Stream" not in nl_df.columns:
        with st.expander("Master playlist not found in session — build it now", expanded=True):
            st.info("Your master playlist isn't cached in the current session. Click below to build it (this can take a bit).")
            if st.button("Build master playlist for backtest", type="primary"):
                nl_df = None  # force rebuild below

    def _build_master_playlist_for_backtest() -> pd.DataFrame:
        cores = [canonical_core_key(c) for c in (cores_for_cache or [])]
        cores = [c for c in cores if c]
        if not cores:
            # Fall back to whatever is selected in session state
            cores = [canonical_core_key(c) for c in st.session_state.get("cores_selected", [])]
            cores = [c for c in cores if c]
        if not cores:
            return pd.DataFrame()

        cache = _load_baseline_from_disk(cfg.window_days)
        cache_ok = False
        if isinstance(cache, dict):
            cached_cores = set(cache.get("cores", []) or [])
            if cached_cores and all(c in cached_cores for c in cores):
                cache_ok = True

        rows = []
        progress = st.progress(0)
        for i, core in enumerate(cores, start=1):
            if cache_ok:
                stream_stats = cache["core_stream_stats"][core]
                pos_map = cache["core_pos_maps"][core]
            else:
                stream_stats = compute_stream_stats(df_all, core_key=core, structures=("AABC",), window_days=cfg.window_days)
                pos_map = pos_map_for_core(df_all, core_key=core, structures=["AABC"])
            bucket_rows = build_northern_star_buckets(stream_stats, pos_map, cfg)
            bucket_rows = bucket_rows.copy()
            bucket_rows["Core"] = core
            rows.append(bucket_rows)
            progress.progress(int(i / max(1, len(cores)) * 100))
        progress.empty()

        if not rows:
            return pd.DataFrame()

        out = pd.concat(rows, ignore_index=True)

        # Universal score (keep identical to Northern Lights tab)
        out = out.copy()
        out["RecentStrength"] = out["RecentHitRate"] * 100.0
        out["DuePressure"] = out["DueScore"]
        out["PosStrength"] = out["PosPctScore"]
        out["UniversalScore"] = 0.45 * out["RecentStrength"] + 0.35 * out["DuePressure"] + 0.20 * out["PosStrength"]
        out["UniversalRank"] = out["UniversalScore"].rank(ascending=False, method="min").astype(int)

        # Column order preference
        cols_front = [
            "UniversalRank", "UniversalScore", "Core", "Stream",
            "Bucket", "BucketPick", "BaseRank", "DueRank",
            "RecentHitRate", "DueScore", "PosPctScore",
            "DaysSinceLastHit", "HitsWindow", "DrawsWindow"
        ]
        out = out[[c for c in cols_front if c in out.columns] + [c for c in out.columns if c not in cols_front]]
        out = out.sort_values(["UniversalRank", "Core", "Stream"]).reset_index(drop=True)
        return out

    if nl_df is None:
        nl_df = _build_master_playlist_for_backtest()
        if nl_df is not None and not nl_df.empty:
            st.session_state["nl_df_current"] = nl_df

    if not isinstance(nl_df, pd.DataFrame) or nl_df.empty:
        st.warning("No master playlist available to backtest. Build it first (Northern Lights tab or button above).")
        return

    playlist_mode = st.radio(
        "What to backtest",
        options=["Top N overall (by UniversalScore)", "Top 1 per stream (best core per stream)", "All playlist entries"],
        index=0,
        horizontal=True,
    )

    # Build the picks table (Core + Stream)
    nl_df_sorted = nl_df.sort_values(["UniversalScore", "UniversalRank"], ascending=[False, True]).copy()
    if playlist_mode.startswith("Top N overall"):
        max_n = max(1, min(500, int(len(nl_df_sorted))))
        default_n = min(39, max_n)
        top_n = st.slider("Top N entries to play each day", 1, max_n, default_n)
        picks = nl_df_sorted.head(top_n)[["Core", "Stream"]].dropna().drop_duplicates().reset_index(drop=True)
    elif playlist_mode.startswith("Top 1 per stream"):
        picks = (
            nl_df_sorted.sort_values(["Stream", "UniversalScore"], ascending=[True, False])
            .groupby("Stream", as_index=False)
            .head(1)[["Core", "Stream"]]
            .dropna().drop_duplicates()
            .reset_index(drop=True)
        )
    else:
        picks = nl_df_sorted[["Core", "Stream"]].dropna().drop_duplicates().reset_index(drop=True)

    if picks.empty:
        st.warning("No playlist picks available for the selected backtest mode.")
        return

    # Cost settings (optional)
    st.markdown("#### Cost assumptions (optional)")
    colc1, colc2, colc3 = st.columns([1, 1, 2])
    with colc1:
        cost_per_play = st.number_input("Cost per play", min_value=0.0, value=0.25, step=0.05)
    with colc2:
        payout_per_win = st.number_input("Payout per win", min_value=0.0, value=247.50, step=1.0)
    with colc3:
        if structure_mode.startswith("Family mode"):
            default_numbers = 3
        else:
            default_numbers = 1
        numbers_per_pick = st.number_input(
            "Number of box numbers per (Core+Stream) pick",
            min_value=1,
            value=int(default_numbers),
            step=1,
            help="If you play all members of a family core, set this to 3. If you play only one number per pick, set to 1."
        )

    # Filter history to date range
    df_range = df_all.copy()
    df_range["Date"] = pd.to_datetime(df_range["Date"])
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df_range = df_range[(df_range["Date"] >= start_ts) & (df_range["Date"] <= end_ts)]

    if df_range.empty:
        st.warning("No draws found in the selected date range.")
        return

    # Ensure BoxKey4 exists
    if "BoxKey4" not in df_range.columns:
        if "Result" not in df_range.columns:
            st.error("History file is missing Result/BoxKey4 needed for matching.")
            return
        df_range = df_range.copy()
        df_range["BoxKey4"] = df_range["Result"].astype(str).str.zfill(4).map(box_key)

    # Precompute draw counts per stream for opportunity counting
    draws_by_stream = df_range["Stream"].value_counts().to_dict()

    # Evaluate hits per pick
    core_to_streams = picks.groupby("Core")["Stream"].apply(list).to_dict()

    records = []
    total_opportunities = 0
    total_hits = 0
    total_unique_hit_days = set()

    for core, streams in core_to_streams.items():
        if not streams:
            continue
        members = set(members_from_core(core, structures=structures))
        if not members:
            continue

        df_s = df_range[df_range["Stream"].isin(streams)]
        if df_s.empty:
            continue

        df_hits = df_s[df_s["BoxKey4"].isin(members)]
        # opportunities = sum draws for each stream used by this core
        core_opps = sum(int(draws_by_stream.get(s, 0)) for s in streams)
        core_hits = int(len(df_hits))

        total_opportunities += core_opps
        total_hits += core_hits
        total_unique_hit_days.update(df_hits["Date"].dt.date.unique().tolist())

        # Stream-level breakdown
        hits_by_stream = df_hits.groupby("Stream").size().to_dict()
        for s in streams:
            opp = int(draws_by_stream.get(s, 0))
            h = int(hits_by_stream.get(s, 0))
            records.append({

"Core": core,
"ActualMemberLabel": None,
"ActualFamilyMember": None,
"PredMemberTop1": None,
"PredMemberTop2": None,
"MemberHitTop1": None,
"MemberHitTop2": None,
"MemberTrainN": 0,
"TrainCnt_AABC": 0,
"TrainCnt_ABBC": 0,
"TrainCnt_ABCC": 0,
                "Stream": s,
                "Opportunities": opp,
                "Hits": h,
                "HitRate": (h / opp) if opp else 0.0
            })

    if not records:
        st.warning("No matching hits found for the selected settings.")
        return

    bt_df = pd.DataFrame(records).sort_values(["Hits", "HitRate"], ascending=[False, False]).reset_index(drop=True)

    days_in_range = int((end_ts.normalize() - start_ts.normalize()).days) + 1
    plays_per_day = int(len(picks)) * int(numbers_per_pick)
    total_plays = int(total_opportunities) * int(numbers_per_pick)
    est_cost = float(total_plays) * float(cost_per_play)
    est_payout = float(total_hits) * float(payout_per_win)
    est_profit = est_payout - est_cost

    colm1, colm2, colm3, colm4 = st.columns(4)
    colm1.metric("Hits", f"{total_hits}")
    colm2.metric("Unique hit days", f"{len(total_unique_hit_days)} / {days_in_range}")
    colm3.metric("Opportunities", f"{total_opportunities}")
    colm4.metric("Plays/day (assumed)", f"{plays_per_day}")

    st.markdown("#### Estimated cost / payout (using your assumptions)")
    colp1, colp2, colp3 = st.columns(3)
    colp1.metric("Estimated cost", f"${est_cost:,.2f}")
    colp2.metric("Estimated payout", f"${est_payout:,.2f}")
    colp3.metric("Estimated profit", f"${est_profit:,.2f}")

    st.markdown("#### Backtest breakdown (per Core + Stream)")
    st.dataframe(bt_df, use_container_width=True)

    # Summaries
    st.markdown("#### Summary by core")
    core_sum = bt_df.groupby("Core", as_index=False).agg(
        Opportunities=("Opportunities", "sum"),
        Hits=("Hits", "sum")
    )
    core_sum["HitRate"] = core_sum["Hits"] / core_sum["Opportunities"].replace(0, np.nan)
    core_sum = core_sum.sort_values(["Hits", "HitRate"], ascending=[False, False]).reset_index(drop=True)
    st.dataframe(core_sum, use_container_width=True)

    # Download
    csv_bytes = bt_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download backtest breakdown (CSV)",
        data=csv_bytes,
        file_name="backtest_core_stream_breakdown.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.set_page_config(page_title="Pick 4 Northern Star", layout="wide", initial_sidebar_state="expanded")

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

    st.divider()
    st.subheader("Seed Traits + Cadence (v51)")
    traits_pos_file = st.file_uploader("Seed traits POSITIVE CSV (optional; autoloads if present)", type=["csv"], key="traits_pos_file")
    traits_neg_file = st.file_uploader("Seed traits NEGATIVE CSV (optional; autoloads if present)", type=["csv"], key="traits_neg_file")
    cadence_md_file = st.file_uploader("Family cadence report (.md optional; autoloads if present)", type=["md","txt"], key="cadence_md_file")

    enable_seed_traits = st.checkbox("Enable Seed Traits boost (soft only)", value=True, key="enable_seed_traits")
    seed_traits_weight = st.slider("Seed Traits weight", 0.0, 1.0, 0.35, 0.05, key="seed_traits_weight")
    enable_cadence = st.checkbox("Enable Cadence boost (soft only)", value=True, key="enable_cadence")
    cadence_weight = st.slider("Cadence weight", 0.0, 1.0, 0.25, 0.05, key="cadence_weight")

    # Keep these weights conservative by default
    due_weight = st.slider("DuePressure weight", 0.0, 1.0, 0.20, 0.05, key="due_weight")
    pos_weight = st.slider("Position-percentile weight", 0.0, 1.0, 0.25, 0.05, key="pos_weight")

    map_file = map24_file  # backward-compatible alias

    exclude_md = st.checkbox("Exclude Maryland (MD)", value=True, help="Optional global exclusion. When enabled (default), MD rows are removed from both the baseline and 24h files before ranking.")
    st.session_state["exclude_md"] = exclude_md

    st.divider()
    st.subheader("Trigger Map (39-play list) — optional boost")
    _apply = st.checkbox("Apply Trigger Map boost", value=False)
    st.session_state["_apply_trigger_map"] = _apply
    _pts = st.slider("Trigger boost points", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
    st.session_state["_trigger_boost_points"] = float(_pts)


    st.divider()
    st.divider()
    with st.expander("Build checklist (do not omit)", expanded=False):
        # --- Live status (auto) ---
        st.markdown("### Live status")
        _sel_now = st.session_state.get("cores_for_cache_ms", []) or st.session_state.get("selected_cores", []) or []
        st.write({
            "hardcoded_daily_doubles_cores": len(CORE_PRESETS),
            "selected_cores_now": len(_sel_now),
            "selected_cores_list": _sel_now[:25] + (["…"] if len(_sel_now) > 25 else []),
        })

        st.markdown("""
**A. Core + cache**
- **A1** Multi-core selection (core dropdown + multi-select)
- **A2** Cache Builder: build baseline cache for selected cores
- **A3** Show tabs for all selected cores (optional) in Core view
- **A4** Core ranking percentile map (tie-breaker) in Northern Lights view
- **A5** Bucket method: Top 12 + DueIndex 13–60 (8 picks)
- **A6** Straights module optional last (does not run unless enabled)

**B. Northern Star / Northern Lights**
- **B1** Northern Star (per-core) ranking view
- **B2** Northern Star buckets per core (Base + Due)
- **B3** Northern Lights master playlist (cross-core)
- **B4** Master playlist scoring is deterministic (stable tie-break)
- **B5** Optional Trigger Map boost (39-play list)

**C. Maps / percentiles**
- **C1** Global Northern Star position percentile map (1–78)
- **C2** Per-core percentile map tabs for selected cores

**D. Cadence & behavior (soft-only)**
- **D1** Cadence report integration (soft boost / transparency)
- **D2** Core-specific behavior tables (per-core stats cached)

**E. Self-maintenance**
- **E1** Local rolling ~3-year baseline store (append from 24h)
- **E2** Purge rows older than ~3 years (automatic)
- **E3** Store status panel: rows + date range + last updated
- **E4** One-click store rebuild/reset (safety)

If any item is missing in a build, treat it as a regression and restore it before adding new features.
""")


    st.subheader("Self-update rolling baseline (optional)")
    use_store = st.checkbox("Use local rolling ~3-year baseline store", value=False, help="Keeps a local rolling baseline by appending new rows from the 24h file and purging rows older than ~3 years. This improves speed and keeps your baseline fresh without you manually editing the all-states file.")
    st.session_state["use_store"] = use_store

    if use_store:
        store_df_preview = load_baseline_store()
        store_meta = _read_meta(BASELINE_STORE_BASE)
        store_rows = int(store_meta.get("rows", store_df_preview.shape[0] if store_df_preview is not None else 0) or 0)
        store_max = store_meta.get("max_date", "") or (str(store_df_preview["Date"].max()) if store_df_preview is not None and not store_df_preview.empty else "")

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
_csel1, _csel2 = st.columns([1,1])
with _csel1:
    if st.button("Select all cores", key="btn_select_all_cores"):
        st.session_state["cores_for_cache_ms"] = list(cores)
        _rerun()
with _csel2:
    if st.button("Clear selection", key="btn_clear_all_cores"):
        st.session_state["cores_for_cache_ms"] = []
        _rerun()

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


# ---- v51: Seed Traits + Cadence data (autoload + optional uploads)
DEFAULT_TRAITS_POS_PATH = "family_seed_traits_DOUBLES_top_positive_EXPANDED.csv"
DEFAULT_TRAITS_NEG_PATH = "family_seed_traits_DOUBLES_top_negative_EXPANDED.csv"
DEFAULT_CADENCE_MD_PATH = "family_cadence_report.md"

# Load traits (uploaded overrides local if provided)
seed_traits_pos_df = _read_local_or_uploaded_csv(globals().get("traits_pos_file", None), DEFAULT_TRAITS_POS_PATH)
seed_traits_neg_df = _read_local_or_uploaded_csv(globals().get("traits_neg_file", None), DEFAULT_TRAITS_NEG_PATH)
seed_traits_pos_lookup = _build_traits_lookup(seed_traits_pos_df)
seed_traits_neg_lookup = _build_traits_lookup(seed_traits_neg_df)

cadence_report_text = _read_local_or_uploaded_text(globals().get("cadence_md_file", None), DEFAULT_CADENCE_MD_PATH)

# Precompute per-stream seed + last5 union digits (for Seed Traits feature)
_prev_seed_by_stream: Dict[str, str] = {}
_last5_union_by_stream: Dict[str, set] = {}

try:
    # Determine the most recent 4-digit seed per stream (prefer 24h map if present)
    if df_24h is not None and not df_24h.empty and "Stream" in df_24h.columns and "Result" in df_24h.columns:
        _tmp = df_24h.copy()
        if "Date" in _tmp.columns:
            _tmp = _tmp.sort_values("Date")
        # take last per stream
        _prev = _tmp.groupby("Stream", as_index=False).tail(1)
        _prev_seed_by_stream = dict(zip(_prev["Stream"].astype(str), _prev["Result"].astype(str)))
    if not _prev_seed_by_stream:
        _tmp = df_all.copy()
        _tmp = _tmp.sort_values("Date")
        _prev = _tmp.groupby("Stream", as_index=False).tail(1)
        _prev_seed_by_stream = dict(zip(_prev["Stream"].astype(str), _prev["Result"].astype(str)))

    # last5 union digits per stream from df_all (most recent 5 rows)
    _tmp = df_all.sort_values("Date")
    for s, g in _tmp.groupby("Stream"):
        tail = g.tail(5)
        digs = set("".join(tail["Result"].astype(str).tolist()))
        _last5_union_by_stream[str(s)] = digs
except Exception:
    pass

today = datetime.date.today()


def _render_backtest_walk_forward(df_all: pd.DataFrame, cfg: "RankConfig", cores_for_cache: list[str]) -> None:
    """Walk-forward backtest (no future leakage).

    - For each test_date:
      - train_df = rows with Date < test_date
      - build per-core stream ranking/buckets from train_df ONLY
      - score winner rows on test_date against those buckets
    """
    if df_all is None or df_all.empty:
        st.warning("No data loaded.")
        return
    if "Date" not in df_all.columns or "Stream" not in df_all.columns or "Result" not in df_all.columns:
        st.error("Backtest requires columns: Date, Stream, Result.")
        return

    # Pick cores
    all_cores = list(CORE_PRESETS)
    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    with c1:
        use_all = st.checkbox("Test ALL cores", value=False, key="bt_use_all_cores")
    with c2:
        include_rare = st.checkbox("Include AAAB/AABB/AAAA members", value=False, key="bt_include_rare")
    with c3:
        max_dates = st.number_input("Max test dates", min_value=1, max_value=3650, value=120, step=10, key="bt_max_dates")


    st.markdown("##### Member‑pick tracking (optional)")
    m1, m2 = st.columns([1.1, 1.9])
    with m1:
        member_track = st.checkbox("Track member accuracy (Top1/Top2)", value=True, key="bt_member_track")
    with m2:
        member_basis_label = st.selectbox(
            "Member predictor basis",
            ["Per‑core (all streams)", "Per‑core + stream"],
            index=0,
            key="bt_member_basis",
            help="Per‑core uses all streams to learn which member is most common for that core. Per‑core+stream learns separately per stream (more specific, but fewer samples).",
        )
    member_basis = "core_stream" if member_basis_label.startswith("Per‑core + stream") else "core"

    if use_all:
        cores_sel = all_cores
    else:
        default_cores = cores_for_cache if cores_for_cache else ["389"]
        cores_sel = st.multiselect(
            "Cores to test",
            options=all_cores,
            default=[c for c in default_cores if c in all_cores] or [all_cores[0]],
            key="bt_cores_sel",
        )

    if not cores_sel:
        st.info("Select at least one core to backtest.")
        return

    # Date range
    dmin = pd.to_datetime(df_all["Date"], errors="coerce").min()
    dmax = pd.to_datetime(df_all["Date"], errors="coerce").max()
    if pd.isna(dmin) or pd.isna(dmax):
        st.error("Could not parse Date values for backtest.")
        return

    default_start = (dmax - pd.Timedelta(days=90)).date() if (dmax - dmin).days > 120 else dmin.date()
    start_date, end_date = st.date_input(
        "Test date range (inclusive)",
        value=(default_start, dmax.date()),
        min_value=dmin.date(),
        max_value=dmax.date(),
        key="bt_date_range",
    )

    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    only_hit_days = st.checkbox("Evaluate only days where a selected core member hit (faster)", value=True, key="bt_only_hit_days")

    run = st.button("Run walk-forward backtest", key="bt_run_btn")
    if not run:
        st.info("Click **Run walk-forward backtest** to generate results.")
        return

    df = df_all.copy()
    df = df[df["Date"].notna()]
    df = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)]
    if df.empty:
        st.warning("No rows in the selected date range.")
        return

    # Build member->core reverse index
    member_to_cores: dict[str, list[str]] = {}
    for core in cores_sel:
        members = []
        # Always include the core's main family members (AABC/ABBC/ABCC)
        members.extend(members_from_core(core, "AABC"))
        # Optionally include higher-rarity structures for the same core
        if include_rare:
            members.extend(members_from_core(core, "AAAB"))
            members.extend(members_from_core(core, "AABB"))
            members.extend(members_from_core(core, "AAAA"))
        for mem in members:
            member_to_cores.setdefault(box_key(mem), []).append(core)

    # Winners by date
    winners_by_date = {d: g for d, g in df.groupby(df["Date"].dt.normalize())}

    # Determine dates to evaluate
    all_dates_sorted = sorted(winners_by_date.keys())
    if only_hit_days:
        candidate_dates = []
        for d in all_dates_sorted:
            g = winners_by_date[d]
            hit = False
            for w in g["Result"].astype(str).tolist():
                if member_to_cores.get(box_key(w.strip()), None):
                    hit = True
                    break
            if hit:
                candidate_dates.append(d)
        dates_to_test = candidate_dates
    else:
        dates_to_test = all_dates_sorted

    if not dates_to_test:
        st.warning("No hit days found for the selected cores in this date range.")
        return

    if len(dates_to_test) > int(max_dates):
        dates_to_test = dates_to_test[-int(max_dates):]

    # Cache per (core, as_of_date, window_days) within this run
    per_core_cache: dict[tuple[str, pd.Timestamp, int], pd.DataFrame] = {}
    per_core_buckets: dict[tuple[str, pd.Timestamp, int], dict] = {}

    # Member-pick prediction cache for this run: (core, test_date, window_days, basis, stream) -> dict
    member_pred_cache: dict[tuple[str, pd.Timestamp, int, str, str | None], dict[str, Any]] = {}

    rows = []
    for test_date in dates_to_test:
        train_df = df_all[df_all["Date"] < test_date]
        if train_df.empty:
            continue

        day_winners = winners_by_date.get(test_date)
        if day_winners is None or day_winners.empty:
            continue

        for _, wr in day_winners.iterrows():
            stream = str(wr.get("Stream", "")).strip()
            winner = str(wr.get("Result", "")).strip()
            wk = box_key(winner)
            hit_cores = member_to_cores.get(wk, [])
            if not hit_cores:
                continue

            for core in hit_cores:
                key = (core, test_date, int(cfg.window_days))
                if key not in per_core_cache:
                    stats_df = compute_stream_stats(train_df, core, window_days=int(cfg.window_days))
                    per_core_cache[key] = stats_df
                    per_core_buckets[key] = bucket_recommendations(stats_df, cfg)

                stats_df = per_core_cache[key]
                buckets = per_core_buckets[key]
                base_streams = set(buckets["Top12BaseScore"]["Stream"].astype(str).tolist()) if "Top12BaseScore" in buckets else set()
                due_streams = set(buckets["Due8"]["Stream"].astype(str).tolist()) if "Due8" in buckets else set()
                predicted = stream in base_streams or stream in due_streams
                bucket = "Both" if (stream in base_streams and stream in due_streams) else ("BaseScore" if stream in base_streams else ("Due8" if stream in due_streams else "None"))

                # Pull per-stream stats if present
                stat_row = None
                try:
                    stat_row = stats_df.loc[stats_df["Stream"].astype(str) == stream].iloc[0]
                except Exception:
                    stat_row = None



                # Member labels + walk-forward member-pick prediction (family only: AABC/ABBC/ABCC)

                actual_member_label = core_member_label(core, winner, include_rare=bool(include_rare)) if member_track else None

                actual_family_member = actual_member_label if actual_member_label in ("AABC", "ABBC", "ABCC") else None


                pred_member_top1 = None

                pred_member_top2 = None

                pred_member_n = 0

                train_cnt_aabc = 0

                train_cnt_abbc = 0

                train_cnt_abcc = 0

                member_hit_top1 = None

                member_hit_top2 = None


                if member_track:

                    mk = (

                        str(core),

                        pd.to_datetime(test_date).normalize(),

                        int(cfg.window_days),

                        str(member_basis),

                        (str(stream) if member_basis == "core_stream" else None),

                    )

                    if mk not in member_pred_cache:

                        member_pred_cache[mk] = predict_core_member(

                            df_all,

                            core,

                            pd.to_datetime(test_date).normalize(),

                            window_days=int(cfg.window_days),

                            basis=str(member_basis),

                            stream=(str(stream) if member_basis == "core_stream" else None),

                            include_rare=False,  # compare only AABC/ABBC/ABCC

                        )

                    mp = member_pred_cache.get(mk, {})

                    pred_member_top1 = mp.get("top1")

                    pred_member_top2 = mp.get("top2")

                    pred_member_n = int(mp.get("n") or 0)

                    cnts = mp.get("counts") or {}

                    train_cnt_aabc = int(cnts.get("AABC") or 0)

                    train_cnt_abbc = int(cnts.get("ABBC") or 0)

                    train_cnt_abcc = int(cnts.get("ABCC") or 0)


                    if actual_family_member and pred_member_top1:

                        member_hit_top1 = (actual_family_member == pred_member_top1)

                        member_hit_top2 = (actual_family_member == pred_member_top1) or (pred_member_top2 is not None and actual_family_member == pred_member_top2)

                rows.append({
                    "Date": test_date.date(),
                    "Stream": stream,
                    "Winner": winner,
                    "Core": core,
                    "Predicted": bool(predicted),
                    "Bucket": bucket,
                    "RankPos": (int(stat_row["RankPos"]) if (stat_row is not None and "RankPos" in stat_row and pd.notna(stat_row["RankPos"])) else None),
                    "BaseScoreRank": (int(stat_row["BaseScoreRank"]) if (stat_row is not None and "BaseScoreRank" in stat_row and pd.notna(stat_row["BaseScoreRank"])) else None),
                    "HitsWindow": (int(stat_row["HitsWindow"]) if (stat_row is not None and "HitsWindow" in stat_row and pd.notna(stat_row["HitsWindow"])) else None),
                    "DaysSinceLastHit": (int(stat_row["DaysSinceLastHit"]) if (stat_row is not None and "DaysSinceLastHit" in stat_row and pd.notna(stat_row["DaysSinceLastHit"])) else None),
                    "AsOfMaxDate": pd.to_datetime(train_df["Date"], errors="coerce").max().date() if "Date" in train_df.columns else None,
                    "ActualMemberLabel": actual_member_label,
                    "ActualFamilyMember": actual_family_member,
                    "PredMemberTop1": pred_member_top1,
                    "PredMemberTop2": pred_member_top2,
                    "MemberHitTop1": member_hit_top1,
                    "MemberHitTop2": member_hit_top2,
                    "MemberTrainN": pred_member_n,
                    "TrainCnt_AABC": train_cnt_aabc,
                    "TrainCnt_ABBC": train_cnt_abbc,
                    "TrainCnt_ABCC": train_cnt_abcc,
                })

    if not rows:
        st.warning("No matching core-family wins found in the evaluated dates.")
        return

    out = pd.DataFrame(rows).sort_values(["Date", "Core", "Predicted"], ascending=[True, True, False])

    # Summary
    total = len(out)
    hits = int(out["Predicted"].sum())
    st.success(f"Evaluated {total} core-family wins; predicted {hits} ({(hits/total*100):.1f}%).")


    # Optional: member-pick accuracy (Top1/Top2) for family members (AABC/ABBC/ABCC)
    if member_track and (not out.empty):
        st.markdown("#### Member pick accuracy (Top1/Top2)")
        st.caption("These stats answer: when a core hit, was the *predicted* family member the *actual* family member? (Top1 = exact pick; Top2 = in top-2 picks).")
        need_cols = ["ActualFamilyMember","PredMemberTop1","PredMemberTop2","MemberHitTop1","MemberHitTop2","MemberTrainN"]
        missing = [c for c in need_cols if c not in out.columns]
        if missing:
            st.warning(f"Member columns missing from output: {missing}. (This should not happen; please re-run.)")
        else:
            member_df = out.dropna(subset=["ActualFamilyMember"]).copy()
            if member_df.empty:
                st.info("No family-member hits in this test window (nothing to score for member accuracy).")
            else:
                agg = member_df.groupby("Core", dropna=False).agg(
                    N=("Core","size"),
                    Top1Hit=("MemberHitTop1","sum"),
                    Top2Hit=("MemberHitTop2","sum"),
                    AvgTrainN=("MemberTrainN","mean"),
                    MedTrainN=("MemberTrainN","median"),
                ).reset_index()
                agg["Top1Rate"] = (agg["Top1Hit"] / agg["N"]).round(4)
                agg["Top2Rate"] = (agg["Top2Hit"] / agg["N"]).round(4)
                st.dataframe(agg.sort_values(["Top2Rate","Top1Rate","N"], ascending=False), use_container_width=True)
                st.caption("Tip: if Top2Rate is strong but Top1Rate is weak, treat this as a *top-2 member shortlist* (play 2 members, not 1).")

    # Trust check: walk-forward (no leakage)
    leak_ok = True
    if (not out.empty) and ("AsOfMaxDate" in out.columns):
        try:
            _max_train = pd.to_datetime(out["AsOfMaxDate"], errors="coerce")
            _test = pd.to_datetime(out["Date"], errors="coerce")
            leak_ok = bool((_max_train <= (_test - pd.Timedelta(days=1))).fillna(True).all())
        except Exception:
            leak_ok = True
    st.caption("Leakage check: " + ("✅ OK" if leak_ok else "❌ FAILED") + " — AsOfMaxDate should be <= test_date-1 for all rows.")

    # -------------------------
    # Strategy Finder (rows/lines)
    # -------------------------
    st.markdown("#### Strategy Finder (minimize plays)")
    st.caption(
        "Goal: find the *specific row lines* where winners concentrate most, so you can play fewer rows per core while keeping as many winners as possible."
    )

    if only_hit_days:
        st.info(
            "You have **Evaluate only days where a selected core member hit** enabled. "
            "Strategy metrics below are computed on those *hit-days only* (faster, but it can inflate day-hit rates). "
            "For true daily rates across the whole date range, re-run with that box unchecked."
        )

    # Controls
    sf1, sf2, sf3 = st.columns([1.2, 1.2, 1.2])
    with sf1:
        rank_choice = st.selectbox(
            "Which chart rows?",
            ["RankPos (overall stream position)", "BaseScoreRank (base score chart position)"],
            index=0,
            key="sf_rank_choice",
            help="RankPos is the overall stream position from the per-core stream ranking. BaseScoreRank is the rank on the BaseScore chart.",
        )
    rank_col = "RankPos" if rank_choice.startswith("RankPos") else "BaseScoreRank"

    with sf2:
        cost_per_play = st.number_input(
            "Cost per play ($)",
            min_value=0.0,
            max_value=10.0,
            value=0.25,
            step=0.05,
            key="sf_cost_per_play",
        )
    with sf3:
        member_mode = st.selectbox(
            "Member play mode (affects plays + scoring)",
            [
                "Play all 3 family members (AABC+ABBC+ABCC)",
                "Play Top2 member picks (requires tracking)",
                "Play Top1 member pick (requires tracking)",
            ],
            index=0,
            key="sf_member_mode",
            help="All-3 counts a win whenever the core hit in that stream and you played the stream. Top2/Top1 count wins only if the predicted member(s) match the actual member.",
        )

    # Determine member multiplier + scoring filter
    member_mult = 3
    member_filter_col = None
    if member_mode.startswith("Play Top2"):
        member_mult = 2
        member_filter_col = "MemberHitTop2"
    elif member_mode.startswith("Play Top1"):
        member_mult = 1
        member_filter_col = "MemberHitTop1"

    # If member mode selected but tracking not available, fall back safely
    if member_filter_col is not None:
        if (not member_track) or (member_filter_col not in out.columns):
            st.warning("Top1/Top2 scoring requires **Track member accuracy**. Falling back to **Play all 3** for Strategy Finder.")
            member_mult = 3
            member_filter_col = None

    # Prepare rank dataframe
    df_rank = out.copy()
    if rank_col not in df_rank.columns:
        st.warning(f"Strategy Finder needs column '{rank_col}', but it was not found in backtest output.")
        df_rank = pd.DataFrame()
    else:
        df_rank[rank_col] = pd.to_numeric(df_rank[rank_col], errors="coerce").astype("Int64")
        df_rank = df_rank.dropna(subset=[rank_col])
        df_rank[rank_col] = df_rank[rank_col].astype(int)

    if df_rank.empty:
        st.info("No ranked rows available to analyze for Strategy Finder.")
    else:
        # Apply member scoring filter if requested
        if member_filter_col is not None and member_filter_col in df_rank.columns:
            df_rank[member_filter_col] = df_rank[member_filter_col].fillna(False).astype(bool)
            df_rank_scored = df_rank[df_rank[member_filter_col]].copy()
        else:
            df_rank_scored = df_rank

        total_wins = len(df_rank_scored)
        total_days = int(df_rank_scored["Date"].nunique())
        cores_in_test = sorted(df_rank_scored["Core"].astype(str).unique().tolist())
        ncores = len(cores_in_test)

        if total_wins == 0:
            st.info("No wins are scorable under the selected member mode in this window.")
        else:
            # Row hotness table
            rc = df_rank_scored.groupby(rank_col).agg(
                Wins=("Core", "size"),
                DaysWithWin=("Date", "nunique"),
            ).reset_index().rename(columns={rank_col: "Row"})
            rc["WinPct"] = (rc["Wins"] / total_wins * 100).round(2)
            rc["DayHitPct"] = (rc["DaysWithWin"] / max(1, total_days) * 100).round(2)

            rc = rc.sort_values(["Wins", "DaysWithWin", "Row"], ascending=[False, False, True])

            hottest_row = int(rc.iloc[0]["Row"])
            hottest_wins = int(rc.iloc[0]["Wins"])
            hottest_dayhit = int(rc.iloc[0]["DaysWithWin"])
            # Plays/day assumes you play this row for every tested core every day
            plays_per_day_row1 = ncores * 1 * member_mult
            cost_per_day_row1 = plays_per_day_row1 * float(cost_per_play)
            st.markdown(
                f"**Hottest single row:** Row **{hottest_row}** on **{rank_col}** "
                f"captured **{hottest_wins}/{total_wins} wins** ({(hottest_wins/total_wins*100):.1f}%), "
                f"and hit on **{hottest_dayhit}/{total_days} days** ({(hottest_dayhit/max(1,total_days)*100):.1f}%). "
                f"Playing only that row across **{ncores} cores** costs ~**{plays_per_day_row1} plays/day** (≈ ${cost_per_day_row1:,.2f}/day at ${cost_per_play:.2f})."
            )

            with st.expander("Row hotness table (all rows)", expanded=False):
                st.dataframe(rc, use_container_width=True, hide_index=True)

            # Evaluate top-K row strategies (specific line sets, not ranges)
            max_row = int(rc["Row"].max())
            max_k_default = min(9, max(1, min(15, max_row)))
            k_max = st.slider(
                "Evaluate Top‑K hottest rows (specific row lines)",
                min_value=1,
                max_value=min(15, max_row),
                value=max_k_default,
                step=1,
                key="sf_kmax",
                help="Top‑K is built from the K hottest rows by win count (not a contiguous range).",
            )

            top_rows = rc["Row"].astype(int).tolist()

            strat_rows = []
            for k in range(1, int(k_max) + 1):
                rows_k = top_rows[:k]
                sub = df_rank_scored[df_rank_scored["Core"].astype(str).isin(cores_in_test) & df_rank_scored[rank_col].isin(rows_k)]
                cap_wins = int(len(sub))
                cap_days = int(sub["Date"].nunique())
                cap_pct = (cap_wins / total_wins * 100.0) if total_wins else 0.0
                day_pct = (cap_days / max(1, total_days) * 100.0)

                plays_per_day = ncores * k * member_mult
                cost_per_day = plays_per_day * float(cost_per_play)
                # Over the tested days, how much spend per captured win?
                spend_total = cost_per_day * total_days
                cost_per_win = (spend_total / cap_wins) if cap_wins > 0 else None
                strat_rows.append({
                    "K (rows)": k,
                    "Rows (specific lines)": ",".join(str(r) for r in rows_k),
                    "CapturedWins": cap_wins,
                    "CapturePct": round(cap_pct, 2),
                    "DaysWith≥1Win": cap_days,
                    "DayHitPct": round(day_pct, 2),
                    "Plays/Day": int(plays_per_day),
                    "Cost/Day($)": round(cost_per_day, 2),
                    "Cost/CapturedWin($)": (round(cost_per_win, 2) if cost_per_win is not None else None),
                })

            strat_df = pd.DataFrame(strat_rows)
            st.markdown("##### Top‑K row strategies (play these specific lines for every tested core)")
            st.dataframe(strat_df, use_container_width=True, hide_index=True)

            # Manual selection (exact rows)
            st.markdown("##### Try a custom set of row lines")
            default_manual = top_rows[:min(3, len(top_rows))]
            manual_rows = st.multiselect(
                "Select specific rows to play (exact lines, not ranges)",
                options=sorted(top_rows),
                default=default_manual,
                key="sf_manual_rows",
            )
            if manual_rows:
                subm = df_rank_scored[df_rank_scored[rank_col].isin([int(x) for x in manual_rows])].copy()
                cap_wins_m = int(len(subm))
                cap_days_m = int(subm["Date"].nunique())
                plays_per_day_m = ncores * len(manual_rows) * member_mult
                cost_per_day_m = plays_per_day_m * float(cost_per_play)
                st.write(
                    f"Custom rows captured **{cap_wins_m}/{total_wins} wins** ({(cap_wins_m/total_wins*100):.1f}%) "
                    f"across **{cap_days_m}/{total_days} days** ({(cap_days_m/max(1,total_days)*100):.1f}%). "
                    f"Plays/day = **{plays_per_day_m}** (≈ ${cost_per_day_m:,.2f}/day)."
                )

                # Per-core breakdown for the chosen rows
                pc = subm.groupby("Core").size().reset_index(name="CapturedWins")
                total_by_core = df_rank_scored.groupby("Core").size().reset_index(name="TotalWins")
                pc = pc.merge(total_by_core, on="Core", how="right").fillna({"CapturedWins": 0})
                pc["CapturePct"] = (pc["CapturedWins"] / pc["TotalWins"] * 100).round(1)
                pc = pc.sort_values(["CapturePct", "TotalWins"], ascending=[False, False])
                with st.expander("Per-core capture for these rows", expanded=False):
                    st.dataframe(pc, use_container_width=True, hide_index=True)
            else:
                st.info("Select at least one row to see custom strategy metrics.")

            # Core-by-core Top2 member recommendation quick table (for the current backtest window)
            if member_track and ("ActualFamilyMember" in out.columns) and (not out.dropna(subset=["ActualFamilyMember"]).empty):
                st.markdown("##### Core-by-core Top2 member recommendation (from training window)")
                st.caption("This summarizes which member label (AABC/ABBC/ABCC) actually hit most often in this backtest window.")
                md = out.dropna(subset=["ActualFamilyMember"]).copy()
                dist = md.pivot_table(index="Core", columns="ActualFamilyMember", values="Date", aggfunc="size", fill_value=0)
                for col in ["AABC","ABBC","ABCC"]:
                    if col not in dist.columns:
                        dist[col] = 0
                dist = dist[["AABC","ABBC","ABCC"]]
                dist["Total"] = dist.sum(axis=1)
                # Top2 members
                def _top2(row):
                    pairs = [(k, int(row[k])) for k in ["AABC","ABBC","ABCC"]]
                    pairs.sort(key=lambda x: (-x[1], x[0]))
                    return pairs[0][0], pairs[1][0]
                top2 = dist.apply(_top2, axis=1, result_type="expand")
                dist["Top1"] = top2[0]
                dist["Top2"] = top2[1]
                dist["Top1Pct"] = dist.apply(lambda r: round((int(r[r["Top1"]]) / (int(r["Total"]) or 1)) * 100, 1), axis=1)
                dist = dist.reset_index().sort_values(["Top1Pct","Total"], ascending=[False, False])
                st.dataframe(dist[["Core","AABC","ABBC","ABCC","Total","Top1","Top2","Top1Pct"]], use_container_width=True, hide_index=True)



# -------------------------
# Member strategy comparisons (walk-forward, no cheat)
# -------------------------
if member_track and (not out.empty):
    st.markdown("#### Member Strategy Finder (MODE vs LAST vs overrides)")
    st.caption(
        "These comparisons are **walk-forward safe**: for each test_date, member predictions are generated using only rows with Date < test_date. "
        "This helps decide whether you should play 1 member, 2 members, or all 3 for a given core."
    )

    # Compute only on rows where the winner was one of the family members (AABC/ABBC/ABCC)
    mc = out.dropna(subset=["ActualFamilyMember"]).copy()
    mc = mc[mc["ActualFamilyMember"].astype(str).isin(["AABC","ABBC","ABCC"])].copy()

    if mc.empty:
        st.info("No family-member rows in this backtest window to compare member strategies.")
    else:
        # Build predictions per row under multiple strategies
        preds = []
        for i, r in mc.iterrows():
            try:
                core = str(r.get("Core","")).zfill(3)
                stream = str(r.get("Stream",""))
                td = pd.to_datetime(r.get("Date"))
                variants = _member_prediction_variants(
                    df_all=df_all,
                    traits_pos_df=seed_traits_pos_df,
                    core_key=core,
                    test_date=td,
                    window_days=int(cfg.window_days),
                    stream=stream,
                    basis=str(member_basis),
                )
                preds.append(variants)
            except Exception:
                preds.append({"MODE": None, "LAST_GLOBAL": None, "LAST_HIER": None, "SEED_OVERRIDE": None, "TRAIT_OVERRIDE": None})

        var_df = pd.DataFrame(preds)
        for c in ["MODE","LAST_GLOBAL","LAST_HIER","SEED_OVERRIDE","TRAIT_OVERRIDE"]:
            mc[f"PredMember_{c}"] = var_df[c].values
            mc[f"Hit_{c}"] = (mc["ActualFamilyMember"].astype(str) == mc[f"PredMember_{c}"].astype(str))

        # Summary by core
        sum_rows = []
        for core, g in mc.groupby("Core"):
            n = int(len(g))
            row = {"Core": core, "N": n}
            for c in ["MODE","LAST_GLOBAL","LAST_HIER","SEED_OVERRIDE","TRAIT_OVERRIDE"]:
                row[f"Top1_{c}"] = int(g[f"Hit_{c}"].sum())
                row[f"Top1Rate_{c}"] = round(float(g[f"Hit_{c}"].mean()), 4) if n else 0.0
            sum_rows.append(row)
        sum_df = pd.DataFrame(sum_rows).sort_values(["N"], ascending=False)

        st.markdown("##### Top1 member accuracy by core (compare strategies)")
        st.dataframe(sum_df, use_container_width=True, hide_index=True)

        # Overall summary
        overall = {"Metric": ["Rows (family-member only)"]}
        overall["Value"] = [len(mc)]
        overall_df = pd.DataFrame(overall)
        st.dataframe(overall_df, use_container_width=True, hide_index=True)

        # Recommended Top2 members per core (latest as-of end_dt)
        st.markdown("##### Core-by-core Top2 member recommendations (for play reduction)")
        st.caption(
            "Top2 is built from the **training window right before the most recent test_date** in this run. "
            "Use this when Top1 is weak but Top2 is strong (play 2 members instead of all 3)."
        )

        try:
            last_test = pd.to_datetime(out["Date"], errors="coerce").max()
        except Exception:
            last_test = pd.to_datetime(end_dt)

        recs = []
        for core in sorted(set(mc["Core"].astype(str).tolist())):
            # MODE distribution from the existing predictor (returns top1/top2 by counts)
            mp = predict_core_member(df_all, core, last_test, int(cfg.window_days), basis=("core_stream" if str(member_basis)=="core_stream" else "core"), stream=None, include_rare=False)
            t1, t2 = mp.get("top1"), mp.get("top2")
            ntrain = int(mp.get("n") or 0)
            recs.append({"Core": core, "Top1(MODE)": t1, "Top2(MODE)": t2, "TrainN": ntrain})
        rec_df = pd.DataFrame(recs).sort_values(["TrainN"], ascending=False)
        st.dataframe(rec_df, use_container_width=True, hide_index=True)

        with st.expander("Download member strategy comparison rows (copy/paste ready)", expanded=False):
            dl_cols = ["Date","Stream","Core","Winner","ActualFamilyMember"] +                           [f"PredMember_{c}" for c in ["MODE","LAST_GLOBAL","LAST_HIER","SEED_OVERRIDE","TRAIT_OVERRIDE"]] +                           [f"Hit_{c}" for c in ["MODE","LAST_GLOBAL","LAST_HIER","SEED_OVERRIDE","TRAIT_OVERRIDE"]]
            dl_cols = [c for c in dl_cols if c in mc.columns]
            st.dataframe(mc[dl_cols].sort_values(["Date","Stream","Core"]), use_container_width=True)
            st.download_button(
                "Download member strategy rows CSV",
                data=mc[dl_cols].to_csv(index=False).encode("utf-8"),
                file_name="member_strategy_comparisons.csv",
                mime="text/csv",
            )

    st.markdown("#### Hit/Miss detail (copy/paste ready)")
    show_cols = [
        "Date", "Stream", "Core", "Winner", "Predicted", "Bucket", "RankPos", "BaseScoreRank", "HitsWindow", "DaysSinceLastHit",
        "ActualFamilyMember", "PredMemberTop1", "PredMemberTop2", "MemberHitTop1", "MemberHitTop2", "MemberTrainN", "AsOfMaxDate",
    ]
    safe_cols = [c for c in show_cols if c in out.columns]
    st.dataframe(out[safe_cols].sort_values(["Date","Stream","Core"], ascending=True), use_container_width=True)
    st.download_button(
        "Download walk-forward rows CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="walkforward_rows.csv",
        mime="text/csv",
    )

    st.markdown("#### By core")
    by_core = out.groupby("Core").agg(Total=("Predicted","size"), Predicted=("Predicted","sum")).reset_index()
    by_core["PredictedPct"] = (by_core["Predicted"] / by_core["Total"] * 100).round(1)
    st.dataframe(by_core.sort_values(["PredictedPct","Total"], ascending=[False, False]), use_container_width=True, hide_index=True)

    st.markdown("#### By stream")
    by_stream = out.groupby("Stream").agg(Total=("Predicted","size"), Predicted=("Predicted","sum")).reset_index()
    by_stream["PredictedPct"] = (by_stream["Predicted"] / by_stream["Total"] * 100).round(1)
    st.dataframe(by_stream.sort_values(["Predicted","Total"], ascending=[False, False]).head(80), use_container_width=True, hide_index=True)

    # Manual day replay
    st.markdown("#### Manual day replay (mock what you'd do daily)")
    unique_days = sorted(out["Date"].unique())
    sel_day = st.selectbox("Pick a day to inspect", options=unique_days, index=max(0, len(unique_days)-1), key="bt_replay_day")
    sel_day_ts = pd.Timestamp(sel_day)

    # Show that day's core hits and where they sat on the chart
    day_rows = out[out["Date"] == sel_day].copy()
    st.write(f"Core-family wins on {sel_day}: {len(day_rows)}")
    st.dataframe(day_rows.sort_values(["Core","Predicted"], ascending=[True, False]), use_container_width=True, hide_index=True)

    # For each core that hit: show predicted stream buckets for that day (from training)
    for core in sorted(day_rows["Core"].unique()):
        st.markdown(f"**Core {core}: predicted streams as of {sel_day}**")

        if member_track:
            try:
                mp_overall = predict_core_member(
                    df_all,
                    core,
                    pd.to_datetime(sel_day_ts).normalize(),
                    window_days=int(cfg.window_days),
                    basis="core",
                    include_rare=False,
                )
            except Exception:
                mp_overall = {}
            if mp_overall:
                st.caption(
                    f"Member pick (overall): Top1={mp_overall.get('top1')}, Top2={mp_overall.get('top2')} (train hits={mp_overall.get('n')})"
                )
        train_df = df_all[df_all["Date"] < sel_day_ts]
        stats_df = compute_stream_stats(train_df, core, window_days=int(cfg.window_days))
        buckets = bucket_recommendations(stats_df, cfg)
        base_df = buckets.get("Top12BaseScore", pd.DataFrame()).copy()
        due_df = buckets.get("Due8", pd.DataFrame()).copy()
        if not base_df.empty:
            base_df["Bucket"] = "BaseScore"
        if not due_df.empty:
            due_df["Bucket"] = "Due8"
        pred_df = pd.concat([base_df, due_df], ignore_index=True)
        if pred_df.empty:
            st.info("No bucket recommendations for this core/day.")
            continue
        # mark if this stream was an actual win that day
        win_streams = set(day_rows[day_rows["Core"] == core]["Stream"].astype(str).tolist())
        pred_df["WonThatDay"] = pred_df["Stream"].astype(str).isin(win_streams)
        cols = [c for c in ["Bucket","Stream","RankPos","BaseScoreRank","HitsWindow","DaysSinceLastHit","WonThatDay"] if c in pred_df.columns]
        st.dataframe(pred_df[cols].sort_values(["WonThatDay","Bucket"], ascending=[False, True]), use_container_width=True, hide_index=True)


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


def compute_stream_stats(df_all: pd.DataFrame, core: str, window_days: int | None = None, exclude_md: bool = False) -> pd.DataFrame:
    """Back-compat wrapper used by the Northern Lights block."""
    if window_days is None:
        window_days = 180
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
    top_n: int = 12,
    due_ranks: Tuple[int, int] = (13, 60),
    seed_core_key: str = "core",
    include_24h: bool = True,
    df_24: Optional[pd.DataFrame] = None,
    core: str = "000",
    **kwargs,
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
    pos_df, _ = position_percentile_map(stream_stats_df)
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

tab_labels = ["Northern Star (v51)", "Northern Lights (Master playlist)", "Core view", "Backtest (optional)"]
tabs = st.tabs(tab_labels)
_t_ns = tabs[0]
_t_nl = tabs[1]
_t_core = tabs[2]
_t_bt = tabs[3]

# --- Northern Lights master playlist (best -> worst across streams/cores) ---
if _t_nl is None:
    _t_nl = st.container()
with _t_nl:
    st.subheader("Northern Lights master playlist")
    st.caption("Aggregated bucket picks across your selected cores. Use this as your universal stream playlist.")
    # Optional performance toggle: build Northern Lights across ALL tracked cores (ignores selection)
    nl_use_all_cores = st.checkbox(
        "Use ALL tracked cores for Northern Lights (can be slower)",
        value=bool(st.session_state.get("_nl_use_all_cores", False)),
        key="_nl_use_all_cores",
        help="If enabled, the playlist ranks every stream using every tracked core. This can be slower unless baseline cache exists.",
    )


    # Ensure cores_for_cache is always defined (selected cores for cache building / views)
    cores_for_cache = list(st.session_state.get('cores_for_cache') or st.session_state.get('selected_cores') or [])
    if not cores_for_cache:
        cores_for_cache = [core_for_view] if 'core_for_view' in locals() else (cores[:1] if 'cores' in locals() and cores else ['000'])
    cores_for_cache = [str(c).zfill(3) for c in cores_for_cache]
    # If requested, ignore selection and use the full tracked core list.
    if nl_use_all_cores:
        cores_for_cache = [str(c).zfill(3) for c in CORE_PRESETS]
        st.info(f"Northern Lights is using ALL tracked cores ({len(cores_for_cache)}). If this feels slow, build baseline cache first in Cache Builder.")

    if not cores_for_cache:
        st.info("Select one or more cores above to populate the playlist.")
    else:
        cfg = st.session_state.get("_cfg", RankConfig())
        include_24h = bool(st.session_state.get("include_24h", True))

        # Build a master list: (core, stream) -> universal score
        # v51: In ALL-CORES mode, enforce strict cache-only for safety/performance.
        stats_by_core: Dict[str, pd.DataFrame] = {}
        if nl_use_all_cores:
            expected_last = last_all if isinstance(last_all, str) else None
            missing = []
            for _c in cores_for_cache:
                ss, _pos_df, _meta = _load_baseline_from_disk(_c, cfg.window_days, expected_last_date=expected_last)
                if ss is None or ss.empty:
                    missing.append(_c)
                else:
                    stats_by_core[_c] = ss
            if missing:
                st.error("ALL-CORES mode is cache-only. Missing baseline caches for: " + ", ".join(missing))
                st.caption("Build caches in the Cache Builder section, then rerun.")
                st.stop()

        rows = []
        for core in cores_for_cache:
            try:
                stats_df = stats_by_core.get(core) if nl_use_all_cores else compute_stream_stats(df_all, core, window_days=window_days, exclude_md=False)
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
                    # Per-stream RankPos percentile map (cache-backed)
                    last_s = None
                    try:
                        last_s = most_recent_date_for_stream(df_all, stream)
                    except Exception:
                        last_s = None

                    pos_df = get_pos_map_cached(str(core), int(window_days), stats_df, last_date=last_s)

                    # Position strength (by RankPos)
                    try:
                        rankpos = int(meta.get("RankPos", 9999) or 9999)
                    except Exception:
                        rankpos = 9999
                    p = 0.0
                    if pos_df is not None and not pos_df.empty and "RankPos" in pos_df.columns:
                        try:
                            _strength_col = "PctStrength" if "PctStrength" in pos_df.columns else ("HitCountPctile" if "HitCountPctile" in pos_df.columns else None)
                            if _strength_col:
                                _m = pos_df[pos_df["RankPos"].astype(int) == int(rankpos)]
                                if not _m.empty:
                                    p = float(_m.iloc[0][_strength_col])
                        except Exception:
                            p = 0.0

                    # Base signals from bucket meta
                    hits_pw = float(meta.get("HitsPerWeek", 0.0) or 0.0)
                    days_since = float(meta.get("DaysSinceLastHit", 0.0) or 0.0)
                    due_bucket_pressure = float(meta.get("DuePressure", 0.0) or 0.0)

                    # Seed Traits score (soft)
                    seed = _prev_seed_by_stream.get(str(stream))
                    seed_score = 0.0
                    if st.session_state.get("enable_seed_traits", True) and seed_traits_pos_lookup:
                        seed_score, _seed_matches = compute_seed_traits_score(
                            str(core),
                            seed,
                            str(stream),
                            pos_lookup=seed_traits_pos_lookup,
                            neg_lookup=seed_traits_neg_lookup,
                            last5_union_digits_by_stream=_last5_union_by_stream,
                        )

                    # Cadence score (soft) — mean gap baseline from window hits
                    try:
                        _total_hits = float(stats_df["HitsWindow"].sum()) if "HitsWindow" in stats_df.columns else 0.0
                    except Exception:
                        _total_hits = 0.0
                    mean_gap_days = (window_days / _total_hits) if _total_hits > 0 else 0.0
                    cadence_score = (
                        compute_cadence_score(days_since, mean_gap_days)
                        if (st.session_state.get("enable_cadence", True) and mean_gap_days > 0)
                        else 0.0
                    )

                    # Universal score (soft additive; no eliminations)
                    due_w = float(st.session_state.get("due_weight", 0.20))
                    pos_w = float(st.session_state.get("pos_weight", 0.25))
                    st_w = float(st.session_state.get("seed_traits_weight", 0.35))
                    cad_w = float(st.session_state.get("cadence_weight", 0.25))

                    universal = (
                        hits_pw
                        + (min(days_since, 50.0) * 0.01 * due_w)
                        + (p * 0.01 * pos_w)
                        + (seed_score * st_w if st.session_state.get("enable_seed_traits", True) else 0.0)
                        + (cadence_score * cad_w if st.session_state.get("enable_cadence", True) else 0.0)
                    )

                    rows.append({
                        "Core": str(core),
                        "Stream": str(stream),
                        "BucketPick": str(meta.get("BucketPick", "")),
                        "UniversalScore": float(universal),
                        "HitsPerWeek": float(hits_pw),
                        "DaysSinceLastHit": float(days_since),
                        "DueBucketPressure": float(due_bucket_pressure),
                        "DuePressure": float(due_bucket_pressure),
                        "PctStrength": float(p),
                        "Seed": seed,
                        "SeedTraitsScore": float(seed_score),
                        "CadenceScore": float(cadence_score),
                        "TriggerBoost": float(meta.get("TriggerBoost", 0.0) or 0.0),
                        "Hits": float(meta.get("Hits", 0.0) or 0.0),
                        "RankPos": int(rankpos) if isinstance(rankpos, int) else int(meta.get("RankPos", 9999) or 9999),
                        "BaseScore": float(meta.get("BaseScore", 0.0) or 0.0),
                        "DueIndex": float(meta.get("DueIndex", 0.0) or 0.0),
                    })
                except Exception:
                    # Fallback: still emit a row without the position/traits features
                    try:
                        hits_pw = float(meta.get("HitsPerWeek", 0.0) or 0.0)
                    except Exception:
                        hits_pw = 0.0
                    try:
                        days_since = float(meta.get("DaysSinceLastHit", 0.0) or 0.0)
                    except Exception:
                        days_since = 0.0
                    rows.append({
                        "Core": str(core),
                        "Stream": str(stream),
                        "BucketPick": str(meta.get("BucketPick", "")),
                        "UniversalScore": float(hits_pw),
                        "HitsPerWeek": float(hits_pw),
                        "DaysSinceLastHit": float(days_since),
                        "DueBucketPressure": float(meta.get("DuePressure", meta.get("DueBucketPressure", 0.0)) or 0.0),
                        "DuePressure": float(meta.get("DuePressure", meta.get("DueBucketPressure", 0.0)) or 0.0),
                        "PctStrength": 0.0,
                        "Seed": _prev_seed_by_stream.get(str(stream)),
                        "SeedTraitsScore": 0.0,
                        "CadenceScore": 0.0,
                        "TriggerBoost": float(meta.get("TriggerBoost", 0.0) or 0.0),
                        "Hits": float(meta.get("Hits", 0.0) or 0.0),
                        "RankPos": int(meta.get("RankPos", 9999) or 9999),
                        "BaseScore": float(meta.get("BaseScore", 0.0) or 0.0),
                        "DueIndex": float(meta.get("DueIndex", 0.0) or 0.0),
                    })

        if not rows:
            st.warning("No playlist rows were produced. Double-check that your history file contains your selected cores in AABC structure.")
        else:
            nl_df = pd.DataFrame(rows)
            # Optional: Trigger Map boost (soft weighting) for the fixed 39-play list
            apply_trigger_map = st.session_state.get("_apply_trigger_map", False)
            trigger_boost_points = float(st.session_state.get("_trigger_boost_points", 2.0) or 2.0)
            if apply_trigger_map and df_24 is not None and not df_24.empty and "BucketPick" in nl_df.columns:
                try:
                    df_prev = df_24.copy()
                    # Use the last row per Stream as "previous winner" for that stream
                    if "Date" in df_prev.columns:
                        df_prev["_DateSort"] = pd.to_datetime(df_prev["Date"], errors="coerce")
                        df_prev = df_prev.sort_values(["Stream", "_DateSort"])
                    else:
                        df_prev = df_prev.sort_values(["Stream"])
                    prev_map = df_prev.groupby("Stream")["Result"].last().to_dict() if "Result" in df_prev.columns else {}
                    nl_df["PrevResult"] = nl_df["Stream"].map(prev_map).fillna("")
                    nl_df["TriggerBoost"] = nl_df.apply(
                        lambda r: trigger_map_boost(str(r.get("BucketPick","")), str(r.get("PrevResult","")), boost_points=trigger_boost_points),
                        axis=1,
                    )
                    nl_df["UniversalScore"] = nl_df["UniversalScore"].astype(float) + nl_df["TriggerBoost"].astype(float)
                except Exception:
                    # Never break the playlist if trigger map cannot apply
                    pass

            # Ensure DuePressure exists (legacy compatibility)
            if "DuePressure" not in nl_df.columns:
                if "DueBucketPressure" in nl_df.columns:
                    nl_df["DuePressure"] = nl_df["DueBucketPressure"].astype(float)
                else:
                    nl_df["DuePressure"] = 0.0

            nl_df = nl_df.sort_values(["UniversalScore", "HitsPerWeek", "DuePressure"], ascending=[False, False, False]).reset_index(drop=True)
            nl_df.insert(0, "Rank", nl_df.index + 1)
            st.session_state["nl_df_current"] = nl_df.copy()

            st.dataframe(nl_df, width="stretch", height=520)

            # Northern Star percentile map (playlist)
            # This summarizes how much "hit weight" concentrates by rank position in the *final* per-stream playlist.
            with st.expander("Northern Star percentile map (playlist positions)", expanded=False):
                try:
                    # Ensure DuePressure exists (legacy compatibility)
                    if "DuePressure" not in nl_df.columns:
                        if "DueBucketPressure" in nl_df.columns:
                            nl_df["DuePressure"] = nl_df["DueBucketPressure"].astype(float)
                        else:
                            nl_df["DuePressure"] = 0.0
                    # Keep only the best row per Stream (highest UniversalScore) -> one entry per stream
                    _best = nl_df.sort_values(
                        ["UniversalScore", "HitsPerWeek", "DuePressure"],
                        ascending=[False, False, False]
                    ).groupby("Stream", as_index=False).head(1).reset_index(drop=True)

                    # Assign playlist rank positions 1..N (typically 78 streams)
                    _best.insert(0, "RankPos", _best.index + 1)
                    _best["HitsWindow"] = _best.get("Hits", 0).astype(int)

                    _pos, _ = position_percentile_map(_best[["RankPos", "HitsWindow"]].copy())
                    st.caption("RankPos = position in the final per-stream playlist. HitCount = historical hits (in the selected window) of the #1 pick for that stream.")
                    st.dataframe(_pos, width="stretch", height=320)
                except Exception as _e:
                    st.warning(f"Could not build the playlist percentile map: {_e}")

            # Northern Star buckets (per core)
            cfg = st.session_state.get("_cfg", RankConfig())
            with st.expander("Northern Star buckets (per core)", expanded=True):
                if cores_for_cache:
                    _b_tabs = st.tabs([f"Core {c}" for c in cores_for_cache]) if len(cores_for_cache) > 1 else [st.container()]
                    for _tab, _c in zip(_b_tabs, cores_for_cache):
                        with _tab:
                            _core_str = str(_c).zfill(3)
                            _stats_df = compute_stream_stats(df_all, _core_str, window_days, exclude_md)
                            if _stats_df is None or _stats_df.empty:
                                st.info(f"No AABC stream stats for core {_core_str}.")
                                continue
                            _b = bucket_recommendations(_stats_df, cfg)
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.caption("Top 12 (BaseScore)")
                                st.write(_b.get("Top12", []))
                            with c2:
                                st.caption(f"Due {getattr(cfg, 'top_due', 8)} (DueIndex)")
                                st.write(_b.get("Due8", []))
                            with c3:
                                st.caption("Combined (Top+Due)")
                                st.write(_b.get("Combined", []))
                else:
                    st.info("Select one or more cores above to view buckets.")

            
            # Percentile map(s) for selected core(s) (tie-breaker visibility in Northern Lights view)
            with st.expander("Core ranking percentile map (tie-breaker)"):
                if cores_for_cache:
                    _tabs = st.tabs([f"Core {c}" for c in cores_for_cache]) if len(cores_for_cache) > 1 else [st.container()]
                    for _tab, _c in zip(_tabs, cores_for_cache):
                        with _tab:
                            _core_str = str(_c).zfill(3)
                            # get_position_percentiles_cached() expects the active window + per-core stream stats.
                            # In this view we build / reuse the same stream-stats used by the core ranking.
                            _stream_stats = compute_stream_stats(df_all, _core_str, window_days, exclude_md)
                            _pm = get_position_percentiles_cached(_core_str, window_days, _stream_stats)
                            st.dataframe(_pm, width="stretch", height=240)
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

if _t_ns is None:
    _t_ns = st.container()
with _t_ns:
    st.header("Northern Star (v51)")
    st.caption("This tab restores the Northern Star scoring view and engines (Rare / Ultra-Rare) while keeping Core View unchanged. Percentile maps by position are shown here as GLOBAL (all selected cores) and PER-CORE maps.")

    # Global RankPos percentile map across selected cores (cache-only for safety)
    st.subheader("Global RankPos percentile map (all selected cores, cache-only)")
    ns_use_all_tracked = st.checkbox("Use ALL tracked cores (cache-only)", value=False, key="ns_use_all_tracked")
    ns_cores = CORE_PRESETS if ns_use_all_tracked else list(st.session_state.get("cores_for_cache_ms", []))
    if not ns_cores:
        st.info("Select cores in the multi-core section above (or enable ALL tracked cores).")
    else:
        expected_last = last_all if isinstance(last_all, str) else None
        global_map, missing = build_allcores_rankpos_pctmap(ns_cores, window_days=cfg.window_days, expected_last_date=expected_last, cache_only=True)
        if missing:
            st.error("Missing baseline caches for these cores (global map is cache-only): " + ", ".join(missing))
            st.caption("Build caches in the Cache Builder section, then rerun.")
        else:
            st.dataframe(global_map, width="stretch")

    st.divider()
    st.subheader("Per-core Northern Star scoring (SeedTraits + Cadence, soft)")
    view_core_ns = st.selectbox("Core (Northern Star view)", options=list(dict.fromkeys([view_core] + list(ns_cores))), key="view_core_ns")
    core_key_ns = canonical_core_key(view_core_ns)

    try:
        stats_ns = compute_stream_stats(df_all, core_key_ns, window_days=cfg.window_days, exclude_md=False)
    except Exception as e:
        st.error(f"Could not compute stats for core {core_key_ns}: {e}")
        stats_ns = pd.DataFrame()

    if stats_ns is not None and not stats_ns.empty:
        # Position map per-core (distinct from global map)
        pos_map_ns = get_position_percentiles_cached(core_key_ns, cfg.window_days, stats_ns)
        pos_strength_by_rank = dict(zip(pos_map_ns["RankPos"], pos_map_ns["PctStrength"]))

        # Cadence base: average gap for this core across streams (in days)
        total_hits = float(stats_ns["HitsWindow"].sum()) if "HitsWindow" in stats_ns.columns else 0.0
        mean_gap_days = (cfg.window_days / total_hits) if total_hits > 0 else 0.0

        # Seed Traits + Cadence per stream
        ns_rows = []
        for _, r in stats_ns.iterrows():
            stream = str(r.get("Stream", ""))
            rankpos = int(r.get("RankPos", 9999))
            pos_strength = float(pos_strength_by_rank.get(rankpos, 0.0))
            seed = _prev_seed_by_stream.get(stream)
            seed_score, seed_matches = compute_seed_traits_score(
                core_key_ns, seed, stream,
                pos_lookup=seed_traits_pos_lookup,
                neg_lookup=seed_traits_neg_lookup,
                last5_union_digits_by_stream=_last5_union_by_stream,
            )
            cadence = compute_cadence_score(float(r.get("DaysSinceLastHit", 0.0)), mean_gap_days) if mean_gap_days > 0 else 0.0

            # Soft combined score
            hits_pw = float(r.get("HitsPerWeek", 0.0))
            due_pressure = float(r.get("DaysSinceLastHit", 0.0))
            ns_score = (
                hits_pw
                + (min(due_pressure, 50.0) * 0.01 * float(st.session_state.get("due_weight", 0.20)))
                + (pos_strength * 0.01 * float(st.session_state.get("pos_weight", 0.25)))
                + (seed_score * float(st.session_state.get("seed_traits_weight", 0.35)) if st.session_state.get("enable_seed_traits", True) else 0.0)
                + (cadence * float(st.session_state.get("cadence_weight", 0.25)) if st.session_state.get("enable_cadence", True) else 0.0)
            )
            ns_rows.append({
                "Stream": stream,
                "RankPos": rankpos,
                "HitsPerWeek": hits_pw,
                "DaysSinceLastHit": due_pressure,
                "PosPctStrength": pos_strength,
                "SeedTraitsScore": seed_score,
                "CadenceScore": cadence,
                "NSScore": ns_score,
                "Seed": seed,
            })
        ns_df = pd.DataFrame(ns_rows).sort_values(["NSScore","HitsPerWeek"], ascending=False)

        st.dataframe(ns_df.head(50), width="stretch")
        st.caption("NSScore is a soft additive score; it does NOT remove streams. Use it to prioritize without harming coverage.")

        with st.expander("Per-core RankPos percentile map (position-based)"):
            st.dataframe(pos_map_ns, width="stretch")

        # Engines (restored UI)
        st.divider()
        st.subheader("Rare Engine (AABC-family; historical lift)")
        if st.session_state.get("r1", True) or st.session_state.get("r2", True) or st.session_state.get("r3", True) or st.session_state.get("r4", True):
            # evaluate_rare_engine signature expects:
            #   (df_all, core, df_24h, enable_r1, enable_r2, enable_r3, enable_r4, window_days_recent)
            # Keep the UI-driven switches and pass the optional 24h map (may be empty).
            try:
                try:
                    rare_df, _rare_summary = evaluate_rare_engine(
                    df_all,
                    core_key_ns,
                    df_24h,
                    enable_r1=r1,
                    enable_r2=r2,
                    enable_r3=r3,
                    enable_r4=r4,
                    window_days_recent=cfg.window_days,
                    )
                except TypeError as _te:
                    if 'window_days_recent' in str(_te) and 'unexpected keyword argument' in str(_te):
                        rare_df, _rare_summary = evaluate_rare_engine(
                        df_all,
                        core_key_ns,
                        df_24h,
                        enable_r1=r1,
                        enable_r2=r2,
                        enable_r3=r3,
                        enable_r4=r4,
                        )
                    else:
                        raise
                st.dataframe(_to_dataframe(rare_df), width="stretch")
            except Exception as e:
                st.error(f"Rare Engine error: {e}")
                st.dataframe(pd.DataFrame(), width="stretch")
        else:
            st.info("Enable at least one Rare Engine checkbox above to view results.")

        st.subheader("Ultra-Rare Engine (AABB/AAAB/etc; historical lift)")
        if st.session_state.get("q1", True) or st.session_state.get("q2", True) or st.session_state.get("q3", True) or st.session_state.get("q4", True):
            try:
                try:
                    ultra_df, _ultra_summary = evaluate_ultra_rare_engine(
                    df_all,
                    core_key_ns,
                    df_24h,
                    enable_q1=q1,
                    enable_q2=q2,
                    enable_q3=q3,
                    enable_q4=q4,
                    window_days_recent=cfg.window_days,
                    )
                except TypeError as _te:
                    if 'window_days_recent' in str(_te) and 'unexpected keyword argument' in str(_te):
                        ultra_df, _ultra_summary = evaluate_ultra_rare_engine(
                        df_all,
                        core_key_ns,
                        df_24h,
                        enable_q1=q1,
                        enable_q2=q2,
                        enable_q3=q3,
                        enable_q4=q4,
                        )
                    else:
                        raise
                st.dataframe(_to_dataframe(ultra_df), width="stretch")
            except Exception as e:
                st.error(f"Ultra-Rare Engine error: {e}")
                st.dataframe(pd.DataFrame(), width="stretch")
        else:
            st.info("Enable at least one Ultra-Rare checkbox above to view results.")

        # Seed Traits match details (debug / transparency)
        with st.expander("Seed Traits matches (debug)"):
            pick_stream = st.selectbox("Stream to inspect", options=list(ns_df["Stream"].head(25)), key="ns_inspect_stream")
            seed = _prev_seed_by_stream.get(str(pick_stream))
            score, matches = compute_seed_traits_score(
                core_key_ns, seed, str(pick_stream),
                pos_lookup=seed_traits_pos_lookup,
                neg_lookup=seed_traits_neg_lookup,
                last5_union_digits_by_stream=_last5_union_by_stream,
            )
            st.write({"core": core_key_ns, "stream": str(pick_stream), "seed": seed, "score": score})
            if matches:
                st.dataframe(pd.DataFrame(matches, columns=["trait","value","lift","sign"]), width="stretch")
            else:
                st.caption("No matching traits found (or trait files not loaded).")
    else:
        st.info("No stats available for this core. Build or load data/caches and rerun.")


if _t_core is None:
    _t_core = st.container()
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
        st.dataframe(stats_df, width="stretch", height=420)

        # Buckets (Top 12 BaseScore + Due 8 from ranks 13–60)
        # NOTE: build_northern_star_buckets() is a *per-stream* helper used by the master playlist.
        # For the per-core view we want the actual bucket lists, which are produced by bucket_recommendations().
        buckets = bucket_recommendations(stats_df, cfg)
        top_bucket = buckets.get("Top12", [])
        due_bucket = buckets.get("Due8", [])
        combined_bucket = buckets.get("Combined", [])

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
                    st.dataframe(pos_map, width="stretch", height=420)
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
if _t_bt is None:
    _t_bt = st.container()
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
        render_backtest(df_all=df_all, cfg=cfg, cores_for_cache=cores_for_cache, df_24h=df_24h)
    except NameError:
        st.warning("Backtest utility is not available in this build.")
    except Exception as e:
        st.error(f"Backtest failed: {e}")
# Northern Star (core) RankPos map is the same distribution, but we cache it separately for clarity
_core_pct_cached = _load_pctmap_from_disk(f"CORE_{view_core}", cfg.window_days, expected_last_date=last_all)
if _core_pct_cached is None:
    try:
        _save_pctmap_to_disk(f"CORE_{view_core}", cfg.window_days, pos_pct, asof_last_date=last_all)
        _core_pct_cached = _load_pctmap_from_disk(f"CORE_{view_core}", cfg.window_days, expected_last_date=last_all)
    except Exception:
        _core_pct_cached = None

if _core_pct_cached is not None and not _core_pct_cached.empty:
    st.caption("Northern Star (this core) RankPos percentiles (cached for stability).")


def _rerun():
    """Compatibility rerun helper (Streamlit versions)."""
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass