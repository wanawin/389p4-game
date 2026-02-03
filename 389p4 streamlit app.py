# pk4_northern_star_app_2026-02-02_v17.py
# Streamlit app: Pick 4 "Northern Star" core stream ranking + Rare/Ultra-Rare engines (AAAB+AABB, AAAA)
# Notes:
# - Designed to work with LotteryPost-style "all-states" consolidated history files (user-provided).
# - Accepts .csv and .txt (tab or comma) inputs.
# - Core ranking uses combined all-states file with a window switch (180/365; default 180).
# - Bucket method: Top 12 BaseScore + 8 DueIndex from ranks 13–60.
# - Box is always included; straights are optional last.
# - Previous-day file is optional, used only for downranking.
# - Output: full ranked list + hotspot percentile map + optional straights shortlist.

from __future__ import annotations

import io
import os
import re
import sys
import math
import json
import time
import zipfile
import hashlib
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Any, Set

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# App Meta / Constants
# -----------------------------

APP_TITLE = "Pick 4 Northern Star — Core Family Ranking + Rare Engines"
APP_VERSION = "v20 (2026-02-03)"
DEFAULT_WINDOW_DAYS = 180

# "Family" in this app refers to 3-member doubles families like 389 = {3389, 3889, 3899}
# Member structures:
# - AABC: 0019 style
# - ABBC: 0119 style
# - ABCC: 0199 style

# NOTE: The app does NOT assume a single stream. It ranks across all streams found in the input.
# NOTE: Any exclusions should be via user options (checkboxes), not hardcoded.


# -----------------------------
# Utilities
# -----------------------------

def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(str(x).strip())
    except Exception:
        return default


def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def _infer_delimiter(sample: str) -> str:
    # Heuristic: prefer tab if present, else comma
    if "\t" in sample:
        return "\t"
    return ","


def _read_uploaded_file(upload) -> Tuple[str, bytes]:
    # Returns (filename, content_bytes)
    if upload is None:
        return ("", b"")
    return (upload.name, upload.getvalue())


def _read_text_bytes_as_df(content_bytes: bytes) -> pd.DataFrame:
    text = content_bytes.decode("utf-8", errors="replace")
    # sniff delimiter by first ~5 lines
    sample = "\n".join(text.splitlines()[:5])
    delim = _infer_delimiter(sample)
    return pd.read_csv(io.StringIO(text), sep=delim, engine="python")


def _read_csv_bytes_as_df(content_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content_bytes))


def load_table_from_upload(upload) -> pd.DataFrame:
    name, b = _read_uploaded_file(upload)
    if not b:
        return pd.DataFrame()
    lower = name.lower()
    if lower.endswith(".csv"):
        return _read_csv_bytes_as_df(b)
    if lower.endswith(".txt"):
        return _read_text_bytes_as_df(b)
    # fallback: try csv
    try:
        return _read_csv_bytes_as_df(b)
    except Exception:
        return _read_text_bytes_as_df(b)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Normalize column names: trim and standardize
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Try to find key columns
    # Expected: Date, State, Game, Draw (optional), Result (required)
    # The user previously confirmed 'Result' is the Pick-4 number column in their dataset
    colmap = {c.lower(): c for c in df.columns}

    def pick(*cands: str) -> Optional[str]:
        for cand in cands:
            if cand.lower() in colmap:
                return colmap[cand.lower()]
        return None

    date_col = pick("date", "drawdate", "draw_date")
    result_col = pick("result", "winning", "numbers", "number")
    state_col = pick("state", "jurisdiction")
    game_col = pick("game", "gamename", "game_name")
    draw_col = pick("draw", "draw_time", "time", "drawtime")
    stream_col = pick("stream", "stream_name", "streamname")

    # Ensure required Result
    if result_col is None:
        # attempt fuzzy search
        for c in df.columns:
            if "result" in c.lower():
                result_col = c
                break
    if result_col is None:
        raise ValueError("Could not find a 'Result' column. Please ensure your file has a Result column with 4-digit numbers.")

    # rename into canonical
    ren = {result_col: "Result"}
    if date_col: ren[date_col] = "Date"
    if state_col: ren[state_col] = "State"
    if game_col: ren[game_col] = "Game"
    if draw_col: ren[draw_col] = "Draw"
    if stream_col: ren[stream_col] = "Stream"

    df = df.rename(columns=ren)

    # Parse/clean Date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date

    # Clean Result as 4-digit string (keep leading zeros)
    df["Result"] = df["Result"].astype(str).str.strip()
    df["Result"] = df["Result"].str.replace(r"\.0$", "", regex=True)
    df["Result"] = df["Result"].str.zfill(4)
    df = df[df["Result"].str.fullmatch(r"\d{4}", na=False)].copy()

    # Ensure Stream exists (fallback from State+Game+Draw)
    if "Stream" not in df.columns:
        parts = []
        for c in ["State", "Game", "Draw"]:
            if c in df.columns:
                parts.append(df[c].astype(str))
        if parts:
            df["Stream"] = parts[0]
            for p in parts[1:]:
                df["Stream"] = df["Stream"] + " | " + p
        else:
            df["Stream"] = "UNKNOWN"

    # Ensure Game exists
    if "Game" not in df.columns:
        df["Game"] = "Pick 4"

    return df


# -----------------------------
# Family logic
# -----------------------------

def family_code_from_result(result: str) -> Optional[str]:
    """
    Determine if a 4-digit result is a "doubles family member" of the form AABC, ABBC, ABCC
    where {A,B,C} are digits with exactly one digit repeated twice and two single digits.
    Family code is the sorted unique digits placed as: X Y Z where:
      - X = repeated digit
      - Y,Z = the two singles sorted ascending
    Example:
      3389 -> repeated digit 3, singles 8,9 => family "389"
      3889 -> repeated digit 8, singles 3,9 => family "389"
      3899 -> repeated digit 9, singles 3,8 => family "389"
    Returns 3-char family code or None if not exactly a doubles-family member.
    """
    if not isinstance(result, str) or not re.fullmatch(r"\d{4}", result):
        return None
    digits = list(result)
    counts = {d: digits.count(d) for d in set(digits)}
    if sorted(counts.values()) != [1, 1, 2]:
        return None
    repeated = [d for d, c in counts.items() if c == 2][0]
    singles = sorted([d for d, c in counts.items() if c == 1])
    return f"{repeated}{singles[0]}{singles[1]}"


def member_type_for_family(result: str, fam: str) -> str:
    """
    Return which member structure the result is within its family:
    - AABC (e.g., 0019)
    - ABBC (e.g., 0119)
    - ABCC (e.g., 0199)
    """
    # structure is about where the repeated digit is placed in the 4-digit string:
    # But for your cadence summary, "AABC/ABBC/ABCC" refers to positional template of the repeated digit.
    # We'll compute based on counts in order.
    digits = list(result)
    # Identify repeated digit
    rep = max(set(digits), key=lambda d: digits.count(d))
    # Find positions of rep
    pos = [i for i, d in enumerate(digits) if d == rep]
    # AABC => positions [0,1]
    # ABBC => positions [1,2]
    # ABCC => positions [2,3]
    if pos == [0, 1]:
        return "AABC"
    if pos == [1, 2]:
        return "ABBC"
    if pos == [2, 3]:
        return "ABCC"
    return "OTHER"


def is_ultra_rare(result: str) -> bool:
    # AAAA (all 4 digits same)
    return bool(re.fullmatch(r"(\d)\1\1\1", result))


def is_rare(result: str) -> bool:
    # AAAB or AABB (3-of-a-kind or two pairs)
    if not re.fullmatch(r"\d{4}", result):
        return False
    digits = list(result)
    counts = sorted([digits.count(d) for d in set(digits)])
    return counts in ([1, 3], [2, 2])


# -----------------------------
# Ranking model
# -----------------------------

@dataclass
class FamilyStats:
    family: str
    total_hits: int
    hit_days: int
    hits_per_week: float
    hit_days_per_week: float
    last_hit: Optional[dt.date]
    days_since_last: Optional[int]
    member_counts: Dict[str, int]
    streams_hit: int


def build_family_stats(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """
    Compute per-family stats across all streams in the window.
    """
    if df.empty:
        return pd.DataFrame()

    # Apply window
    if "Date" in df.columns and df["Date"].notna().any():
        max_date = max([d for d in df["Date"].dropna().tolist()])
        cutoff = max_date - dt.timedelta(days=window_days - 1)
        dfw = df[df["Date"].between(cutoff, max_date, inclusive="both")].copy()
        days_span = (max_date - cutoff).days + 1
        weeks_span = days_span / 7.0
    else:
        dfw = df.copy()
        weeks_span = max(len(dfw) / 7.0, 1.0)

    # Identify doubles-family members
    dfw["Family"] = dfw["Result"].apply(family_code_from_result)
    fam_df = dfw[dfw["Family"].notna()].copy()
    if fam_df.empty:
        return pd.DataFrame()

    # Hit day = Date if present else fallback to row index bucket (not ideal, but keeps app alive)
    if "Date" in fam_df.columns and fam_df["Date"].notna().any():
        fam_df["HitDay"] = fam_df["Date"]
    else:
        fam_df["HitDay"] = pd.to_datetime(fam_df.index, unit="D", origin="unix").dt.date

    fam_df["MemberType"] = fam_df.apply(lambda r: member_type_for_family(r["Result"], r["Family"]), axis=1)

    # Aggregate
    g = fam_df.groupby("Family", dropna=False)
    rows = []
    for fam, sub in g:
        total_hits = int(len(sub))
        hit_days = int(sub["HitDay"].nunique())
        last_hit = None
        if "Date" in sub.columns and sub["Date"].notna().any():
            last_hit = max(sub["Date"].dropna().tolist())
            days_since = (max(sub["Date"].dropna().tolist()) - last_hit).days  # always 0
            # We'll compute days_since relative to global max_date if available
            if "Date" in df.columns and df["Date"].notna().any():
                global_max = max([d for d in df["Date"].dropna().tolist()])
                days_since = (global_max - last_hit).days
        else:
            days_since = None

        member_counts = sub["MemberType"].value_counts().to_dict()
        streams_hit = int(sub["Stream"].nunique()) if "Stream" in sub.columns else 1

        rows.append(
            FamilyStats(
                family=fam,
                total_hits=total_hits,
                hit_days=hit_days,
                hits_per_week=total_hits / weeks_span,
                hit_days_per_week=hit_days / weeks_span,
                last_hit=last_hit,
                days_since_last=days_since,
                member_counts=member_counts,
                streams_hit=streams_hit,
            )
        )

    out = pd.DataFrame([r.__dict__ for r in rows])
    # Expand member counts into columns
    for k in ["AABC", "ABBC", "ABCC", "OTHER"]:
        out[k] = out["member_counts"].apply(lambda d: int(d.get(k, 0)))
    out = out.drop(columns=["member_counts"])
    out = out.sort_values(["hit_days_per_week", "hits_per_week", "total_hits"], ascending=False).reset_index(drop=True)
    return out


def bucket_score(rank: int) -> int:
    """
    Bucket method per your spec:
      - Top 12 = BaseScore bucket (strong)
      - Ranks 13–60 can earn DueIndex points (8 slots)
    This app implements a conservative version that does NOT overfit:
      - BaseScore: (12 - rank + 1) for top 12, else 0
      - DueIndex: if rank in 13–60, award points by percentile band within that range
    """
    if rank <= 12:
        return 13 - rank
    if 13 <= rank <= 60:
        # Map rank 13..60 into 8 buckets (lower rank => higher due points)
        span = 60 - 13 + 1  # 48
        pos = rank - 13  # 0..47
        bucket = int(pos / (span / 8.0))  # 0..7
        # Invert so best gets 8
        return 8 - bucket
    return 0


def apply_previous_day_downrank(ranked: pd.DataFrame, prev_df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional previous-day file: used ONLY to downrank families that hit yesterday.
    It does NOT exclude them.
    """
    if ranked.empty or prev_df.empty:
        return ranked

    prev_df = prev_df.copy()
    if "Result" not in prev_df.columns:
        prev_df = normalize_columns(prev_df)

    prev_df["Family"] = prev_df["Result"].apply(family_code_from_result)
    hit_fams = set(prev_df["Family"].dropna().unique().tolist())
    if not hit_fams:
        return ranked

    ranked = ranked.copy()
    ranked["DownrankedYesterday"] = ranked["family"].isin(hit_fams)
    # Soft penalty: move down a few positions by subtracting a small score
    ranked["SoftPenalty"] = ranked["DownrankedYesterday"].astype(int) * 2
    ranked["FinalScore"] = ranked["Score"] - ranked["SoftPenalty"]
    ranked = ranked.sort_values(["FinalScore", "hit_days_per_week", "hits_per_week"], ascending=False).reset_index(drop=True)
    return ranked


# -----------------------------
# UI
# -----------------------------

def render_sidebar() -> Dict[str, Any]:
    st.sidebar.title("Controls")

    window_days = st.sidebar.radio(
        "History window (days)",
        options=[180, 365],
        index=0,
        help="Switch between last 180 days (default) vs last 365 days.",
    )

    show_rare = st.sidebar.checkbox("Show Rare Engine (AAAB + AABB)", value=True)
    show_ultra = st.sidebar.checkbox("Show Ultra-Rare Engine (AAAA)", value=True)

    straights_mode = st.sidebar.checkbox("Generate straights shortlist (optional)", value=False)
    straights_top_n = st.sidebar.slider("Straights shortlist size", min_value=10, max_value=150, value=40, step=5)

    max_families = st.sidebar.slider("Max families to display", min_value=25, max_value=250, value=120, step=5)

    return dict(
        window_days=window_days,
        show_rare=show_rare,
        show_ultra=show_ultra,
        straights_mode=straights_mode,
        straights_top_n=straights_top_n,
        max_families=max_families,
    )


def render_header():
    st.title(APP_TITLE)
    st.caption(f"{APP_VERSION} — Family chasing across all streams (no single-stream targeting).")


def render_uploads() -> Tuple[pd.DataFrame, pd.DataFrame]:
    st.subheader("Upload files")
    st.write("Upload your consolidated all-states history file. Optional: upload a previous-day file for soft downranking only.")

    main_upload = st.file_uploader("All-states history (.csv or .txt)", type=["csv", "txt"], key="main")
    prev_upload = st.file_uploader("Previous-day file (optional) (.csv or .txt)", type=["csv", "txt"], key="prev")

    main_df = load_table_from_upload(main_upload) if main_upload else pd.DataFrame()
    prev_df = load_table_from_upload(prev_upload) if prev_upload else pd.DataFrame()

    if not main_df.empty:
        main_df = normalize_columns(main_df)
    if not prev_df.empty:
        prev_df = normalize_columns(prev_df)

    return main_df, prev_df


def render_family_table(ranked: pd.DataFrame, max_rows: int):
    if ranked.empty:
        st.info("No doubles families found in the selected window.")
        return

    show = ranked.head(max_rows).copy()

    # Display-friendly
    show["LastHit"] = show["last_hit"].astype(str)
    cols = [
        "rank",
        "family",
        "Score",
        "FinalScore",
        "hit_days_per_week",
        "hits_per_week",
        "hit_days",
        "total_hits",
        "streams_hit",
        "AABC",
        "ABBC",
        "ABCC",
        "DownrankedYesterday",
        "LastHit",
    ]
    for c in cols:
        if c not in show.columns:
            show[c] = ""

    show = show[cols]
    show = show.rename(
        columns={
            "rank": "Rank",
            "family": "Family",
            "hit_days_per_week": "Hit-days/Wk",
            "hits_per_week": "Hits/Wk",
            "hit_days": "Hit-days",
            "total_hits": "Hits",
            "streams_hit": "Streams Hit",
            "DownrankedYesterday": "Hit Yesterday? (soft)",
        }
    )
    st.dataframe(show, use_container_width=True, height=720)


def render_rare_engines(df: pd.DataFrame, window_days: int):
    if df.empty:
        return

    st.subheader("Rare / Ultra-Rare Engines")

    if "Date" in df.columns and df["Date"].notna().any():
        max_date = max([d for d in df["Date"].dropna().tolist()])
        cutoff = max_date - dt.timedelta(days=window_days - 1)
        dfw = df[df["Date"].between(cutoff, max_date, inclusive="both")].copy()
    else:
        dfw = df.copy()

    dfw["is_rare"] = dfw["Result"].apply(is_rare)
    dfw["is_ultra"] = dfw["Result"].apply(is_ultra_rare)

    rare = dfw[dfw["is_rare"]].copy()
    ultra = dfw[dfw["is_ultra"]].copy()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Rare (AAAB + AABB)**")
        st.write(f"Count in window: {len(rare)}")
        if not rare.empty:
            st.dataframe(rare[["Date", "Stream", "Result"]].tail(50), use_container_width=True, height=350)

    with c2:
        st.markdown("**Ultra-Rare (AAAA)**")
        st.write(f"Count in window: {len(ultra)}")
        if not ultra.empty:
            st.dataframe(ultra[["Date", "Stream", "Result"]].tail(50), use_container_width=True, height=350)


def generate_straights_from_family_members(family_code: str) -> List[str]:
    """
    Generate the 3 canonical doubles-family members for a family code XYZ where
    X, Y, Z are digits and the repeated digit is each of X/Y/Z respectively:
      - XYZZ? Not; for family code like "389":
        members are 3389, 3889, 3899
    We'll return [XXYZ, XYYZ, XYZZ] in the same order as cadence report (AABC, ABBC, ABCC)
    """
    if not re.fullmatch(r"\d{3}", family_code):
        return []
    a, b, c = list(family_code)
    # AABC: aa b c
    m1 = f"{a}{a}{b}{c}"
    # ABBC: a bb c
    m2 = f"{a}{b}{b}{c}"
    # ABCC: a b cc
    m3 = f"{a}{b}{c}{c}"
    # But this only matches when repeated digit is first char.
    # For consistency with your report's family naming (389 = digits {3,8,9}),
    # the "AABC/ABBC/ABCC" are member templates within that digit set:
    # 3389, 3889, 3899. That corresponds to repeat 3, repeat 8, repeat 9.
    # So we must generate all three repeats:
    digits = sorted([a, b, c])
    members = [
        f"{digits[0]}{digits[0]}{digits[1]}{digits[2]}",
        f"{digits[0]}{digits[1]}{digits[1]}{digits[2]}",
        f"{digits[0]}{digits[1]}{digits[2]}{digits[2]}",
    ]
    # Now reorder to match (repeat first digit of family code, then second, then third)
    order = [a, b, c]
    out = []
    for rep in order:
        other = sorted([d for d in [a, b, c] if d != rep])
        out.append(f"{rep}{rep}{other[0]}{other[1]}")
    return out


def build_straights_shortlist(ranked: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    Optional: produce a shortlist of straight plays derived from top ranked families.
    This does NOT attempt to predict exact straight ordering beyond canonical family members.
    """
    if ranked.empty:
        return pd.DataFrame()
    fams = ranked.head(top_n // 3 + 5)["family"].tolist()
    rows = []
    for fam in fams:
        members = generate_straights_from_family_members(fam)
        for m in members:
            rows.append({"Family": fam, "Member": m})
    out = pd.DataFrame(rows).head(top_n).copy()
    return out


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    render_header()

    controls = render_sidebar()
    df, prev_df = render_uploads()

    if df.empty:
        st.info("Upload a file to begin.")
        return

    window_days = int(controls["window_days"])
    stats = build_family_stats(df, window_days=window_days)

    if stats.empty:
        st.warning("No doubles families found in your file (AABC/ABBC/ABCC).")
        return

    # Rank families
    ranked = stats.copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["Score"] = ranked["rank"].apply(bucket_score)

    # Default: FinalScore equals Score (will be overwritten if prev-day is provided)
    ranked["FinalScore"] = ranked["Score"]
    ranked["DownrankedYesterday"] = False
    ranked = apply_previous_day_downrank(ranked, prev_df)

    st.subheader("Family Ranking (across all streams)")
    st.write("This ranks doubles families by recent hit-day cadence and a conservative bucket score. No families are excluded by default.")
    render_family_table(ranked, max_rows=int(controls["max_families"]))

    if controls["show_rare"] or controls["show_ultra"]:
        render_rare_engines(df, window_days=window_days)

    if controls["straights_mode"]:
        st.subheader("Optional Straights Shortlist (derived from top families)")
        straights = build_straights_shortlist(ranked, top_n=int(controls["straights_top_n"]))
        if straights.empty:
            st.info("No straights generated.")
        else:
            st.dataframe(straights, use_container_width=True, height=520)

    # Downloads
    st.subheader("Export")
    # Export ranked table as CSV
    csv_bytes = ranked.to_csv(index=False).encode("utf-8")
    st.download_button("Download ranked families CSV", data=csv_bytes, file_name=f"pk4_northern_star_ranked_families_{APP_VERSION}.csv", mime="text/csv")


if __name__ == "__main__":
    main()
