# streamlit_app.py
# Pick-4 (3389 / 3889 / 3899 families) prediction helper:
# - Master ranking: all streams (State/Game) scored for likelihood a family hits next
# - Straight ordering learner: per State/Game, rank the 12 unique straights for each family
#
# Notes:
# - Accepts BOTH .txt and .csv inputs.
# - TXT parsing is resilient to: tabs, multiple spaces, commas, Fireball/Wild Ball trailing text.
# - Playable list upload is OPTIONAL (marks PlayableByUser=Yes/No; never filters rows).

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from itertools import permutations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Config
# -----------------------------

st.set_page_config(page_title="Pick 4 3389/3889/3899 — Master Ranking + Straights", layout="wide")

FAMILY_KEYS = {
    "3389": "3389",
    "3889": "3889",
    "3899": "3899",
}
FAMILY_SET = set(FAMILY_KEYS.keys())

DIGIT_RE = re.compile(r"\d")

# "1-6-5" / prev-draw context sets (user rule)
GOOD_DIGITS = {0, 2, 3, 4, 7, 8, 9}
BAD_DIGITS = {1, 5, 6}


# -----------------------------
# Utilities
# -----------------------------


def _safe_decode(uploaded) -> str:
    """Read a Streamlit UploadedFile as text."""
    raw = uploaded.getvalue()
    # Try UTF-8 first; fall back to latin-1 to avoid hard failures.
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def normalize_state(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_game(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_4digit_result(txt: str) -> str | None:
    """Extract a 4-digit Pick 4 result from a messy text fragment.

    Key goal: NEVER mistake a date (e.g., 2025-12-24, "Dec 27, 2025") for the result.
    Supports formats like:
      - 3389
      - 3-3-8-9
      - 3 3 8 9
      - '... 3-3-8-9, Fireball: 9'
      - whole lines that also contain dates/state/game
    """
    if txt is None:
        return None
    s = str(txt).strip()
    if not s:
        return None

    low = s.lower()

    # If this fragment looks like a DATE token, do not treat it as a result.
    # (Prevents '2025' or '2025-12-24' from being parsed as the Pick 4 number.)
    if re.search(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}", s):
        return None
    if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b", low):
        # Date-like text such as "Sat, Dec 27, 2025"
        # Let parse_date_any handle it elsewhere; this function should not.
        if re.search(r"\b\d{4}\b", s):
            return None

    # Prefer explicit 4-digit sequences separated by dashes/spaces (common result formats)
    m_sep = list(re.finditer(r"(?<!\d)(\d)\s*[-\s]\s*(\d)\s*[-\s]\s*(\d)\s*[-\s]\s*(\d)(?!\d)", s))
    if m_sep:
        a, b, c, d = m_sep[-1].groups()
        return f"{a}{b}{c}{d}"

    # Next, look for standalone 4-digit groups.
    m4 = list(re.finditer(r"(?<!\d)(\d{4})(?!\d)", s))
    if not m4:
        return None

    # Choose the last 4-digit token that is NOT a plausible year.
    for m in reversed(m4):
        token = m.group(1)
        try:
            val = int(token)
        except Exception:
            continue
        if 1900 <= val <= 2100:
            continue
        return token

    return None
def prevdraw_counts_from_straight(straight: str) -> tuple[int, int]:
    """Return (good_count, bad_count) for a 4-digit straight string."""
    digs = [int(ch) for ch in str(straight)]
    good = sum(d in GOOD_DIGITS for d in digs)
    bad = sum(d in BAD_DIGITS for d in digs)
    return good, bad


def prevdraw_status(good: int, bad: int) -> str:
    """Human label for the user's '1-6-5' context rule."""
    # Strong pass: >=3 good digits AND <=1 bad digit
    if good >= 3 and bad <= 1:
        return "PASS_STRONG"
    # Weak pass: still acceptable context
    if good >= 2 and bad <= 2:
        return "PASS_WEAK"
    # Otherwise, de-prioritize
    return "AVOID"


def attach_prevday_context(master_df: pd.DataFrame, prevday_df: pd.DataFrame, asof: pd.Timestamp | None) -> pd.DataFrame:
    """
    Merge previous-day latest results (by State+Game) into master ranking.

    prevday_df must contain at least: ['state','game','date','straight'].
    """
    if prevday_df is None or prevday_df.empty:
        return master_df

    df = prevday_df.copy()
    if asof is not None and "date" in df.columns:
        df = df[df["date"] <= asof]

    # Keep last known result per stream (State+Game)
    if "date" in df.columns:
        df = df.sort_values("date").groupby(["state", "game"], as_index=False).tail(1)
    else:
        df = df.groupby(["state", "game"], as_index=False).tail(1)

    df["state_norm"] = df["state"].map(normalize_state)
    df["game_norm"] = df["game"].map(normalize_game)
    df["PrevDrawDate"] = df.get("date")
    df["PrevDrawStraight"] = df.get("straight")
    df["PrevDrawBox"] = df.get("box")

    gb = df["PrevDrawStraight"].apply(lambda s: prevdraw_counts_from_straight(s) if pd.notna(s) else (np.nan, np.nan))
    df["PrevGoodCount"] = [x[0] for x in gb]
    df["PrevBadCount"] = [x[1] for x in gb]
    df["PrevStatus"] = [prevdraw_status(int(g), int(b)) if pd.notna(g) and pd.notna(b) else "UNKNOWN"
                        for g, b in zip(df["PrevGoodCount"], df["PrevBadCount"])]

    # Numeric score in [-1, 1] to optionally blend into the main score
    df["PrevDrawScore"] = ((df["PrevGoodCount"] - df["PrevBadCount"]) / 5.0).clip(-1, 1)

    keep_cols = ["state_norm", "game_norm", "PrevDrawDate", "PrevDrawStraight", "PrevDrawBox",
                 "PrevGoodCount", "PrevBadCount", "PrevStatus", "PrevDrawScore"]

    out = master_df.copy()
    out["state_norm"] = out["State"].map(normalize_state)
    out["game_norm"] = out["Game"].map(normalize_game)

    out = out.merge(df[keep_cols], on=["state_norm", "game_norm"], how="left")
    out.drop(columns=["state_norm", "game_norm"], inplace=True, errors="ignore")

    return out




def box_key(num4: str) -> str:
    return "".join(sorted(num4))


def unique_perms(num4: str) -> List[str]:
    return sorted({"".join(p) for p in permutations(list(num4), 4)})


def parse_date_any(s: str) -> Optional[pd.Timestamp]:
    """Parse date from common formats in your TXT/CSV."""
    if s is None:
        return None
    txt = str(s).strip()
    if not txt:
        return None

    # Typical lines: "Sat, Dec 27, 2025" or "2025/12/26"
    # Let pandas try first.
    try:
        dt = pd.to_datetime(txt, errors="coerce", infer_datetime_format=True)
        if pd.isna(dt):
            return None
        return dt
    except Exception:
        return None


# -----------------------------
# Parsing (HITS + STREAM files)
# -----------------------------

REQUIRED_COLS = ["date", "state", "game", "result"]

# Target Pick-4 families (sorted digit strings)
DEFAULT_FAMILIES = ["3389", "3889", "3899"]


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        mapping[c] = lc
    df = df.rename(columns=mapping)

    # Common aliases
    if "drawdate" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"drawdate": "date"})
    if "draw" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"draw": "date"})
    if "province" in df.columns and "state" not in df.columns:
        df = df.rename(columns={"province": "state"})

    return df


def _parse_csv_flexible(text: str) -> Optional[pd.DataFrame]:
    """Attempt CSV/TSV parsing using pandas.

    NOTE: many files that look like "text" are actually TSV where the
    date contains commas (e.g., "Fri, Dec 26, 2025"). If we let pandas
    sniff delimiters, it can wrongly choose comma and shred the date.
    So we try tab-first when tabs are present.
    """

    # Grab the first non-empty line for delimiter heuristics
    first_line = ""
    for ln in text.splitlines():
        if ln.strip():
            first_line = ln
            break

    # Build a small list of parsing attempts, ordered by likelihood.
    attempts: List[Dict[str, object]] = []
    if "\t" in first_line and first_line.count("\t") >= 2:
        attempts.append(dict(sep="\t", engine="python"))
    # Common cases
    attempts.extend([
        dict(sep=None, engine="python"),
        dict(sep=",", engine="python"),
        dict(sep=";", engine="python"),
    ])

    df = None
    for kwargs in attempts:
        try:
            df = pd.read_csv(io.StringIO(text), **kwargs)
            break
        except Exception:
            df = None
            continue

    if df is None:
        return None
    df = _standardize_columns(df)

    # If it already contains required columns, great.
    if all(c in df.columns for c in REQUIRED_COLS):
        return df

    # Sometimes the file is delimiter-separated but without header.
    # Try heuristic: 4 columns in order date, state, game, result.
    if df.shape[1] >= 4 and "date" not in df.columns:
        cols = list(df.columns)
        df2 = df.rename(columns={cols[0]: "date", cols[1]: "state", cols[2]: "game", cols[3]: "result"})
        if all(c in df2.columns for c in REQUIRED_COLS):
            # Validate that at least one date is parseable; otherwise this
            # is likely a bad delimiter (e.g., comma split inside the date).
            sample = df2["date"].head(8).astype(str).tolist()
            if any(parse_date_any(x) is not None for x in sample):
                return df2

    return None


def _infer_stream_from_filename(name: str, master_streams: list[tuple[str, str]] | None = None) -> tuple[str | None, str | None]:
    """Best-effort inference of (State, Game) from a filename.

    This is intentionally heuristic: it helps when users upload single-stream exports that only contain Date/Result.
    """
    if not name:
        return (None, None)
    lower = name.lower()

    # Common abbreviations at file start: "tx night p4 ...", "la p4 ...", "ar p4 ..."
    abbr_map = {
        "al":"Alabama","ak":"Alaska","az":"Arizona","ar":"Arkansas","ca":"California","co":"Colorado","ct":"Connecticut","de":"Delaware",
        "fl":"Florida","ga":"Georgia","hi":"Hawaii","id":"Idaho","il":"Illinois","in":"Indiana","ia":"Iowa","ks":"Kansas","ky":"Kentucky",
        "la":"Louisiana","me":"Maine","md":"Maryland","ma":"Massachusetts","mi":"Michigan","mn":"Minnesota","ms":"Mississippi","mo":"Missouri",
        "mt":"Montana","ne":"Nebraska","nv":"Nevada","nh":"New Hampshire","nj":"New Jersey","nm":"New Mexico","ny":"New York","nc":"North Carolina",
        "nd":"North Dakota","oh":"Ohio","ok":"Oklahoma","or":"Oregon","pa":"Pennsylvania","ri":"Rhode Island","sc":"South Carolina","sd":"South Dakota",
        "tn":"Tennessee","tx":"Texas","ut":"Utah","vt":"Vermont","va":"Virginia","wa":"Washington","wv":"West Virginia","wi":"Wisconsin","wy":"Wyoming",
        # extras
        "dc":"Washington, D.C.","pr":"Puerto Rico","on":"Ontario"
    }
    state_guess = None
    m = re.match(r"^\s*([a-z]{2})\b", lower)
    if m and m.group(1) in abbr_map:
        state_guess = abbr_map[m.group(1)]

    # If not found, try full state words present in filename
    if state_guess is None and master_streams:
        states = sorted({s for s,_ in master_streams}, key=len, reverse=True)
        for s in states:
            if s.lower() in lower:
                state_guess = s
                break

    # Game guess by keyword, then snap to a real game name if master_streams is provided
    game_guess = None
    kw = None
    for k in ["morning","midday","day","evening","night","10pm","9pm","8pm","7pm","6pm","5pm","4pm","3pm","2pm","1pm","am","pm"]:
        if k in lower:
            kw = k
            break

    if kw and master_streams and state_guess:
        # Choose the first game under this state that contains the keyword
        candidates = [g for s,g in master_streams if s == state_guess]
        # Prefer longer/more specific matches
        candidates_sorted = sorted(candidates, key=lambda g: (-len(g), g))
        for g in candidates_sorted:
            gl = g.lower()
            if kw in gl:
                game_guess = g
                break

    return (state_guess, game_guess)


def _parse_txt_lines(text: str, *, default_state: str | None = None, default_game: str | None = None, master_streams: list[tuple[str, str]] | None = None, filename: str | None = None) -> pd.DataFrame:
    """Parse LotteryPost-like TXT (tab separated) OR looser text exports.

    Supports:
      - 4+ columns: Date, State, Game, Result[, ...]
      - 2-3 columns (single-stream exports): Date, Result[, Fireball/WildBall...]
      - comma-only lines like: "Sat, Dec 27, 2025\tTexas\tDaily 4 Night\t3-9-3-8, Fireball: 9"
      - space-separated fallback (best-effort): "Sat, Dec 27, 2025 Texas Daily 4 Night 3-9-3-8"
    """
    if not text:
        return pd.DataFrame(columns=["Date", "State", "Game", "Result", "RawLine"])

    inferred_state, inferred_game = (None, None)
    if filename:
        inferred_state, inferred_game = _infer_stream_from_filename(filename, master_streams=master_streams)

    state_fallback = default_state or inferred_state
    game_fallback = default_game or inferred_game

    out = []
    for raw in text.splitlines():
        line = raw.strip("\ufeff").strip()
        if not line:
            continue
        low = line.lower()

        # Skip obvious headers
        if ("date" in low and "state" in low and "game" in low and "result" in low) or low.startswith("date\t") or low.startswith("date,"):
            continue

        # Split by tabs if present, else try multiple spaces, else commas
        if "\t" in line:
            parts = [p.strip() for p in re.split(r"\t+", line) if p.strip()]
        else:
            # If it's clearly comma-separated with embedded commas in the date, keep as one string and parse by regex below.
            parts = [p.strip() for p in re.split(r"\s{2,}", line) if p.strip()]
            if len(parts) <= 1:
                parts = [p.strip() for p in line.split(",") if p.strip()]

        # Identify date and result among parts
        date_idx = None
        result_idx = None
        parsed_dt = None
        parsed_res = None

        for i, p in enumerate(parts):
            if date_idx is None:
                dt = parse_date_any(p)
                if dt is not None:
                    date_idx = i
                    parsed_dt = dt
            if result_idx is None:
                r = extract_4digit_result(p)
                if r is not None:
                    result_idx = i
                    parsed_res = r
            if date_idx is not None and result_idx is not None:
                break

        # If we couldn't isolate via parts, try a regex fallback on the whole line
        if parsed_dt is None or parsed_res is None:
            # Find a date at start
            mdate = re.match(r"^(?P<d>(?:\w{3},\s*)?\w{3}\s+\d{1,2},\s+\d{4}|\d{4}[-/]\d{2}[-/]\d{2})", line)
            if mdate:
                parsed_dt = parse_date_any(mdate.group("d"))
            parsed_res = parsed_res or extract_4digit_result(line)

        if parsed_dt is None or parsed_res is None:
            continue

        # Collect state/game from remaining parts, if present
        rem = []
        if parts:
            for j, p in enumerate(parts):
                if j == date_idx or j == result_idx:
                    continue
                # Skip Fireball/Wild Ball tail pieces if they show up as separate columns
                if "fireball" in p.lower() or "wild" in p.lower():
                    continue
                rem.append(p)

        state = None
        game = None

        if len(rem) >= 2:
            state, game = rem[0], rem[1]
        elif len(rem) == 1:
            # If it looks like a state name, treat as state; otherwise treat as game
            if rem[0].istitle() and " " in rem[0]:
                state = rem[0]
            else:
                game = rem[0]

        state = state or state_fallback or "(Unknown)"
        game = game or game_fallback or "(Unknown)"

        out.append({
            "Date": pd.Timestamp(parsed_dt).normalize(),
            "State": str(state),
            "Game": str(game),
            "Result": parsed_res,
            "RawLine": line,
        })

    if not out:
        return pd.DataFrame(columns=["Date", "State", "Game", "Result", "RawLine"])

    df = pd.DataFrame(out)
    df = df.dropna(subset=["Date", "Result"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_history(file, filter_families: bool = True, families: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Load a 5-year Pick-4 history file (TXT/CSV) into a normalized dataframe.

    Returns lowercase columns expected by the rest of the app:
      - date (datetime64)
      - state (str)
      - game (str)
      - result (4-digit str)
      - family (sorted digits str)
      - rawline (original line, when available)
    """
    if file is None:
        return pd.DataFrame(columns=["date","state","game","result","family","rawline"])

    name = getattr(file, "name", "") or ""
    raw = file.getvalue() if hasattr(file, "getvalue") else file.read()

    # --- Parse
    df = None
    if name.lower().endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(raw), dtype=str)
        except Exception:
            # fallback: treat as text
            df = _parse_txt_lines(raw.decode("utf-8", errors="ignore").splitlines())
    else:
        df = _parse_txt_lines(raw.decode("utf-8", errors="ignore").splitlines())

    if df is None or df.empty:
        return pd.DataFrame(columns=["date","state","game","result","family","rawline"])

    # --- Normalize column names to lowercase
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Accept common variants
    rename_map = {}
    if "drawdate" in df.columns and "date" not in df.columns:
        rename_map["drawdate"] = "date"
    if "winning" in df.columns and "result" not in df.columns:
        rename_map["winning"] = "result"
    if "numbers" in df.columns and "result" not in df.columns:
        rename_map["numbers"] = "result"
    if "pick" in df.columns and "game" not in df.columns:
        rename_map["pick"] = "game"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure required columns exist
    for col in ["date","state","game","result"]:
        if col not in df.columns:
            # Try title-case legacy columns (from _parse_txt_lines)
            alt = col.title()
            if alt.lower() in df.columns:
                continue
            raise KeyError(f"Missing required column '{col}'. Found: {list(df.columns)}")

    if "rawline" not in df.columns:
        df["rawline"] = ""

    # --- Clean fields
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["state"] = df["state"].astype(str).str.strip()
    df["game"]  = df["game"].astype(str).str.strip()

    # Normalize result to 4 digits (strip spaces, dashes, and any Fireball/Wild Ball tail text)
    df["result"] = df["result"].astype(str).apply(extract_4digit_result)
    df = df[df["result"].str.len() == 4].copy()
    df = df.dropna(subset=["date"])

    # Family is the sorted digits string (box key)
    df["family"] = df["result"].apply(box_key)

    # Filter to the target families if requested
    if filter_families:
        fams = list(families) if families is not None else list(DEFAULT_FAMILIES)
        fams_norm = {box_key(str(f)) for f in fams if str(f).strip()}
        df = df[df["family"].isin(fams_norm)].copy()

    df = df.sort_values(["date","state","game"]).reset_index(drop=True)

    # Keep a consistent column set
    keep_cols = ["date","state","game","result","family","rawline"]
    return df[keep_cols]

def compute_stream_stats(df: pd.DataFrame,
                         analysis_start: pd.Timestamp,
                         analysis_end: pd.Timestamp,
                         schedule_laplace_alpha: float,
                         schedule_combine_mode: str) -> pd.DataFrame:
    """Compute stats per (state, game) using ONLY the hits history."""

    # Ensure a stable family key is available (older files may not include it).
    if "family" not in df.columns and "result" in df.columns:
        df = df.copy()
        df["family"] = df["result"].apply(box_key)

    groups = df.groupby(["state", "game"], sort=False)
    rows = []

    total_days = max(1, int((analysis_end.date() - analysis_start.date()).days) + 1)

    for (state, game), g in groups:
        g = g.sort_values("date")
        hits = int(len(g))

        last_hit_date = g["date"].max() if hits else None
        days_since = None
        if last_hit_date is not None:
            days_since = int((analysis_end.normalize() - last_hit_date.normalize()).days)

        # HitRate: hits per day in window
        hit_rate = safe_div(hits, total_days)

        # Consistency: fraction of months in window with >=1 hit
        if hits:
            months_with = g["date"].dt.to_period("M").nunique()
        else:
            months_with = 0
        months_total = max(1, (analysis_end.to_period("M") - analysis_start.to_period("M")).n + 1)
        consistency = safe_div(months_with, months_total)

        # Reliability: soft boost for sample size
        reliability = math.log1p(hits)

        # Family shares
        counts_by_family = g["family"].value_counts().to_dict()
        share_3389 = safe_div(counts_by_family.get("3389", 0), hits)
        share_3889 = safe_div(counts_by_family.get("3889", 0), hits)
        share_3899 = safe_div(counts_by_family.get("3899", 0), hits)

        # Gap-based overdue and expected next hit
        dates = g["date"].sort_values().dt.normalize().tolist()
        gaps = []
        for i in range(1, len(dates)):
            gap = int((dates[i] - dates[i - 1]).days)
            if gap > 0:
                gaps.append(gap)

        overdue_pct = 0.0
        expected_gap = None
        predicted_next = None
        due_tempered = 0.0

        if gaps and days_since is not None:
            gaps_arr = np.array(gaps, dtype=float)
            expected_gap = float(np.mean(gaps_arr))

            # OverduePercentile: how deep are we relative to historical gaps
            overdue_pct = float(np.mean(gaps_arr <= float(days_since)))

            # Temper by proximity to median-ish gaps (avoid over-weighting extreme droughts)
            med = float(np.median(gaps_arr))
            prox = math.exp(-abs(float(days_since) - med) / (med + 1.0))
            due_tempered = float(overdue_pct * (0.25 + 0.75 * prox))

            # Predicted next hit date
            if last_hit_date is not None:
                predicted_next = last_hit_date.normalize() + pd.Timedelta(days=int(round(expected_gap)))

        schedule_boost = compute_schedule_boost(dates, analysis_end.normalize(), laplace_alpha=schedule_laplace_alpha,
                                                combine_mode=schedule_combine_mode)

        rows.append({
            "State": state,
            "Game": game,
            "Hits": hits,
            "LastHitDate": last_hit_date.date().isoformat() if last_hit_date is not None else "",
            "DaysSinceLastHit": days_since if days_since is not None else "",
            "HitRate": hit_rate,
            "Consistency": consistency,
            "Reliability": reliability,
            "Share_3389": share_3389,
            "Share_3889": share_3889,
            "Share_3899": share_3899,
            "OverduePercentile": overdue_pct,
            "DueTempered": due_tempered,
            "ScheduleBoost": schedule_boost,
            "ExpectedGapDays": expected_gap if expected_gap is not None else "",
            "PredictedNextHitDate": predicted_next.date().isoformat() if predicted_next is not None else "",
        })

    return pd.DataFrame(rows)


def score_master_table(stats: pd.DataFrame,
                       w_hit: float,
                       w_due: float,
                       w_sched: float,
                       w_cons: float,
                       w_rel: float) -> pd.DataFrame:
    if stats.empty:
        return stats

    df = stats.copy()

    # Normalize components to comparable 0..1 range
    def minmax(col: str) -> pd.Series:
        x = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        mn, mx = float(x.min()), float(x.max())
        if mx <= mn:
            return pd.Series([0.0] * len(x), index=x.index)
        return (x - mn) / (mx - mn)

    hit_n = minmax("HitRate")
    due_n = minmax("DueTempered")
    sched_n = minmax("ScheduleBoost")
    cons_n = minmax("Consistency")
    rel_n = minmax("Reliability")

    score = (
        w_hit * hit_n
        + w_due * due_n
        + w_sched * sched_n
        + w_cons * cons_n
        + w_rel * rel_n
    )

    df["Score"] = score
    df = df.sort_values(["Score", "Hits"], ascending=[False, False]).reset_index(drop=True)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))
    return df


# -----------------------------
# Straights learner (per stream)
# -----------------------------


def straight_ranking_for_stream(
    df_stream: pd.DataFrame,
    state: str,
    game: str,
    family: str,
    # Preferred parameter names (match UI controls):
    laplace_alpha: float = 1.0,
    recency_half_life_days: int = 120,
    blend_recency: float = 0.30,
    asof: Optional[pd.Timestamp] = None,
    # Back-compat aliases (older internal naming):
    alpha: Optional[float] = None,
    half_life: Optional[int] = None,
    recency_mix: Optional[float] = None,
) -> pd.DataFrame:
    """Return a 12-straight ranking table for one (state, game, family).

    If the selected stream has 0 hits for this family, we fall back to a global
    distribution (all uploaded stream rows) for that family. If that is also 0,
    we use a uniform distribution.

    Returns: DataFrame with 12 rows (one per straight), including an 'Evidence' column.
    """
    # Alias mapping so older call-sites keep working:
    if alpha is not None:
        laplace_alpha = float(alpha)
    if half_life is not None:
        recency_half_life_days = int(half_life)
    if recency_mix is not None:
        blend_recency = float(recency_mix)

    perms = unique_perms(family)

    if asof is None:
        asof = pd.Timestamp(df_stream["date"].max()).normalize() if not df_stream.empty else pd.Timestamp.today().normalize()
    else:
        asof = pd.Timestamp(asof).normalize()

    # Evidence: state/game specific hits for this family
    g = df_stream[(df_stream["state"] == state) & (df_stream["game"] == game) & (df_stream["family"] == family)].copy()
    used_mode = "state_game"

    # Global fallback evidence: any stream hits for this family
    g_global = df_stream[df_stream["family"] == family].copy() if not df_stream.empty else pd.DataFrame(columns=df_stream.columns if not df_stream.empty else [])

    if g.empty:
        if not g_global.empty:
            g = g_global
            used_mode = "global_fallback"
        else:
            used_mode = "uniform"

    # Counts
    counts = {p: 0 for p in perms}
    last_seen_map: dict[str, pd.Timestamp] = {}

    if used_mode != "uniform":
        # result is already canonical as 4 digits (e.g., '9383')
        for p, sub in g.groupby("result"):
            if p in counts:
                counts[p] = int(len(sub))
                last_seen_map[p] = pd.Timestamp(sub["date"].max()).normalize()

    total = sum(counts.values())

    # Freq prob (Laplace)
    denom = (total + laplace_alpha * len(perms))
    freq_prob = {p: (counts[p] + laplace_alpha) / denom for p in perms}

    # Recency weight per permutation
    rec_w = {}
    for p in perms:
        if p in last_seen_map:
            ds = (asof - last_seen_map[p]).days
            rec_w[p] = exp_decay(ds, recency_half_life_days)
        else:
            rec_w[p] = 0.0

    # Normalize recency weights to look like a probability distribution
    rec_sum = sum(rec_w.values())
    if rec_sum > 0:
        rec_prob = {p: rec_w[p] / rec_sum for p in perms}
    else:
        rec_prob = {p: 1.0 / len(perms) for p in perms}

    # Final blend
    out_rows = []
    for p in perms:
        score = (1.0 - blend_recency) * freq_prob[p] + blend_recency * rec_prob[p]
        out_rows.append({
            "Straight": p,
            "Count": counts[p],
            "Prob": float(score),
        })

    out = pd.DataFrame(out_rows)
    out = out.sort_values(["Prob", "Count", "Straight"], ascending=[False, False, True]).reset_index(drop=True)
    out.insert(0, "Rank", range(1, len(out) + 1))

    info = {
        "mode": used_mode,
        "state_game_hits": int(df_stream[(df_stream["state"] == state) & (df_stream["game"] == game) & (df_stream["family"] == family)].shape[0]) if not df_stream.empty else 0,
        "global_hits": int(g_global.shape[0]) if not df_stream.empty else 0,
        "asof": asof,
        "family": family,
        "state": state,
        "game": game,
    }

    # Add evidence mode for UI warnings/fallback transparency
    out["Evidence"] = used_mode

    return out



# -----------------------------
# UI
# -----------------------------

st.title("Pick 4 3389 / 3889 / 3899 — Master Ranking + 12-Straights Learner")

with st.sidebar:
    st.header("Inputs")
    st.caption("HITS = 5-year hits list across all states/games. STREAM = per-state/game 24-month history for straight ordering.")

    hits_file = st.file_uploader(
        "Upload 5-year HIT history (TXT or CSV)",
        type=["txt", "csv"],
        help="TXT can be tab-separated or space-separated. Must contain Date, State, Game, Result. Fireball/Wild Ball is okay.",
    )

    st.markdown("---")
    st.subheader("As-Of scoring date")
    asof_date = st.date_input("As-Of Date", value=None)
    assume_no_hits_after = st.checkbox(
        "Assume there were NO hits after the last file date through As-Of Date",
        value=True,
        help="If checked, days-since-last-hit is computed up to As-Of Date even when your file stops earlier.",
    )

    st.markdown("---")
    st.subheader("Optional: Playable list")
    playable_file = st.file_uploader(
        "Upload a Playable list (TXT or CSV with columns State,Game) to MARK playable streams (no filtering)",
        type=["txt", "csv"],
        help="If TXT: each line should be State<TAB>Game or State,Game.",
    )


# Optional: previous-day winners list (for the user's 1-6-5 context rule)
st.subheader("Optional: Previous-day winners (1-6-5 context)")
prevday_files = st.file_uploader(
    "Upload previous-day winners/results (TXT or CSV) to label/boost streams",
    type=["txt", "csv"],
    accept_multiple_files=True,
    key="prevday_files",
    help=(
        "Upload a small file with the most recent results per stream (State/Game/Result, with a Date if you have it). "
        "The app uses your rule: PASS_STRONG if >=3 digits in {0,2,3,4,7,8,9} AND <=1 digit in {1,5,6}; "
        "otherwise it flags AVOID. This does not drop rows; it only adds columns and can optionally boost ordering."
    ),
)
prevday_mode = st.selectbox(
    "How to use previous-day context",
    ["Off (add columns only)", "Boost ordering (recommended)"],
    index=0,
    key="prevday_mode",
)
# --- Defaults (so app never crashes when previous-day context is OFF) ---
# These values are only *used* when prevday_mode is not actively supplying them.
# When the UI sliders are shown (Boost ordering mode), the user-selected values override these.
schedule_laplace_alpha = 2.0
schedule_mode = "Multiply (recommended)"
schedule_blend = 0.30  # 0=freq only, 1=recency only

w_hit = 0.40
w_due = 0.20
w_sched = 0.20
w_cons = 0.10
w_rel = 0.10

# Straights-learning defaults (used if the optional STREAM upload isn't provided)
straight_recency_half_life_days = 120
straight_laplace_alpha = 1.0
straight_blend_recency_frequency = 0.30
straight_mix = straight_blend_recency_frequency
stream_files = []
prevday_weight = 0.0
if prevday_mode == "Boost ordering (recommended)":
    prevday_weight = st.slider(
        "Prev-day boost weight (how strongly to move PASS_STRONG streams upward)",
        0.0,
        0.30,
        0.12,
        0.01,
        key="prevday_weight",
    )

    st.markdown("---")
    st.subheader("Model controls")
    # Fixed weights (but we still expose them as read-only text + hidden slider option)
    st.caption("Weights are fixed to your approved mix (you can change later if you want).")

    schedule_laplace_alpha = st.slider(
        "Schedule smoothing α (higher = weaker schedule boost / less overfit)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.1,
    )
    schedule_mode = st.selectbox(
        "ScheduleBoost combine mode",
        options=["Multiply (weekday*month)", "Average (weekday+month)/2"],
        index=0,
    )

    # Default fixed weights (match your on-screen mix style)
    w_hit = 0.50
    w_due = 0.30
    w_sched = 0.10
    w_cons = 0.08
    w_rel = 0.02

    st.markdown("**Fixed scoring weights**")
    st.write(f"- {w_hit:.2f} HitRate")
    st.write(f"- {w_due:.2f} OverduePercentile (tempered by GapProximity)")
    st.write(f"- {w_sched:.2f} ScheduleBoost")
    st.write(f"- {w_cons:.2f} Consistency")
    st.write(f"- {w_rel:.2f} Reliability")

    st.markdown("---")
    st.subheader("Straights learning")
    stream_files = st.file_uploader(
        "Upload 24-month STREAM history file(s) (TXT or CSV)",
        type=["txt", "csv"],
        accept_multiple_files=True,
        help="You can upload one per state/game, or many at once. Must contain Date, State, Game, Result.",
    )

    straight_recency_half_life_days = st.slider("Recency half-life (days)", 1, 365, 120)
    straight_laplace_alpha = st.slider("Smoothing laplace_alpha (Laplace)", 0.0, 5.0, 1.0, 0.05)
    straight_mix = st.slider("Blend recency vs frequency (0=freq only, 1=recency only)", 0.0, 1.0, 0.30, 0.01)


# ---------------
# Load data
# ---------------

if hits_file is None:
    st.info("Upload your 5-year HIT history to see the master ranking.")
    st.stop()

hits_df = load_history(hits_file, filter_families=True)

if hits_df.empty:
    st.error("After parsing, the HITS file contains 0 rows matching families 3389/3889/3899.")
    st.stop()

file_start = hits_df["date"].min().normalize()
file_end = hits_df["date"].max().normalize()

# Default As-Of = file end (or user override)
if asof_date is None:
    asof_ts = file_end
else:
    asof_ts = pd.Timestamp(asof_date)

analysis_end = asof_ts.normalize() if assume_no_hits_after else file_end
analysis_start = file_start

combine_mode = "multiply" if schedule_mode.lower().startswith("multiply") else "average"

st.caption(
    f"History USED: {analysis_start.date()} → {file_end.date()} (file end) | "
    f"As-Of scoring date: {asof_ts.date()} | Analysis window end: {analysis_end.date()} | "
    f"days_window={(analysis_end.date() - analysis_start.date()).days + 1} | "
    f"streams found: {hits_df.groupby(['state','game']).ngroups}"
)

# Playable list
playable_df = load_playable_list(playable_file) if playable_file is not None else pd.DataFrame(columns=["state", "game"])
playable_set = set()
if not playable_df.empty:
    playable_set = set(zip(playable_df["state"], playable_df["game"]))

# Compute per-stream stats + score
stats_df = compute_stream_stats(
    hits_df,
    analysis_start=analysis_start,
    analysis_end=analysis_end,
    schedule_laplace_alpha=float(schedule_laplace_alpha),
    schedule_combine_mode=combine_mode,
)

scored = score_master_table(
    stats_df,
    w_hit=w_hit,
    w_due=w_due,
    w_sched=w_sched,
    w_cons=w_cons,
    w_rel=w_rel,
)

if not scored.empty:
    scored["PlayableByUser"] = np.where(
        scored.apply(lambda r: (r["State"], r["Game"]) in playable_set, axis=1),
        "Yes",
        "No",
    )


# Prev-day winners context (1-6-5 method)
prevday_df = pd.DataFrame()
if "prevday_files" in locals() and prevday_files:
    prev_parts = []
    for f in prevday_files:
        try:
            prev_parts.append(load_history(f, filter_families=False))
        except Exception as e:
            st.warning(f"Could not parse prev-day file '{getattr(f, 'name', 'uploaded')}'. ({e})")
    if prev_parts:
        prevday_df = pd.concat(prev_parts, ignore_index=True)

if not prevday_df.empty:
    scored = attach_prevday_context(scored, prevday_df, analysis_end)

# Always create a 'ScoreWithPrevDay' column for convenience
if not scored.empty:
    scored["ScoreWithPrevDay"] = scored["Score"]

    if not prevday_df.empty and prevday_mode == "Boost ordering (recommended)":
        w = float(prevday_weight or 0.0)
        scored["ScoreWithPrevDay"] = (scored["Score"] + w * scored["PrevDrawScore"].fillna(0.0)).clip(0.0, 1.0)

    # Sort for display: Playable first (if provided), then boosted score, then base score
    scored["_playable_rank"] = (scored.get("PlayableByUser", "No") == "Yes").astype(int)
    scored = scored.sort_values(["_playable_rank", "ScoreWithPrevDay", "Score"], ascending=[False, False, False])
    scored.drop(columns=["_playable_rank"], inplace=True, errors="ignore")


# -----------------------------
# Master ranking display
# -----------------------------

st.header("A) Master Ranking — All States / All Games (Most → Least Likely)")

# Show a compact explanation
with st.expander("What this score is doing (quick)", expanded=False):
    st.markdown(
        """

    # --- parameter aliasing / back-compat ---
    # If caller passed legacy names (laplace_alpha/recency_half_life_days/blend_recency), use them.
    if laplace_alpha is not None:
        laplace_laplace_alpha = laplace_alpha
    if recency_half_life_days is not None:
        recency_recency_half_life_days_days = recency_half_life_days
    if blend_recency is not None:
        blend_recency = blend_recency

    # Normalize types/ranges defensively
    laplace_laplace_alpha = float(laplace_laplace_alpha)
    recency_recency_half_life_days_days = int(recency_recency_half_life_days_days)
    blend_recency = float(blend_recency)
    if blend_recency < 0:
        blend_recency = 0.0
    if blend_recency > 1:
        blend_recency = 1.0

- **HitRate**: streams that hit these families more often (in your file window) score higher.
- **OverduePercentile (tempered)**: streams that are **"due"** based on their own historical *hit gaps* score higher.
- **ScheduleBoost**: boosts streams that historically hit more often on the **same weekday/month** as the As-Of date.
- **Consistency / Reliability**: small stabilizers so tiny samples don’t dominate.

This is *not* using the winning 12/27 results directly — it is using hit-gap behavior learned from the history you uploaded.
        """
    )

st.dataframe(scored, use_container_width=True, height=520)

csv_bytes = scored.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download master ranking CSV",
    data=csv_bytes,
    file_name="pk4_master_ranking_3389_3889_3899.csv",
    mime="text/csv",
)

# -----------------------------
# Straights learner display
# -----------------------------

st.header("B) Straight ordering learning (state-specific 12 straights per family)")

if not stream_files:
    st.info("Optional: upload a previous-day winners/results file above if you want straights learning (Top 12) to run.")
    stream_df = pd.DataFrame(columns=REQUIRED_COLS + ["family"])
    stream_stats = {"n_rows": 0, "by_family": {}, "by_state": {}, "by_game": {}, "by_state_game": {}}
    stream_scores = {}
else:
    stream_dfs = []
    for f in stream_files:
        try:
            t = _safe_decode(f.getvalue())
            sdf = _parse_txt_lines(t)
            if sdf is not None and len(sdf) > 0:
                # normalize columns to required schema
                tmp = sdf.copy()
                tmp.columns = [c.strip().lower() for c in tmp.columns]
                if "result" in tmp.columns:
                    tmp["result"] = tmp["result"].astype(str).str.strip().str.zfill(4)
                    tmp = tmp[tmp["result"].str.match(r"^\d{4}$", na=False)]
                # align required cols; fill missing with blanks
                for c in REQUIRED_COLS:
                    if c not in tmp.columns:
                        tmp[c] = ""
                tmp = tmp[REQUIRED_COLS].copy()
                tmp["family"] = tmp["result"].apply(box_key)
                stream_dfs.append(tmp)
        except Exception:
            continue

    if not stream_dfs:
        st.warning("No usable rows found in the previous-day file(s). Straights learning will be skipped.")
        stream_df = pd.DataFrame(columns=REQUIRED_COLS + ["family"])
        stream_stats = {"n_rows": 0, "by_family": {}, "by_state": {}, "by_game": {}, "by_state_game": {}}
        stream_scores = {}
    else:
        stream_df = pd.concat(stream_dfs, ignore_index=True)
        stream_df = stream_df[stream_df["result"].str.match(r"^\d{4}$", na=False)].copy()
        stream_df["family"] = stream_df["result"].apply(box_key)

        stream_stats = compute_stream_stats(
            stream_df,
            asof=analysis_end,
        )
        stream_scores = build_straight_scores(
            stream_df,
            asof=analysis_end,
            recency_half_life_days=int(straight_recency_half_life_days),
            laplace_alpha=float(straight_laplace_alpha),
            blend_recency_frequency=float(straight_mix),
        )

st.subheader("C) Straight ordering (Top 12) by State/Game (optional)")
if not stream_scores:
    st.info("No previous-day winners file uploaded (or no usable rows). Straight ordering is skipped.")
else:
    colA, colB = st.columns([1, 2], vertical_alignment="top")
    with colA:
        sel_state = st.selectbox(
            "State",
            options=sorted(stream_df["state"].dropna().unique().tolist()),
            index=0,
            key="straight_state",
        )
        games_for_state = sorted(stream_df.loc[stream_df["state"] == sel_state, "game"].dropna().unique().tolist())
        sel_game = st.selectbox(
            "Game",
            options=games_for_state,
            index=0,
            key="straight_game",
        )

    with colB:
        rows = []
        for fam in DEFAULT_FAMILIES:
            tbl = top_straights_table_for_family(
                family=fam,
                stream_scores=stream_scores,
                state=sel_state,
                game=sel_game,
                top_n=12,
                asof=analysis_end,
            )
            tbl.insert(0, "Family", fam)
            tbl.insert(1, "State", sel_state)
            tbl.insert(2, "Game", sel_game)
            rows.append(tbl)

        bundle = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        st.dataframe(bundle, use_container_width=True, hide_index=True)

        st.download_button(
            "Download straight ranking CSV for selected State/Game",
            data=bundle.to_csv(index=False).encode("utf-8"),
            file_name=f"pk4_straight_ranking_{sel_state}_{sel_game}.csv".replace(" ", "_").replace("/", "-"),
            mime="text/csv",
        )