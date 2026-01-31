# streamlit_app.py
# Pick-4 (active families families) prediction helper:
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

st.set_page_config(page_title="Pick 4 Families — Master Ranking + Straights", layout="wide")

# -----------------------------
# Families
# -----------------------------
# The app originally targeted the families box families.
# It now supports many families, each optionally with its own learner parameters.
# A "family" in this app is the box-sorted 4-digit key (e.g., 3899, 0199, 0013).

DEFAULT_CORE_FAMILIES: set[str] = {"3389", "3889", "3899"}

def canonical_family_key(x: str) -> str:
    """Normalize to a 4-digit box-key: digits only, zero-pad to 4, then sort."""
    if x is None:
        return ""
    s = str(x).strip()
    # keep digits only
    digs = "".join([c for c in s if c.isdigit()])
    if not digs:
        return ""
    # If the source drops leading zeros (e.g., '199' for '0199'), restore them
    if len(digs) < 4:
        digs = digs.zfill(4)
    # If longer, keep the last 4 digits (rare, but avoids crashes)
    if len(digs) > 4:
        digs = digs[-4:]
    return "".join(sorted(digs))

# Default families used when no external family list is provided (back-compat).
DEFAULT_FAMILIES: set[str] = set(DEFAULT_CORE_FAMILIES)
FAMILIES: list[str] = sorted(DEFAULT_FAMILIES)

# Back-compat: older code paths referenced `sorted_str()` for box-keying.
def sorted_str(num4: str) -> str:
    return box_key(num4)

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


def extract_4digit_result(s: str) -> Optional[str]:
    """Extract a Pick-4 result from a messy field.

    Primary goal: reliably recover *four digits* even when the source is messy.
    Secondary fallback: if the field clearly contains only *three* digits (common when a
    leading zero is omitted), left-pad with a zero.

    Examples handled:
      - 3-9-3-8 -> 3938
      - 3938 -> 3938
      - 3-9-3-8, Fireball: 9 -> 3938
      - 389 -> 0389   (only when the field is essentially just those three digits)
    """
    if s is None:
        return None
    txt = str(s)
    digits = DIGIT_RE.findall(txt)
    if len(digits) >= 4:
        return "".join(digits[:4])

    # Fallback: allow 3-digit results only when the field is basically just that number
    # (e.g., "389" or "3 8 9"), to avoid accidentally grabbing digits from dates, etc.
    if len(digits) == 3:
        compact = re.sub(r"[^0-9]", "", txt)
        if len(compact) == 3 and compact == "".join(digits):
            return "0" + compact

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


# -----------------------------
# Structure + Core family helpers
# -----------------------------

def structure_from_member_key(member_key: str) -> str:
    """Return structure for a 4-digit BOX key (sorted digits), e.g. 3389 -> AABC."""
    s = str(member_key).strip()
    if len(s) != 4 or not s.isdigit():
        return "UNKNOWN"
    counts = sorted([s.count(ch) for ch in set(s)], reverse=True)
    if counts == [4]:
        return "AAAA"   # quads
    if counts == [3, 1]:
        return "AAAB"   # triples
    if counts == [2, 2]:
        return "AABB"   # double-doubles
    if counts == [2, 1, 1]:
        return "AABC"   # doubles
    if counts == [1, 1, 1, 1]:
        return "ABCD"   # all distinct
    return "OTHER"


def core_from_member_key(member_key: str) -> str:
    """Core key is the sorted unique digits. Length depends on structure."""
    s = str(member_key).strip()
    if len(s) != 4 or not s.isdigit():
        return ""
    return "".join(sorted(set(s)))


def members_from_core(structure: str, core: str) -> List[str]:
    """Generate BOX-member keys (sorted digits) for a core family."""
    core = "".join([c for c in str(core).strip() if c.isdigit()])
    structure = str(structure).strip().upper()
    if structure in ("AABC", "AAAB") and len(core) != 3:
        return []
    if structure == "AABB" and len(core) != 2:
        return []
    if structure == "AAAA" and len(core) != 1:
        return []
    digits = list(core)

    out = []
    if structure == "AABC":
        for rep in digits:
            others = [d for d in digits if d != rep]
            if len(others) != 2:
                continue
            out.append(box_key(rep + rep + others[0] + others[1]))
    elif structure == "AAAB":
        for trip in digits:
            others = [d for d in digits if d != trip]
            for single in others:
                out.append(box_key(trip + trip + trip + single))
    elif structure == "AABB":
        a, b = digits
        out.append(box_key(a + a + b + b))
    elif structure == "AAAA":
        d = digits[0]
        out.append(d * 4)
    elif structure == "ABCD":
        if len(core) == 4:
            out.append(box_key(core))
    return sorted(set(out))


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

    # Broader aliases seen in lottery exports
    # Date
    for dc in ("draw_date", "draw date", "date_drawn", "drawingdate", "drawing_date"):
        if dc in df.columns and "date" not in df.columns:
            df = df.rename(columns={dc: "date"})
            break
    # State
    for sc in ("jurisdiction", "state_name", "state/province", "st"):
        if sc in df.columns and "state" not in df.columns:
            df = df.rename(columns={sc: "state"})
            break
    # Game
    for gc in ("game_name", "game type", "lottery", "product", "game_type"):
        if gc in df.columns and "game" not in df.columns:
            df = df.rename(columns={gc: "game"})
            break
    # Result / Winning numbers
    for rc in (
        "winningnumbers",
        "winning_numbers",
        "winning numbers",
        "numbers",
        "result",
        "results",
        "winningnumber",
        "winning_number",
        "win",
        "winning",
    ):
        if rc in df.columns and "result" not in df.columns:
            df = df.rename(columns={rc: "result"})
            break

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


def load_history(uploaded, filter_families: bool = True, *, keep_families: set[str] | None = None, master_streams: list[tuple[str, str]] | None = None, required_cols: list[str] | None = None) -> pd.DataFrame:
    """Load either CSV or TXT into a canonical df with required columns.

    - MASTER history: filter_families=True (keeps only the families box families by default)
    - STREAM history: filter_families=False (keeps all results, used to learn ordering)
    """
    if uploaded is None:
        return pd.DataFrame(columns=REQUIRED_COLS)

    name = (getattr(uploaded, "name", "") or "").lower()
    text = _safe_decode(uploaded)

    df: Optional[pd.DataFrame] = None

    if name.endswith(".csv"):
        df = _parse_csv_flexible(text)
        if df is None or df.empty:
            df = _parse_txt_lines(text, master_streams=master_streams, filename=getattr(uploaded, "name", None))
    elif name.endswith(".txt"):
        # Many “.txt” uploads in this workflow are actually TSV/CSV exports.
        # If we parse as free-form text first, we can mistakenly extract the year
        # (e.g., 2025) as the 4-digit "result" and then filter everything out.
        # So: try tabular parsing first, and only fall back to line parsing if needed.
        df_csv = _parse_csv_flexible(text)
        if df_csv is not None and not df_csv.empty and df_csv.shape[1] >= 4:
            df = df_csv
        else:
            # Strong TSV fallback (headerless exports are common).
            df_tsv = None
            if "\t" in text:
                try:
                    df_tsv = pd.read_csv(io.StringIO(text), sep="\t", header=None)
                except Exception:
                    df_tsv = None
            if df_tsv is not None and not df_tsv.empty and df_tsv.shape[1] >= 4:
                df_tsv = df_tsv.iloc[:, :4].copy()
                df_tsv.columns = ["date", "state", "game", "result"]
                df = df_tsv
            else:
                df = _parse_txt_lines(text, master_streams=master_streams, filename=getattr(uploaded, "name", None))
                if df is None or df.empty:
                    df = df_csv
    else:
        df = _parse_csv_flexible(text)
        if df is None or df.empty:
            df = _parse_txt_lines(text, master_streams=master_streams, filename=getattr(uploaded, "name", None))

    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_COLS)

    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    # Columns we will keep from the parsed file (plus derived fields).
    base_cols = list(required_cols) if required_cols else list(REQUIRED_COLS)
    for c in ("date", "state", "game", "result"):
        if c not in base_cols:
            base_cols.append(c)

    # Normalize column names
    col_map = {}
    for c in df.columns:
        lc = c.strip().lower()

        # Exact matches (common exports)
        if lc in ("date", "drawdate", "draw_date", "draw date"):
            col_map[c] = "date"
            continue
        if lc in ("state", "jurisdiction"):
            col_map[c] = "state"
            continue
        if lc in ("game", "draw", "drawtime", "draw_time", "draw time", "game/draw"):
            # Some exports include a separate "Draw" column (Morning/Evening).
            # If a proper "Game" column is present, do NOT overwrite it with the draw-time column.
            if lc == "draw" and "game" in col_map.values():
                continue
            col_map[c] = "game"
            continue
        if lc in ("result", "numbers", "winningnumbers", "winning_numbers", "winning numbers", "winning_number", "winning number"):
            col_map[c] = "result"
            continue

        # Fuzzy matches for exports like "Date (EST)" or "Result -> Fireball"
        if "date" in lc and c not in col_map:
            col_map[c] = "date"
            continue
        if ("state" in lc or lc.startswith("st")) and c not in col_map:
            col_map[c] = "state"
            continue
        # Avoid mapping columns like "draw" or "draw time" to game unless they look like a game label
        if ("game" in lc or "draw time" in lc or lc.startswith("game/")) and c not in col_map:
            col_map[c] = "game"
            continue
        if "result" in lc or "winning" in lc or "numbers" in lc:
            col_map[c] = "result"
            continue

    df = df.rename(columns=col_map)

    # Ensure required columns exist (some parsers return Title-case)
    if "date" not in df.columns and "Date" in df.columns:
        df["date"] = df["Date"]
    if "state" not in df.columns and "State" in df.columns:
        df["state"] = df["State"]
    if "game" not in df.columns and "Game" in df.columns:
        df["game"] = df["Game"]
    if "result" not in df.columns and "Result" in df.columns:
        df["result"] = df["Result"]

    if "date" not in df.columns or "result" not in df.columns:
        # Tab-delimited export fallback (common for LotteryPost-style text exports)
        if "	" in text:
            try:
                df_tab = pd.read_csv(io.StringIO(text), sep="	", header=None, dtype=str)
                # Heuristic: common layouts are 4 cols (date, state, game, result) or 6 cols (rank, date, state, game, draw, result).
                if df_tab.shape[1] == 4:
                    df_tab.columns = ["date", "state", "game", "result"]
                elif df_tab.shape[1] == 5:
                    df_tab.columns = ["date", "state", "game", "draw", "result"]
                elif df_tab.shape[1] >= 6:
                    cols = ["rank", "date", "state", "game", "draw", "result"] + [f"extra_{k}" for k in range(df_tab.shape[1]-6)]
                    df_tab.columns = cols[: df_tab.shape[1]]
                df = df_tab
            except Exception:
                pass
        if "date" not in df.columns or "result" not in df.columns:
            return pd.DataFrame(columns=REQUIRED_COLS)

    # Fill missing state/game (single-stream exports)
    if "state" not in df.columns:
        df["state"] = "(Unknown)"
    if "game" not in df.columns:
        df["game"] = "(Unknown)"

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])

    df["result"] = df["result"].astype(str).apply(extract_4digit_result)
    df = df.dropna(subset=["result"])

    df["state"] = df["state"].astype(str)
    df["game"] = df["game"].astype(str)

    # Filter families only for MASTER history, not for STREAM learning
        # Canonicalize columns to match REQUIRED_COLS (lowercase)
    # (Some parsers may return Title-case; normalize both directions.)
    df = df.rename(columns={"Date": "date", "State": "state", "Game": "game", "Result": "result"})
    df = df[base_cols].copy()

    # Derive family key (sorted digits) for later stats and optional filtering.
    # NOTE: family is the box-sorted key (e.g., active families).
    df["family"] = df["result"].astype(str).apply(sorted_str)

    df["structure"] = df["family"].apply(structure_from_member_key)
    df["core"] = df["family"].apply(core_from_member_key)
    if filter_families:
        families_keep = keep_families if keep_families is not None else DEFAULT_FAMILIES
        df = df[df["family"].isin(families_keep)].copy()
    df = df.sort_values(["state", "game", "date"]).reset_index(drop=True)
    return df


def load_playable_list(uploaded) -> pd.DataFrame:
    """Load playable streams list. Accepts TXT or CSV.

    Must include State and Game either as header or as 2 columns.
    """
    if uploaded is None:
        return pd.DataFrame(columns=["state", "game"])

    text = _safe_decode(uploaded)
    name = (uploaded.name or "").lower()

    df: Optional[pd.DataFrame] = None
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(io.StringIO(text))
        except Exception:
            df = None
    if df is None:
        # TXT or fallback: parse as simple lines
        rows = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith("state") and "game" in low:
                continue
            if "\t" in line:
                parts = [p.strip() for p in line.split("\t") if p.strip()]
            else:
                # try comma, else 2+ spaces
                if "," in line:
                    parts = [p.strip() for p in line.split(",") if p.strip()]
                else:
                    parts = [p.strip() for p in re.split(r"\s{2,}", line) if p.strip()]
            if len(parts) < 2:
                continue
            rows.append({"state": normalize_state(parts[0]), "game": normalize_game(parts[1])})
        df = pd.DataFrame(rows)

    df = _standardize_columns(df)
    # Accept either state/game or State/Game
    if "state" not in df.columns or "game" not in df.columns:
        # If exactly 2 cols, map them
        if df.shape[1] >= 2:
            cols = list(df.columns)
            df = df.rename(columns={cols[0]: "state", cols[1]: "game"})

    if "state" not in df.columns or "game" not in df.columns:
        return pd.DataFrame(columns=["state", "game"])

    df = df[["state", "game"]].copy()
    df["state"] = df["state"].astype(str).map(normalize_state)
    df["game"] = df["game"].astype(str).map(normalize_game)
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    return df


# -----------------------------
# Scoring (Master Ranking)
# -----------------------------


# -----------------------------
# Family priors + recommendation
# -----------------------------

def load_family_counts_csv(uploaded) -> pd.DataFrame:
    """Load the optional family-counts CSV used as a global frequency prior.

    Expected columns (case-insensitive):
      - box_key (or family)
      - count
    Optional:
      - members
      - example
    """
    if uploaded is None:
        return pd.DataFrame(columns=["family", "count", "prior_prob"])

    try:
        text = _safe_decode(uploaded)
        df = pd.read_csv(io.StringIO(text))
    except Exception:
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            return pd.DataFrame(columns=["family", "count", "prior_prob"])

    if df is None or df.empty:
        return pd.DataFrame(columns=["family", "count", "prior_prob"])

    cols = {c.lower().strip(): c for c in df.columns}
    key_col = cols.get("box_key") or cols.get("family") or cols.get("boxkey") or cols.get("box")
    cnt_col = cols.get("count") or cols.get("hits") or cols.get("freq") or cols.get("frequency")

    if key_col is None or cnt_col is None:
        return pd.DataFrame(columns=["family", "count", "prior_prob"])

    out = df[[key_col, cnt_col]].copy()
    out.columns = ["family", "count"]
    out["family"] = out["family"].apply(canonical_family_key)
    out = out[out["family"].str.fullmatch(r"\d{4}", na=False)]
    out["count"] = pd.to_numeric(out["count"], errors="coerce").fillna(0).astype(int)
    out = out.groupby("family", as_index=False)["count"].sum()
    total = int(out["count"].sum())
    if total <= 0:
        out["prior_prob"] = 0.0
    else:
        out["prior_prob"] = out["count"] / float(total)
    return out.sort_values(["count", "family"], ascending=[False, True]).reset_index(drop=True)


def _recency_score(days_since: float, half_life_days: float, mode: str) -> float:
    """Return a 0..1 score from a days-since metric."""
    if days_since is None or (isinstance(days_since, float) and np.isnan(days_since)):
        return 0.0
    d = float(max(0.0, days_since))
    # Hot = recently seen gets higher score. Due = long-unseen gets higher score.
    hot = float(exp_decay(d, half_life_days))
    if mode == "Hot":
        return hot
    if mode == "Due":
        return 1.0 - hot
    return 0.0


def recommend_families(
    hits_df: pd.DataFrame,
    priors_df: pd.DataFrame,
    *,
    asof: pd.Timestamp,
    base_streams_only: set[tuple[str, str]] | None,
    top_n: int = 15,
    w_prior: float = 0.40,
    w_hist: float = 0.40,
    w_recency: float = 0.20,
    recency_mode: str = "Hot",
    recency_half_life_days: int = 60,
) -> pd.DataFrame:
    """Recommend families using a blended score.

    - prior: global 3-year count file (optional)
    - hist: hit-rate in the HITS file (optionally restricted to Playable streams)
    - recency: hot/due boost based on last seen date in the HITS file
    """
    if hits_df is None or hits_df.empty:
        # If no hits file, fall back to priors only
        if priors_df is None or priors_df.empty:
            return pd.DataFrame(columns=["family", "score", "prior_prob", "hist_prob", "days_since", "recency_score"])
        out = priors_df.copy()
        out["hist_prob"] = 0.0
        out["days_since"] = np.nan
        out["recency_score"] = 0.0
        out["score"] = out["prior_prob"]
        return out.sort_values(["score", "count", "family"], ascending=[False, False, True]).head(top_n).reset_index(drop=True)

    df = hits_df.copy()
    if "family" not in df.columns:
        df["family"] = df["result"].astype(str).apply(sorted_str)

    if base_streams_only:
        df = df[df.apply(lambda r: (r["state"], r["game"]) in base_streams_only, axis=1)]

    if df.empty:
        return recommend_families(pd.DataFrame(), priors_df, asof=asof, base_streams_only=None, top_n=top_n, w_prior=w_prior, w_hist=w_hist, w_recency=w_recency, recency_mode=recency_mode, recency_half_life_days=recency_half_life_days)

    vc = df["family"].value_counts()
    total = int(vc.sum())
    hist = vc.rename("count_hist").reset_index().rename(columns={"index": "family"})
    hist["hist_prob"] = hist["count_hist"] / float(total) if total > 0 else 0.0

    last = df.groupby("family")["date"].max().rename("last_seen").reset_index()
    merged = hist.merge(last, on="family", how="left")

    merged["days_since"] = (pd.to_datetime(asof).normalize() - pd.to_datetime(merged["last_seen"]).dt.normalize()).dt.days
    merged["recency_score"] = merged["days_since"].apply(lambda d: _recency_score(d, recency_half_life_days, recency_mode))

    if priors_df is not None and not priors_df.empty:
        merged = merged.merge(priors_df[["family", "count", "prior_prob"]], on="family", how="left")
    else:
        merged["count"] = 0
        merged["prior_prob"] = 0.0

    merged["prior_prob"] = merged["prior_prob"].fillna(0.0)
    merged["hist_prob"] = merged["hist_prob"].fillna(0.0)

    # Normalize weights if user sets funky values
    wsum = float(max(1e-9, (w_prior + w_hist + w_recency)))
    wp, wh, wr = float(w_prior)/wsum, float(w_hist)/wsum, float(w_recency)/wsum

    merged["score"] = wp * merged["prior_prob"] + wh * merged["hist_prob"] + wr * merged["recency_score"]

    merged = merged.sort_values(["score", "count_hist", "family"], ascending=[False, False, True]).reset_index(drop=True)
    return merged.head(int(max(1, top_n))).copy()


def exp_decay(days: float, recency_half_life_days: float) -> float:
    if recency_half_life_days <= 0:
        return 0.0
    return float(math.exp(-math.log(2) * (days / recency_half_life_days)))


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


@dataclass
class StreamStats:
    state: str
    game: str
    hits: int
    last_hit_date: Optional[pd.Timestamp]
    days_since_last_hit: Optional[int]
    hit_rate: float
    consistency: float
    reliability: float
    share_3389: float
    share_3889: float
    share_3899: float
    top_member: str
    top_member_share: float
    member_mix: str
    overdue_percentile: float
    due_tempered: float
    schedule_boost: float
    expected_gap_days: Optional[float]
    predicted_next_hit: Optional[pd.Timestamp]


def compute_schedule_boost(dates: Sequence[pd.Timestamp],
                           target: pd.Timestamp,
                           laplace_alpha: float,
                           combine_mode: str) -> float:
    if len(dates) == 0:
        return 0.0

    # weekday: 0=Mon..6=Sun
    wds = [int(d.weekday()) for d in dates]
    months = [int(d.month) for d in dates]

    total = len(dates)
    wd_counts = np.bincount(wds, minlength=7)
    mo_counts = np.bincount(months, minlength=13)  # 1..12 used

    wd_prob = safe_div(wd_counts[target.weekday()] + laplace_alpha, total + laplace_alpha * 7)
    mo_prob = safe_div(mo_counts[target.month] + laplace_alpha, total + laplace_alpha * 12)

    if combine_mode.lower().startswith("mult"):
        return float(wd_prob * mo_prob)
    # default: average
    return float((wd_prob + mo_prob) / 2.0)


def compute_stream_stats(df: pd.DataFrame,
                         analysis_start: pd.Timestamp,
                         analysis_end: pd.Timestamp,
                         schedule_laplace_alpha: float,
                         schedule_combine_mode: str) -> pd.DataFrame:
    """Compute stats per (state, game) using ONLY the hits history."""

    # Ensure family column exists (box-sorted key) for per-family breakdowns.
    if "family" not in df.columns:
        df = df.copy()
        df["family"] = df["result"].astype(str).apply(sorted_str)

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

        # Member shares (within the selected hit families)
        counts_by_family = g["family"].value_counts().to_dict()

        # Keep legacy 389-member shares for backwards compatibility (0 if not present)
        share_3389 = safe_div(counts_by_family.get("3389", 0), hits)
        share_3889 = safe_div(counts_by_family.get("3889", 0), hits)
        share_3899 = safe_div(counts_by_family.get("3899", 0), hits)

        # Generic mix string + top-member share (works for any family set)
        if counts_by_family:
            top_member, top_cnt = max(counts_by_family.items(), key=lambda kv: kv[1])
            top_member_share = safe_div(top_cnt, hits)
            mix_parts = []
            for k, v in sorted(counts_by_family.items(), key=lambda kv: kv[1], reverse=True)[:5]:
                mix_parts.append(f"{k}:{safe_div(v, hits):.2f}")
            member_mix = ", ".join(mix_parts)
        else:
            top_member, top_member_share, member_mix = "", 0.0, ""


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
            "TopMember": top_member,
            "TopMemberShare": top_member_share,
            "MemberMix": member_mix,
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





def _unique_permutations_from_family(family_key: str) -> list[str]:
    """Return all unique straight orderings (permutations) for a 4-digit family box-key."""
    family_key = canonical_family_key(str(family_key))
    if not re.fullmatch(r"\d{4}", family_key or ""):
        return []
    digits = list(family_key)
    from itertools import permutations
    perms = sorted({"".join(p) for p in permutations(digits, 4)})
    return perms


def compute_straight_ordering_stats(stream_slice: pd.DataFrame, family_key: str) -> pd.DataFrame:
    """Counts straight orderings inside one family from STREAM history.

    Expects stream_slice to include a 4-digit string column named 'result'.
    Returns a dataframe with columns: Straight, Count, Pct.
    """
    if stream_slice is None or getattr(stream_slice, "empty", True):
        return pd.DataFrame(columns=["Straight", "Count", "Pct"])

    if "result" not in stream_slice.columns:
        return pd.DataFrame(columns=["Straight", "Count", "Pct"])

    s = stream_slice["result"].astype(str).str.strip()
    s = s[s.str.fullmatch(r"\d{4}", na=False)]
    if s.empty:
        return pd.DataFrame(columns=["Straight", "Count", "Pct"])

    counts = s.value_counts()
    total = int(counts.sum())
    orders = _unique_permutations_from_family(family_key)
    if not orders:
        # Fallback: just show what we saw
        out = (
            counts.rename_axis("Straight")
            .reset_index(name="Count")
            .assign(Pct=lambda d: d["Count"] / max(total, 1))
            .sort_values(["Count", "Straight"], ascending=[False, True])
            .reset_index(drop=True)
        )
        return out

    rows = []
    for o in orders:
        c = int(counts.get(o, 0))
        rows.append((o, c, (c / total) if total else 0.0))

    out = pd.DataFrame(rows, columns=["Straight", "Count", "Pct"])
    out = out.sort_values(["Count", "Straight"], ascending=[False, True]).reset_index(drop=True)
    return out

# -----------------------------
# UI
# -----------------------------


def build_12_straights_table(
    ordering_stats: pd.DataFrame,
    family_choice: str | int | None = None,
    model_probs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a stable table of all unique straights (permutations) for a family.

    `ordering_stats` is expected to include: Straight, Count, Pct (from compute_straight_ordering_stats).
    If `model_probs` is provided, it may include: Straight, Prob, Evidence.
    """
    fam = canonical_family_key(str(family_choice) if family_choice is not None else "")
    if not re.fullmatch(r"\d{4}", fam or ""):
        # Try to infer family from the first straight; fall back to the original core family.
        if isinstance(ordering_stats, pd.DataFrame) and not ordering_stats.empty and "Straight" in ordering_stats.columns:
            fam = canonical_family_key(str(ordering_stats["Straight"].iloc[0]))
        else:
            fam = "3389"

    digits = list(fam)
    all_straights = sorted(set("".join(p) for p in itertools.permutations(digits, 4)))

    stats = ordering_stats.copy() if isinstance(ordering_stats, pd.DataFrame) else pd.DataFrame()
    if stats.empty:
        stats = pd.DataFrame(columns=["Straight", "Count", "Pct"])
    if "Straight" not in stats.columns:
        stats["Straight"] = ""
    if "Count" not in stats.columns:
        stats["Count"] = 0
    if "Pct" not in stats.columns:
        stats["Pct"] = 0.0

    stats["Straight"] = stats["Straight"].astype(str)
    stats["Count"] = pd.to_numeric(stats["Count"], errors="coerce").fillna(0).astype(int)
    stats["Pct"] = pd.to_numeric(stats["Pct"], errors="coerce").fillna(0.0).astype(float)

    base = pd.DataFrame({"Straight": all_straights})
    merged = base.merge(stats[["Straight", "Count", "Pct"]], on="Straight", how="left")
    merged["Count"] = merged["Count"].fillna(0).astype(int)
    merged["Pct"] = merged["Pct"].fillna(0.0).astype(float)

    # Optional model probabilities
    if isinstance(model_probs, pd.DataFrame) and (not model_probs.empty) and ("Straight" in model_probs.columns):
        mp = model_probs.copy()
        mp["Straight"] = mp["Straight"].astype(str)
        if "Prob" in mp.columns:
            mp["ModelProb"] = pd.to_numeric(mp["Prob"], errors="coerce").fillna(0.0).astype(float)
        elif "ModelProb" not in mp.columns:
            mp["ModelProb"] = 0.0
        if "Evidence" not in mp.columns:
            mp["Evidence"] = ""
        mp = mp[["Straight", "ModelProb", "Evidence"]].drop_duplicates("Straight")
        merged = merged.merge(mp, on="Straight", how="left")
        merged["ModelProb"] = merged["ModelProb"].fillna(0.0).astype(float)
        merged["Evidence"] = merged["Evidence"].fillna("").astype(str)

    sort_cols = ["Count", "Straight"]
    asc = [False, True]
    if "ModelProb" in merged.columns:
        sort_cols = ["Count", "ModelProb", "Straight"]
        asc = [False, False, True]

    merged = merged.sort_values(sort_cols, ascending=asc).reset_index(drop=True)
    merged.insert(0, "Rank", merged.index + 1)

    cols = ["Rank", "Straight", "Count", "Pct"]
    if "ModelProb" in merged.columns:
        cols += ["ModelProb", "Evidence"]
    return merged[cols]

st.title("Pick 4 Families — Master Ranking + 12-Straights Learner")

with st.sidebar:
    st.header("Inputs")
    st.caption("HITS = 5-year hits list across all states/games. STREAM = per-state/game 24-month history for straight ordering.")

    st.markdown("### Families (optional)")
    family_counts_file = st.file_uploader(
        "Upload family list + 3-year frequency (CSV)",
        type=["csv"],
        help="CSV expected columns: box_key, count (plus optional members/example). Used as a global frequency prior for recommending families.",
    )
    st.caption("If you skip this, the app defaults to the original core families: 3389 / 3889 / 3899.")


    hits_file = st.file_uploader(
        "Upload 5-year HIT history (TXT or CSV)",
        type=["txt", "csv"],
        help="TXT can be tab-separated or space-separated. Must contain Date, State, Game, Result. Fireball/Wild Ball is okay.",
    )

    st.markdown("---")
    st.caption("STREAM = per-state/game 24-month history for straight ordering learner.")
    stream_files_sidebar = st.file_uploader(
        "Upload 24-month STREAM history (TXT or CSV — you can upload multiple)",
        type=["txt", "csv"],
        accept_multiple_files=True,
    )
    st.markdown("---")
    st.subheader("As-Of scoring date")
    asof_date = st.date_input("As-Of Date", value=None)
    assume_no_hits_after = st.checkbox(
        "Assume there were NO hits after the last file date through As-Of Date",
        value=True,
        help="If checked, days-since-last-hit is computed up to As-Of Date even when your file stops earlier.",
    )

    st.subheader("History window (ranking)")
    window_label = st.selectbox(
        "Use results from:",
        ["Last 180 days (default)", "Last 365 days", "All available"],
        index=0,
        key="history_window",
    )
    window_days = 180 if window_label.startswith("Last 180") else (365 if window_label.startswith("Last 365") else None)


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
prevday_file = prevday_files[0] if prevday_files else None
prevday_mode = st.selectbox(
    "How to use previous-day context",
    ["Off (add columns only)", "Boost ordering (recommended)"],
    index=0,
    key="prevday_mode",
)

# --- Core model defaults (must exist even when prev-day context is OFF) ---
# Earlier iterations accidentally defined schedule_* and weight variables only
# inside the "Boost ordering" branch, which caused NameError crashes when the
# optional 1-6-5 component was left Off.

# Schedule learning defaults (used by straight ordering regardless of prev-day mode)
schedule_mode = "Multiply (weekday*month)"
# Back-compat: some earlier revisions referred to the straight Laplace slider as `laplace_alpha`.
# Define a safe default here so the app never crashes before the sidebar sliders run.
laplace_alpha = float(st.session_state.get("straight_laplace_alpha", 1.0))
schedule_laplace_alpha = float(st.session_state.get("schedule_laplace_alpha", laplace_alpha))

# Base evidence weights (used by the straight ordering learner)
w_hit = 0.50
w_stream = 0.30
w_sched = 0.10
w_cons = 0.08
w_rel = 0.02
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
st.caption("Weights are fixed to your approved mix (you can change later if you want).")

# These controls/vars must exist even when the optional prev-day component is OFF.
schedule_laplace_alpha = st.slider(
    "Schedule smoothing α (higher = weaker schedule boost / less overfit)",
    min_value=0.0,
    max_value=10.0,
    value=float(st.session_state.get("schedule_laplace_alpha", 2.0)),
    step=0.1,
    key="schedule_laplace_alpha",
)
schedule_mode = st.selectbox(
    "ScheduleBoost combine mode",
    options=["Multiply (weekday*month)", "Average (weekday+month)/2"],
    index=0,
    key="schedule_mode",
)

# Fixed weights (match your on-screen mix style)
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
st.caption("Upload 24-month STREAM history (TXT/CSV) in the sidebar to enable the 12-Straights Learner below.")

straight_recency_half_life_days = st.slider("Recency half-life (days)", 1, 365, 120)
straight_laplace_alpha = st.slider("Smoothing laplace_alpha (Laplace)", 0.0, 5.0, 1.0, 0.05)
straight_mix = st.slider("Blend recency vs frequency (0=freq only, 1=recency only)", 0.0, 1.0, 0.30, 0.01)


# ---------------
# Load data
# ---------------

if hits_file is None:
    st.info("Upload your 5-year HIT history to see the master ranking.")
    st.stop()

hits_df = load_history(hits_file, filter_families=False)

if hits_df.empty:
    st.error("After parsing, the HITS file contains 0 rows with valid 4-digit results.")
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
if window_days is not None:
    # Use a rolling window ending at the max date in the file (as-of).
    analysis_start = max(file_start, analysis_end - pd.Timedelta(days=window_days - 1))

# Apply the window filter for everything downstream (ranking + due logic + stats)
hits_df = hits_df[(hits_df["date"] >= pd.Timestamp(analysis_start)) & (hits_df["date"] <= (pd.Timestamp(analysis_end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)))]


# Safety: schedule_mode may not be defined in some builds; default to "multiply"
schedule_mode = st.session_state.get("schedule_mode", "Multiply")
schedule_mode = str(schedule_mode)
combine_mode = "multiply" if str(schedule_mode).lower().startswith("mult") else "average"

st.caption(
    f"History USED: {analysis_start.date()} → {analysis_end.date()} | File max date: {file_end.date()} | "
    f"As-Of scoring date: {asof_ts.date()} | days_window={(analysis_end.date() - analysis_start.date()).days + 1} | "
    f"streams found: {hits_df.groupby(['state','game']).ngroups}"
)

# Playable list
playable_df = load_playable_list(playable_file) if playable_file is not None else pd.DataFrame(columns=["state", "game"])
playable_set = set()
if not playable_df.empty:
    playable_set = set(zip(playable_df["state"], playable_df["game"]))

# -----------------------------
# Families selection (active families for this run)
# -----------------------------
priors_df = load_family_counts_csv(family_counts_file) if "family_counts_file" in locals() else pd.DataFrame(columns=["family","count","prior_prob"])

# Candidate families are anything observed in the HITS window plus anything listed in the priors file.
families_from_hits = sorted(set(hits_df["family"].dropna().astype(str).tolist())) if ("family" in hits_df.columns and not hits_df.empty) else []
families_from_priors = sorted(set(priors_df["family"].dropna().astype(str).tolist())) if (priors_df is not None and not priors_df.empty) else []
available_families = sorted(set(families_from_hits).union(families_from_priors))

# Sensible fallback if nothing is discoverable yet
if not available_families:
    available_families = sorted(DEFAULT_CORE_FAMILIES)

with st.sidebar:
    st.markdown("---")
    st.header("Families")
    st.caption("Active families affect the master ranking (what counts as a 'hit') and which families appear in the straights learner.")
    fam_mode = st.selectbox("Family selection mode", ["Auto recommend (top N)", "Manual select"], index=0)

    base_on_playable = False
    if playable_set:
        base_on_playable = st.checkbox("Base recommendations on Playable streams only", value=True)
    else:
        st.caption("Tip: upload a Playable list to base recommendations on only the streams you actually play.")

    recency_pref = st.radio("Recency preference", ["Hot", "Due", "Off"], index=0, horizontal=True)
    recency_half = st.slider("Family recency half-life (days)", 1, 365, 60)
    top_n_fams = st.slider("Top N families to recommend", 3, 50, 15)

    st.markdown("**Score weights** (auto mode)")
    w_prior = st.slider("Weight: Global prior (3-year file)", 0.0, 1.0, 0.40, 0.05)
    w_hist = st.slider("Weight: In-file frequency (HITS)", 0.0, 1.0, 0.40, 0.05)
    w_rec = st.slider("Weight: Recency boost", 0.0, 1.0, 0.20, 0.05)

# Build auto recommendations (even if user chooses manual, we can show the table)
rec_df = recommend_families(
    hits_df,
    priors_df,
            asof=asof_ts,
    base_streams_only=(playable_set if (base_on_playable and playable_set) else None),
    top_n=int(top_n_fams),
    w_prior=float(w_prior),
    w_hist=float(w_hist),
    w_recency=(0.0 if recency_pref == "Off" else float(w_rec)),
    recency_mode=("Hot" if recency_pref == "Hot" else ("Due" if recency_pref == "Due" else "Hot")),
    recency_half_life_days=int(recency_half),
)

default_active = list(DEFAULT_CORE_FAMILIES)
if fam_mode.startswith("Auto"):
    if rec_df is not None and not rec_df.empty and "family" in rec_df.columns:
        default_active = rec_df["family"].astype(str).tolist()
else:
    # In manual mode, default to the original core families if present, otherwise the top priors
    if not set(DEFAULT_CORE_FAMILIES).intersection(set(available_families)) and (priors_df is not None and not priors_df.empty):
        default_active = priors_df.head(min(10, len(priors_df)))["family"].astype(str).tolist()

with st.sidebar:
    active_families = st.multiselect(
        "Active families for this run",
        options=available_families,
        default=[f for f in default_active if f in available_families],
        help="These are 4-digit box-keys (sorted digits).",
    )
    # Safety fallback
    if not active_families:
        active_families = sorted(DEFAULT_CORE_FAMILIES)

st.markdown("**Overlay blend (Static vs Dynamic)**")
overlay_w_dynamic = st.slider(
    "Dynamic weight (higher = trust dynamic more)",
    min_value=0.0,
    max_value=1.0,
    value=0.65,
    step=0.05,
    help="This controls the Combined Overlay rank (Dynamic + Static). Static uses long-term stability; Dynamic uses recent + due pressure.",
)
st.session_state["overlay_w_dynamic"] = overlay_w_dynamic

# Optional override: pick a CORE family by structure (e.g., Doubles core "389" => members 3389/3889/3899)
st.markdown("**Core-family override (optional)**")
use_core_override = st.checkbox(
    "Use core-family selector (by structure)",
    value=False,
    help="Turn this on if you want to select families like 389 (core digits) instead of 4-digit box keys. The app will expand the core into the correct member box-keys automatically.",
)
active_core_label = ""
if use_core_override:
    structure_label = st.selectbox(
        "Structure (for core families)",
        ["AABC (Doubles)", "AAAB (Triples)", "AABB (Double-Doubles)", "ABCD (All distinct)", "AAAA (Quads)"],
        index=0,
    )
    structure_map = {
        "AABC (Doubles)": "AABC",
        "AAAB (Triples)": "AAAB",
        "AABB (Double-Doubles)": "AABB",
        "ABCD (All distinct)": "ABCD",
        "AAAA (Quads)": "AAAA",
    }
    structure_choice = structure_map.get(structure_label, "AABC")

    cores_avail = sorted(
        hits_df[hits_df.get("structure", "").astype(str) == structure_choice]["core"].dropna().astype(str).unique().tolist()
    ) if ("structure" in hits_df.columns and "core" in hits_df.columns) else []
    if not cores_avail:
        st.warning("No core families found yet for that structure (need a HITS file with results).")
    else:
        core_choice = st.selectbox("Core family digits", cores_avail, index=0)
        core_choice = str(core_choice).strip()
        core_members = members_from_core(structure_choice, core_choice)
        if core_members:
            active_families = core_members
            active_core_label = f"{structure_choice}:{core_choice}"
            st.caption("Expanded members: " + ", ".join(core_members))
        else:
            st.warning("That core selection did not produce any member keys.")

    use_family_param_overrides = st.checkbox(
        "Use per-family straights-learner parameters",
        value=False,
        help="If enabled, you can set different straight-ordering learner parameters per family. If off, the global sliders apply to every family.",
    )

    fam_params_df = pd.DataFrame()
    if use_family_param_overrides:
        seed_rows = []
        for fam in active_families:
            seed_rows.append({
                "family": fam,
                "recency_half_life_days": int(st.session_state.get("fam_param__half_life__" + fam, st.session_state.get("straight_recency_half_life_days", straight_recency_half_life_days))),
                "laplace_alpha": float(st.session_state.get("fam_param__laplace__" + fam, st.session_state.get("straight_laplace_alpha", straight_laplace_alpha))),
                "blend_recency": float(st.session_state.get("fam_param__blend__" + fam, st.session_state.get("straight_mix", straight_mix))),
            })
        fam_params_df = pd.DataFrame(seed_rows)

        edited = st.data_editor(
            fam_params_df,
            use_container_width=True,
            num_rows="fixed",
            hide_index=True,
            column_config={
                "family": st.column_config.TextColumn("Family", disabled=True),
                "recency_half_life_days": st.column_config.NumberColumn("Recency half-life (days)", min_value=1, max_value=365, step=1),
                "laplace_alpha": st.column_config.NumberColumn("Laplace alpha", min_value=0.0, max_value=10.0, step=0.05),
                "blend_recency": st.column_config.NumberColumn("Blend recency (0..1)", min_value=0.0, max_value=1.0, step=0.01),
            },
        )

        # Persist per-family values so switching families doesn't reset them.
        if isinstance(edited, pd.DataFrame) and not edited.empty:
            for _, r in edited.iterrows():
                fam = str(r.get("family", "")).strip()
                if fam:
                    st.session_state["fam_param__half_life__" + fam] = int(pd.to_numeric(r.get("recency_half_life_days", 120), errors="coerce") or 120)
                    st.session_state["fam_param__laplace__" + fam] = float(pd.to_numeric(r.get("laplace_alpha", 1.0), errors="coerce") or 1.0)
                    st.session_state["fam_param__blend__" + fam] = float(pd.to_numeric(r.get("blend_recency", 0.30), errors="coerce") or 0.30)

        st.session_state["family_params_overrides_enabled"] = True
    else:
        st.session_state["family_params_overrides_enabled"] = False


ACTIVE_FAMILIES = set(active_families)
FAMILIES = sorted(ACTIVE_FAMILIES)

# Show the recommendation table in the main page for transparency
with st.expander("Family recommendations (auto-scored)", expanded=False):
    if rec_df is None or rec_df.empty:
        st.write("No family recommendations available (need HITS file, or a family-counts CSV).")
    else:
        show_cols = [c for c in ["family","score","prior_prob","hist_prob","days_since","recency_score","count_hist","count"] if c in rec_df.columns]
        st.dataframe(rec_df[show_cols], use_container_width=True)

# Apply ACTIVE_FAMILIES for the master ranking definition of a "hit"
hits_df_active = hits_df[hits_df["family"].isin(ACTIVE_FAMILIES)].copy()

# Compute per-stream stats + score
stats_df = compute_stream_stats(
    hits_df_active,
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



# -------------------------------------------------
# Add STATIC vs DYNAMIC rank diagnostics (for on-screen comparison)
# StaticRank: where the stream ranks if you sort purely by historical family hits (Hits/HitRate)
# DynamicRank: today's rank after Score/PrevDay adjustments
# -------------------------------------------------
if not scored.empty:
    try:
        _static_rank_df = scored[["State","Game","Hits","HitRate"]].copy()
        _static_rank_df = _static_rank_df.sort_values(["Hits","HitRate"], ascending=[False, False]).reset_index(drop=True)
        _static_rank_df["StaticRank"] = np.arange(1, len(_static_rank_df) + 1)

        rank_map = dict(zip(zip(_static_rank_df["State"], _static_rank_df["Game"]), _static_rank_df["StaticRank"]))
        scored["_k"] = list(zip(scored["State"], scored["Game"]))
        scored["StaticRank"] = scored["_k"].map(rank_map).astype("Int64")

        # Dynamic rank is the current order (already sorted above)
        scored["DynamicRank"] = np.arange(1, len(scored) + 1).astype("Int64")

        scored["RankDelta_StaticMinusDynamic"] = (scored["StaticRank"] - scored["DynamicRank"]).astype("Int64")

        # Combined overlay: reward streams that score high in BOTH (dynamic + static)
        n_streams = max(1, len(scored))
        denom = (n_streams - 1) if n_streams > 1 else 1
        scored["StaticScore01"] = 1.0 - ((scored["StaticRank"].astype(float) - 1.0) / denom)
        scored["DynamicScore01"] = 1.0 - ((scored["DynamicRank"].astype(float) - 1.0) / denom)

        # User-controlled blend (default leans dynamic, but still respects static)
        overlay_w_dynamic = st.session_state.get("overlay_w_dynamic", 0.65)
        scored["OverlayScore"] = (overlay_w_dynamic * scored["DynamicScore01"]) + ((1.0 - overlay_w_dynamic) * scored["StaticScore01"])
        scored["OverlayRank"] = scored["OverlayScore"].rank(method="first", ascending=False).astype("Int64")

        # Convenience flags
        scored["Top20_Static"] = np.where(scored["StaticRank"] <= 20, "Yes", "No")
        scored["Top20_Dynamic"] = np.where(scored["DynamicRank"] <= 20, "Yes", "No")
        scored["Top20_Both"] = np.where((scored["StaticRank"] <= 20) & (scored["DynamicRank"] <= 20), "Yes", "No")

        scored.drop(columns=["_k"], inplace=True, errors="ignore")
    except Exception:
        # Never break the app if a column is missing
        pass


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


# Quick compare view: STATIC vs DYNAMIC ranks side-by-side (no need to open any exported list)
with st.expander("Static vs Dynamic comparison (recommended)", expanded=True):

    sort_choice = st.radio(
        "Sort this view by:",
        options=["DynamicRank", "OverlayRank", "StaticRank"],
        index=1 if ("OverlayRank" in scored.columns if scored is not None else False) else 0,
        horizontal=True,
        help="OverlayRank blends Dynamic + Static (see sidebar slider).",
    )
    topn = st.slider(
        f"Show top N streams (by {sort_choice})",
        min_value=10,
        max_value=min(78, len(scored) if not scored.empty else 78),
        value=min(50, len(scored) if not scored.empty else 50),
        step=1,
    )
    if scored is None or scored.empty:
        st.write("No ranking table available yet.")
    else:
        key_cols = [
            "DynamicRank","OverlayRank","StaticRank","RankDelta_StaticMinusDynamic",
            "Top20_Both","Top20_Dynamic","Top20_Static",
            "State","Game",
            "ScoreWithPrevDay","Score","OverlayScore",
            "Hits","HitRate",
            "DaysSinceLastHit","DueTempered","ScheduleBoost",
        ]
        # Optional columns if present
        for c in ["PrevDrawScore","PrevDrawResult","PrevDrawDate"]:
            if c in scored.columns:
                key_cols.append(c)

        show_cols = [c for c in key_cols if c in scored.columns]
        view_df = scored.sort_values(by=sort_choice, ascending=True).head(int(topn))
        st.dataframe(view_df[show_cols], use_container_width=True, height=520)

        # Small summary: overlap between static and dynamic top20
        if "Top20_Both" in scored.columns:
            both = int((scored["Top20_Both"] == "Yes").sum())
            st.caption(f"Overlap: {both} streams are Top‑20 in BOTH Static and Dynamic ranks for the current run.")


st.dataframe(scored, use_container_width=True, height=520)

csv_bytes = scored.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download master ranking CSV",
    data=csv_bytes,
    file_name="pk4_master_ranking_3389_3889_3899.csv",
    mime="text/csv",
)

# Also offer a TXT export (tab-separated) because it's easier to open/read than CSV in many setups
txt_bytes = scored.to_csv(index=False, sep="\t").encode("utf-8")
st.download_button(
    "Download master ranking TXT (tab-separated)",
    data=txt_bytes,
    file_name="pk4_master_ranking_3389_3889_3899.txt",
    mime="text/plain",
)


# -----------------------------
# Straights learner display
# -----------------------------

# -----------------------------
# B) Straights Learner
# -----------------------------
st.header("B) Straight ordering learning (12‑Straights Learner)")
st.caption(
    "Optional: upload 24‑month STREAM history in the sidebar to learn the most common straight orderings "
    "inside each family (families). This never affects the 5‑year HIT ranking unless you choose to use it."
)

stream_df = pd.DataFrame(columns=["date", "state", "game", "result", "digits", "family"])
if stream_files_sidebar:
    dfs = []
    for f in stream_files_sidebar:
        try:
            df_f = load_history(f, filter_families=False, required_cols=REQUIRED_COLS)
            dfs.append(df_f)
        except Exception as e:
            st.warning(f"STREAM file '{getattr(f, 'name', 'upload')}' could not be parsed and was skipped: {e}")
    if dfs:
        stream_df = pd.concat(dfs, ignore_index=True)
        # Normalize state/game for consistency
        stream_df["state_norm"] = stream_df["state"].map(normalize_state)
        stream_df["game_norm"] = stream_df["game"].map(normalize_game)
else:
    st.info("No STREAM file uploaded yet. Upload it in the left sidebar under **Inputs** to enable the 12‑Straights Learner.")

if stream_df.empty:
    st.warning("Straights Learner is idle (no STREAM rows loaded). The Master Ranking above still works.")
else:
    # Pick a stream (state+game) to learn straight orderings from
    available_streams = (
        stream_df[["state_norm", "game_norm"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["state_norm", "game_norm"])
    )
    stream_labels = [
        f"{s} — {g}" for s, g in zip(available_streams["state_norm"], available_streams["game_norm"])
    ]
    label_to_stream = dict(zip(stream_labels, zip(available_streams["state_norm"], available_streams["game_norm"])))

    colS1, colS2 = st.columns([2, 1])
    with colS1:
        chosen_label = st.selectbox(
            "Choose a stream for the 12‑Straights Learner",
            options=stream_labels,
            index=0,
            help="This uses the 24‑month STREAM history to learn straight ordering frequencies.",
        )
    with colS2:
        family_choice = st.selectbox("Family", options=FAMILIES, index=0)

    s_state, s_game = label_to_stream[chosen_label]
    stream_slice = stream_df[
        (stream_df["state_norm"] == s_state)
        & (stream_df["game_norm"] == s_game)
        & (stream_df["family"] == family_choice)
    ].copy()

    if stream_slice.empty:
        st.warning("No STREAM rows for that stream+family. Try a different stream or family.")
    else:
        ordering_stats = compute_straight_ordering_stats(stream_slice, family_choice)

        # Optional per-family parameter overrides (if enabled in the sidebar)
        use_over = bool(st.session_state.get("family_params_overrides_enabled", False))
        half = int(st.session_state.get("fam_param__half_life__" + family_choice, st.session_state.get("straight_recency_half_life_days", straight_recency_half_life_days))) if use_over else int(st.session_state.get("straight_recency_half_life_days", straight_recency_half_life_days))
        lap = float(st.session_state.get("fam_param__laplace__" + family_choice, st.session_state.get("straight_laplace_alpha", straight_laplace_alpha))) if use_over else float(st.session_state.get("straight_laplace_alpha", straight_laplace_alpha))
        blend = float(st.session_state.get("fam_param__blend__" + family_choice, st.session_state.get("straight_mix", straight_mix))) if use_over else float(st.session_state.get("straight_mix", straight_mix))

        model_rank = straight_ranking_for_stream(
            stream_df,
    s_state,
            s_game,
            family_choice,
            laplace_alpha=lap,
            recency_half_life_days=half,
            blend_recency=blend,
    asof=asof_ts,
        )

        straights_table = build_12_straights_table(ordering_stats, family_choice=family_choice, model_probs=model_rank)

        st.subheader("Straights output (all unique permutations for this family)")
        st.dataframe(straights_table, use_container_width=True)

        st.download_button(
            "Download Straights (CSV)",
            data=straights_table.to_csv(index=False).encode("utf-8"),
            file_name=f"straights_{s_state}_{s_game}_{family_choice}.csv".replace(" ", "_"),
            mime="text/csv",
        )

# -----------------------------
# C) Play lists & exclusions
# -----------------------------
st.header("C) Play lists, exclusions, and exports")

score_col = "ScoreWithPrevDay" if "ScoreWithPrevDay" in scored.columns else "Score"
work = scored.copy()

# Qualification flags
work["IsPlayable"] = True
if playable_set:
    work["IsPlayable"] = work["PlayableByUser"].fillna(False)

# 1-6-5 “bad-heavy” flag (only if prevday file uploaded)
work["PrevBadHeavy"] = False
if "PrevBadCount" in work.columns and "PrevGoodCount" in work.columns:
    work["PrevBadHeavy"] = (work["PrevBadCount"].fillna(0) >= 2) & (work["PrevGoodCount"].fillna(0) <= 2)

colC1, colC2, colC3 = st.columns([1, 1, 1])
with colC1:
    top_n = st.number_input("Top N streams to show", min_value=10, max_value=500, value=75, step=5)
with colC2:
    show_bottom = st.number_input("Bottom N streams to show (avoid list)", min_value=0, max_value=500, value=25, step=5)
with colC3:
    hide_not_playable = st.checkbox("Hide streams not on my Playable List", value=False, disabled=(not playable_set))

view = work.sort_values(score_col, ascending=False)

if hide_not_playable and playable_set:
    view = view[view["IsPlayable"]]

st.subheader("Recommended (Top N)")
top_view = view.head(int(top_n)).copy()
st.dataframe(top_view, use_container_width=True)

st.download_button(
    "Download Top N (CSV)",
    data=top_view.to_csv(index=False).encode("utf-8"),
    file_name="pk4_3389_3889_3899_top_streams.csv",
    mime="text/csv",
)

if int(show_bottom) > 0:
    st.subheader("Avoid / lowest-ranked (Bottom N)")
    bottom_view = view.tail(int(show_bottom)).sort_values(score_col, ascending=True).copy()
    st.dataframe(bottom_view, use_container_width=True)
    st.download_button(
        "Download Bottom N (CSV)",
        data=bottom_view.to_csv(index=False).encode("utf-8"),
        file_name="pk4_3389_3889_3899_bottom_streams.csv",
        mime="text/csv",
    )

# Explicit exclusions list (reasons)
exclusions = work.copy()
reasons = []
if playable_set:
    reasons.append(("Not on playable list", ~exclusions["IsPlayable"]))
if prevday_file is not None and "PrevBadHeavy" in exclusions.columns:
    reasons.append(("Bad-heavy previous draw (1-6-5)", exclusions["PrevBadHeavy"]))

if reasons:
    mask = False
    for _, msk in reasons:
        mask = mask | msk.fillna(False)
    excl = exclusions[mask].copy()

    def _reason_row(row):
        r = []
        for label, msk in reasons:
            if bool(msk.loc[row.name]):
                r.append(label)
        return "; ".join(r) if r else ""

    # Compute reasons per row (stable)
    excl["_exclude_reason"] = ""
    for label, msk in reasons:
        excl.loc[msk.fillna(False), "_exclude_reason"] = excl.loc[msk.fillna(False), "_exclude_reason"].apply(
            lambda s, lab=label: (s + "; " + lab).strip("; ").strip()
        )

    excl = excl.sort_values(score_col, ascending=False)

    st.subheader("Streams to consider excluding (with reasons)")
    st.dataframe(excl, use_container_width=True)
    st.download_button(
        "Download Exclusions (CSV)",
        data=excl.to_csv(index=False).encode("utf-8"),
        file_name="pk4_3389_3889_3899_exclusions.csv",
        mime="text/csv",
    )
else:
    st.info("No exclusion rules active (upload a Playable List and/or a previous‑day file to generate exclusions).")