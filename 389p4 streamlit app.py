# pick4_learning_ranker_FULL_TESTING_v2.py
# Run: streamlit run pick4_learning_ranker_FULL_TESTING_v2.py

import streamlit as st
import pandas as pd
import numpy as np
import itertools
import re

st.set_page_config(layout="wide")
st.title("Pick 4 Learning Ranker — Best States/Games + State-Specific 12 Straights (3389/3889/3899)")

# ------------------------------------------------------------
# Target digit families
# ------------------------------------------------------------
FAMILIES = {
    ("3", "3", "8", "9"): "3389",
    ("3", "8", "8", "9"): "3889",
    ("3", "8", "9", "9"): "3899",
}
FAMILY_NAMES = ["3389", "3889", "3899"]
REQUIRED_COLS = ["date", "state", "game", "result"]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def zfill4(x) -> str:
    s = str(x).strip()
    s = "".join([c for c in s if c.isdigit()])
    return s.zfill(4)[-4:]


def safe_to_datetime(series):
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)


def infer_family(num4: str):
    digs = tuple(sorted(list(zfill4(num4))))
    return FAMILIES.get(digs)


def month_key(dt: pd.Timestamp) -> str:
    return f"{dt.year}-{dt.month:02d}"


def normalize(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    denom = s.max() - s.min()
    if denom == 0:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - s.min()) / (denom + 1e-12)


def perms_for_family(fam_key):
    return sorted(set("".join(p) for p in itertools.permutations(fam_key, 4)))  # 12 perms


def recency_weight(days_since: float, half_life_days: float) -> float:
    if days_since is None or np.isinf(days_since):
        return 0.0
    half_life_days = max(float(half_life_days), 1.0)
    return float(0.5 ** (float(days_since) / half_life_days))


def load_any_csv(file_obj):
    try:
        return pd.read_csv(file_obj, sep=None, engine="python")
    except Exception:
        file_obj.seek(0)
        try:
            return pd.read_csv(file_obj, sep="\t", engine="python")
        except Exception:
            file_obj.seek(0)
            return None


def has_required_cols(df: pd.DataFrame) -> bool:
    cols = [c.strip().lower() for c in df.columns]
    return all(c in cols for c in REQUIRED_COLS)


def split_cols_loose(text: str):
    parts = re.split(r"(?:\t+|\s{2,})", text.strip())
    return [p.strip() for p in parts if p.strip()]


def extract_date_and_rest(line: str):
    m = re.match(r"^\s*(.+?\b\d{4}\b)\s+(.*)$", line.strip())
    if not m:
        return None, None
    return m.group(1).strip(), m.group(2).strip()


def extract_result_token(line: str):
    candidates = re.findall(r"(\d(?:[-\s]\d){3}|\b\d{4}\b)", line)
    if not candidates:
        return None
    return candidates[-1]


def parse_raw_text_to_df(text: str) -> pd.DataFrame:
    rows = []
    lines = [ln.strip("\n\r") for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        low = ln.strip().lower()
        if low.startswith("date") and "state" in low and "result" in low:
            continue

        date_str, rest = extract_date_and_rest(ln)
        if not date_str or not rest:
            continue

        res_token = extract_result_token(ln)
        if not res_token:
            continue
        result = zfill4(res_token)

        split_idx = rest.rfind(res_token)
        if split_idx != -1:
            left = rest[:split_idx].strip()
        else:
            idx = ln.rfind(res_token)
            left = ln[:idx].strip()
            if left.startswith(date_str):
                left = left[len(date_str):].strip()

        parts = split_cols_loose(left)

        state = None
        game = None

        if len(parts) >= 2:
            state = parts[0]
            game = parts[1]
        elif len(parts) == 1:
            s = parts[0]
            keywords = ["Pick", "Cash", "Win", "Daily", "Four", "4"]
            found_at = None
            for kw in keywords:
                pos = s.find(kw)
                if pos > 0:
                    found_at = pos
                    break
            if found_at:
                state = s[:found_at].strip()
                game = s[found_at:].strip()
            else:
                continue
        else:
            continue

        rows.append({"date": date_str, "state": state, "game": game, "result": result})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["date"] = safe_to_datetime(df["date"])
    df["state"] = df["state"].astype(str).str.strip()
    df["game"] = df["game"].astype(str).str.strip()
    df["result"] = df["result"].apply(zfill4)
    df = df.dropna(subset=["date", "state", "game", "result"])
    return df


def parse_upload(file_obj) -> pd.DataFrame:
    df_csv = load_any_csv(file_obj)
    if df_csv is not None and not df_csv.empty:
        df_csv.columns = [c.strip().lower() for c in df_csv.columns]
        if has_required_cols(df_csv):
            df = df_csv.copy()
            df["date"] = safe_to_datetime(df["date"])
            df["state"] = df["state"].astype(str).str.strip()
            df["game"] = df["game"].astype(str).str.strip()
            df["result"] = df["result"].apply(zfill4)
            df = df.dropna(subset=["date", "state", "game", "result"])
            return df

    file_obj.seek(0)
    raw_bytes = file_obj.read()
    if isinstance(raw_bytes, bytes):
        text = raw_bytes.decode("utf-8", errors="ignore")
    else:
        text = str(raw_bytes)

    return parse_raw_text_to_df(text)


# ------------------------------------------------------------
# Upload UI
# ------------------------------------------------------------
st.markdown("### Upload Data (CSV with headers OR TXT aligned columns is OK)")
hits_file = st.file_uploader("Upload 5-year HITS file (your 3389/3889/3899 hits list)", type=["csv", "txt"])
stream_files = st.file_uploader(
    "Upload 2-year STREAM history file(s) (FULL draw history per state/game). Optional for straight recency boosting.",
    type=["csv", "txt"],
    accept_multiple_files=True
)

with st.expander("Learning controls (optional)", expanded=False):
    st.markdown("**State/Game ranking weights (HITS-based):**")
    w_rate = st.slider("Weight: Hit frequency", 0.0, 1.0, 0.45, 0.05)
    w_rec  = st.slider("Weight: Recency (drought)", 0.0, 1.0, 0.30, 0.05)
    w_cons = st.slider("Weight: Consistency (months with hits)", 0.0, 1.0, 0.20, 0.05)
    w_samp = st.slider("Weight: Reliability (sample size)", 0.0, 1.0, 0.05, 0.05)
    total_w = w_rate + w_rec + w_cons + w_samp
    if total_w == 0:
        st.warning("All weights are 0 — using equal weights internally.")
        w_rate = w_rec = w_cons = w_samp = 1.0
        total_w = 4.0

    st.markdown("**Straight ordering learning (state-specific 12 straights per family):**")
    st.caption("Straights are learned from the HITS file (true hit-order). STREAM file only boosts recency if it contains these families.")
    half_life = st.slider("Recency half-life (days)", 30, 365, 120, 15)
    alpha = st.slider("Smoothing alpha (Laplace)", 0.1, 5.0, 1.0, 0.1)
    recency_mix = st.slider("Blend recency vs frequency (0=freq only, 1=recency only)", 0.0, 1.0, 0.30, 0.05)

if not hits_file:
    st.info("Upload the 5-year HITS file to run.")
    st.stop()

# ------------------------------------------------------------
# Parse uploads
# ------------------------------------------------------------
hits = parse_upload(hits_file)
if hits.empty:
    st.error("Could not parse the HITS file into date/state/game/result (0 rows parsed).")
    st.stop()

stream = None
if stream_files:
    stream_list = []
    for f in stream_files:
        df = parse_upload(f)
        if not df.empty:
            stream_list.append(df)
    if stream_list:
        stream = pd.concat(stream_list, ignore_index=True)

# Add family fields
hits["family"] = hits["result"].apply(infer_family)
hits = hits[hits["family"].isin(FAMILY_NAMES)]

if hits.empty:
    st.error("After parsing, HITS contains 0 rows matching families 3389/3889/3899.")
    st.stop()

stream_target = None
if stream is not None and not stream.empty:
    stream["family"] = stream["result"].apply(infer_family)
    stream_target = stream[stream["family"].isin(FAMILY_NAMES)].copy()

# ------------------------------------------------------------
# A) Master ranking — ALL State/Game streams (from HITS)
# ------------------------------------------------------------
global_min = hits["date"].min()
global_max = hits["date"].max()
global_days = max((global_max - global_min).days + 1, 1)
global_months = hits["date"].apply(month_key).nunique()

rows = []
for (state, game), g in hits.groupby(["state", "game"]):
    g = g.sort_values("date")
    hits_count = int(len(g))

    hit_rate = hits_count / global_days
    days_since_last = int((global_max - g["date"].max()).days)
    recency_score = 1.0 / (days_since_last + 1.0)

    months_with_hit = int(g["date"].apply(month_key).nunique())
    consistency = months_with_hit / max(global_months, 1)

    reliability = np.log1p(hits_count)

    fam_counts = g["family"].value_counts().to_dict()
    total_f = sum(fam_counts.values()) if fam_counts else 0
    share_3389 = fam_counts.get("3389", 0) / total_f if total_f else 0.0
    share_3889 = fam_counts.get("3889", 0) / total_f if total_f else 0.0
    share_3899 = fam_counts.get("3899", 0) / total_f if total_f else 0.0

    rows.append({
        "State": state,
        "Game": game,
        "Hits": hits_count,
        "LastHitDate": g["date"].max().date(),
        "DaysSinceLastHit": days_since_last,
        "HitRate": hit_rate,
        "RecencyScore": recency_score,
        "Consistency": consistency,
        "Reliability": reliability,
        "Share_3389": share_3389,
        "Share_3889": share_3889,
        "Share_3899": share_3899,
    })

state_df = pd.DataFrame(rows)
state_df["Score"] = (
    (w_rate / total_w) * normalize(state_df["HitRate"]) +
    (w_rec  / total_w) * normalize(state_df["RecencyScore"]) +
    (w_cons / total_w) * normalize(state_df["Consistency"]) +
    (w_samp / total_w) * normalize(state_df["Reliability"])
)

state_df = state_df.sort_values("Score", ascending=False).reset_index(drop=True)
state_df.insert(0, "Rank", state_df.index + 1)

st.markdown("## A) Master Ranking — All States / All Games (Most → Least Likely)")
st.caption(f"HITS window: {global_min.date()} → {global_max.date()}  |  HITS rows used: {len(hits)}")
st.dataframe(state_df, use_container_width=True, height=520)

st.download_button(
    "Download MASTER State/Game Ranking (CSV)",
    data=state_df.to_csv(index=False).encode("utf-8"),
    file_name="MASTER_state_game_ranking.csv",
    mime="text/csv",
)

# ------------------------------------------------------------
# B) Winner-specific likelihood per state/game
# ------------------------------------------------------------
st.markdown("## B) Winner-Specific Likelihood per State/Game (Which family is more likely there)")
family_df = state_df[[
    "Rank", "State", "Game", "Score", "Hits", "LastHitDate", "DaysSinceLastHit",
    "Share_3389", "Share_3889", "Share_3899"
]].copy()
st.dataframe(family_df, use_container_width=True, height=420)

st.download_button(
    "Download Winner-Specific Shares (CSV)",
    data=family_df.to_csv(index=False).encode("utf-8"),
    file_name="WINNER_specific_shares_by_state_game.csv",
    mime="text/csv",
)

# ------------------------------------------------------------
# C) State-specific 12-straight ordering per family
#    Primary: learned from HITS (true winning straight order)
#    Optional boost: if STREAM contains these families, use it as recency booster
# ------------------------------------------------------------
st.markdown("## C) State-Specific Straight Ordering (All 12 shown, graded, ranked)")
if stream_target is None or stream_target.empty:
    st.warning("STREAM file contains 0 occurrences of 3389/3889/3899-family numbers. Straights will be learned from HITS only (still state-specific).")

fam_perm = {name: perms_for_family(key) for key, name in FAMILIES.items()}
rank_lookup = {(r.State, r.Game): int(r.Rank) for r in state_df.itertuples(index=False)}

available_pairs = list(state_df[["State", "Game"]].itertuples(index=False, name=None))
selected_pair = st.selectbox(
    "Select a State/Game:",
    options=available_pairs,
    index=0,
    format_func=lambda x: f"{x[0]} — {x[1]}  (MASTER rank: {rank_lookup.get(x, 'N/A')})"
)

def tables_from_hits_and_stream(state: str, game: str):
    # HIT source
    h = hits[(hits["state"] == state) & (hits["game"] == game)].copy()
    if h.empty:
        return None

    # Optional stream source
    s = None
    if stream_target is not None and not stream_target.empty:
        s = stream_target[(stream_target["state"] == state) & (stream_target["game"] == game)].copy()
        if s.empty:
            s = None

    # Latest date for recency reference
    latest_date = h["date"].max()
    if s is not None:
        latest_date = max(latest_date, s["date"].max())

    out = {}
    for fam in FAMILY_NAMES:
        perms = fam_perm[fam]

        hf = h[h["family"] == fam].copy()
        h_total = int(len(hf))
        h_counts = hf["result"].value_counts().to_dict()
        h_last = hf.groupby("result")["date"].max().to_dict() if h_total else {}

        if s is not None:
            sf = s[s["family"] == fam].copy()
            s_total = int(len(sf))
            s_counts = sf["result"].value_counts().to_dict()
            s_last = sf.groupby("result")["date"].max().to_dict() if s_total else {}
        else:
            s_total = 0
            s_counts = {}
            s_last = {}

        # Combine counts (HITS + STREAM) for probability
        # STREAM is usually shorter/more recent; we don't overweight it—just add counts.
        total = h_total + s_total
        rows = []
        for p in perms:
            c = int(h_counts.get(p, 0) + s_counts.get(p, 0))

            # Laplace smoothing probability among 12 perms
            if total > 0:
                prob = (c + alpha) / (total + alpha * len(perms))
            else:
                prob = 1.0 / len(perms)

            # Recency: prefer stream last_seen if it exists, else hits
            last_seen_dt = None
            if p in s_last:
                last_seen_dt = s_last[p]
            elif p in h_last:
                last_seen_dt = h_last[p]

            if last_seen_dt is not None:
                days_since = int((latest_date - last_seen_dt).days)
                rec_w = recency_weight(days_since, half_life)
                last_seen = last_seen_dt.date()
            else:
                days_since = None
                rec_w = 0.0
                last_seen = None

            score = (1.0 - recency_mix) * prob + recency_mix * rec_w

            rows.append({
                "Straight": p,
                "Count(H+S)": c,
                "Prob_Smoothed": float(prob),
                "LastSeen": last_seen,
                "DaysSinceSeen": days_since,
                "Score": float(score),
                "HITS_Count": int(h_counts.get(p, 0)),
                "STREAM_Count": int(s_counts.get(p, 0)),
            })

        df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", df.index + 1)
        out[fam] = df

    return out

tables = tables_from_hits_and_stream(*selected_pair)
if tables is None:
    st.warning("No HITS rows found for this state/game (unexpected).")
else:
    rr = state_df[(state_df["State"] == selected_pair[0]) & (state_df["Game"] == selected_pair[1])].iloc[0]
    st.caption(
        f"Winner-family likelihood (HITS): 3389={rr['Share_3389']:.1%} | 3889={rr['Share_3889']:.1%} | 3899={rr['Share_3899']:.1%} "
        f"(HITS={int(rr['Hits'])}, LastHit={rr['LastHitDate']}, DaysSince={int(rr['DaysSinceLastHit'])})"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 3389 family — 12 straights")
        st.dataframe(tables["3389"], use_container_width=True, height=520)
    with c2:
        st.markdown("### 3889 family — 12 straights")
        st.dataframe(tables["3889"], use_container_width=True, height=520)
    with c3:
        st.markdown("### 3899 family — 12 straights")
        st.dataframe(tables["3899"], use_container_width=True, height=520)

# ------------------------------------------------------------
# D) Full straight rankings export (ALL state/games, ALL families, ALL 12 perms)
# ------------------------------------------------------------
st.markdown("## D) Download FULL Straight Rankings (All states/games • all 3 families • all 12 straights)")

export_rows = []
for state, game in available_pairs:
    tables = tables_from_hits_and_stream(state, game)
    if tables is None:
        continue
    master_rank = rank_lookup.get((state, game), None)
    for fam in FAMILY_NAMES:
        df = tables[fam].copy()
        df["State"] = state
        df["Game"] = game
        df["Family"] = fam
        df["MASTER_Rank"] = master_rank
        export_rows.append(df)

export_df = pd.concat(export_rows, ignore_index=True)
export_df = export_df[[
    "State", "Game", "MASTER_Rank",
    "Family", "Rank", "Straight",
    "Score", "Prob_Smoothed", "Count(H+S)", "HITS_Count", "STREAM_Count",
    "LastSeen", "DaysSinceSeen"
]].sort_values(
    by=["MASTER_Rank", "State", "Game", "Family", "Rank"],
    ascending=[True, True, True, True, True],
    na_position="last"
).reset_index(drop=True)

st.download_button(
    "Download FULL Straight Rankings (CSV)",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name="FULL_state_game_family_straight_rankings.csv",
    mime="text/csv",
)

st.success("Learning complete. Upload updated histories anytime — the app relearns immediately.")
