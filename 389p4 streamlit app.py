# pick4_learning_ranker_FULL.py
# Run: streamlit run pick4_learning_ranker_FULL.py

import streamlit as st
import pandas as pd
import numpy as np
import itertools
from datetime import datetime

st.set_page_config(layout="wide")
st.title("Pick 4 Learning Ranker — Best States/Games + State-Specific 12 Straights (3389/3889/3899)")

# -----------------------------
# Assumed headers (same for BOTH uploads)
# -----------------------------
REQUIRED_COLS = ["date", "state", "game", "result"]

# Target digit families
FAMILIES = {
    ("3", "3", "8", "9"): "3389",
    ("3", "8", "8", "9"): "3889",
    ("3", "8", "9", "9"): "3899",
}
FAMILY_NAMES = ["3389", "3889", "3899"]

# -----------------------------
# Helpers
# -----------------------------
def zfill4(x) -> str:
    s = str(x).strip()
    # keep digits only
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
    # fam_key is tuple of digits like ('3','3','8','9')
    perms = sorted(set("".join(p) for p in itertools.permutations(fam_key, 4)))
    return perms  # 12 perms

def recency_weight(days_since: float, half_life_days: float) -> float:
    # Exponential decay: 0.5^(days/half_life)
    if days_since is None or np.isinf(days_since):
        return 0.0
    half_life_days = max(float(half_life_days), 1.0)
    return float(0.5 ** (float(days_since) / half_life_days))

def load_any(file_obj):
    # Auto-detect delimiter
    try:
        return pd.read_csv(file_obj, sep=None, engine="python")
    except Exception:
        file_obj.seek(0)
        return pd.read_csv(file_obj, sep="\t", engine="python")

def ensure_required_cols(df: pd.DataFrame, label: str):
    cols = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in cols]
    if missing:
        st.error(f"{label} is missing required columns: {missing}. Required: {REQUIRED_COLS}")
        st.stop()
    df = df.copy()
    df.columns = cols
    return df

# -----------------------------
# Upload UI
# -----------------------------
st.markdown("### Upload Data (headers must match: date, state, game, result)")

hits_file = st.file_uploader(
    "Upload 5-year HITS file (only these families; all states/games)",
    type=["csv", "txt"],
    accept_multiple_files=False,
)

stream_files = st.file_uploader(
    "Upload 2-year STREAM history file(s) (full Pick-4 results for ordering, same headers). You can upload multiple and I will combine them.",
    type=["csv", "txt"],
    accept_multiple_files=True,
)

with st.expander("Learning controls (optional)", expanded=False):
    st.markdown("These controls affect ranking/weighting (the app still uses the real counts).")

    st.markdown("**State/Game ranking weights (most → least likely):**")
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
    half_life = st.slider("Recency half-life (days)", 30, 365, 120, 15)
    alpha = st.slider("Smoothing alpha (Laplace)", 0.1, 5.0, 1.0, 0.1)
    recency_mix = st.slider("Blend recency vs frequency (0=freq only, 1=recency only)", 0.0, 1.0, 0.30, 0.05)

if not hits_file or not stream_files:
    st.info("Upload the 5-year HITS file and at least one 2-year STREAM file to run learning + ranking.")
    st.stop()

# -----------------------------
# Load + validate
# -----------------------------
hits_raw = load_any(hits_file)
hits = ensure_required_cols(hits_raw, "HITS file")

stream_list = []
for f in stream_files:
    df = load_any(f)
    df = ensure_required_cols(df, f"STREAM file '{getattr(f, 'name', 'uploaded')}'")
    stream_list.append(df)

stream = pd.concat(stream_list, ignore_index=True)

# Clean & parse types
hits = hits.copy()
hits["date"] = safe_to_datetime(hits["date"])
hits["state"] = hits["state"].astype(str).str.strip()
hits["game"] = hits["game"].astype(str).str.strip()
hits["result"] = hits["result"].apply(zfill4)
hits["family"] = hits["result"].apply(infer_family)
hits = hits.dropna(subset=["date", "state", "game", "result"])
hits = hits[hits["family"].isin(FAMILY_NAMES)]

stream = stream.copy()
stream["date"] = safe_to_datetime(stream["date"])
stream["state"] = stream["state"].astype(str).str.strip()
stream["game"] = stream["game"].astype(str).str.strip()
stream["result"] = stream["result"].apply(zfill4)
stream["family"] = stream["result"].apply(infer_family)
stream = stream.dropna(subset=["date", "state", "game", "result"])

# Focus on just the target families for ordering (still state-specific)
stream_target = stream[stream["family"].isin(FAMILY_NAMES)].copy()

if hits.empty:
    st.error("After parsing, the HITS file contains 0 valid rows for families 3389/3889/3899. Check the file content.")
    st.stop()

if stream_target.empty:
    st.error("After parsing, the STREAM file(s) contain 0 rows matching families 3389/3889/3899. Ensure stream files include real Pick-4 results.")
    st.stop()

# -----------------------------
# A) Master ranking — ALL State/Game streams (from HITS)
# -----------------------------
global_min = hits["date"].min()
global_max = hits["date"].max()
global_days = max((global_max - global_min).days + 1, 1)
global_months = hits["date"].apply(month_key).nunique()

rows = []
for (state, game), g in hits.groupby(["state", "game"]):
    g = g.sort_values("date")
    hits_count = int(len(g))

    # Comparable frequency metric across streams using the same global window
    hit_rate = hits_count / global_days  # hits-per-day over the whole 5y window (comparable across streams)

    days_since_last = int((global_max - g["date"].max()).days)
    recency_score = 1.0 / (days_since_last + 1.0)

    months_with_hit = int(g["date"].apply(month_key).nunique())
    consistency = months_with_hit / max(global_months, 1)

    # Reliability: log scale (prevents huge streams dominating too hard)
    reliability = np.log1p(hits_count)

    fam_counts = g["family"].value_counts().to_dict()
    total_f = sum(fam_counts.values()) if fam_counts else 0
    share_3389 = fam_counts.get("3389", 0) / total_f if total_f else 0.0
    share_3889 = fam_counts.get("3889", 0) / total_f if total_f else 0.0
    share_3899 = fam_counts.get("3899", 0) / total_f if total_f else 0.0

    rows.append({
        "State": state,
        "Game": game,
        "Score_HitRate": hit_rate,
        "Score_Recency": recency_score,
        "Score_Consistency": consistency,
        "Score_Reliability": reliability,
        "Hits": hits_count,
        "DaysSinceLastHit": days_since_last,
        "LastHitDate": g["date"].max().date(),
        "Share_3389": share_3389,
        "Share_3889": share_3889,
        "Share_3899": share_3899,
    })

state_df = pd.DataFrame(rows)

# Normalize and blend
state_df["Score"] = (
    (w_rate / total_w) * normalize(state_df["Score_HitRate"]) +
    (w_rec  / total_w) * normalize(state_df["Score_Recency"]) +
    (w_cons / total_w) * normalize(state_df["Score_Consistency"]) +
    (w_samp / total_w) * normalize(state_df["Score_Reliability"])
)

state_df = state_df.sort_values("Score", ascending=False).reset_index(drop=True)
state_df.insert(0, "Rank", state_df.index + 1)

st.markdown("## A) Master Ranking — All States / All Games (Most → Least Likely)")
st.caption(f"Learned from HITS window: {global_min.date()} → {global_max.date()}  |  Rows used: {len(hits)}")
st.dataframe(state_df, use_container_width=True, height=520)

# Download master
st.download_button(
    "Download MASTER State/Game Ranking (CSV)",
    data=state_df.to_csv(index=False).encode("utf-8"),
    file_name="MASTER_state_game_ranking.csv",
    mime="text/csv",
)

# -----------------------------
# B) State-specific Family likelihood (winner-specific within stream)
# (This is already inside state_df as Share_* but we export a focused sheet too)
# -----------------------------
family_df = state_df[[
    "Rank", "State", "Game", "Score", "Hits", "LastHitDate", "DaysSinceLastHit",
    "Share_3389", "Share_3889", "Share_3899"
]].copy()

st.markdown("## B) Winner-Specific Likelihood per State/Game (Which of the 3 is more likely there)")
st.dataframe(family_df, use_container_width=True, height=420)

st.download_button(
    "Download Winner-Specific Shares (CSV)",
    data=family_df.to_csv(index=False).encode("utf-8"),
    file_name="WINNER_specific_shares_by_state_game.csv",
    mime="text/csv",
)

# -----------------------------
# C) State-specific 12-straight ordering per family (from STREAM)
# -----------------------------
st.markdown("## C) State-Specific Straight Ordering (All 12 shown, graded, ranked)")

# Precompute perms
fam_perm = {name: perms_for_family(key) for key, name in FAMILIES.items()}

# We will compute straight rankings for ALL state+games that exist in the STREAM target data
# but display via selector (otherwise UI is enormous). Full CSV export includes everything.
stream_groups = stream_target.groupby(["state", "game"])

# Build a fast lookup for top-ranked state/game order (so the dropdown is already best-first)
rank_lookup = {(r.State, r.Game): int(r.Rank) for r in state_df.itertuples(index=False)}
available_pairs = sorted(
    [(s, g) for (s, g) in stream_groups.groups.keys()],
    key=lambda x: rank_lookup.get(x, 10**9)
)

if not available_pairs:
    st.warning("No state/game pairs found in STREAM target rows (3389/3889/3899).")
    st.stop()

default_pair = available_pairs[0]
selected_pair = st.selectbox(
    "Select a State/Game to view its 12-straight tables (dropdown is ordered by MASTER rank when available):",
    options=available_pairs,
    index=0,
    format_func=lambda x: f"{x[0]} — {x[1]}  (MASTER rank: {rank_lookup.get(x, 'N/A')})"
)

def compute_straight_tables_for_pair(state: str, game: str):
    s = stream_target[(stream_target["state"] == state) & (stream_target["game"] == game)].copy()
    if s.empty:
        return None

    out = {}
    latest_date = s["date"].max()

    for fam in FAMILY_NAMES:
        sf = s[s["family"] == fam].copy()
        perms = fam_perm[fam]

        total_fam = int(len(sf))
        counts = sf["result"].value_counts().to_dict()
        last_seen_map = sf.groupby("result")["date"].max().to_dict() if total_fam else {}

        rows = []
        for p in perms:
            c = int(counts.get(p, 0))

            # Smoothed probability within this family for this state/game
            if total_fam > 0:
                prob = (c + alpha) / (total_fam + alpha * len(perms))
            else:
                prob = 1.0 / len(perms)

            # Recency component per permutation
            if p in last_seen_map:
                days_since = int((latest_date - last_seen_map[p]).days)
                rec_w = recency_weight(days_since, half_life)
                last_seen = last_seen_map[p].date()
            else:
                days_since = None
                rec_w = 0.0
                last_seen = None

            score = (1.0 - recency_mix) * prob + recency_mix * rec_w

            rows.append({
                "Straight": p,
                "Count": c,
                "Prob_Smoothed": float(prob),
                "LastSeen": last_seen,
                "DaysSinceSeen": days_since,
                "Score": float(score),
            })

        df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", df.index + 1)
        out[fam] = df

    return out

tables = compute_straight_tables_for_pair(*selected_pair)
if tables is None:
    st.warning("No target-family stream rows found for this selection.")
else:
    # Show family likelihood from hits (if available)
    hit_row = state_df[(state_df["State"] == selected_pair[0]) & (state_df["Game"] == selected_pair[1])]
    if not hit_row.empty:
        rr = hit_row.iloc[0]
        st.caption(
            f"Winner-specific likelihood (from 5y HITS): "
            f"3389={rr['Share_3389']:.1%} | 3889={rr['Share_3889']:.1%} | 3899={rr['Share_3899']:.1%}  "
            f"(HITS={int(rr['Hits'])}, LastHit={rr['LastHitDate']}, DaysSince={int(rr['DaysSinceLastHit'])})"
        )
    else:
        st.caption("This state/game does not appear in the HITS file (cannot show winner-specific shares).")

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

# -----------------------------
# D) Full straight rankings export (ALL state/games, ALL families, ALL 12 perms)
# -----------------------------
st.markdown("## D) Download FULL Straight Rankings (All states/games • all 3 families • all 12 straights)")

# Build full export (can be large; still deterministic)
export_rows = []
for (state, game), s in stream_groups:
    s = s.copy()
    latest_date = s["date"].max()

    for fam in FAMILY_NAMES:
        sf = s[s["family"] == fam].copy()
        perms = fam_perm[fam]
        total_fam = int(len(sf))
        counts = sf["result"].value_counts().to_dict()
        last_seen_map = sf.groupby("result")["date"].max().to_dict() if total_fam else {}

        # Compute scores for each permutation
        tmp = []
        for p in perms:
            c = int(counts.get(p, 0))
            if total_fam > 0:
                prob = (c + alpha) / (total_fam + alpha * len(perms))
            else:
                prob = 1.0 / len(perms)

            if p in last_seen_map:
                days_since = int((latest_date - last_seen_map[p]).days)
                rec_w = recency_weight(days_since, half_life)
                last_seen = last_seen_map[p].date()
            else:
                days_since = None
                rec_w = 0.0
                last_seen = None

            score = (1.0 - recency_mix) * prob + recency_mix * rec_w
            tmp.append((p, c, prob, last_seen, days_since, score))

        # Rank within family for this state/game
        tmp_sorted = sorted(tmp, key=lambda x: x[-1], reverse=True)
        for rank, (p, c, prob, last_seen, days_since, score) in enumerate(tmp_sorted, start=1):
            export_rows.append({
                "State": state,
                "Game": game,
                "Family": fam,
                "Straight": p,
                "Rank": rank,
                "Score": float(score),
                "Count": int(c),
                "Prob_Smoothed": float(prob),
                "LastSeen": last_seen,
                "DaysSinceSeen": days_since,
                "FamilyTotalCount": total_fam,
                "MASTER_Rank": rank_lookup.get((state, game), None),
            })

export_df = pd.DataFrame(export_rows)
# Helpful sort: by master rank (if known), then state/game, then family, then straight rank
export_df = export_df.sort_values(
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

st.success("Learning complete. Upload updated histories anytime — the app relearns immediately and regenerates rankings.")
