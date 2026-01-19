import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import itertools

st.set_page_config(layout="wide")
st.title("Pick 4 (3389 / 3889 / 3899) — Master Ranking with History-Learned Recency+Due")

# -----------------------------
# Families
# -----------------------------
FAMILIES = {
    ("3", "3", "8", "9"): "3389",
    ("3", "8", "8", "9"): "3889",
    ("3", "8", "9", "9"): "3899",
}
FAMILY_NAMES = ["3389", "3889", "3899"]

def zfill4(x) -> str:
    s = str(x).strip()
    s = "".join([c for c in s if c.isdigit()])
    return s.zfill(4)[-4:]

def infer_family(num4: str):
    digs = tuple(sorted(list(zfill4(num4))))
    return FAMILIES.get(digs)

# -----------------------------
# Robust parsing (TXT aligned cols or CSV)
# Expected fields: date, state, game, result (result may be like 3-9-8-8)
# -----------------------------
def safe_to_datetime(series):
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)

def load_any_csv(file_obj):
    try:
        return pd.read_csv(file_obj, sep=None, engine="python")
    except Exception:
        file_obj.seek(0)
        try:
            return pd.read_csv(file_obj, sep="\t", engine="python")
        except Exception:
            return None

def split_cols_loose(text: str):
    return [p.strip() for p in re.split(r"(?:\t+|\s{2,})", text.strip()) if p.strip()]

def extract_date_and_rest(line: str):
    m = re.match(r"^\s*(.+?\b\d{4}\b)\s+(.*)$", line.strip())
    if not m:
        return None, None
    return m.group(1).strip(), m.group(2).strip()

def extract_result_token(line: str):
    # last occurrence of either "3-8-9-8" or "3898"
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
        left = rest[:split_idx].strip() if split_idx != -1 else rest

        parts = split_cols_loose(left)
        if len(parts) < 2:
            continue
        state, game = parts[0], parts[1]

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
    # Try CSV
    df_csv = load_any_csv(file_obj)
    if df_csv is not None and not df_csv.empty:
        df_csv.columns = [c.strip().lower() for c in df_csv.columns]
        needed = {"date", "state", "game", "result"}
        if needed.issubset(set(df_csv.columns)):
            df = df_csv.copy()
            df["date"] = safe_to_datetime(df["date"])
            df["state"] = df["state"].astype(str).str.strip()
            df["game"] = df["game"].astype(str).str.strip()
            df["result"] = df["result"].apply(zfill4)
            df = df.dropna(subset=["date", "state", "game", "result"])
            return df

    # Fallback: raw text
    file_obj.seek(0)
    raw = file_obj.read()
    text = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
    return parse_raw_text_to_df(text)

# -----------------------------
# History-learned Recency+Due: hazard curve from gaps
# hazard[t] = events at gap length t / gaps at risk at t
# t starts at 1 (tomorrow after last hit)
# -----------------------------
def build_gap_hazard(hit_dates_by_stream, t_max=365, smooth_ema=0.25):
    gap_lengths = []
    for dates in hit_dates_by_stream.values():
        if len(dates) < 2:
            continue
        dates = sorted(dates)
        gaps = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
        gaps = [g for g in gaps if g > 0]
        gap_lengths.extend(gaps)

    if not gap_lengths:
        # no gaps: return flat tiny hazard
        return np.full(t_max+1, 1e-6), {"gap_count": 0, "t_max": t_max}

    gaps = np.array(gap_lengths, dtype=int)
    # cap extremely large gaps so tail doesn’t create fake spikes
    gaps = np.clip(gaps, 1, t_max)

    # at_risk[t] = number of gaps with length >= t
    # events[t]  = number of gaps with length == t
    at_risk = np.zeros(t_max+1, dtype=float)
    events = np.zeros(t_max+1, dtype=float)

    for t in range(1, t_max+1):
        at_risk[t] = np.sum(gaps >= t)
        events[t] = np.sum(gaps == t)

    hazard = np.zeros(t_max+1, dtype=float)
    for t in range(1, t_max+1):
        hazard[t] = (events[t] / at_risk[t]) if at_risk[t] > 0 else 0.0

    # Smooth with EMA (prevents “spiky” due-only behavior)
    haz_s = hazard.copy()
    for t in range(2, t_max+1):
        haz_s[t] = smooth_ema * hazard[t] + (1.0 - smooth_ema) * haz_s[t-1]

    meta = {"gap_count": int(len(gap_lengths)), "t_max": int(t_max)}
    return haz_s, meta

def normalize_01(x):
    x = np.asarray(x, dtype=float)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if np.isclose(mx, mn):
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

# -----------------------------
# UI
# -----------------------------
hits_file = st.file_uploader("Upload 5-year HIT history (TXT or CSV)", type=["txt", "csv"])
if not hits_file:
    st.stop()

hits = parse_upload(hits_file)
if hits.empty:
    st.error("Could not parse the file into columns: date, state, game, result.")
    st.stop()

hits["family"] = hits["result"].apply(infer_family)
hits = hits[hits["family"].isin(FAMILY_NAMES)].copy()
if hits.empty:
    st.error("Parsed file, but found 0 rows matching families 3389/3889/3899.")
    st.stop()

# Window
hits = hits.sort_values("date")
start_date = hits["date"].min().date()
end_date = hits["date"].max().date()
days_window = (end_date - start_date).days + 1

st.caption(f"HITS window: {start_date} → {end_date} | rows used: {len(hits):,} | streams: {hits.groupby(['state','game']).ngroups:,}")

# Build per-stream hit dates
hit_dates_by_stream = {}
for (state, game), g in hits.groupby(["state", "game"], sort=False):
    hit_dates_by_stream[(state, game)] = list(pd.to_datetime(g["date"]).dt.date)

# Learning controls
with st.expander("Scoring controls", expanded=True):
    colA, colB, colC = st.columns(3)
    with colA:
        t_max = st.slider("Max drought days modeled (cap)", 90, 730, 365, 15)
        ema = st.slider("Hazard smoothing (EMA)", 0.05, 0.60, 0.25, 0.05)
    with colB:
        rec_half_life = st.slider("Classic recency half-life (days)", 7, 365, 90, 1)
        due_half_life = st.slider("Classic due half-life (days)", 7, 365, 120, 1)
    with colC:
        w_rate = st.slider("Weight: HitRate", 0.0, 2.0, 0.60, 0.05)
        w_hist_rd = st.slider("Weight: History-Learned Recency+Due", 0.0, 2.0, 0.80, 0.05)
        w_cons = st.slider("Weight: Consistency (months with hits)", 0.0, 2.0, 0.25, 0.05)
        w_rel = st.slider("Weight: Reliability (sample size)", 0.0, 2.0, 0.15, 0.05)

# Build hazard curve
haz_s, haz_meta = build_gap_hazard(hit_dates_by_stream, t_max=t_max, smooth_ema=ema)

# Per-stream stats
rows = []
for (state, game), dates in hit_dates_by_stream.items():
    dates_sorted = sorted(dates)
    hits_n = len(dates_sorted)
    last_hit = dates_sorted[-1]
    days_since = (end_date - last_hit).days

    # hit rate per calendar day in window (simple & consistent)
    hit_rate = hits_n / days_window

    # classic recency/due (transparent)
    recency_score = np.exp(-days_since / float(rec_half_life))
    due_score = 1.0 - np.exp(-days_since / float(due_half_life))

    # history-learned combined: hazard at t = days_since + 1
    t = int(min(max(days_since + 1, 1), t_max))
    hist_rd = float(haz_s[t])

    # consistency: months with >=1 hit / total months in window
    g = hits[(hits["state"] == state) & (hits["game"] == game)]
    months_with_hits = g["date"].dt.to_period("M").nunique()
    total_months = pd.period_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="M").nunique()
    consistency = months_with_hits / total_months if total_months else 0.0

    # reliability: log(1+hits) (keeps small streams from lying)
    reliability = float(np.log1p(hits_n))

    # family shares
    fam_counts = g["family"].value_counts().to_dict()
    share_3389 = fam_counts.get("3389", 0) / hits_n if hits_n else 0.0
    share_3889 = fam_counts.get("3889", 0) / hits_n if hits_n else 0.0
    share_3899 = fam_counts.get("3899", 0) / hits_n if hits_n else 0.0

    rows.append({
        "State": state,
        "Game": game,
        "Hits": hits_n,
        "LastHitDate": last_hit,
        "DaysSinceLastHit": days_since,
        "HitRate": hit_rate,
        "RecencyScore": recency_score,
        "DueScore": due_score,
        "HistRecencyDue": hist_rd,
        "Consistency": consistency,
        "Reliability": reliability,
        "Share_3389": share_3389,
        "Share_3889": share_3889,
        "Share_3899": share_3899,
    })

df = pd.DataFrame(rows)

# Normalize components for scoring
df["HitRate_n"] = normalize_01(df["HitRate"])
df["HistRD_n"]  = normalize_01(df["HistRecencyDue"])
df["Cons_n"]    = normalize_01(df["Consistency"])
df["Rel_n"]     = normalize_01(df["Reliability"])

df["Score"] = (
    w_rate * df["HitRate_n"] +
    w_hist_rd * df["HistRD_n"] +
    w_cons * df["Cons_n"] +
    w_rel * df["Rel_n"]
)

df = df.sort_values("Score", ascending=False).reset_index(drop=True)
df.insert(0, "Rank", df.index + 1)

# Display master table
st.markdown("## A) Master Ranking — All States / All Games (Most → Least likely)")
st.dataframe(df, use_container_width=True, height=600)

st.download_button(
    "Download master ranking (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="pk4_3389_3889_3899_master_ranking.csv",
    mime="text/csv",
)

# -----------------------------
# Coverage: how many streams/day to catch ≥1 winner
# Uses independence assumption: P(hit≥1)=1-∏(1-p_i)
# Here p_i is HitRate (per day) from the 5-year window.
# -----------------------------
st.markdown("## B) How many streams/day to catch ≥1 winner? (based on top-K by Score)")
max_k = min(50, len(df))
k_values = list(range(1, max_k + 1))

p_list = df["HitRate"].to_numpy()  # per-day probability proxy
cov_rows = []
prod = 1.0
for k in k_values:
    prod *= (1.0 - float(p_list[k-1]))
    p_at_least_one = 1.0 - prod
    expected_hits = float(np.sum(p_list[:k]))
    cov_rows.append({
        "K_streams_per_day": k,
        "P(catch ≥1 winner today)": p_at_least_one,
        "Expected winners per day (anywhere)": expected_hits,
        "BoxOnlyCost_per_day_$ (K * 0.75)": 0.75 * k,
    })

cov = pd.DataFrame(cov_rows)
st.dataframe(cov, use_container_width=True, height=450)

st.download_button(
    "Download coverage table (CSV)",
    data=cov.to_csv(index=False).encode("utf-8"),
    file_name="pk4_master_coverage_vs_k.csv",
    mime="text/csv",
)

with st.expander("Hazard curve details (what the app learned)", expanded=False):
    st.write(f"Gap count used to learn hazard: {haz_meta['gap_count']:,} | capped at {haz_meta['t_max']} days.")
    haz_df = pd.DataFrame({
        "t_days_since_last_hit_plus_1": np.arange(1, t_max+1),
        "Hazard_smoothed_P(hit_tomorrow)": haz_s[1:t_max+1]
    })
    st.dataframe(haz_df.head(60), use_container_width=True)
    st.dataframe(haz_df.tail(60), use_container_width=True)
