import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import timedelta, date
import itertools

st.set_page_config(layout="wide")
st.title("Pick 4 (3389/3889/3899) — State/Game Predictor (Safer Due + Schedule + Optional Straights)")

# -----------------------------
# Families
# -----------------------------
FAMILIES = {
    ("3", "3", "8", "9"): "3389",
    ("3", "8", "8", "9"): "3889",
    ("3", "8", "9", "9"): "3899",
}
FAMILY_NAMES = ["3389", "3889", "3899"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# -----------------------------
# Utilities
# -----------------------------
def safe_to_datetime(series):
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)

def zfill4(x) -> str:
    s = str(x).strip()
    s = "".join([c for c in s if c.isdigit()])
    return s.zfill(4)[-4:]

def infer_family(num4: str):
    digs = tuple(sorted(list(zfill4(num4))))
    return FAMILIES.get(digs)

def normalize_01(s):
    s = pd.Series(s).astype(float)
    denom = s.max() - s.min()
    if denom == 0 or np.isnan(denom):
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - s.min()) / (denom + 1e-12)

def dom_bin(d: date):
    # bins: 1-7, 8-14, 15-21, 22-28, 29-31
    day = d.day
    if day <= 7: return 0
    if day <= 14: return 1
    if day <= 21: return 2
    if day <= 28: return 3
    return 4

# -----------------------------
# Robust parsing (tab-separated, Fireball appended, wrapped Fireball lines)
# -----------------------------
FB_PREFIX_RE = re.compile(r"^(Fireball|Wild Ball|Superball|Sum It Up|Lucky Sum)\s*:\s*\d+\s*$", flags=re.I)
DIGIT_ONLY_RE = re.compile(r"^\d+\s*$")

def stitch_wrapped_lines(text: str) -> str:
    """
    Joins continuation lines like:
      "... 3-9-3-8, Fireball:"
      "9"
    or:
      "Fireball: 9"
    back onto the previous record line.
    """
    lines = [ln.rstrip("\n\r") for ln in text.splitlines()]
    out = []
    for ln in lines:
        if not ln.strip():
            continue

        # If it has a tab, treat as a proper record line
        if "\t" in ln:
            out.append(ln)
            continue

        # No tabs: possibly a wrapped continuation (Fireball etc.)
        if not out:
            continue

        s = ln.strip()
        prev = out[-1].rstrip()

        if FB_PREFIX_RE.match(s):
            if prev.endswith(":"):
                out[-1] = prev + " " + s
            continue

        if DIGIT_ONLY_RE.match(s):
            if prev.endswith(":"):
                out[-1] = prev + " " + s
            continue

        # otherwise ignore stray line
        continue

    return "\n".join(out)

def extract_pick4_from_result_field(result_field: str):
    """
    Extract FIRST pick4 result from a field like:
      "3-9-3-8, Fireball: 9" -> 3938
      "3 9 3 8"              -> 3938
      "3938 Fireball: 9"     -> 3938
    """
    if result_field is None:
        return None
    s = str(result_field)

    m = re.search(r"(\d)\s*[-\s]\s*(\d)\s*[-\s]\s*(\d)\s*[-\s]\s*(\d)", s)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}{m.group(4)}"

    m = re.search(r"\b(\d{4})\b", s)
    if m:
        return m.group(1)

    return None

def load_any_csv(file_obj):
    try:
        return pd.read_csv(file_obj, sep=None, engine="python")
    except Exception:
        file_obj.seek(0)
        try:
            return pd.read_csv(file_obj, sep="\t", engine="python")
        except Exception:
            return None

def parse_upload(file_obj) -> pd.DataFrame:
    # 1) Try structured CSV with headers
    df_csv = load_any_csv(file_obj)
    if df_csv is not None and not df_csv.empty:
        df_csv.columns = [c.strip().lower() for c in df_csv.columns]
        needed = {"date", "state", "game", "result"}
        if needed.issubset(set(df_csv.columns)):
            df = df_csv.copy()
            df["date"] = safe_to_datetime(df["date"])
            df["state"] = df["state"].astype(str).str.strip()
            df["game"] = df["game"].astype(str).str.strip()
            df["result"] = df["result"].apply(lambda x: zfill4(extract_pick4_from_result_field(x) or x))
            df = df.dropna(subset=["date", "state", "game", "result"])
            return df

    # 2) Raw text: stitch wrapped lines, then parse tab columns
    file_obj.seek(0)
    raw = file_obj.read()
    text = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
    text = stitch_wrapped_lines(text)

    rows = []
    for ln in text.splitlines():
        if not ln.strip():
            continue
        low = ln.strip().lower()
        if low.startswith("date") and ("state" in low) and ("result" in low):
            continue
        if "\t" not in ln:
            continue

        parts = ln.split("\t")
        if len(parts) < 4:
            continue

        date_str = parts[0].strip()
        state = parts[1].strip()
        game = parts[2].strip()
        result_field = " ".join(p.strip() for p in parts[3:] if p.strip())

        pick4 = extract_pick4_from_result_field(result_field)
        if not pick4:
            continue

        rows.append({"date": date_str, "state": state, "game": game, "result": zfill4(pick4)})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = safe_to_datetime(df["date"])
    df["state"] = df["state"].astype(str).str.strip()
    df["game"] = df["game"].astype(str).str.strip()
    df["result"] = df["result"].apply(zfill4)
    df = df.dropna(subset=["date", "state", "game", "result"])
    return df

# -----------------------------
# Per-stream features
# -----------------------------
def gaps_in_days(dates_sorted):
    if len(dates_sorted) < 2:
        return []
    return [(dates_sorted[i] - dates_sorted[i-1]).days for i in range(1, len(dates_sorted))
            if (dates_sorted[i] - dates_sorted[i-1]).days > 0]

def overdue_percentile(current_drought, past_gaps):
    if not past_gaps:
        return 0.5
    past = np.array(past_gaps, dtype=float)
    return float(np.mean(past <= current_drought))

def gap_proximity_score(current_drought, past_gaps):
    """
    "Sweet spot" control:
    - High if current drought is near typical (median)
    - Low if far from typical
    Uses robust scale (IQR). Returns 0..1.
    """
    if not past_gaps or len(past_gaps) < 4:
        return 0.5
    g = np.array(past_gaps, dtype=float)
    med = float(np.median(g))
    q25 = float(np.quantile(g, 0.25))
    q75 = float(np.quantile(g, 0.75))
    iqr = max(q75 - q25, 1.0)

    z = abs(float(current_drought) - med) / iqr
    # exponential decay: near median => ~1, far => ->0
    score = float(np.exp(-z))
    return max(0.0, min(1.0, score))

def weekday_weights(dates_sorted, alpha=1.0):
    counts = np.zeros(7, dtype=float)
    for d in dates_sorted:
        counts[d.weekday()] += 1
    counts += float(alpha)
    w = counts / counts.mean()
    return w

def dom_weights(dates_sorted, alpha=1.0):
    counts = np.zeros(5, dtype=float)
    for d in dates_sorted:
        counts[dom_bin(d)] += 1
    counts += float(alpha)
    w = counts / counts.mean()
    return w

# -----------------------------
# Uploads
# -----------------------------
hits_file = st.file_uploader("Upload HIT history (TXT or CSV) — may include Fireball lines", type=["txt", "csv"])
if not hits_file:
    st.stop()

hits = parse_upload(hits_file)
if hits.empty:
    st.error("Could not parse the file into date/state/game/result.")
    st.stop()

hits["family"] = hits["result"].apply(infer_family)
hits = hits[hits["family"].isin(FAMILY_NAMES)].copy()
if hits.empty:
    st.error("Parsed file, but found 0 rows matching families 3389/3889/3899.")
    st.stop()

# Optional playable list (NOT required for prediction)
playable_file = st.file_uploader(
    "Optional: Upload a Playable list (CSV with columns State,Game) to MARK playable streams (no filtering).",
    type=["csv"]
)
playable_set = set()
if playable_file:
    p = pd.read_csv(playable_file)
    p.columns = [c.strip() for c in p.columns]
    if "State" in p.columns and "Game" in p.columns:
        playable_set = set((str(a).strip(), str(b).strip()) for a, b in zip(p["State"], p["Game"]))

# Window + prediction date
hits = hits.sort_values("date")
start_date = hits["date"].min().date()
end_date = hits["date"].max().date()
days_window = (end_date - start_date).days + 1
prediction_date = end_date

st.caption(
    f"History USED: {start_date} → {end_date} | rows: {len(hits):,} | streams found: {hits.groupby(['state','game']).ngroups:,} | prediction ref date: {prediction_date}"
)

# -----------------------------
# Controls (weights fixed to your request; pattern smoothing adjustable)
# -----------------------------
with st.expander("Model controls (the scoring weights are fixed to your approved mix)", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        alpha_pat = st.slider("Pattern smoothing (weekday/month). Higher = weaker boost.", 0.25, 5.0, 1.0, 0.25)
        schedule_mode = st.selectbox(
            "ScheduleBoost mode",
            ["Multiply (weekday*month)", "Average ((weekday+month)/2)"],
            index=0
        )
    with c2:
        st.write("**Fixed scoring weights**")
        st.write("- 0.50 HitRate")
        st.write("- 0.30 OverduePercentile (tempered by GapProximity)")
        st.write("- 0.10 Reliability")
        st.write("- 0.10 ScheduleBoost")

# -----------------------------
# Build per-stream tables
# -----------------------------
stream_dates = {}
for (state, game), g in hits.groupby(["state", "game"], sort=False):
    stream_dates[(state, game)] = list(pd.to_datetime(g["date"]).dt.date)

rows = []
today_wd = prediction_date.weekday()
today_db = dom_bin(prediction_date)

for (state, game), dates in stream_dates.items():
    dates_sorted = sorted(dates)
    hits_n = len(dates_sorted)
    last_hit = dates_sorted[-1]
    drought = (prediction_date - last_hit).days

    # base frequency
    hit_rate = hits_n / max(days_window, 1)

    # reliability (sample size stabilizer)
    reliability = float(np.log1p(hits_n))

    # due features (stream-normalized)
    gaps = gaps_in_days(dates_sorted)
    over_p = overdue_percentile(drought, gaps)
    prox = gap_proximity_score(drought, gaps)

    # "tempered due" so extreme droughts don't dominate
    # If a stream is very overdue but far from its typical gaps, this reduces the due effect.
    tempered_overdue = over_p * (0.40 + 0.60 * prox)  # keeps some due signal but enforces sweet-spot

    # schedule boost (learned from hit dates)
    wday_w = weekday_weights(dates_sorted, alpha=float(alpha_pat))
    dom_w = dom_weights(dates_sorted, alpha=float(alpha_pat))
    weekday_boost = float(wday_w[today_wd])
    dom_boost = float(dom_w[today_db])

    if schedule_mode.startswith("Multiply"):
        sched_boost = weekday_boost * dom_boost
    else:
        sched_boost = 0.5 * (weekday_boost + dom_boost)

    # shares
    gg = hits[(hits["state"] == state) & (hits["game"] == game)]
    fam_counts = gg["family"].value_counts().to_dict()
    share_3389 = fam_counts.get("3389", 0) / hits_n if hits_n else 0.0
    share_3889 = fam_counts.get("3889", 0) / hits_n if hits_n else 0.0
    share_3899 = fam_counts.get("3899", 0) / hits_n if hits_n else 0.0

    playable = "Unknown"
    if playable_set:
        playable = "Yes" if (state, game) in playable_set else "No"

    # simple next-hit estimate from median gap (for pattern detection)
    # (Not used in Score; just informational)
    if gaps:
        med_gap = int(round(float(np.median(gaps))))
        next_est = last_hit + timedelta(days=med_gap)
    else:
        med_gap = None
        next_est = None

    rows.append({
        "State": state,
        "Game": game,
        "PlayableByUser": playable,
        "Hits": hits_n,
        "LastHitDate": last_hit,
        "DaysSinceLastHit": drought,
        "HitRate": hit_rate,
        "OverduePercentile": over_p,
        "GapProximity": prox,
        "TemperedOverdue": tempered_overdue,
        "WeekdayBoostToday": weekday_boost,
        "DayOfMonthBoostToday": dom_boost,
        "ScheduleBoostToday": sched_boost,
        "Reliability": reliability,
        "NextHit_EstDate_MedianGap": next_est,
        "MedianGapDays": med_gap,
        "Share_3389": share_3389,
        "Share_3889": share_3889,
        "Share_3899": share_3899,
    })

df = pd.DataFrame(rows)

# Normalize components
df["n_rate"] = normalize_01(df["HitRate"])
df["n_due"]  = normalize_01(df["TemperedOverdue"])
df["n_rel"]  = normalize_01(df["Reliability"])
df["n_sched"] = normalize_01(df["ScheduleBoostToday"])

# Fixed scoring weights (your request)
df["Score"] = (
    0.50 * df["n_rate"] +
    0.30 * df["n_due"] +
    0.10 * df["n_rel"] +
    0.10 * df["n_sched"]
)

df = df.sort_values("Score", ascending=False).reset_index(drop=True)
df.insert(0, "Rank", df.index + 1)

# -----------------------------
# Coverage estimate (top-K)
# -----------------------------
def coverage_table(df_ranked, max_k=60):
    max_k = min(max_k, len(df_ranked))
    p_list = df_ranked["HitRate"].to_numpy()
    prod = 1.0
    rows = []
    for k in range(1, max_k + 1):
        prod *= (1.0 - float(p_list[k-1]))
        p_at_least_one = 1.0 - prod
        expected_hits = float(np.sum(p_list[:k]))
        rows.append({
            "K_streams_per_day": k,
            "P(catch ≥1 winner today)": p_at_least_one,
            "Expected winners per day (anywhere)": expected_hits,
            "BoxOnlyCost_per_day_$ (K * 0.75)": 0.75 * k,
        })
    return pd.DataFrame(rows)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Master Prediction List", "Schedule Planner", "Straight Estimator (optional upload)"])

with tab1:
    st.markdown("### Master Prediction List (Most → Least likely)")
    st.caption("Score uses safer due logic: tempered overdue + sweet-spot control (GapProximity).")
    st.dataframe(
        df[[
            "Rank","State","Game","PlayableByUser","Score",
            "Hits","LastHitDate","DaysSinceLastHit",
            "HitRate","OverduePercentile","GapProximity","TemperedOverdue",
            "ScheduleBoostToday","WeekdayBoostToday","DayOfMonthBoostToday",
            "NextHit_EstDate_MedianGap","MedianGapDays",
            "Share_3389","Share_3889","Share_3899"
        ]],
        use_container_width=True,
        height=650
    )

    st.download_button(
        "Download master predictions (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="pk4_master_predictions_v2.csv",
        mime="text/csv"
    )

    st.markdown("### Coverage vs Top-K (scale-back tool)")
    cov = coverage_table(df, max_k=60)
    st.dataframe(cov, use_container_width=True, height=420)
    st.download_button(
        "Download coverage table (CSV)",
        data=cov.to_csv(index=False).encode("utf-8"),
        file_name="pk4_coverage_vs_k_v2.csv",
        mime="text/csv"
    )

with tab2:
    st.markdown("### Schedule Planner (next 21 days)")
    st.caption("This does NOT force you to play only these days — it shows where the model sees the best timing advantages.")

    horizon_days = 21
    schedule_rows = []

    # Precompute per-stream weekday/dom weights so we can score day-by-day properly
    per_stream_wday = {}
    per_stream_dom = {}
    for (state, game), dates in stream_dates.items():
        dates_sorted = sorted(dates)
        per_stream_wday[(state, game)] = weekday_weights(dates_sorted, alpha=float(alpha_pat))
        per_stream_dom[(state, game)] = dom_weights(dates_sorted, alpha=float(alpha_pat))

    for k in range(0, horizon_days):
        d = prediction_date + timedelta(days=k)
        wd = d.weekday()
        db = dom_bin(d)

        tmp = df.copy()

        # recompute schedule boost for this day
        boosts = []
        for r in tmp.itertuples(index=False):
            wday_w = per_stream_wday[(r.State, r.Game)]
            dom_w = per_stream_dom[(r.State, r.Game)]
            weekday_boost = float(wday_w[wd])
            dom_boost = float(dom_w[db])
            if schedule_mode.startswith("Multiply"):
                boosts.append(weekday_boost * dom_boost)
            else:
                boosts.append(0.5 * (weekday_boost + dom_boost))
        tmp["ScheduleBoostDay"] = boosts

        tmp["n_sched_day"] = normalize_01(tmp["ScheduleBoostDay"])

        # same fixed scoring, but swap in the day-specific schedule component
        tmp["ScoreDay"] = (
            0.50 * tmp["n_rate"] +
            0.30 * tmp["n_due"] +
            0.10 * tmp["n_rel"] +
            0.10 * tmp["n_sched_day"]
        )

        tmp = tmp.sort_values("ScoreDay", ascending=False).head(25)
        tmp = tmp.reset_index(drop=True)
        tmp["RankThatDay"] = tmp.index + 1
        tmp["Date"] = d
        tmp["Weekday"] = WEEKDAYS[wd]

        schedule_rows.append(tmp[[
            "Date","Weekday","RankThatDay","State","Game","PlayableByUser","ScoreDay","ScheduleBoostDay",
            "OverduePercentile","GapProximity","HitRate","LastHitDate"
        ]])

    sched = pd.concat(schedule_rows, ignore_index=True)
    st.dataframe(sched, use_container_width=True, height=650)

    st.download_button(
        "Download schedule (CSV)",
        data=sched.to_csv(index=False).encode("utf-8"),
        file_name="pk4_schedule_next_21_days_v2.csv",
        mime="text/csv"
    )

with tab3:
    st.markdown("### Straight Estimator (kept; only runs when you upload 24-month stream history)")
    st.info("Upload a 24-month per-stream file (same columns: date/state/game/result). The app will rank all 12 straights per family for the selected stream.")

    stream_file = st.file_uploader("Upload 24-month per-stream history (TXT or CSV)", type=["txt","csv"], key="stream_upload")
    if not stream_file:
        st.stop()

    stream = parse_upload(stream_file)
    if stream.empty:
        st.error("Could not parse the stream history file.")
        st.stop()

    stream["result"] = stream["result"].apply(zfill4)
    stream["family"] = stream["result"].apply(infer_family)
    stream = stream[stream["family"].isin(FAMILY_NAMES)].copy()
    if stream.empty:
        st.error("Stream file parsed, but contains 0 rows for families 3389/3889/3899.")
        st.stop()

    stream_pairs = sorted(stream.groupby(["state","game"]).groups.keys())
    sel = st.selectbox("Pick a State/Game to rank straights:", stream_pairs, format_func=lambda x: f"{x[0]} — {x[1]}")
    sstate, sgame = sel
    s = stream[(stream["state"] == sstate) & (stream["game"] == sgame)].copy()

    def perms_for_family(name):
        fam_key = None
        for k, v in FAMILIES.items():
            if v == name:
                fam_key = k
                break
        perms = sorted(set("".join(p) for p in itertools.permutations(fam_key, 4)))
        return perms

    alpha = 1.0
    tables = {}
    for fam in FAMILY_NAMES:
        sf = s[s["family"] == fam].copy()
        perms = perms_for_family(fam)
        total = len(sf)
        counts = sf["result"].value_counts().to_dict()
        rows = []
        for p in perms:
            c = int(counts.get(p, 0))
            prob = (c + alpha) / (total + alpha * len(perms)) if total else 1.0 / len(perms)
            rows.append({"Straight": p, "Count": c, "Prob": float(prob)})
        t = pd.DataFrame(rows).sort_values("Prob", ascending=False).reset_index(drop=True)
        t.insert(0, "Rank", t.index + 1)
        tables[fam] = t

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 3389 — 12 straights")
        st.dataframe(tables["3389"], use_container_width=True, height=520)
    with c2:
        st.markdown("#### 3889 — 12 straights")
        st.dataframe(tables["3889"], use_container_width=True, height=520)
    with c3:
        st.markdown("#### 3899 — 12 straights")
        st.dataframe(tables["3899"], use_container_width=True, height=520)
