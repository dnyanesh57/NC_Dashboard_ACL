# digiqc_dashboard_NC_V2.7_SJCPL.py
# Digital NC Register — Streamlit (SJCPL Brand)
# - Keeps V2.6 working framework intact
# - Fixes fullscreen legend/title overlap (legend docked below + safe margins)
# - Adds donuts, sunburst, heatmaps, Pareto, backlog area, calendar heatmap, stacked mixes, bubble chart
# - NEW: Activity-Wise also shows "Normal (Non-R2C)" and "All NCs" sections for recurring-issue tracking
# - Brand-locked palette: Blue(#00AEDA), Black(#000000), Grey(#939598), White(#FFFFFF)
# --------------------------------------------------------------
# Run:
#   pip install -U streamlit plotly pandas numpy openpyxl
#   streamlit run digiqc_dashboard_NC_V2.7_SJCPL.py

from typing import Optional, Any, Tuple, List
import datetime as dt
import re, io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------- Safe Styler import (older pandas compatible) ----------
try:
    from pandas.io.formats.style import Styler  # type: ignore
except Exception:
    Styler = Any  # type: ignore

# ---------- Page ----------
st.set_page_config(page_title="Digital NC Register — SJCPL", page_icon="🧭", layout="wide")

# ---------- SJCPL Brand (locked) ----------
WHITE = "#FFFFFF"
BLACK = "#000000"
GREY  = "#939598"
BLUE  = "#00AEDA"

# Helpers to create brand-based gradient colours (no extra hues)
def _hex_to_rgb(h: str) -> Tuple[int,int,int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0,2,4))  # type: ignore

def _rgb_to_hex(r: Tuple[int,int,int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*r)

def blend(c1: str, c2: str, t: float) -> str:
    r1,g1,b1 = _hex_to_rgb(c1); r2,g2,b2 = _hex_to_rgb(c2)
    r = int(round(r1 + (r2 - r1)*t))
    g = int(round(g1 + (g2 - g1)*t))
    b = int(round(b1 + (b2 - b1)*t))
    return _rgb_to_hex((max(0,min(255,r)), max(0,min(255,g)), max(0,min(255,b))))

# Build a long, non-repeating sequence using only brand endpoints via gradients
def distinct_brand_colors(n: int) -> List[str]:
    """Return n visually distinct colours using only BLUE↔BLACK↔GREY↔WHITE gradients.
       Ensures the 4 base swatches appear at most once each."""
    anchors = [BLUE, BLACK, GREY, WHITE]
    seq: List[str] = []
    order = [BLUE, BLACK, GREY, WHITE]
    for c in order:
        if len(seq) < n:
            seq.append(c)
    if len(seq) >= n:
        return seq[:n]
    legs = [(BLUE, BLACK), (BLACK, GREY), (GREY, WHITE), (WHITE, BLUE)]
    needed = n - len(seq)
    steps_per_leg = max(2, int(np.ceil(needed / len(legs))) + 1)
    extras: List[str] = []
    for a,b in legs:
        for i in range(1, steps_per_leg):
            t = i/steps_per_leg
            extras.append(blend(a,b,t))
    seen = set(x.upper() for x in seq)
    out = [c for c in extras if c.upper() not in seen]
    seq.extend(out[:needed])
    return seq[:n]

# Status colours mapped to brand-only choices.
SJCPL_STATUS = {
    "Closed":     BLACK,      # Terminal / final
    "Resolved":   BLACK,      # Effective resolution (same visual as Closed)
    "Approved":   GREY,       # Neutral
    "In Process": BLUE,       # Active
    "In-Process": BLUE,       # Active
    "Open":       BLUE,       # Active
    "Redo":       GREY,       # Neutral/needs action
    "Rejected":   GREY,       # Neutral/negative mapped to grey
    "Responded":  BLUE,       # Progress event
}

SJCPL_METRICS = {
    "Total":    BLUE,
    "Resolved": BLACK,
    "R2C":      GREY,
    "Open":     BLUE,
    "RespOnly": GREY,
}

# Continuous gradient using only brand colours
BRAND_GRADIENT = [
    [0.0, WHITE],
    [0.5, BLUE],
    [1.0, BLACK],
]

THEMES = {
    "SJCPL": {
        "template": "plotly_white",
        "status_map": SJCPL_STATUS,
        "metric_map": SJCPL_METRICS,
        "gradient": BRAND_GRADIENT
    }
}
theme = "SJCPL"

# ---------- Plotly styling (fullscreen-safe legend) ----------
def style_fig(fig, theme_name: str):
    fig.update_layout(
        template=THEMES[theme_name]["template"],
        font=dict(family="Roboto, Arial, sans-serif", size=12, color=BLACK),

        # Legend docked below the plot area — avoids title/subtitle overlap in fullscreen
        legend=dict(
            title='',
            orientation="h",
            yanchor="top",
            y=-0.28,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(0,0,0,0.08)",
            borderwidth=1
        ),

        hoverlabel=dict(bgcolor="rgba(255,255,255,0.95)", 
                        font=dict(family="Roboto, Arial, sans-serif", color=BLACK)),
        bargap=0.22,

        # Extra headroom/footroom for fullscreen
        margin=dict(l=10, r=10, t=90, b=120),
        autosize=True
    )

    # Axes & grids
    fig.update_xaxes(showgrid=True, gridcolor="#ECEFF1", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#ECEFF1", zeroline=False)

    # ✅ Only apply marker line to trace types that actually support it
    _safe_types = ("bar", "histogram", "scatter", "pie", "box", "violin", "treemap", "sunburst")
    for t in _safe_types:
        fig.update_traces(
            selector=dict(type=t),
            marker=dict(line=dict(width=0.5, color="rgba(0,0,0,0.20)"))
        )

    return fig


def show_chart(fig, key: str):
    try:
        st.plotly_chart(fig, use_container_width=True, key=key)
    except Exception as e:
        st.warning(f"Chart failed ({key}): {e}")

# ---------- Explicit date/time parsing ----------
_date_ddmmyyyy_slash = re.compile(r"^\s*(\d{1,2})/(\d{1,2})/(\d{2,4})\s*$")
_date_ddmmyyyy_dash  = re.compile(r"^\s*(\d{1,2})-(\d{1,2})-(\d{2,4})\s*$")
_date_yyyymmdd_dash  = re.compile(r"^\s*(\d{4})-(\d{1,2})-(\d{1,2})\s*$")
_time_hhmm_ampm      = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([ap]\.?m\.?)\s*$", re.I)
_time_hhmm_24        = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$")

def _norm_year(y: int) -> int:
    if y < 100: return 2000 + y if y < 70 else 1900 + y
    return y

def _normalize_date_str(s: str) -> str:
    if not s or s.lower() in ("nan", "nat", "none"): return ""
    s = s.strip()
    m = _date_ddmmyyyy_slash.match(s) or _date_ddmmyyyy_dash.match(s)
    if m:
        d, mth, y = int(m.group(1)), int(m.group(2)), _norm_year(int(m.group(3)))
        try: return f"{y:04d}-{mth:02d}-{d:02d}"
        except Exception: return ""
    m = _date_yyyymmdd_dash.match(s)
    if m:
        y, mth, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try: return f"{y:04d}-{mth:02d}-{d:02d}"
        except Exception: return ""
    return ""

def _normalize_time_str(s: str) -> str:
    if not s or s.lower() in ("nan", "nat", "none"): return ""
    s0 = s.strip().lower().replace(".", "")
    m = _time_hhmm_ampm.match(s0)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2)); ss = int(m.group(3)) if m.group(3) else 0
        ampm = m.group(4).lower()
        if "pm" in ampm and hh < 12: hh += 12
        if "am" in ampm and hh == 12: hh = 0
        if not (0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59): return ""
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    m = _time_hhmm_24.match(s0)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2)); ss = int(m.group(3)) if m.group(3) else 0
        if not (0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59): return ""
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    return ""

def _normalize_series_date(s: pd.Series) -> pd.Series:
    return s.astype(str).map(_normalize_date_str)

def _normalize_series_time(s: pd.Series) -> pd.Series:
    return s.astype(str).map(_normalize_time_str)

def combine_datetime(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    d = _normalize_series_date(date_series)
    t = _normalize_series_time(time_series)
    has_date = d != ""
    t = np.where((t == "") & has_date, "00:00:00", t)
    full = np.where(has_date, d + " " + t, "")
    out = pd.to_datetime(full, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    return out  # keep naive

# ---------- Business rules ----------
def _safe_get(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] if col in df.columns else pd.Series([np.nan] * len(df), index=df.index, name=col)

def extract_location_variable(raw: pd.Series) -> pd.Series:
    def _extract(val):
        if pd.isna(val): return val
        s = str(val)
        return s.rsplit("/", 1)[-1].strip() if "/" in s else s.strip()
    return raw.apply(_extract)

def humanize_td(td: pd.Series) -> pd.Series:
    def _fmt(x):
        if pd.isna(x): return ""
        total_seconds = int(x.total_seconds())
        if total_seconds < 0: return ""
        days, rem = divmod(total_seconds, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, _ = divmod(rem, 60)
        parts = []
        if days: parts.append(f"{days}d")
        if hours: parts.append(f"{hours}h")
        if minutes: parts.append(f"{minutes}m")
        return " ".join(parts) if parts else "0m"
    return td.apply(_fmt)

def style_status_rows(df: pd.DataFrame, theme_name: str) -> Styler:
    status_map = THEMES[theme_name]["status_map"]
    def highlight(row):
        status = str(row.get("Current Status", "")).strip()
        bg = status_map.get(status, WHITE)
        txt = WHITE if bg in (BLACK, GREY) else BLACK
        return [f"background-color: {bg}; color: {txt};"] * len(row)
    try:
        return df.style.apply(highlight, axis=1)
    except Exception:
        return df  # fallback

def ensure_last_status_change(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee _LastStatusChangeDT/_LastStatusEvent exist, computing from available timestamps."""
    def ensure_series(df, out_col, dcol, tcol):
        if out_col not in df.columns:
            if dcol in df.columns or tcol in df.columns:
                df[out_col] = combine_datetime(_safe_get(df, dcol), _safe_get(df, tcol))
            else:
                df[out_col] = pd.NaT
        return df

    # Make sure the base datetime columns exist
    df = ensure_series(df, "_RaisedOnDT",    "Raised On Date",   "Raised On Time")
    df = ensure_series(df, "_RespondedOnDT", "Responded On Date","Responded On Time")
    df = ensure_series(df, "_RejectedOnDT",  "Rejected On Date", "Rejected On Time")
    df = ensure_series(df, "_ClosedOnDT",    "Closed On Date",   "Closed On Time")

    # Effective resolution (closed else valid responded>raised)
    if "_EffectiveResolutionDT" not in df.columns:
        eff = df["_ClosedOnDT"].copy()
        mask_eff = eff.isna() & df["_RespondedOnDT"].notna() & df["_RaisedOnDT"].notna() & (df["_RespondedOnDT"] > df["_RaisedOnDT"])
        eff.loc[mask_eff] = df.loc[mask_eff, "_RespondedOnDT"]
        df["_EffectiveResolutionDT"] = eff

    # Compute last status change + event
    ev_cols = ["_RespondedOnDT","_RejectedOnDT","_ClosedOnDT","_EffectiveResolutionDT"]
    existing = [c for c in ev_cols if c in df.columns]
    if existing:
        sentinel = pd.Timestamp("1900-01-01 00:00:00")
        evdf = df[existing].copy()
        evdf_f = evdf.fillna(sentinel)
        last_ts  = evdf_f.max(axis=1)
        none_mask = evdf.notna().sum(axis=1) == 0
        last_ts  = last_ts.mask(none_mask, pd.NaT)
        last_col = evdf_f.idxmax(axis=1)
        rev = {
            "_RespondedOnDT":"Responded",
            "_RejectedOnDT":"Rejected",
            "_ClosedOnDT":"Closed",
            "_EffectiveResolutionDT":"Effective",
        }
        df["_LastStatusChangeDT"] = last_ts
        df["_LastStatusEvent"]    = last_col.map(rev).where(~none_mask, None)
    else:
        df["_LastStatusChangeDT"] = pd.NaT
        df["_LastStatusEvent"]    = None
    return df

# ---------- Derived columns (effective closure + flags + last status change) ----------
def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure future/rolling-out fields exist
    for col in [
        "Root Cause Analysis","Correction","Corrective Action",
        "Labour Cost","Material Cost","Machinery Cost","Other Cost","Total Cost"
    ]:
        if col not in df.columns:
            df[col] = np.nan

    # Location Variable fix
    df["Location Variable (Fixed)"] = extract_location_variable(_safe_get(df, "Location Variable")) if "Location Variable" in df.columns else np.nan

    # Datetimes
    df["_RaisedOnDT"]     = combine_datetime(_safe_get(df, "Raised On Date"), _safe_get(df, "Raised On Time"))
    df["_DeadlineDT"]     = combine_datetime(_safe_get(df, "Deadline Date"), _safe_get(df, "Deadline Time"))
    df["_RespondedOnDT"]  = combine_datetime(_safe_get(df, "Responded On Date"), _safe_get(df, "Responded On Time"))
    df["_RejectedOnDT"]   = combine_datetime(_safe_get(df, "Rejected On Date"), _safe_get(df, "Rejected On Time"))
    df["_ClosedOnDT"]     = combine_datetime(_safe_get(df, "Closed On Date"), _safe_get(df, "Closed On Time"))

    # Effective resolution (vectorized): closed else responded>raised
    eff = df["_ClosedOnDT"].copy()
    mask_eff = eff.isna() & df["_RespondedOnDT"].notna() & df["_RaisedOnDT"].notna() & (df["_RespondedOnDT"] > df["_RaisedOnDT"])
    eff.loc[mask_eff] = df.loc[mask_eff, "_RespondedOnDT"]
    df["_EffectiveResolutionDT"] = eff

    # Timings
    df["Computed Closure Time (Hrs)"] = (df["_EffectiveResolutionDT"] - df["_RaisedOnDT"]).dt.total_seconds() / 3600.0
    df["Responding Time (Hrs)"]       = (df["_RespondedOnDT"] - df["_RaisedOnDT"]).dt.total_seconds() / 3600.0

    # Responded but NOT Closed
    df["_RespondedNotClosed_Flag"] = (
        df["_ClosedOnDT"].isna() &
        df["_RespondedOnDT"].notna() & df["_RaisedOnDT"].notna() &
        (df["_RespondedOnDT"] > df["_RaisedOnDT"])
    ).astype(int)

    # Close-after-response
    mask_car = df["_EffectiveResolutionDT"].notna() & df["_RespondedOnDT"].notna() & (df["_EffectiveResolutionDT"] >= df["_RespondedOnDT"])
    df["Close After Response (Hrs)"] = np.where(
        mask_car,
        (df["_EffectiveResolutionDT"] - df["_RespondedOnDT"]).dt.total_seconds() / 3600.0,
        np.nan
    )

    # SLA
    df["SLA Met"] = np.where(
        df["_DeadlineDT"].notna() & df["_EffectiveResolutionDT"].notna(),
        df["_EffectiveResolutionDT"] <= df["_DeadlineDT"],
        np.nan
    )

    # Total cost fallback
    parts = ["Labour Cost","Material Cost","Machinery Cost","Other Cost"]
    if "Total Cost" in df.columns:
        part_sum = df[[p for p in parts if p in df.columns]].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        df["Total Cost"] = pd.to_numeric(df["Total Cost"], errors="coerce").fillna(part_sum)

    # -------- Rejected → Closed flags (inferred + strict) --------
    def _nz(series_name: str) -> pd.Series:
        s = _safe_get(df, series_name)
        return s.where(s.notna(), "").astype(str).str.strip()

    has_reject_evidence = (
        df["_RejectedOnDT"].notna() |
        _nz("Rejected By").ne("") | _nz("Rejected Comment").ne("") |
        _nz("Rejected On Date").ne("") | _nz("Rejected On Time").ne("")
    )
    cur_status = _nz("Current Status").str.lower()
    closedish  = cur_status.str.contains(r"\b(closed|approved|resolved|complete)\b", regex=True)
    has_close_evidence = (
        df["_ClosedOnDT"].notna() | closedish |
        _nz("Closed By").ne("") | _nz("Closed Comment").ne("") |
        _nz("Closed On Date").ne("") | _nz("Closed On Time").ne("")
    )
    df["_R2C_Flag"] = (has_reject_evidence & has_close_evidence).astype(int)

    both_dt = df["_RejectedOnDT"].notna() & df["_ClosedOnDT"].notna()
    df["_R2C_Strict_Flag"] = both_dt.astype(int)
    dur_hours = np.where(both_dt, (df["_ClosedOnDT"] - df["_RejectedOnDT"]).dt.total_seconds() / 3600.0, np.nan)
    df["R2C Hours (>=0)"] = np.where(np.isfinite(dur_hours), np.maximum(dur_hours, 0.0), np.nan)

    # -------- Last Status Change (vectorized) --------
    event_cols = {
        "Responded": "_RespondedOnDT",
        "Rejected":  "_RejectedOnDT",
        "Closed":    "_ClosedOnDT",
        "Effective": "_EffectiveResolutionDT",
    }
    available = [c for c in event_cols.values() if c in df.columns]
    if available:
        sentinel = pd.Timestamp("1900-01-01 00:00:00")
        evdf = df[available].copy()
        evdf_f = evdf.fillna(sentinel)
        last_ts = evdf_f.max(axis=1)
        none_mask = evdf.notna().sum(axis=1) == 0
        last_ts = last_ts.mask(none_mask, pd.NaT)
        last_col = evdf_f.idxmax(axis=1)
        last_col = last_col.mask(none_mask, None)
        rev = {v: k for k, v in event_cols.items()}
        df["_LastStatusChangeDT"] = last_ts
        df["_LastStatusEvent"] = last_col.map(rev).where(~none_mask, None)
    else:
        df["_LastStatusChangeDT"] = pd.NaT
        df["_LastStatusEvent"] = None

    # -------- Calendar splits for extra timelines --------
    if df["_RaisedOnDT"].notna().any():
        df["_RaisedDate"] = df["_RaisedOnDT"].dt.date
        df["_RaisedWeek"] = df["_RaisedOnDT"].dt.to_period("W").astype(str)
        df["_RaisedDOW"]  = df["_RaisedOnDT"].dt.day_name()
        df["_RaisedHour"] = df["_RaisedOnDT"].dt.hour
    else:
        df["_RaisedDate"] = np.nan
        df["_RaisedWeek"] = np.nan
        df["_RaisedDOW"]  = np.nan
        df["_RaisedHour"] = np.nan

    return df

def _to_label(x) -> str:
    if pd.isna(x): return "—"
    if isinstance(x, (list, tuple, set)):
        return ", ".join("" if pd.isna(y) else str(y).strip() for y in x)
    s = str(x).strip()
    return s if s else "—"

def bar_top_counts(df: pd.DataFrame, col: str, topn: int = 10, template="plotly_white", theme_name: str="SJCPL"):
    if col not in df.columns:
        return px.bar(pd.DataFrame({col: [], "count": []}), x="count", y=col, template=template)
    labels = df[col].apply(_to_label)
    vc = labels.value_counts(dropna=False).head(topn)
    counts = pd.DataFrame({col: vc.index.astype(str).tolist(), "count": vc.values})
    fig = px.bar(
        counts.sort_values("count", ascending=True),
        x="count", y=col, orientation="h",
        color=col, color_discrete_sequence=distinct_brand_colors(len(counts)),
        template=template, text_auto=True
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return style_fig(fig, theme_name)

def _clean_hours_for_hist(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if len(x) > 0:
        cap = np.nanpercentile(x, 99)
        x = x[x <= cap]
    return x

def metrics_summary(df: pd.DataFrame, theme_name: str):
    total_issues = len(df)
    resolved = (df["_EffectiveResolutionDT"].notna()).sum()
    open_issues = total_issues - resolved
    median_response = pd.to_timedelta(df["Responding Time (Hrs)"], unit="h").median(skipna=True)
    median_close    = pd.to_timedelta(df["Computed Closure Time (Hrs)"], unit="h").median(skipna=True)
    sla_known = df["SLA Met"].dropna() if "SLA Met" in df.columns else pd.Series(dtype=float)
    sla_rate = (sla_known.mean() * 100) if len(sla_known) else np.nan

    def _fmt(td):
        if pd.isna(td): return "—"
        secs = int(td.total_seconds())
        d, r = divmod(secs, 86400); h, r = divmod(r, 3600); m, _ = divmod(r, 60)
        if d or h or m: return f"{d}d {h}h {m}m"
        return "0m"

    t1, t2, t3, t4, t5, t6 = st.columns(6)
    with t1: st.metric("Total Issues", f"{total_issues}")
    with t2: st.metric("Resolved (Closed/Effective)", f"{resolved}")
    with t3: st.metric("Open / Unresolved", f"{open_issues}")
    with t4: st.metric("Median Closure Time", _fmt(median_close))
    with t5: st.metric("Median Responding Time", _fmt(median_response))
    with t6: st.metric("SLA Met Rate", f"{sla_rate:.1f}%" if pd.notna(sla_rate) else "—")

# ---------- Header (robust gradient + white text) ----------
APP_TITLE = "🧭 Digital — NC Insights Dashboard"
APP_SUB   = "SJCPL visual theme · Roboto · Brand colors only"
HEADER_BG = f"linear-gradient(90deg, {BLACK} 0%, {BLUE} 100%)"
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');
html, body, [class*="css"], .stApp {{
  font-family: 'Roboto', sans-serif;
}}
div#sj-titlebar {{
  background-image: {HEADER_BG} !important;
  background-color: {BLACK} !important;
  color: {WHITE} !important;
  padding: 14px 18px; border-radius: 14px; margin: 6px 0 18px 0;
}}
div#sj-titlebar h1, div#sj-titlebar p {{
  color: {WHITE} !important; margin: 0;
}}
div#sj-titlebar p {{ margin-top: 4px; opacity: .9; }}
</style>
<div id="sj-titlebar">
  <h1>{APP_TITLE}</h1>
  <p>{APP_SUB}</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar (brand-locked) ----------
with st.sidebar:
    st.title("🧭 Digital NC Register")
    st.caption("SJCPL brand-locked UI (Roboto + Blue/Black/Grey/White)")
    logo_url = st.text_input("Logo URL (optional)", value="")
    st.markdown("#### Data Source")
    uploaded = st.file_uploader("Upload Issue Register (CSV/XLSX)", type=["csv","xlsx","xls"], key="uploader")

# ---------- Load Data (requested logic) ----------
def normalize_colname(c: str) -> str:
    """Keep official column labels intact, but clean whitespace and unicode quirks."""
    s = str(c).replace("\u2013", "-").replace("\u2014", "-").replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_data(file: Optional[io.BytesIO]) -> pd.DataFrame:
    """
    Robust reader:
      - If file is None: pull demo CSV from GitHub
      - Else: try Excel, else CSV with multiple encodings
      - Normalize column names and drop duplicate columns
    """
    if file is None:
        default_path = "https://raw.githubusercontent.com/dnyanesh57/NC_Dashboard/main/data/CSV-INSTRUCTION-DETAIL-REPORT-09-15-2025-09-30-25.csv"
        try:
            df = pd.read_csv(default_path)
        except Exception:
            st.error("No file uploaded and demo CSV not available.")
            st.stop()
    else:
        name = getattr(file, "name", "uploaded.csv").lower()
        if name.endswith(".xlsx") or name.endswith(".xls"):
            df = pd.read_excel(file)
        else:
            for enc in [None, "utf-8", "utf-8-sig", "latin-1"]:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding=enc)
                    break
                except Exception:
                    continue
            else:
                st.error("Could not read the uploaded CSV with common encodings.")
                st.stop()
    df = df.rename(columns={c: normalize_colname(c) for c in df.columns})
    df = df.loc[:, ~pd.Series(df.columns).duplicated().values]
    return df

@st.cache_data(show_spinner=False)
def preprocess(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df.columns = [c.strip() for c in df.columns]
    return add_derived_columns(df)

if uploaded is not None:
    df_raw = load_data(uploaded)
else:
    df_raw = load_data(None)

df = preprocess(df_raw)

# ---------- Show logo if provided ----------
st.markdown(
    f"""
    <div style="display:flex;justify-content:flex-end;margin:-6px 0 8px 0;">
      {"<img style='height:32px;object-fit:contain;' src='" + logo_url + "' />" if logo_url else ""}
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- Global filters ----------
def get_date_range_inputs(df: pd.DataFrame) -> Tuple[Optional[dt.date], Optional[dt.date]]:
    enable = st.checkbox("Enable Raised On date filter", value=False, key="datefilter-enable")
    if not enable:
        return None, None
    if df["_RaisedOnDT"].notna().any():
        dmin = pd.to_datetime(df["_RaisedOnDT"].min()).date()
        dmax = pd.to_datetime(df["_RaisedOnDT"].max()).date()
    else:
        today = dt.date.today()
        dmin = dmax = today
    picked = st.date_input("Raised On — Range", value=(dmin, dmax), key="datefilter-range")
    if isinstance(picked, tuple) and len(picked) == 2:
        return picked[0], picked[1]
    return picked, picked

def filtered_view(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.markdown("#### Filters")
        def options(col: str):
            return sorted(df.get(col, pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        proj          = st.multiselect("Project Name", options("Project Name"), key="f-proj")
        status        = st.multiselect("Current Status", options("Current Status"), key="f-status")
        types_l0      = st.multiselect("Type L0", options("Type L0"), key="f-typeL0")
        types_l1      = st.multiselect("Type L1", options("Type L1"), key="f-typeL1")
        types_l2      = st.multiselect("Type L2", options("Type L2"), key="f-typeL2")
        tags1         = st.multiselect("Tag 1", options("Tag 1"), key="f-tag1")
        tags2         = st.multiselect("Tag 2", options("Tag 2"), key="f-tag2")
        raised_by     = st.multiselect("Raised By", options("Raised By"), key="f-raisedby")
        assigned_team = st.multiselect("Assigned Team", options("Assigned Team"), key="f-ateam")
        assigned_user = st.multiselect("Assigned Team User", options("Assigned Team User"), key="f-auser")
        date_min, date_max = get_date_range_inputs(df)

    m = np.ones(len(df), dtype=bool)
    def match(col: str, sel: list):
        if sel:
            return df.get(col).astype(str).isin([str(x) for x in sel]).to_numpy()
        return np.ones(len(df), dtype=bool)

    if proj:          m &= match("Project Name", proj)
    if status:        m &= match("Current Status", status)
    if types_l0:      m &= match("Type L0", types_l0)
    if types_l1:      m &= match("Type L1", types_l1)
    if types_l2:      m &= match("Type L2", types_l2)
    if tags1:         m &= match("Tag 1", tags1)
    if tags2:         m &= match("Tag 2", tags2)
    if raised_by:     m &= match("Raised By", raised_by)
    if assigned_team: m &= match("Assigned Team", assigned_team)
    if assigned_user: m &= match("Assigned Team User", assigned_user)
    if date_min:      m &= (df["_RaisedOnDT"].dt.date >= date_min).fillna(False).to_numpy()
    if date_max:      m &= (df["_RaisedOnDT"].dt.date <= date_max).fillna(False).to_numpy()

    return df.loc[m].copy()

df_filtered = filtered_view(df)
st.divider()

metric_colors = THEMES[theme]["metric_map"]
status_colors = THEMES[theme]["status_map"]

mask_r2c_inferred   = (df_filtered["_R2C_Flag"] == 1)
mask_responly       = (df_filtered["_RespondedNotClosed_Flag"] == 1)

r2c_count_scope     = int(mask_r2c_inferred.sum())
resp_only_count     = int(mask_responly.sum())
total_ncs_scope     = len(df_filtered)

# ---------- Tabs (added "Status") ----------
tabs = st.tabs([
    "Overview",
    "Status",
    "Project Status",
    "Project Explorer",
    "Tower-Wise",
    "User-Wise",
    "Activity-Wise",
    "Timelines",
    "NC-View",
    "Sketch-View",
    "NC Table",
])

# ---------- Overview ----------
with tabs[0]:
    st.header("Overview")
    metrics_summary(df_filtered, theme)

    comp = pd.DataFrame({"Metric": ["Total","Rejected→Closed"], "Count": [total_ncs_scope, r2c_count_scope]})
    fig_comp = px.bar(
        comp.sort_values("Count"),
        x="Count", y="Metric", orientation="h",
        text_auto=True, title="Comparative — Total vs Rejected→Closed (inferred)",
        color="Metric",
        color_discrete_sequence=distinct_brand_colors(2)
    )
    show_chart(style_fig(fig_comp, theme), key="ov-comp-r2c")

    if "Assigned Team User" in df_filtered.columns and resp_only_count > 0:
        resp_counts = (df_filtered.loc[mask_responly, "Assigned Team User"]
                       .fillna("—").astype(str)
                       .value_counts().rename_axis("Assignee").reset_index(name="Responded not Closed"))
        fig_resp_only = px.bar(
            resp_counts.sort_values("Responded not Closed"),
            x="Responded not Closed", y="Assignee", orientation="h",
            title="Responded but NOT Closed — by Assignee",
            color_discrete_sequence=distinct_brand_colors(1)
        )
        show_chart(style_fig(fig_resp_only, theme), key="ov-responded-not-closed-assignee")

    if "Raised By" in df_filtered.columns and resp_only_count > 0:
        rb_counts = (df_filtered.loc[mask_responly, "Raised By"]
                     .fillna("—").astype(str)
                     .value_counts().rename_axis("Raised By").reset_index(name="Responded not Closed"))
        fig_rb = px.bar(
            rb_counts.sort_values("Responded not Closed"),
            x="Responded not Closed", y="Raised By", orientation="h",
            title="Responded but NOT Closed — by Raiser",
            color_discrete_sequence=distinct_brand_colors(1)
        )
        show_chart(style_fig(fig_rb, theme), key="ov-responded-not-closed-raisedby")

    if "Current Status" in df_filtered.columns:
        vc = df_filtered["Current Status"].value_counts(dropna=False).reset_index()
        vc.columns = ["Current Status","Count"]
        fig_sd = px.bar(vc, x="Current Status", y="Count", text_auto=True, title="Current Status Distribution",
                        color="Current Status", color_discrete_sequence=distinct_brand_colors(len(vc)))
        fig_sd.update_xaxes(tickangle=0, tickfont=dict(size=11))
        show_chart(style_fig(fig_sd, theme), key="ov-status-dist")

    st.subheader("Response vs Closure — Time Distributions")
    c3, c4 = st.columns(2)
    with c3:
        series = _clean_hours_for_hist(df_filtered["Responding Time (Hrs)"])
        if len(series):
            fig = px.histogram(
                x=series, nbins=30,
                labels={"x": "Responding Time (Hrs)"},
                opacity=0.9, color_discrete_sequence=[metric_colors["RespOnly"]]
            )
            fig.update_traces(showlegend=False)
            fig.update_layout(title="Responding Time (Hrs)")
            show_chart(style_fig(fig, theme), key="tab0-hist-responding")
        else:
            st.info("No data for Responding Time.")
    with c4:
        series = _clean_hours_for_hist(df_filtered["Computed Closure Time (Hrs)"])
        if len(series):
            fig = px.histogram(
                x=series, nbins=30,
                labels={"x": "Computed Closure Time (Hrs)"},
                opacity=0.9, color_discrete_sequence=[metric_colors["Resolved"]]
            )
            fig.update_traces(showlegend=False)
            fig.update_layout(title="Computed Closure Time (Hrs)")
            show_chart(style_fig(fig, theme), key="tab0-hist-closure")
        else:
            st.info("No data for Computed Closure Time.")

    # ---- Added Overview visuals (donuts + sunburst) ----
    st.subheader("More Overview Visuals")
    cA, cB, cC = st.columns(3)
    with cA:
        if "SLA Met" in df_filtered.columns and len(df_filtered):
            work = df_filtered.copy()
            work["SLA State"] = work["SLA Met"].map({True: "Met", False: "Missed"}).fillna("Unknown")
            vc = work["SLA State"].value_counts().rename_axis("SLA State").reset_index(name="Count")
            fig = px.pie(vc, names="SLA State", values="Count", hole=0.6,
                         color="SLA State", color_discrete_sequence=distinct_brand_colors(len(vc)))
            fig.update_layout(title="SLA — Met vs Missed vs Unknown")
            show_chart(style_fig(fig, theme), key="ov-sla-donut")
    with cB:
        if "_R2C_Flag" in df_filtered.columns and len(df_filtered):
            temp = df_filtered.copy()
            temp["R2C State"] = np.where(temp["_R2C_Flag"] == 1, "R2C", "Non-R2C")
            vc = temp["R2C State"].value_counts().rename_axis("R2C").reset_index(name="Count")
            fig = px.pie(vc, names="R2C", values="Count", hole=0.6,
                         color="R2C", color_discrete_sequence=distinct_brand_colors(len(vc)))
            fig.update_layout(title="R2C — Overall Split (inferred)")
            show_chart(style_fig(fig, theme), key="ov-r2c-donut")
    with cC:
        if all(c in df_filtered.columns for c in ["Type L0","Type L1","Type L2"]):
            sb = df_filtered.copy()
            for c in ["Type L0","Type L1","Type L2"]:
                sb[c] = sb[c].fillna("—").astype(str)
            fig = px.sunburst(sb, path=["Type L0","Type L1","Type L2"],
                              color="Type L0", color_discrete_sequence=distinct_brand_colors(sb["Type L0"].nunique()+2))
            fig.update_layout(title="Activity Hierarchy — L0 / L1 / L2")
            show_chart(style_fig(fig, theme), key="ov-sunburst")

# ---------- Status (extended) ----------
with tabs[1]:
    st.header("Status")
    st.caption("See which NCs changed status Today, in the Last 3 days, or in This week.")

    # Ensure required columns exist (robust even if preprocess didn’t run)
    df_filtered = ensure_last_status_change(df_filtered)
    last_change = pd.to_datetime(df_filtered["_LastStatusChangeDT"], errors="coerce")

    period = st.selectbox("Show changes from", ["Today", "Last 3 days", "This week", "All"], index=0, key="status-period")

    now = pd.Timestamp.now()
    today = now.normalize()
    start_of_week = today - pd.Timedelta(days=today.weekday())  # Monday

    if period == "Today":
        m = last_change.dt.normalize() == today
        window_label = f"Today ({today.date()})"
    elif period == "Last 3 days":
        cutoff = today - pd.Timedelta(days=2)  # today + previous 2 = 3 days incl. today
        m = last_change >= cutoff
        window_label = f"Last 3 days (since {cutoff.date()})"
    elif period == "This week":
        m = last_change >= start_of_week
        window_label = f"This week (since {start_of_week.date()})"
    else:
        m = last_change.notna()
        window_label = "All available"

    changed = df_filtered.loc[m & last_change.notna()].copy()

    count_changed = len(changed)
    total_in_scope = len(df_filtered)

    c1,c2,c3,cx = st.columns(4)
    with c1: st.metric("NCs with Status Change", f"{count_changed}")
    with c2: st.metric("Window", window_label)
    with c3:
        pct = (count_changed/total_in_scope*100.0) if total_in_scope else 0.0
        st.metric("% of Filtered NCs", f"{pct:.1f}%")
    with cx:
        if count_changed and changed["_LastStatusChangeDT"].notna().any():
            age_hrs = (now - changed["_LastStatusChangeDT"]).dt.total_seconds()/3600.0
            med = np.nanmedian(age_hrs) if np.isfinite(age_hrs).any() else np.nan
            st.metric("Median hrs since change", f"{med:.1f}" if pd.notna(med) else "—")

    # KPIs for R2C & Responded-not-Closed within window
    if count_changed:
        k1,k2 = st.columns(2)
        with k1:
            if "_R2C_Flag" in changed.columns:
                st.metric("R2C in window (inferred)", int(changed["_R2C_Flag"].sum()))
        with k2:
            if "_RespondedNotClosed_Flag" in changed.columns:
                st.metric("Responded-not-Closed in window", int(changed["_RespondedNotClosed_Flag"].sum()))

    if count_changed == 0:
        st.info("No status changes found for the selected period and filters.")
    else:
        # Bar: by Last Status Event
        evt_counts = (changed["_LastStatusEvent"].fillna("Unknown")
                      .value_counts().rename_axis("Event").reset_index(name="Count"))
        evt_colors = {
            "Responded": BLUE,
            "Rejected": GREY,
            "Closed": BLACK,
            "Effective": blend(BLUE, BLACK, 0.55),  # brand-based gradient tone
            "Unknown": blend(GREY, WHITE, 0.35)
        }
        fig_evt = px.bar(evt_counts, x="Event", y="Count", text_auto=True,
                         color="Event",
                         color_discrete_map=evt_colors)
        fig_evt.update_layout(title="Status Changes — by Event")
        show_chart(style_fig(fig_evt, theme), key="st-evt-bar")

        cA, cB = st.columns(2)
        with cA:
            # Top Projects
            if "Project Name" in changed.columns:
                top_proj = (changed["Project Name"].fillna("—").value_counts()
                            .rename_axis("Project").reset_index(name="Count")).head(15)
                fig_proj = px.bar(top_proj.sort_values("Count"),
                                  x="Count", y="Project", orientation="h",
                                  title="Status Changes — by Project (Top 15)",
                                  color_discrete_sequence=distinct_brand_colors(1))
                show_chart(style_fig(fig_proj, theme), key="st-proj-bar")
        with cB:
            # Top Assignees
            if "Assigned Team User" in changed.columns:
                top_user = (changed["Assigned Team User"].fillna("—").astype(str)
                            .value_counts().rename_axis("Assignee").reset_index(name="Count")).head(15)
                fig_user = px.bar(top_user.sort_values("Count"),
                                  x="Count", y="Assignee", orientation="h",
                                  title="Status Changes — by Assignee (Top 15)",
                                  color_discrete_sequence=distinct_brand_colors(1))
                show_chart(style_fig(fig_user, theme), key="st-user-bar")

        # Mini timeline: changes per day in window
        changed["Change Date"] = changed["_LastStatusChangeDT"].dt.date
        daily = changed.groupby("Change Date").size().reset_index(name="Count")
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=daily["Change Date"], y=daily["Count"], mode="lines+markers",
                                      line=dict(color=BLUE)))
        fig_line.update_layout(title="Daily Status Changes", xaxis_title="", yaxis_title="Count")
        show_chart(style_fig(fig_line, theme), key="st-daily-line")

        # --- Added daily breakdowns & age buckets ---
        tmp = changed.copy()
        tmp["Change Date"] = pd.to_datetime(tmp["_LastStatusChangeDT"]).dt.date

        if "_LastStatusEvent" in tmp.columns:
            ev_day = tmp.groupby(["Change Date","_LastStatusEvent"]).size().reset_index(name="Count")
            fig_ev_day = px.bar(ev_day, x="Change Date", y="Count", color="_LastStatusEvent",
                                color_discrete_sequence=distinct_brand_colors(ev_day["_LastStatusEvent"].nunique()+2))
            fig_ev_day.update_layout(title="Per-day changes by Event")
            show_chart(style_fig(fig_ev_day, theme), key="st-ev-perday")

        if "Current Status" in tmp.columns:
            st_day = tmp.groupby(["Change Date","Current Status"]).size().reset_index(name="Count")
            fig_st_day = px.bar(st_day, x="Change Date", y="Count", color="Current Status",
                                color_discrete_sequence=distinct_brand_colors(st_day["Current Status"].nunique()+2))
            fig_st_day.update_layout(title="Per-day changes by Status")
            show_chart(style_fig(fig_st_day, theme), key="st-status-perday")

        if "Project Name" in tmp.columns:
            top_p = tmp.groupby("Project Name").size().reset_index(name="Changes").sort_values("Changes", ascending=False).head(15)
            fig_top = px.bar(top_p.sort_values("Changes"), x="Changes", y="Project Name", orientation="h",
                             color_discrete_sequence=distinct_brand_colors(1), text_auto=True)
            fig_top.update_layout(title="Top Projects — # of Changes")
            show_chart(style_fig(fig_top, theme), key="st-top-projects")

        if "_R2C_Flag" in tmp.columns and "Project Name" in tmp.columns:
            r2 = tmp.groupby("Project Name")["_R2C_Flag"].sum().reset_index(name="R2C")
            r2 = r2[r2["R2C"]>0].sort_values("R2C", ascending=False).head(15)
            if len(r2):
                fig_r2p = px.bar(r2, x="Project Name", y="R2C",
                                 color_discrete_sequence=distinct_brand_colors(1), text_auto=True)
                fig_r2p.update_layout(title="R2C in window — by Project")
                show_chart(style_fig(fig_r2p, theme), key="st-r2c-proj")

        if changed["_LastStatusChangeDT"].notna().any():
            hours = (now - changed["_LastStatusChangeDT"]).dt.total_seconds()/3600.0
            bins = [0, 8, 24, 72, 168, 1e9]; labels = ["≤8h","8–24h","1–3d","3–7d",">7d"]
            buck = pd.cut(hours, bins=bins, labels=labels, right=True)
            vc = pd.Series(buck).value_counts().reindex(labels).reset_index()
            vc.columns = ["Age Bucket","Count"]
            fig_age = px.bar(vc, x="Age Bucket", y="Count", color="Age Bucket",
                             color_discrete_sequence=distinct_brand_colors(len(vc)), text_auto=True)
            fig_age.update_layout(title="Age since last change — Buckets")
            show_chart(style_fig(fig_age, theme), key="st-age-buckets")

        # Table of changed NCs
        st.subheader("Changed NCs")
        show_cols = [c for c in [
            "Reference ID","Project Name","Current Status",
            "_LastStatusEvent","_LastStatusChangeDT",
            "Assigned Team","Assigned Team User",
            "Raised On Date","Raised On Time"
        ] if c in changed.columns]
        if show_cols:
            view = changed[show_cols].rename(columns={
                "_LastStatusEvent":"Last Event",
                "_LastStatusChangeDT":"Last Changed At"
            }).sort_values("Last Changed At", ascending=False)
            st.dataframe(view.head(1500), use_container_width=True)

# ---------- Project Status ----------
with tabs[2]:
    st.header("Project Status")
    if "Project Name" in df_filtered.columns:
        grp = df_filtered.groupby("Project Name").agg(
            Total=("Reference ID","count") if "Reference ID" in df_filtered.columns else ("Project Name","count"),
            Resolved=("_EffectiveResolutionDT", lambda x: x.notna().sum()),
            R2C=("_R2C_Flag", "sum"),
            RespOnly=("_RespondedNotClosed_Flag", "sum"),
            Median_Close_Hrs=("Computed Closure Time (Hrs)", "median"),
            SLA_Met=("SLA Met", "mean"),
        ).reset_index()
        if "SLA_Met" in grp.columns:
            grp["SLA_Met"] = (grp["SLA_Met"] * 100).round(1)
        st.dataframe(grp, use_container_width=True)

        melted = grp.melt(id_vars=["Project Name"], value_vars=["Total","Resolved","R2C","RespOnly"], var_name="Metric", value_name="Count")
        fig_proj = px.bar(melted, x="Project Name", y="Count", color="Metric", barmode="group",
                          title="Project — Total vs Resolved vs Rejected→Closed vs Responded-not-Closed",
                          color_discrete_sequence=distinct_brand_colors(melted["Metric"].nunique()))
        fig_proj.update_xaxes(tickangle=30, tickfont=dict(size=11))
        show_chart(style_fig(fig_proj, theme), key="tab1-project-bar")

        # ---- Added: Stacked status mix (top N) ----
        if "Current Status" in df_filtered.columns:
            topN = min(20, len(grp))
            top_projects = grp.sort_values("Total", ascending=False).head(topN)["Project Name"].astype(str).tolist()
            small = df_filtered[df_filtered["Project Name"].astype(str).isin(top_projects)].copy()
            small["Current Status"] = small["Current Status"].fillna("—").astype(str)
            stack = small.groupby(["Project Name","Current Status"]).size().reset_index(name="Count")
            fig_stack = px.bar(stack, x="Project Name", y="Count", color="Current Status", text_auto=True,
                               color_discrete_sequence=distinct_brand_colors(stack["Current Status"].nunique()+2))
            fig_stack.update_layout(title="Top Projects — Status Mix (stacked)")
            fig_stack.update_xaxes(tickangle=30, tickfont=dict(size=11))
            show_chart(style_fig(fig_stack, theme), key="proj-status-stacked")

        # ---- Added: SLA% vs Resolved (bubble) ----
        if "SLA_Met" in grp.columns:
            fig_sc = px.scatter(grp, x="SLA_Met", y="Resolved", size="Total", size_max=28,
                                hover_name="Project Name", color_discrete_sequence=distinct_brand_colors(1))
            fig_sc.update_layout(title="Project — SLA% vs Resolved")
            show_chart(style_fig(fig_sc, theme), key="proj-sla-scatter")
    else:
        st.info("Column 'Project Name' not found.")

# ---------- Project Explorer ----------
with tabs[3]:
    st.header("Project Explorer")
    c1, c2 = st.columns([1,2])
    with c1:
        st.caption("Counts by Types & Tags")
        for colname, key in [("Type L0","typeL0"), ("Type L1","typeL1"), ("Type L2","typeL2"), ("Tag 1","tag1"), ("Tag 2","tag2")]:
            if colname in df_filtered.columns:
                show_chart(bar_top_counts(df_filtered, colname, template=THEMES[theme]["template"], theme_name=theme),
                           key=f"tab2-{key}")
    with c2:
        st.caption("Counts by Status, SLA, R→C and Responded-not-Closed")
        if "Current Status" in df_filtered.columns:
            fig_st = bar_top_counts(df_filtered, "Current Status", template=THEMES[theme]["template"], theme_name=theme)
            fig_st.for_each_trace(lambda t: t.update(marker=dict(line_width=0.5)))
            show_chart(fig_st, key="tab2-status")

        if "SLA Met" in df_filtered.columns:
            work = df_filtered.copy()
            work["SLA State"] = df_filtered["SLA Met"].map({True: "Met", False: "Missed"}).fillna("Unknown")
            show_chart(bar_top_counts(work, "SLA State", template=THEMES[theme]["template"], theme_name=theme),
                       key="tab2-sla")

        if "Assigned Team User" in df_filtered.columns and r2c_count_scope > 0:
            counts = (df_filtered.loc[mask_r2c_inferred, "Assigned Team User"]
                      .fillna("—").astype(str)
                      .value_counts().rename_axis("Assignee").reset_index(name="Rejected→Closed"))
            fig_r2c_scope = px.bar(counts.sort_values("Rejected→Closed"),
                                   x="Rejected→Closed", y="Assignee", orientation="h",
                                   title="Rejected → Closed — by Assignee (inferred, scope)",
                                   color_discrete_sequence=distinct_brand_colors(1))
            show_chart(style_fig(fig_r2c_scope, theme), key="tab2-r2c-assignee")

        if "Assigned Team User" in df_filtered.columns and resp_only_count > 0:
            resp_counts = (df_filtered.loc[mask_responly, "Assigned Team User"]
                           .fillna("—").astype(str)
                           .value_counts().rename_axis("Assignee").reset_index(name="Responded not Closed"))
            fig_resp_scope = px.bar(resp_counts.sort_values("Responded not Closed"),
                                    x="Responded not Closed", y="Assignee", orientation="h",
                                    title="Responded but NOT Closed — by Assignee (scope)",
                                    color_discrete_sequence=distinct_brand_colors(1))
            show_chart(style_fig(fig_resp_scope, theme), key="tab2-responly-assignee")

    # ---- Added Per-project snapshots (donuts + distribution + heatmap) ----
    st.subheader("Per-project Snapshots")
    projects = sorted(df_filtered.get("Project Name", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    sel_proj = st.selectbox("Select a project", ["(All)"]+projects, index=0, key="pe-proj")
    df_scope = df_filtered if sel_proj=="(All)" else df_filtered[df_filtered["Project Name"].astype(str)==sel_proj]

    s1,s2,s3 = st.columns(3)
    with s1:
        if "Current Status" in df_scope.columns and len(df_scope):
            vc = df_scope["Current Status"].fillna("—").astype(str).value_counts().reset_index()
            vc.columns = ["Current Status","Count"]
            fig = px.pie(vc, names="Current Status", values="Count", hole=0.6,
                         color="Current Status", color_discrete_sequence=distinct_brand_colors(len(vc)))
            fig.update_layout(title="Status")
            show_chart(style_fig(fig, theme), key="pe-status-donut")
    with s2:
        who = "Responded By" if "Responded By" in df_scope.columns else ("Assigned Team User" if "Assigned Team User" in df_scope.columns else None)
        if who:
            vc = df_scope[who].fillna("—").astype(str).value_counts().reset_index()
            vc.columns = [who, "Count"]
            fig = px.pie(vc, names=who, values="Count", hole=0.6,
                         color=who, color_discrete_sequence=distinct_brand_colors(len(vc)))
            fig.update_layout(title="Inspectors / Responders")
            show_chart(style_fig(fig, theme), key="pe-inspectors")
    with s3:
        if "Closed By" in df_scope.columns:
            vc = df_scope["Closed By"].fillna("—").astype(str).value_counts().reset_index()
            vc.columns = ["Closed By","Count"]
            fig = px.pie(vc, names="Closed By", values="Count", hole=0.6,
                         color="Closed By", color_discrete_sequence=distinct_brand_colors(len(vc)))
            fig.update_layout(title="Approvers / Closers")
            show_chart(style_fig(fig, theme), key="pe-closers")

    d1,d2 = st.columns(2)
    with d1:
        s = pd.to_numeric(df_scope["Computed Closure Time (Hrs)"], errors="coerce")
        s = s[(s >= 0) & s.notna()]
        if len(s):
            fig = px.box(y=s, points="suspectedoutliers")
            fig.update_traces(showlegend=False, marker_color=distinct_brand_colors(1)[0])
            fig.update_layout(title="Closure Time (hrs) — distribution")
            show_chart(style_fig(fig, theme), key="pe-close-box")
    with d2:
        x = pd.to_numeric(df_scope["Computed Closure Time (Hrs)"], errors="coerce")
        y = pd.to_numeric(df_scope["Total Cost"], errors="coerce") if "Total Cost" in df_scope.columns else pd.Series(index=df_scope.index, dtype=float)
        mask = x.notna() & y.notna() & (x >= 0) & (y >= 0)
        if mask.any():
            fig = px.scatter(df_scope[mask], x="Computed Closure Time (Hrs)", y="Total Cost",
                             hover_data=["Reference ID"] if "Reference ID" in df_scope.columns else None,
                             color_discrete_sequence=distinct_brand_colors(1))
            fig.update_layout(title="Closure time vs Total Cost")
            show_chart(style_fig(fig, theme), key="pe-close-cost")

    if "Type L1" in df_scope.columns and "Current Status" in df_scope.columns:
        a = df_scope[["Type L1","Current Status"]].copy().fillna("—").astype(str)
        pv = pd.pivot_table(a, index="Type L1", columns="Current Status", aggfunc=len, fill_value=0)
        if pv.size:
            fig = px.imshow(pv.values, x=pv.columns.astype(str), y=pv.index.astype(str),
                            labels=dict(x="Current Status", y="Type L1", color="Count"), aspect="auto")
            fig.update_layout(title="Heatmap — Type L1 vs Status")
            show_chart(style_fig(fig, theme), key="pe-heatmap-l1")

# ---------- Tower-Wise ----------
with tabs[4]:
    st.header("Tower-Wise")
    tower_col = "Location L1" if "Location L1" in df_filtered.columns else None

    if tower_col:
        grp = df_filtered.groupby(tower_col).agg(
            Total=("Reference ID","count") if "Reference ID" in df_filtered.columns else (tower_col,"count"),
            Resolved=("_EffectiveResolutionDT", lambda x: x.notna().sum()),
            R2C=("_R2C_Flag", "sum"),
            RespOnly=("_RespondedNotClosed_Flag", "sum"),
        ).reset_index()
        st.dataframe(grp.sort_values("Total", ascending=False), use_container_width=True)

        melted = grp.melt(id_vars=[tower_col], value_vars=["Total","Resolved","R2C","RespOnly"], var_name="Metric", value_name="Count")
        fig_tower = px.bar(melted, x=tower_col, y="Count", color="Metric", barmode="group",
                           title="Tower — Total vs Resolved vs Rejected→Closed vs Responded-not-Closed",
                           color_discrete_sequence=distinct_brand_colors(melted["Metric"].nunique()))
        fig_tower.update_xaxes(tickangle=30, tickfont=dict(size=11))
        show_chart(style_fig(fig_tower, theme), key="tab3-tower-group")

        # ---- Added: Stacked status per tower + heatmap ----
        if "Current Status" in df_filtered.columns:
            top_t = grp.sort_values("Total", ascending=False).head(15)[tower_col].astype(str).tolist()
            small = df_filtered[df_filtered[tower_col].astype(str).isin(top_t)].copy()
            small["Current Status"] = small["Current Status"].fillna("—").astype(str)
            stack = small.groupby([tower_col, "Current Status"]).size().reset_index(name="Count")
            fig2 = px.bar(stack, x=tower_col, y="Count", color="Current Status", text_auto=True,
                          color_discrete_sequence=distinct_brand_colors(stack["Current Status"].nunique()+2))
            fig2.update_layout(title="Top Towers — Status mix (stacked)")
            show_chart(style_fig(fig2, theme), key="tw-status-stacked")

        if "Type L2" in df_filtered.columns:
            a = df_filtered[[tower_col, "Type L2"]].copy().fillna("—").astype(str)
            pv = pd.pivot_table(a, index=tower_col, columns="Type L2", aggfunc=len, fill_value=0)
            if pv.size:
                fig = px.imshow(pv.values, x=pv.columns.astype(str), y=pv.index.astype(str),
                                labels=dict(x="Type L2", y=tower_col, color="Count"), aspect="auto")
                fig.update_layout(title="Heatmap — Tower vs Type L2")
                show_chart(style_fig(fig, theme), key="tw-heatmap")
    else:
        st.info("Column 'Location L1' not found.")

# ---------- User-Wise ----------
with tabs[5]:
    st.header("User-Wise")

    if "Assigned Team User" in df_filtered.columns:
        usr = df_filtered.groupby("Assigned Team User").agg(
            Total=("Reference ID","count") if "Reference ID" in df_filtered.columns else ("Assigned Team User","count"),
            Resolved=("_EffectiveResolutionDT", lambda x: x.notna().sum()),
            R2C=("_R2C_Flag", "sum"),
            RespOnly=("_RespondedNotClosed_Flag", "sum"),
            Median_Close_Hrs=("Computed Closure Time (Hrs)", "median"),
        ).reset_index()

        maxN = max(1, len(usr))
        topN = st.slider("Top N users (grouped bar)", 5, max(5, maxN), min(25, maxN), key="uw-topn")

        long_u = usr.melt(id_vars=["Assigned Team User"], value_vars=["Resolved","R2C","RespOnly"],
                          var_name="Metric", value_name="Count")

        tot = long_u.groupby("Assigned Team User")["Count"].sum().sort_values(ascending=False).head(topN).index
        long_u_top = long_u[long_u["Assigned Team User"].isin(tot)]

        fig_u_grp = px.bar(long_u_top.sort_values("Count"),
                           x="Count", y="Assigned Team User", color="Metric", barmode="group",
                           title="User — Resolved vs Rejected→Closed vs Responded-not-Closed (Top N)",
                           color_discrete_sequence=distinct_brand_colors(long_u_top["Metric"].nunique()),
                           text_auto=True)
        fig_u_grp.update_yaxes(categoryorder="total ascending")
        show_chart(style_fig(fig_u_grp, theme), key="tab4-u-group")

        resp_all = usr[usr["RespOnly"] > 1].sort_values("RespOnly")
        if len(resp_all):
            fig_u_resp_only = px.bar(resp_all,
                                     x="RespOnly", y="Assigned Team User", orientation="h",
                                     title="Responded but NOT Closed — Counts (ALL users, count > 1)",
                                     color_discrete_sequence=distinct_brand_colors(1))
            show_chart(style_fig(fig_u_resp_only, theme), key="tab4-u-resp-only")
        else:
            st.info("No Responded-not-Closed users with count > 1.")

        c1, c2 = st.columns(2)
        with c1:
            fig_u_total = px.bar(usr.sort_values("Total", ascending=False).head(25),
                                 x="Assigned Team User", y="Total", title="Top Assignees by Total",
                                 color_discrete_sequence=distinct_brand_colors(1))
            fig_u_total.update_xaxes(tickangle=30)
            show_chart(style_fig(fig_u_total, theme), key="tab4-u-total")
        with c2:
            fig_u_res = px.bar(usr.sort_values("Resolved", ascending=False).head(25),
                               x="Assigned Team User", y="Resolved", title="Top Assignees by Resolved",
                               color_discrete_sequence=distinct_brand_colors(1))
            fig_u_res.update_xaxes(tickangle=30)
            show_chart(style_fig(fig_u_res, theme), key="tab4-u-resolved")

        strict_rows = df_filtered["_R2C_Strict_Flag"] == 1
        if strict_rows.any():
            med = (df_filtered.loc[strict_rows, ["Assigned Team User","R2C Hours (>=0)"]]
                   .dropna()
                   .groupby("Assigned Team User")["R2C Hours (>=0)"]
                   .median()
                   .reset_index()
                   .rename(columns={"R2C Hours (>=0)":"Median Hours to Close after Rejection"}))
            fig_u_med = px.bar(med.sort_values("Median Hours to Close after Rejection"),
                               x="Median Hours to Close after Rejection", y="Assigned Team User", orientation="h",
                               title="Median CLOSE-after-REJECTION (hours) — by Assignee (strict)",
                               color_discrete_sequence=distinct_brand_colors(1))
            show_chart(style_fig(fig_u_med, theme), key="tab4-u-med")

        # ---- Added: Bubble — workload vs median close (bubble ~ Resolved) ----
        small = usr.copy()
        small = small[small["Total"] > 0]
        if len(small):
            fig_bub = px.scatter(small, x="Median_Close_Hrs", y="Total", size="Resolved",
                                 hover_name="Assigned Team User", size_max=30,
                                 color_discrete_sequence=distinct_brand_colors(1))
            fig_bub.update_layout(title="Users — Median Close (hrs) vs Total (bubble ~ Resolved)")
            show_chart(style_fig(fig_bub, theme), key="uw-bubble")
    else:
        st.info("Column 'Assigned Team User' not found.")

    if "Raised By" in df_filtered.columns and resp_only_count > 0:
        rb_counts = (df_filtered.loc[mask_responly, "Raised By"]
                     .fillna("—").astype(str)
                     .value_counts().rename_axis("Raised By").reset_index(name="Responded not Closed"))
        rb_counts = rb_counts[rb_counts["Responded not Closed"] > 1]
        if len(rb_counts):
            fig_rb_all = px.bar(
                rb_counts.sort_values("Responded not Closed"),
                x="Responded not Closed", y="Raised By", orientation="h",
                title="Raised-by: Responded-not-Closed (count > 1)",
                color_discrete_sequence=distinct_brand_colors(1)
            )
            show_chart(style_fig(fig_rb_all, theme), key="tab4-raisedby-responly")
        else:
            st.info("No Responded-not-Closed raised-by reminders with count > 1 in current filters.")

# ---------- Activity-Wise ----------
with tabs[6]:
    st.header("Activity-Wise")
    st.caption("Totals + R→C (inferred) + Responded-not-Closed; brand gradient for %R→C (White→Blue→Black).")

    # --- helper duplicated to allow reuse with custom subsets ---
    def grouped_measures_for(df_data: pd.DataFrame, by_col: str, key_prefix: str, show_ratio: bool = True, topn: int = 20):
        if by_col not in df_data.columns:
            return
        agg = df_data.groupby(by_col).agg(
            Total=("Reference ID","count") if "Reference ID" in df_data.columns else (by_col,"count"),
            Resolved=("_EffectiveResolutionDT", lambda x: x.notna().sum()),
            R2C=("_R2C_Flag", "sum"),
            RespOnly=("_RespondedNotClosed_Flag", "sum"),
        ).reset_index()
        if len(agg) == 0: return
        agg["R2C%"] = np.where(agg["Total"]>0, agg["R2C"]/agg["Total"]*100, np.nan)
        agg = agg.sort_values("Total", ascending=False).head(topn)

        melted = agg.melt(id_vars=[by_col], value_vars=["Total","R2C","RespOnly"], var_name="Metric", value_name="Count")
        fig = px.bar(
            melted, x=by_col, y="Count", color="Metric", barmode="group",
            title=f"{by_col} — Total vs R→C vs Responded-not-Closed (Top {topn})",
            color_discrete_sequence=distinct_brand_colors(melted["Metric"].nunique()),
            text_auto=True
        )
        fig.update_xaxes(tickangle=30, tickfont=dict(size=10))
        show_chart(style_fig(fig, theme), key=f"{key_prefix}-{by_col}-bars")

        if show_ratio and agg["R2C"].sum() > 0:
            figr = px.bar(
                agg.sort_values("R2C%", ascending=True),
                x="R2C%", y=by_col, orientation="h",
                title=f"{by_col} — % Rejected→Closed (inferred)",
                color="R2C%", color_continuous_scale=THEMES[theme]["gradient"],
                text="R2C%"
            )
            figr.update_traces(texttemplate="%{x:.1f}%")
            show_chart(style_fig(figr, theme), key=f"{key_prefix}-{by_col}-rate")

    # --- Original KPIs on full filtered scope (unchanged) ---
    c1, c2 = st.columns(2)
    with c1:
        grouped_measures_for(df_filtered, "Type L1", "tab5", show_ratio=True, topn=20)
        if "Tag 1" in df_filtered.columns:
            grouped_measures_for(df_filtered, "Tag 1", "tab5-tag1", show_ratio=True, topn=20)
    with c2:
        grouped_measures_for(df_filtered, "Type L2", "tab5", show_ratio=True, topn=20)
        if "Tag 2" in df_filtered.columns:
            grouped_measures_for(df_filtered, "Tag 2", "tab5-tag2", show_ratio=True, topn=20)

    # ---- Original: Pareto & Heatmaps (unchanged) ----
    st.subheader("Pareto & Heatmaps")

    if "Type L0" in df_filtered.columns:
        agg = df_filtered["Type L0"].fillna("—").astype(str).value_counts().reset_index()
        agg.columns = ["Type L0","Total"]
        agg = agg.sort_values("Total", ascending=False).head(20)
        if len(agg):
            agg["Cum%"] = agg["Total"].cumsum() / agg["Total"].sum() * 100
            figp = go.Figure()
            figp.add_bar(x=agg["Type L0"], y=agg["Total"], name="Total", marker_color=distinct_brand_colors(1)[0])
            figp.add_scatter(x=agg["Type L0"], y=agg["Cum%"], name="Cum %", yaxis="y2", mode="lines+markers",
                             line=dict(width=2, color=distinct_brand_colors(2)[1]))
            figp.update_layout(yaxis2=dict(overlaying="y", side="right", title="Cum %"), title="Type L0 — Pareto (Total)")
            show_chart(style_fig(figp, theme), key="aw-pareto-l0")

    if "Type L2" in df_filtered.columns and "Current Status" in df_filtered.columns:
        a = df_filtered[["Type L2","Current Status"]].copy().fillna("—").astype(str)
        pv = pd.pivot_table(a, index="Type L2", columns="Current Status", aggfunc=len, fill_value=0)
        if pv.size:
            fig = px.imshow(pv.values, x=pv.columns.astype(str), y=pv.index.astype(str), aspect="auto",
                            labels=dict(x="Current Status", y="Type L2", color="Count"))
            fig.update_layout(title="Heatmap — Type L2 vs Status")
            show_chart(style_fig(fig, theme), key="aw-heatmap-l2")

    if all(c in df_filtered.columns for c in ["Type L0","Type L1","Type L2"]):
        sb = df_filtered.copy()
        for c in ["Type L0","Type L1","Type L2"]:
            sb[c] = sb[c].fillna("—").astype(str)
        fig = px.sunburst(sb, path=["Type L0","Type L1","Type L2"],
                          color="Type L0", color_discrete_sequence=distinct_brand_colors(sb["Type L0"].nunique()+2))
        fig.update_layout(title="Sunburst — Activity Hierarchy")
        show_chart(style_fig(fig, theme), key="aw-sunburst")

    # ===================== NEW SECTION: Normal (Non-R2C) =====================
    st.subheader("Normal (Non-R2C) — Recurrence & Mix")
    normal_df = df_filtered[df_filtered["_R2C_Flag"] != 1].copy()
    if len(normal_df) == 0:
        st.info("No Non-R2C rows in current filters.")
    else:
        cA, cB = st.columns(2)
        with cA:
            grouped_measures_for(normal_df, "Type L1", "norm", show_ratio=False, topn=20)
        with cB:
            grouped_measures_for(normal_df, "Type L2", "norm", show_ratio=False, topn=20)

        # Donut of status within Normal
        if "Current Status" in normal_df.columns:
            vc = normal_df["Current Status"].fillna("—").astype(str).value_counts().reset_index()
            vc.columns = ["Current Status","Count"]
            fig = px.pie(vc, names="Current Status", values="Count", hole=0.6,
                         color="Current Status", color_discrete_sequence=distinct_brand_colors(len(vc)))
            fig.update_layout(title="Normal — Status Split")
            show_chart(style_fig(fig, theme), key="norm-status-donut")

        # Heatmap L1×Status for Normal
        if "Type L1" in normal_df.columns and "Current Status" in normal_df.columns:
            a = normal_df[["Type L1","Current Status"]].copy().fillna("—").astype(str)
            pv = pd.pivot_table(a, index="Type L1", columns="Current Status", aggfunc=len, fill_value=0)
            if pv.size:
                fig = px.imshow(pv.values, x=pv.columns.astype(str), y=pv.index.astype(str),
                                labels=dict(x="Current Status", y="Type L1", color="Count"), aspect="auto")
                fig.update_layout(title="Normal — Heatmap (Type L1 × Status)")
                show_chart(style_fig(fig, theme), key="norm-heatmap-l1")

        # Sunburst for Normal subset
        if all(c in normal_df.columns for c in ["Type L0","Type L1","Type L2"]):
            nsb = normal_df.copy()
            for c in ["Type L0","Type L1","Type L2"]:
                nsb[c] = nsb[c].fillna("—").astype(str)
            fig = px.sunburst(nsb, path=["Type L0","Type L1","Type L2"],
                              color="Type L0", color_discrete_sequence=distinct_brand_colors(nsb["Type L0"].nunique()+2))
            fig.update_layout(title="Normal — Activity Hierarchy")
            show_chart(style_fig(fig, theme), key="norm-sunburst")

    # ===================== NEW SECTION: All NCs — Recurring Issue Finder =====================
    st.subheader("All NCs — Recurring Issue Finder")
    # Choose field to define "recurrence"
    choices = []
    for col in ["Type L2","Type L1","Description","Recommendation","Tag 1","Tag 2","Location Variable (Fixed)"]:
        if col in df_filtered.columns:
            choices.append(col)
    if not choices:
        st.info("No suitable columns found for recurrence analysis.")
    else:
        pick = st.selectbox("Most recurring by", choices, index=0, key="all-recur-by")
        topn = st.slider("Top N", 5, 50, 20, step=5, key="all-recur-topn")

        # Clean labels a bit for free-text columns
        series = df_filtered[pick].astype(str).str.strip()
        series = series.replace(["", "nan", "NaN", "None"], "—")
        vc = series.value_counts().head(topn).rename_axis(pick).reset_index(name="Count")
        fig = px.bar(vc.sort_values("Count"),
                     x="Count", y=pick, orientation="h",
                     title=f"Most Recurring — {pick} (Top {topn})",
                     color_discrete_sequence=distinct_brand_colors(1),
                     text_auto=True)
        show_chart(style_fig(fig, theme), key="all-recur-top")

        # Optional: show distribution across status for the selected top items
        if "Current Status" in df_filtered.columns and len(vc):
            tops = vc[pick].astype(str).tolist()
            small = df_filtered[df_filtered[pick].astype(str).isin(tops)].copy()
            small["Current Status"] = small["Current Status"].fillna("—").astype(str)
            dist = small.groupby([pick, "Current Status"]).size().reset_index(name="Count")
            fig2 = px.bar(dist, x="Count", y=pick, color="Current Status", orientation="h",
                          title=f"Status Mix for Top {pick}",
                          color_discrete_sequence=distinct_brand_colors(dist["Current Status"].nunique()+2),
                          text_auto=True)
            show_chart(style_fig(fig2, theme), key="all-recur-statusmix")

# ---------- Timelines (extended) ----------
with tabs[7]:
    st.header("Timelines (light)")
    if "_RaisedOnDT" in df_filtered.columns:
        work = df_filtered.copy()
        work["Date"] = work["_RaisedOnDT"].dt.date
        series = work.groupby("Date").agg(
            Raised=("Reference ID", "count") if "Reference ID" in work.columns else ("Date","count"),
            Resolved=("_EffectiveResolutionDT", lambda x: x.notna().sum()),
            R2C=("_R2C_Flag", "sum"),
            RespOnly=("_RespondedNotClosed_Flag", "sum"),
        ).reset_index()
        fig2 = go.Figure()
        mc = metric_colors
        fig2.add_trace(go.Scatter(x=series["Date"], y=series["Raised"], mode="lines", name="Raised", line=dict(color=mc["Total"])))
        fig2.add_trace(go.Scatter(x=series["Date"], y=series["Resolved"], mode="lines", name="Resolved", line=dict(color=mc["Resolved"])))
        fig2.add_trace(go.Scatter(x=series["Date"], y=series["R2C"], mode="lines", name="Rejected→Closed", line=dict(color=mc["R2C"])))
        fig2.add_trace(go.Scatter(x=series["Date"], y=series["RespOnly"], mode="lines", name="Responded-not-Closed", line=dict(color=mc["RespOnly"])))
        fig2.update_layout(title="Daily Flow — Raised vs Resolved vs R→C vs Responded-not-Closed")
        show_chart(style_fig(fig2, theme), key="tab6-lines")

        # ---- Added: Backlog area + Calendar heatmap ----
        series["RaisedCum"]   = series["Raised"].cumsum()
        series["ResolvedCum"] = series["Resolved"].cumsum()
        series["Backlog"]     = series["RaisedCum"] - series["ResolvedCum"]
        area = go.Figure()
        area.add_trace(go.Scatter(x=series["Date"], y=series["Backlog"], name="Open Backlog", mode="lines",
                                  fill="tozeroy", line=dict(width=0.5, color=distinct_brand_colors(1)[0])))
        area.update_layout(title="Open Backlog (cumulative)")
        show_chart(style_fig(area, theme), key="tl-backlog")

        if df_filtered["_RaisedWeek"].notna().any() and df_filtered["_RaisedDOW"].notna().any():
            cal = df_filtered.groupby(["_RaisedWeek","_RaisedDOW"]).size().reset_index(name="Count")
            days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            cal["_RaisedDOW"] = pd.Categorical(cal["_RaisedDOW"], categories=days, ordered=True)
            pv = cal.pivot(index="_RaisedDOW", columns="_RaisedWeek", values="Count").fillna(0)
            fig = px.imshow(pv.values, x=pv.columns, y=pv.index, aspect="auto",
                            labels=dict(x="Week", y="Day", color="Raised"))
            fig.update_layout(title="Calendar Heatmap — Raised (Week × DOW)")
            show_chart(style_fig(fig, theme), key="tl-cal-heatmap")
    else:
        st.info("No Raised On timestamps available.")

# ---------- NC-View ----------
# ---------- NC-View ----------
with tabs[8]:
    st.header("NC-View")

    # ---- Scope by project (optional) ----
    proj_opts = sorted(df_filtered.get("Project Name", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    sel_proj = st.selectbox("Filter by Project (optional)", ["(All)"] + proj_opts, index=0, key="nc-proj")
    df_scope = df_filtered if sel_proj == "(All)" else df_filtered[df_filtered.get("Project Name", "").astype(str) == sel_proj]

    # ---- Build a rich, typeable selector ----
    def _short(s, n=80):
        if pd.isna(s): return "—"
        s = str(s).strip()
        return s if len(s) <= n else s[: n - 1] + "…"

    def _mk_label(r: pd.Series) -> str:
        ref  = str(r.get("Reference ID", "—"))
        proj = str(r.get("Project Name", "—"))
        t1   = str(r.get("Type L1", "—"))
        t2   = str(r.get("Type L2", "—"))
        locv = str(r.get("Location Variable (Fixed)", r.get("Location Variable", "—")))
        desc = _short(r.get("Description", "—"), 70)
        bits = [ref, proj, f"{t1}/{t2}", locv, desc]
        # Join with " — " but suppress trailing empties nicely
        return " — ".join([b for b in bits if b and b != "—"])

    if "Reference ID" in df_scope.columns and len(df_scope):
        # Build an indexable table of options with a searchable label column
        opt_df = df_scope.copy()
        opt_df["_label"] = opt_df.apply(_mk_label, axis=1)

        # Free-text search to narrow down (typable input)
        q = st.text_input("Search NC (by ref / project / type / location / description)", value="", key="nc-search").strip().lower()
        if q:
            mask = opt_df["_label"].str.lower().str.contains(q, na=False)
            opt_df = opt_df[mask]

        if len(opt_df) == 0:
            st.info("No NCs match your search and filters.")
        else:
            # Selectbox remains typeable and shows the rich label; value is index of the row
            # Use index to keep it stable even if duplicate refs exist
            opt_df = opt_df.reset_index(drop=False).rename(columns={"index": "_idx"})
            sel_idx = st.selectbox(
                "Select an NC",
                options=opt_df["_idx"].tolist(),
                format_func=lambda i: opt_df.loc[opt_df["_idx"] == i, "_label"].values[0],
                key="nc-ref"
            )
            row = df_scope.iloc[sel_idx:sel_idx+1]  # single-row DataFrame slice
    else:
        st.info("No Reference IDs available in current filters.")
        row = pd.DataFrame()

    # ---- Details + visuals for the selected NC ----
    if not row.empty:
        r = row.iloc[0]

        # Quick KPIs row
        def _fmt_dt(x):
            return "" if pd.isna(x) else pd.to_datetime(x).strftime("%Y-%m-%d %H:%M")

        top = st.columns(5)
        with top[0]: st.metric("Reference", str(r.get("Reference ID", "—")))
        with top[1]: st.metric("Project",   str(r.get("Project Name", "—")))
        with top[2]: st.metric("Status",    str(r.get("Current Status", "—")))
        with top[3]:
            sla = r.get("SLA Met")
            st.metric("SLA", "Met" if sla is True else ("Missed" if sla is False else "—"))
        with top[4]: st.metric("Assignee",  str(r.get("Assigned Team User", "—")))

        # People & timestamps
        st.caption(
            f"Raised: {_fmt_dt(r.get('_RaisedOnDT'))} | "
            f"Responded: {_fmt_dt(r.get('_RespondedOnDT'))} | "
            f"Rejected: {_fmt_dt(r.get('_RejectedOnDT'))} | "
            f"Closed: {_fmt_dt(r.get('_ClosedOnDT'))} | "
            f"Effective: {_fmt_dt(r.get('_EffectiveResolutionDT'))}"
        )

        # Key commentary & roles (cards)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 📝 Description")
            st.write(r.get("Description", "—"))
            st.markdown("##### ✅ Recommendation")
            st.write(r.get("Recommendation", "—"))

            st.markdown("##### 💬 Responded Comment")
            st.write(r.get("Responded Comment", "—"))
            st.markdown("##### 🚫 Rejected Comment")
            st.write(r.get("Rejected Comment", "—"))
            st.markdown("##### 🏁 Closed Comment")
            st.write(r.get("Closed Comment", "—"))
        with c2:
            st.markdown("##### 👥 People")
            st.write(f"**Raised By:** {r.get('Raised By', '—')}")
            st.write(f"**Responded By:** {r.get('Responded By', '—')}")
            st.write(f"**Rejected By:** {r.get('Rejected By', '—')}")
            st.write(f"**Closed By:** {r.get('Closed By', '—')}")

            st.markdown("##### 🗺️ Location")
            loc_lines = []
            for c in ["Location / Reference", "Location Variable (Fixed)", "Location L0", "Location L1", "Location L2", "Location L3"]:
                if c in row.index:
                    loc_lines.append(f"**{c}:** {r.get(c, '—')}")
            st.write("\n\n".join(loc_lines) if loc_lines else "—")

            st.markdown("##### 🏷️ Types & Tags")
            type_lines = []
            for c in ["Type L0", "Type L1", "Type L2", "Tag 1", "Tag 2"]:
                if c in row.index:
                    type_lines.append(f"**{c}:** {r.get(c, '—')}")
            st.write("\n\n".join(type_lines) if type_lines else "—")

            # URL if present
            url_val = r.get("URL", None)
            if isinstance(url_val, str) and url_val.strip():
                st.markdown(f"##### 🔗 Link\n[{_short(url_val, 60)}]({url_val})")

        st.divider()

        # Keep your original visuals (event strip, Gantt, step, SLA bullet)
        # Event strip
        events = []
        for name, col, color in [
            ("Raised", "_RaisedOnDT", GREY),
            ("Responded", "_RespondedOnDT", BLUE),
            ("Rejected", "_RejectedOnDT", GREY),
            ("Closed", "_ClosedOnDT", BLACK),
            ("Effective", "_EffectiveResolutionDT", blend(BLUE, BLACK, 0.55)),
            ("Deadline", "_DeadlineDT", GREY),
        ]:
            ts = r.get(col)
            if pd.notna(ts):
                events.append(dict(name=name, ts=pd.to_datetime(ts), color=color))
        if events:
            evdf = pd.DataFrame(events).sort_values("ts")
            fig_ev = go.Figure()
            fig_ev.add_trace(go.Scatter(
                x=evdf["ts"], y=[0]*len(evdf),
                mode="markers+text", text=evdf["name"],
                textposition="top center", marker=dict(size=10, color=evdf["color"])
            ))
            fig_ev.update_yaxes(visible=False)
            fig_ev.update_layout(title="Event Strip", xaxis_title="", height=220)
            show_chart(style_fig(fig_ev, theme), key="nc-events")
        else:
            st.info("No timestamps available to draw the event strip.")

        # Gantt segments
        segs = []
        rs = r.get("_RaisedOnDT"); rp = r.get("_RespondedOnDT"); ef = r.get("_EffectiveResolutionDT")
        if pd.notna(rs) and pd.notna(rp) and rp >= rs:
            segs.append(dict(segment="Raised→Responded", start=pd.to_datetime(rs), finish=pd.to_datetime(rp)))
        if pd.notna(rp) and pd.notna(ef) and ef >= rp:
            segs.append(dict(segment="Responded→Effective", start=pd.to_datetime(rp), finish=pd.to_datetime(ef)))
        elif pd.notna(rs) and pd.notna(ef) and ef >= rs and (pd.isna(rp) or rp < rs):
            segs.append(dict(segment="Raised→Effective", start=pd.to_datetime(rs), finish=pd.to_datetime(ef)))
        if segs:
            tdf = pd.DataFrame(segs)
            fig_tl = px.timeline(tdf, x_start="start", x_end="finish", y="segment",
                                 color="segment", color_discrete_sequence=distinct_brand_colors(len(tdf["segment"].unique())))
            fig_tl.update_yaxes(autorange="reversed")
            fig_tl.update_layout(title="Gantt (segments)", height=250, showlegend=False)
            show_chart(style_fig(fig_tl, theme), key="nc-gantt")
        else:
            st.info("Not enough timestamps to render Gantt segments.")

        # Status step chart
        steps = []
        if pd.notna(rs): steps.append(("Raised", pd.to_datetime(rs), 0))
        if pd.notna(rp) and pd.notna(rs) and rp >= rs: steps.append(("Responded", pd.to_datetime(rp), 1))
        rj = r.get("_RejectedOnDT")
        if pd.notna(rj) and (not steps or pd.to_datetime(rj) >= steps[-1][1]): steps.append(("Rejected", pd.to_datetime(rj), 2))
        cl = r.get("_ClosedOnDT")
        if pd.notna(cl): steps.append(("Closed", pd.to_datetime(cl), 3))
        ef2 = r.get("_EffectiveResolutionDT")
        if pd.notna(ef2) and (not cl or pd.to_datetime(ef2) != pd.to_datetime(cl)):
            steps.append(("Effective", pd.to_datetime(ef2), 4))
        if steps:
            xs = [t[1] for t in steps]
            ys = [t[2] for t in steps]
            fig_step = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers", line_shape="hv", line=dict(color=BLUE)))
            fig_step.update_yaxes(
                tickmode="array",
                tickvals=[0,1,2,3,4],
                ticktext=["Raised","Responded","Rejected","Closed","Effective"]
            )
            fig_step.update_layout(title="Status Step Chart", height=260)
            show_chart(style_fig(fig_step, theme), key="nc-step")
        else:
            st.info("Not enough events to render a step chart.")

        # SLA bullet
        dl = r.get("_DeadlineDT"); ef = r.get("_EffectiveResolutionDT")
        if pd.notna(rs) and (pd.notna(dl) or pd.notna(ef)):
            base = pd.to_datetime(rs)
            dl_hours = (pd.to_datetime(dl) - base).total_seconds()/3600 if pd.notna(dl) else None
            ef_hours = (pd.to_datetime(ef) - base).total_seconds()/3600 if pd.notna(ef) else None
            bars = []
            if dl_hours is not None: bars.append(("Deadline", max(0, dl_hours)))
            if ef_hours is not None: bars.append(("Actual", max(0, ef_hours)))
            if bars:
                bdf = pd.DataFrame(bars, columns=["Metric","Hours"])
                fig_b = px.bar(bdf, x="Hours", y="Metric", orientation="h",
                               title="SLA — Deadline vs Actual (hours)",
                               color="Metric",
                               color_discrete_sequence=[GREY, BLACK],
                               text_auto=True)
                show_chart(style_fig(fig_b, theme), key="nc-sla-bullet")
        else:
            st.info("No SLA/Effective times available for bullet chart.")

        st.divider()
        # ---- All raw fields (one-stop detail view) ----
        with st.expander("All fields (raw)"):
            # transpose pretty: two columns if many fields
            raw_series = r.copy()
            # Convert datetimes to readable strings
            for c in raw_series.index:
                v = raw_series[c]
                if pd.api.types.is_datetime64_any_dtype(type(v)) or isinstance(v, pd.Timestamp):
                    raw_series[c] = _fmt_dt(v)
            # Show as a two-column markdown table for readability
            keys = list(raw_series.index)
            half = (len(keys) + 1) // 2
            left_keys, right_keys = keys[:half], keys[half:]
            lc, rc = st.columns(2)
            with lc:
                for k in left_keys:
                    st.markdown(f"**{k}:**  {raw_series.get(k, '—')}")
            with rc:
                for k in right_keys:
                    st.markdown(f"**{k}:**  {raw_series.get(k, '—')}")
    else:
        st.info("Select an NC to view details.")

# ---------- Sketch-View (Treemap) ----------
with tabs[9]:
    st.header("Sketch-View")
    st.caption("Treemap from 'Location / Reference' (pre-aggregated). Brand palette only.")

    path_col = "Location / Reference" if "Location / Reference" in df_filtered.columns else None
    if not path_col:
        st.info("Column 'Location / Reference' not found.")
    else:
        treemap_enable = st.checkbox("Show Treemap", value=True, key="sk-enable")
        max_depth = st.slider("Max depth", 1, 6, 4, step=1, key="sk-depth")
        max_nodes = st.slider("Max nodes", 100, 5000, 800, step=100, help="Limit total rectangles for performance.", key="sk-nodes")
        show_badges = st.checkbox("Show status badges in tiles", value=False, key="sk-badges")

        @st.cache_data(show_spinner=False)
        def build_levels(df_src: pd.DataFrame, col: str, depth: int) -> pd.DataFrame:
            parts = df_src[col].fillna("").astype(str).str.split("/")
            out = pd.DataFrame(index=df_src.index)
            for i in range(depth):
                out[f"Level_{i}"] = parts.str[i].fillna("").str.strip().where(parts.str.len() > i, "")
            out["Count"] = 1
            return out

        def unique_columns(df_in: pd.DataFrame) -> pd.DataFrame:
            cols = []
            seen = {}
            for c in df_in.columns:
                if c not in seen:
                    seen[c] = 1; cols.append(c)
                else:
                    seen[c] += 1; cols.append(f"{c}__{seen[c]}")
            df_in = df_in.copy(); df_in.columns = cols
            return df_in

        if treemap_enable:
            lv = build_levels(df_filtered, path_col, max_depth)
            grp_cols = [c for c in lv.columns if c.startswith("Level_")]
            agg = lv.groupby(grp_cols, dropna=False, as_index=False)["Count"].sum()

            label_cols = []
            if show_badges and "Current Status" in df_filtered.columns:
                tmp = df_filtered.copy()
                tmp["Current Status"] = tmp["Current Status"].fillna("—").astype(str)
                for i in range(max_depth):
                    tmp[f"Level_{i}"] = lv[f"Level_{i}"]
                top_statuses = tmp["Current Status"].value_counts().head(2).index.tolist()
                for status in top_statuses:
                    badge = tmp[tmp["Current Status"] == status].groupby(grp_cols).size()
                    badge = badge.reindex(agg.set_index(grp_cols).index, fill_value=0).values
                    colname = f"{status}"
                    agg[colname] = badge
                    label_cols.append(colname)

            if len(agg) > max_nodes:
                agg = agg.sort_values("Count", ascending=False).head(max_nodes)
            agg = unique_columns(agg)
            path_cols = [c for c in agg.columns if c.startswith("Level_")]

            if show_badges and label_cols:
                def _mk_label(row):
                    name = [row[c] for c in path_cols if row[c]][-1] or "(root)"
                    badges = " ".join([f"{k}:{int(row[k])}" for k in label_cols])
                    return f"{name}<br><span style='font-size:10px;opacity:.8'>{badges}</span>"
                labels = agg.apply(_mk_label, axis=1)
            else:
                labels = None

            fig_t = px.treemap(
                agg, path=path_cols, values="Count",
                color_discrete_sequence=distinct_brand_colors( max(3, len(path_cols)) )
            )
            if labels is not None:
                fig_t.update_traces(texttemplate=labels, hovertemplate="%{label}<br>Count=%{value}<extra></extra>")
            fig_t.update_layout(title="Location / Reference — Treemap (aggregated)")
            show_chart(style_fig(fig_t, theme), key="sk-treemap")

        st.subheader("Issues at Selected Path")
        selected = st.text_input("Filter by path contains", "", key="sketch-filter")
        view = df_filtered[df_filtered[path_col].str.contains(selected, case=False, na=False)] if selected else df_filtered

        show_cols = [c for c in [
            "Reference ID","Project Name", path_col, "Location Variable (Fixed)",
            "Type L0","Type L1","Type L2","Tag 1","Tag 2",
            "Assigned Team","Assigned Team User","Current Status",
            "Responding Time (Hrs)","Computed Closure Time (Hrs)","Close After Response (Hrs)"
        ] if c in view.columns]
        if len(view):
            st.dataframe(view[show_cols].head(1500), use_container_width=True)
        else:
            st.info("No issues match the selected filter.")

# ---------- NC Table ----------
with tabs[10]:
    st.header("NC Table")
    st.caption("Styled table with row shading by Current Status (brand colours).")
    shade = st.toggle("Enable row shading", value=False, key="tbl-shade")

    display_cols = [c for c in [
        "Reference ID","Project Name","Location / Reference","Location Variable (Fixed)",
        "Location L0","Location L1","Location L2","Location L3",
        "Description","Recommendation",
        "Raised By","Raised On Date","Raised On Time",
        "Deadline Date","Deadline Time",
        "Assigned Team","Assigned Team User",
        "Current Status",
        "Responded By","Responded On Date","Responded On Time",
        "Rejected By","Rejected On Date","Rejected On Time",
        "Closed By","Closed On Date","Closed On Time",
        "Responding Time (Hrs)","Computed Closure Time (Hrs)","Close After Response (Hrs)",
        "_RespondedNotClosed_Flag", "_R2C_Flag", "_R2C_Strict_Flag",
        "Type L0","Type L1","Type L2","Tag 1","Tag 2",
        "Root Cause Analysis","Correction","Corrective Action",
        "Labour Cost","Material Cost","Machinery Cost","Other Cost","Total Cost"
    ] if c in df_filtered.columns]

    if not display_cols:
        st.info("No known NC columns found in the dataset.")
    else:
        view = df_filtered[display_cols]
        if shade:
            try:
                st.write(style_status_rows(view.head(1500), theme).to_html(), unsafe_allow_html=True)
            except Exception:
                st.dataframe(view.head(1500), use_container_width=True)
        else:
            st.dataframe(view.head(1500), use_container_width=True)
        csv_data = view.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download filtered table (CSV)", data=csv_data, file_name="digiqc_filtered.csv", mime="text/csv", key="dl-full-table")

st.caption("© Digital Issue Dashboard — Streamlit (SJCPL Brand) — V2.7")
