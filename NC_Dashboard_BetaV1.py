# digiqc_dashboard_SJCPL_v3.py
# Digital NC Register — Streamlit (SJCPL Brand, unique brand-only colours, Status tab, fullscreen-safe legends)
# Run:
#   pip install -U streamlit plotly pandas numpy openpyxl
#   # optional for click-to-filter on bars:
#   pip install streamlit-plotly-events
#   streamlit run digiqc_dashboard_SJCPL_v3.py

from __future__ import annotations
from typing import Optional, Any, Tuple, List, Sequence, Dict
import io
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import datetime as dt

# ---------- Optional interactive clicks on Plotly ----------
try:
    from streamlit_plotly_events import plotly_events  # type: ignore
    HAVE_PLOTLY_EVENTS = True
except Exception:
    HAVE_PLOTLY_EVENTS = False

# ---------- Page ----------
st.set_page_config(page_title="Digital NC Register — SJCPL", page_icon="🧭", layout="wide")

# ---------- SJCPL Brand (locked) ----------
WHITE = "#FFFFFF"
BLACK = "#000000"
GREY  = "#939598"
BLUE  = "#00AEDA"

# Base discrete cycle (seed); we will NOT reuse within a chart — we expand with a gradient when needed
BRAND_SEQ = [BLUE, GREY, BLACK]

# Status colours (UI tints; charts use unique-per-series rule below)
SJCPL_STATUS = {
    "Closed":     BLACK,
    "Resolved":   BLACK,
    "Approved":   GREY,
    "In Process": BLUE,
    "In-Process": BLUE,
    "Open":       BLUE,
    "Redo":       GREY,
    "Rejected":   GREY,
    "Responded":  BLUE,
}

SJCPL_METRICS = {
    "Total":    BLUE,
    "Resolved": BLACK,
    "R2C":      GREY,
    "Open":     BLUE,
    "RespOnly": GREY,
}

THEMES = {
    "SJCPL": {
        "template": "plotly_white",
        "status_map": SJCPL_STATUS,
        "metric_map": SJCPL_METRICS,
    }
}
theme = "SJCPL"

# ---------- Brand-aware colour helpers (unique per chart) ----------
def _hex_to_rgb(h: str) -> tuple[int,int,int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _rgb_to_hex(rgb: tuple[int,int,int]) -> str:
    return "#%02X%02X%02X" % rgb

def _mix(c1: str, c2: str, t: float) -> str:
    r1,g1,b1 = _hex_to_rgb(c1); r2,r2g,g2 = _hex_to_rgb(c2)[0], _hex_to_rgb(c2)[1], _hex_to_rgb(c2)[2]  # unpack safely
    # small micro-optim: avoid multiple calls
    r2, g2, b2 = _hex_to_rgb(c2)
    r = int(round(r1 + (r2 - r1) * t))
    g = int(round(g1 + (g2 - g1) * t))
    b = int(round(b1 + (b2 - b1) * t))
    return _rgb_to_hex((r, g, b))

def sample_brand_gradient(n: int, clamp: tuple[float,float]=(0.06, 0.94)) -> list[str]:
    """
    Evenly sample n distinct colours from the brand gradient WHITE→BLUE→BLACK,
    clamping ends so we never hit pure white/black.
    """
    if n <= 0: return []
    xs = np.linspace(clamp[0], clamp[1], n)
    out: list[str] = []
    for t in xs:
        if t <= 0.5:
            local = (t - 0.0) / 0.5
            out.append(_mix(WHITE, BLUE, local))
        else:
            local = (t - 0.5) / 0.5
            out.append(_mix(BLUE, BLACK, local))
    # ensure uniqueness
    seen = set(); uniq = []
    for i, c in enumerate(out):
        u = c.upper()
        if u in seen:
            tj = float(xs[i]) + 0.001
            tj = min(clamp[1], max(clamp[0], tj))
            if tj <= 0.5:
                cj = _mix(WHITE, BLUE, (tj - 0.0)/0.5)
            else:
                cj = _mix(BLUE, BLACK, (tj - 0.5)/0.5)
            c = cj; u = c.upper()
        uniq.append(c); seen.add(u)
    return uniq

def generate_brand_sequence(n: int) -> list[str]:
    """
    Start with base 3, expand with gradient to guarantee n unique brand-only colours.
    Ensures no brand colour is used more than once in a given chart.
    """
    base = BRAND_SEQ[:]
    if n <= len(base):
        return base[:n]
    extra = sample_brand_gradient(n - len(base))
    # ensure full palette is unique (defensive)
    all_cols = base + extra
    seen = set(); out = []
    for c in all_cols:
        u = c.upper()
        if u not in seen:
            out.append(c); seen.add(u)
        else:
            # nudge
            out.append(_mix(BLUE, BLACK, 0.5))
    return out

def brand_map_for(values: Sequence) -> dict[str, str]:
    """Stable map: each unique value → a distinct brand-derived colour."""
    vals = [str(v) for v in values]
    cols = generate_brand_sequence(len(vals))
    return dict(zip(vals, cols))

# ---------- Plotly styling (fullscreen-safe) ----------
def style_fig(fig: go.Figure, title: Optional[str] = None, subtitle: Optional[str] = None) -> go.Figure:
    if title and subtitle:
        fig.update_layout(title=dict(
            text=f"{title}<br><sup>{subtitle}</sup>",
            x=0, xanchor="left", y=0.98, yanchor="top",
            font=dict(size=16, color=BLACK),
            pad=dict(b=12)
        ))
    elif title:
        fig.update_layout(title=dict(
            text=title, x=0, xanchor="left", y=0.98, yanchor="top",
            font=dict(size=16, color=BLACK),
            pad=dict(b=12)
        ))

    fig.update_layout(
        template=THEMES[theme]["template"],
        font=dict(family="Roboto, Arial, sans-serif", size=12, color=BLACK),
        # Put legend BELOW the plot area to avoid overlap in fullscreen
        legend=dict(
            title='',
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.08)",
            borderwidth=1,
            tracegroupgap=10
        ),
        hoverlabel=dict(bgcolor="rgba(255,255,255,0.95)",
                        font=dict(family="Roboto, Arial, sans-serif", color=BLACK)),
        bargap=0.22,
        margin=dict(l=10, r=10, t=90, b=100),
        autosize=True
    )
    fig.update_xaxes(showgrid=True, gridcolor="#ECEFF1", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#ECEFF1", zeroline=False)
    fig.update_traces(marker_line_width=0.5, marker_line_color="rgba(0,0,0,0.18)")
    return fig

def show_chart(fig: go.Figure, key: str, clickable: bool = False):
    """
    If clickable=True and streamlit-plotly-events is available, return clicked x labels.
    Otherwise, return [] and render normally.
    """
    if clickable and HAVE_PLOTLY_EVENTS:
        events = plotly_events(fig, click_event=True, select_event=False, hover_event=False, key=key)
        try:
            # usually events: [{'curveNumber':..., 'pointIndex':..., 'x': 'label', 'y': count, ...}]
            clicked = [e.get("x") for e in events if "x" in e]
        except Exception:
            clicked = []
        return clicked
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)
        return []

# ---------- Header / CSS ----------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Roboto', Arial, sans-serif; color: {BLACK}; }}
:root {{
  --sj-blue: {BLUE};
  --sj-grey: {GREY};
  --sj-black: {BLACK};
  --sj-white: {WHITE};
}}
.sj-header {{
  background: linear-gradient(90deg, var(--sj-black) 0%, var(--sj-blue) 100%);
  padding: 14px 18px; border-radius: 14px; color: var(--sj-white);
  margin: 6px 0 18px 0;
}}
.sj-header h1 {{ margin: 0; font-weight: 800; letter-spacing: .2px; color: var(--sj-white); }}
.sj-header p {{ margin: 4px 0 0 0; opacity: .92; color: var(--sj-white); }}
.plot-container .legend {{ overflow-x: auto; }}
</style>
<div class="sj-header">
  <h1>🧭 Digital NC Register — SJCPL</h1>
  <p>Brand-locked visuals (Roboto · Blue/Black/Grey/White) — unique brand colours per chart</p>
</div>
""", unsafe_allow_html=True)

# ---------- New robust Load data ----------
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
        default_path = "https://raw.githubusercontent.com/dnyanesh57/NC_Dashboard/main/data/CSV-INSTRUCTION-DETAIL-REPORT-09-08-2025-04-25-44.csv"
        try:
            df = pd.read_csv(default_path)
        except Exception:
            st.error("No file uploaded and demo CSV not reachable.")
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

# ---------- Date/time parsing helpers ----------
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
    # ensure tz-naive
    if pd.api.types.is_datetime64tz_dtype(out):
        out = out.dt.tz_convert(None)
    return out

# ---------- Business helpers ----------
def _safe_get(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] if col in df.columns else pd.Series([np.nan] * len(df), index=df.index, name=col)

def extract_location_variable(raw: pd.Series) -> pd.Series:
    def _extract(val):
        if pd.isna(val): return val
        s = str(val)
        return s.rsplit("/", 1)[-1].strip() if "/" in s else s.strip()
    return raw.apply(_extract)

def style_status_rows(df: pd.DataFrame) -> "Styler":
    status_map = THEMES[theme]["status_map"]
    def highlight(row):
        status = str(row.get("Current Status", "")).strip()
        bg = status_map.get(status, WHITE)
        txt = WHITE if bg in (BLACK, GREY) else BLACK
        return [f"background-color: {bg}; color: {txt};"] * len(row)
    try:
        from pandas.io.formats.style import Styler  # type: ignore
        return df.style.apply(highlight, axis=1)
    except Exception:
        return df  # fallback

# ---------- Derived columns ----------
def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ensure future/rolling-out fields exist
    for col in [
        "Root Cause Analysis","Correction","Corrective Action",
        "Labour Cost","Material Cost","Machinery Cost","Other Cost","Total Cost"
    ]:
        if col not in df.columns:
            df[col] = np.nan

    # Location Variable fix
    if "Location Variable" in df.columns:
        df["Location Variable (Fixed)"] = extract_location_variable(_safe_get(df, "Location Variable"))
    else:
        df["Location Variable (Fixed)"] = np.nan

    # Datetimes
    df["_RaisedOnDT"]     = combine_datetime(_safe_get(df, "Raised On Date"), _safe_get(df, "Raised On Time"))
    df["_DeadlineDT"]     = combine_datetime(_safe_get(df, "Deadline Date"), _safe_get(df, "Deadline Time"))
    df["_RespondedOnDT"]  = combine_datetime(_safe_get(df, "Responded On Date"), _safe_get(df, "Responded On Time"))
    df["_RejectedOnDT"]   = combine_datetime(_safe_get(df, "Rejected On Date"), _safe_get(df, "Rejected On Time"))
    df["_ClosedOnDT"]     = combine_datetime(_safe_get(df, "Closed On Date"), _safe_get(df, "Closed On Time"))

    # Effective resolution (Closed else Responded>Raised)
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

    # R2C flags
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

    # Last Status Change (naive) + Event label for Status tab
    # Pick the latest among known event timestamps
    candidates = {
        "Raised":    df["_RaisedOnDT"],
        "Responded": df["_RespondedOnDT"],
        "Rejected":  df["_RejectedOnDT"],
        "Closed":    df["_ClosedOnDT"],
        "Effective": df["_EffectiveResolutionDT"],
    }
    # Max timestamp per row
    cand_df = pd.concat(candidates, axis=1)
    # Ensure tz-naive
    for c in cand_df.columns:
        s = cand_df[c]
        if pd.api.types.is_datetime64tz_dtype(s):
            cand_df[c] = s.dt.tz_convert(None)
    last_dt = cand_df.max(axis=1)
    # Label of the max
    last_evt = cand_df.idxmax(axis=1)
    # Where all are NaT, set NaN label
    last_evt = last_evt.where(last_dt.notna(), np.nan)
    df["_LastStatusChangeDT"] = last_dt
    df["_LastStatusEvent"] = last_evt

    return df

def metrics_summary(df: pd.DataFrame):
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
        return f"{d}d {h}h {m}m"

    t1, t2, t3, t4, t5, t6 = st.columns(6)
    with t1: st.metric("Total Issues", f"{total_issues}")
    with t2: st.metric("Resolved (Closed/Effective)", f"{resolved}")
    with t3: st.metric("Open / Unresolved", f"{open_issues}")
    with t4: st.metric("Median Closure Time", _fmt(median_close))
    with t5: st.metric("Median Responding Time", _fmt(median_response))
    with t6: st.metric("SLA Met Rate", f"{sla_rate:.1f}%" if pd.notna(sla_rate) else "—")

# ---------- Sidebar ----------
with st.sidebar:
    st.title("🧭 Digital NC Register")
    st.caption("SJCPL brand-locked UI (Roboto + Blue/Black/Grey/White)")
    logo_url = st.text_input("Logo URL (optional)", value="")
    uploaded = st.file_uploader("Upload Issue Register (CSV/XLSX)", type=["csv","xlsx","xls"], key="uploader")

# ---------- Load & preprocess ----------
@st.cache_data(show_spinner=False)
def preprocess(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df.columns = [c.strip() for c in df.columns]
    return add_derived_columns(df)

df_raw = load_data(uploaded)
df = preprocess(df_raw)

# ---------- Filters ----------
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

# ---------- Tabs ----------
tabs = st.tabs([
    "Overview",
    "Project Status",
    "Project Explorer",
    "Tower-Wise",
    "User-Wise",
    "Activity-Wise",
    "Timelines",
    "NC-View",
    "Sketch-View",
    "Status",           # NEW
    "NC Table",
])

# ---------- Overview ----------
with tabs[0]:
    st.header("Overview")
    metrics_summary(df_filtered)
    # Status distribution
    if "Current Status" in df_filtered.columns:
        vc = df_filtered["Current Status"].fillna("—").astype(str).value_counts().reset_index()
        vc.columns = ["Current Status","Count"]
        cmap = brand_map_for(vc["Current Status"].tolist())
        fig_sd = px.bar(vc, x="Current Status", y="Count",
                        color="Current Status", color_discrete_map=cmap,
                        text_auto=True)
        fig_sd.update_xaxes(tickangle=0, tickfont=dict(size=11))
        show_chart(style_fig(fig_sd, title="Current Status Distribution"), key="ov-status-dist")

    st.subheader("Response vs Closure — Time Distributions")
    c3, c4 = st.columns(2)
    with c3:
        series = pd.to_numeric(df_filtered["Responding Time (Hrs)"], errors="coerce")
        series = series[(series >= 0) & (series.notna())]
        if len(series):
            fig = px.histogram(x=series, nbins=30, labels={"x":"Responding Time (Hrs)"})
            fig.update_traces(showlegend=False, marker_color=generate_brand_sequence(1)[0])
            fig.update_layout(title="Responding Time (Hrs)")
            show_chart(style_fig(fig), key="tab0-hist-responding")
        else:
            st.info("No data for Responding Time.")
    with c4:
        series = pd.to_numeric(df_filtered["Computed Closure Time (Hrs)"], errors="coerce")
        series = series[(series >= 0) & (series.notna())]
        if len(series):
            fig = px.histogram(x=series, nbins=30, labels={"x":"Computed Closure Time (Hrs)"})
            fig.update_traces(showlegend=False, marker_color=generate_brand_sequence(1)[0])
            fig.update_layout(title="Computed Closure Time (Hrs)")
            show_chart(style_fig(fig), key="tab0-hist-closure")
        else:
            st.info("No data for Computed Closure Time.")

# ---------- Project Status ----------
with tabs[1]:
    st.header("Project Status")
    if "Project Name" in df_filtered.columns:
        grp = df_filtered.groupby("Project Name").agg(
            Total=("Project Name", "count"),
            Resolved=("_EffectiveResolutionDT", lambda x: x.notna().sum()),
            R2C=("_R2C_Flag", "sum"),
            RespOnly=("_RespondedNotClosed_Flag", "sum"),
            Median_Close_Hrs=("Computed Closure Time (Hrs)", "median"),
            SLA_Met=("SLA Met", "mean"),
        ).reset_index()
        if "SLA_Met" in grp.columns:
            grp["SLA_Met"] = (grp["SLA_Met"] * 100).round(1)
        st.dataframe(grp, use_container_width=True)

        melted = grp.melt(id_vars=["Project Name"], value_vars=["Total","Resolved","R2C","RespOnly"],
                          var_name="Metric", value_name="Count")
        # unique colors per metric (only 4)
        cmap = brand_map_for(["Total","Resolved","R2C","RespOnly"])
        fig_proj = px.bar(melted, x="Project Name", y="Count", color="Metric", barmode="group",
                          color_discrete_map=cmap, text_auto=True)
        fig_proj.update_xaxes(tickangle=30, tickfont=dict(size=11))
        show_chart(style_fig(fig_proj, title="Project — Total vs Resolved vs R2C vs RespOnly"), key="tab1-project-bar")
    else:
        st.info("Column 'Project Name' not found.")

# ---------- Project Explorer ----------
with tabs[2]:
    st.header("Project Explorer")
    def simple_counts_bar(df0: pd.DataFrame, col: str, title: str, key: str):
        if col not in df0.columns: return
        vc = df0[col].fillna("—").astype(str).value_counts().reset_index()
        vc.columns = [col, "Count"]
        cmap = brand_map_for(vc[col].tolist())
        fig = px.bar(vc.sort_values("Count"), x="Count", y=col, orientation="h",
                     color=col, color_discrete_map=cmap, text_auto=True)
        show_chart(style_fig(fig, title=title), key=key)

    c1, c2 = st.columns([1,2])
    with c1:
        st.caption("Counts by Types & Tags")
        for colname, key in [("Type L0","typeL0"), ("Type L1","typeL1"),
                             ("Type L2","typeL2"), ("Tag 1","tag1"), ("Tag 2","tag2")]:
            simple_counts_bar(df_filtered, colname, f"{colname} — Counts (Top)", f"tab2-{key}")
    with c2:
        st.caption("Counts by Status, SLA, R2C and Responded-not-Closed")
        if "Current Status" in df_filtered.columns:
            simple_counts_bar(df_filtered, "Current Status", "Current Status — Counts", "tab2-status")
        if "SLA Met" in df_filtered.columns:
            work = df_filtered.copy()
            work["SLA State"] = work["SLA Met"].map({True:"Met", False:"Missed"}).fillna("Unknown")
            simple_counts_bar(work, "SLA State", "SLA — States", "tab2-sla")
        # R2C by assignee
        if "Assigned Team User" in df_filtered.columns:
            r2c_scope = df_filtered[df_filtered["_R2C_Flag"]==1]
            if len(r2c_scope):
                vc = r2c_scope["Assigned Team User"].fillna("—").astype(str).value_counts().reset_index()
                vc.columns = ["Assignee","R2C"]
                cmap = brand_map_for(vc["Assignee"].tolist())
                fig = px.bar(vc.sort_values("R2C"),
                             x="R2C", y="Assignee", orientation="h",
                             color="Assignee", color_discrete_map=cmap, text_auto=True)
                show_chart(style_fig(fig, title="R2C — by Assignee (inferred)"), key="tab2-r2c-assignee")

# ---------- Tower-Wise ----------
with tabs[3]:
    st.header("Tower-Wise")
    tower_col = "Location L1" if "Location L1" in df_filtered.columns else None
    if tower_col:
        grp = df_filtered.groupby(tower_col).agg(
            Total=(tower_col,"count"),
            Resolved=("_EffectiveResolutionDT", lambda x: x.notna().sum()),
            R2C=("_R2C_Flag", "sum"),
            RespOnly=("_RespondedNotClosed_Flag", "sum"),
        ).reset_index()
        st.dataframe(grp.sort_values("Total", ascending=False), use_container_width=True)

        melted = grp.melt(id_vars=[tower_col], value_vars=["Total","Resolved","R2C","RespOnly"],
                          var_name="Metric", value_name="Count")
        cmap = brand_map_for(["Total","Resolved","R2C","RespOnly"])
        fig_tower = px.bar(melted, x=tower_col, y="Count", color="Metric", barmode="group",
                           color_discrete_map=cmap, text_auto=True)
        fig_tower.update_xaxes(tickangle=30, tickfont=dict(size=11))
        show_chart(style_fig(fig_tower, title="Tower — Totals vs Resolved vs R2C vs RespOnly"), key="tab3-tower")
    else:
        st.info("Column 'Location L1' not found.")

# ---------- User-Wise ----------
with tabs[4]:
    st.header("User-Wise")
    if "Assigned Team User" in df_filtered.columns:
        usr = df_filtered.groupby("Assigned Team User").agg(
            Total=("Assigned Team User","count"),
            Resolved=("_EffectiveResolutionDT", lambda x: x.notna().sum()),
            R2C=("_R2C_Flag", "sum"),
            RespOnly=("_RespondedNotClosed_Flag", "sum"),
            Median_Close_Hrs=("Computed Closure Time (Hrs)", "median"),
        ).reset_index()

        long_u = usr.melt(id_vars=["Assigned Team User"], value_vars=["Resolved","R2C","RespOnly"],
                          var_name="Metric", value_name="Count")
        tot = long_u.groupby("Assigned Team User")["Count"].sum().sort_values(ascending=False).head(15).index
        long_u_top = long_u[long_u["Assigned Team User"].isin(tot)]
        cmap = brand_map_for(["Resolved","R2C","RespOnly"])
        fig_u_grp = px.bar(long_u_top.sort_values("Count"),
                           x="Count", y="Assigned Team User", color="Metric", barmode="group",
                           color_discrete_map=cmap, text_auto=True)
        fig_u_grp.update_yaxes(categoryorder="total ascending")
        show_chart(style_fig(fig_u_grp, title="User — Resolved vs R2C vs RespOnly (Top 15)"), key="tab4-u-group")
    else:
        st.info("Column 'Assigned Team User' not found.")

# ---------- Activity-Wise ----------
with tabs[5]:
    st.header("Activity-Wise")
    st.caption("Totals + R2C (inferred) + Responded-not-Closed; brand gradient for %R2C.")
    def grouped_measures(by_col: str, title_prefix: str):
        if by_col not in df_filtered.columns:
            return
        agg = df_filtered.groupby(by_col).agg(
            Total=(by_col,"count"),
            R2C=("_R2C_Flag", "sum"),
            RespOnly=("_RespondedNotClosed_Flag", "sum"),
        ).reset_index()
        if len(agg) == 0: return
        agg["R2C%"] = np.where(agg["Total"]>0, agg["R2C"]/agg["Total"]*100, np.nan)
        agg = agg.sort_values("Total", ascending=False).head(20)

        # Grouped bars
        melted = agg.melt(id_vars=[by_col], value_vars=["Total","R2C","RespOnly"], var_name="Metric", value_name="Count")
        cmap = brand_map_for(["Total","R2C","RespOnly"])
        fig = px.bar(melted, x=by_col, y="Count", color="Metric", barmode="group",
                     color_discrete_map=cmap, text_auto=True)
        fig.update_xaxes(tickangle=30, tickfont=dict(size=10))
        show_chart(style_fig(fig, title=f"{title_prefix} — Totals vs R2C vs RespOnly"), key=f"act-{by_col}-bars")

        # R2C%
        figp = px.bar(agg.sort_values("R2C%", ascending=True),
                      x="R2C%", y=by_col, orientation="h",
                      color="R2C%", color_continuous_scale=[[0.0, WHITE],[0.5, BLUE],[1.0, BLACK]],
                      text="R2C%")
        figp.update_traces(texttemplate="%{x:.1f}%")
        show_chart(style_fig(figp, title=f"{title_prefix} — % R2C (inferred)"), key=f"act-{by_col}-pct")

    c1, c2 = st.columns(2)
    with c1:
        grouped_measures("Type L1", "Type L1")
        if "Tag 1" in df_filtered.columns:
            grouped_measures("Tag 1", "Tag 1")
    with c2:
        grouped_measures("Type L2", "Type L2")
        if "Tag 2" in df_filtered.columns:
            grouped_measures("Tag 2", "Tag 2")

# ---------- Timelines ----------
with tabs[6]:
    st.header("Timelines")
    if df_filtered["_RaisedOnDT"].notna().any():
        work = df_filtered.copy()
        work["Date"] = work["_RaisedOnDT"].dt.date
        series = work.groupby("Date").agg(
            Raised=("Date","count"),
            Resolved=("_EffectiveResolutionDT", lambda x: x.notna().sum()),
            R2C=("_R2C_Flag", "sum"),
            RespOnly=("_RespondedNotClosed_Flag", "sum"),
        ).reset_index()
        mc = metric_colors
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=series["Date"], y=series["Raised"], mode="lines", name="Raised", line=dict(color=mc["Total"])))
        fig2.add_trace(go.Scatter(x=series["Date"], y=series["Resolved"], mode="lines", name="Resolved", line=dict(color=mc["Resolved"])))
        fig2.add_trace(go.Scatter(x=series["Date"], y=series["R2C"], mode="lines", name="R2C", line=dict(color=mc["R2C"])))
        fig2.add_trace(go.Scatter(x=series["Date"], y=series["RespOnly"], mode="lines", name="RespOnly", line=dict(color=mc["RespOnly"])))
        fig2.update_layout(title="Daily Flow — Raised vs Resolved vs R2C vs RespOnly")
        show_chart(style_fig(fig2), key="timeline-all")
    else:
        st.info("No Raised On timestamps available.")

# ---------- NC-View ----------
with tabs[7]:
    st.header("NC-View")
    proj_opts = sorted(df_filtered.get("Project Name", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    sel_proj = st.selectbox("Filter by Project (optional)", ["(All)"] + proj_opts, index=0, key="nc-proj")
    df_scope = df_filtered if sel_proj == "(All)" else df_filtered[df_filtered.get("Project Name","").astype(str) == sel_proj]

    if "Reference ID" in df_scope.columns and len(df_scope):
        ref_opts = df_scope["Reference ID"].astype(str).tolist()
        sel_ref = st.selectbox("Select Reference ID", ref_opts, key="nc-ref")
        row = df_scope[df_scope["Reference ID"].astype(str) == sel_ref].head(1)
    else:
        st.info("No Reference IDs available in current filters.")
        row = pd.DataFrame()

    if not row.empty:
        r = row.iloc[0]
        def _fmt_ts(x):
            return "" if pd.isna(x) else pd.to_datetime(x).strftime("%Y-%m-%d %H:%M")
        cols = st.columns(4)
        cols[0].metric("Project", str(r.get("Project Name","—")))
        cols[1].metric("Status", str(r.get("Current Status","—")))
        sla = r.get("SLA Met")
        cols[2].metric("SLA", "Met" if sla is True else ("Missed" if sla is False else "—"))
        cols[3].metric("Assignee", str(r.get("Assigned Team User","—")))
        st.caption(
            f"Raised: {_fmt_ts(r.get('_RaisedOnDT'))} | "
            f"Responded: {_fmt_ts(r.get('_RespondedOnDT'))} | "
            f"Rejected: {_fmt_ts(r.get('_RejectedOnDT'))} | "
            f"Closed: {_fmt_ts(r.get('_ClosedOnDT'))} | "
            f"Effective: {_fmt_ts(r.get('_EffectiveResolutionDT'))}"
        )
        # Event strip
        events = []
        for name, col, color in [
            ("Raised", "_RaisedOnDT", GREY),
            ("Responded", "_RespondedOnDT", BLUE),
            ("Rejected", "_RejectedOnDT", GREY),
            ("Closed", "_ClosedOnDT", BLACK),
            ("Effective", "_EffectiveResolutionDT", BLUE),
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
            show_chart(style_fig(fig_ev), key="nc-events")
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
                                 color="segment", color_discrete_sequence=generate_brand_sequence(len(tdf["segment"].unique())))
            fig_tl.update_yaxes(autorange="reversed")
            fig_tl.update_layout(title="Gantt (segments)", height=250, showlegend=False)
            show_chart(style_fig(fig_tl), key="nc-gantt")

# ---------- Sketch-View ----------
with tabs[8]:
    st.header("Sketch-View")
    st.caption("Treemap from 'Location / Reference' (pre-aggregated). Brand palette only.")
    path_col = "Location / Reference" if "Location / Reference" in df_filtered.columns else None
    if not path_col:
        st.info("Column 'Location / Reference' not found.")
    else:
        treemap_enable = st.checkbox("Show Treemap", value=True, key="sk-enable")
        max_depth = st.slider("Max depth", 1, 6, 4, step=1, key="sk-depth")
        max_nodes = st.slider("Max nodes", 100, 5000, 800, step=100, help="Limit total rectangles for performance.", key="sk-nodes")

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
            if len(agg) > max_nodes:
                agg = agg.sort_values("Count", ascending=False).head(max_nodes)
            agg = unique_columns(agg)
            path_cols = [c for c in agg.columns if c.startswith("Level_")]
            fig_t = px.treemap(
                agg, path=path_cols, values="Count",
                color_discrete_sequence=generate_brand_sequence(len(path_cols))
            )
            fig_t.update_layout(title="Location / Reference — Treemap (aggregated)")
            show_chart(style_fig(fig_t), key="sk-treemap")

# ---------- Status (NEW) ----------
with tabs[9]:
    st.header("Status")
    # Time window selector
    period = st.selectbox("Show status changes in:", ["Today", "Last 3 days", "This week", "All available"], index=0)
    last_change = df_filtered.get("_LastStatusChangeDT")
    if last_change is None:
        st.error("Missing _LastStatusChangeDT. Please check preprocessing.")
        st.stop()
    # Make sure it's tz-naive for comparisons
    if pd.api.types.is_datetime64tz_dtype(last_change):
        last_change = last_change.dt.tz_convert(None)

    now = pd.Timestamp.now(tz="UTC").tz_convert(None)
    today = now.normalize()
    start_of_week = today - pd.Timedelta(days=today.weekday())

    if period == "Today":
        m = last_change.dt.normalize() == today
        window_label = f"Today ({today.date()})"
    elif period == "Last 3 days":
        cutoff = today - pd.Timedelta(days=2)  # today + prev 2 = 3 days
        m = last_change >= cutoff
        window_label = f"Last 3 days (since {cutoff.date()})"
    elif period == "This week":
        m = last_change >= start_of_week
        window_label = f"This week (since {start_of_week.date()})"
    else:
        m = last_change.notna()
        window_label = "All available"

    changed = df_filtered.loc[m & last_change.notna()].copy()

    # Top metrics
    total_changed = len(changed)
    r2c_changed = int(changed["_R2C_Flag"].sum()) if "_R2C_Flag" in changed.columns else 0
    resp_only = int(changed["_RespondedNotClosed_Flag"].sum()) if "_RespondedNotClosed_Flag" in changed.columns else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Changed in window", total_changed)
    c2.metric("R2C in window", r2c_changed)
    c3.metric("Responded-not-Closed in window", resp_only)

    # Event counts (click to filter if plugin available)
    sel_events: List[str] = []
    sel_statuses: List[str] = []

    left, right = st.columns(2)
    with left:
        if "_LastStatusEvent" in changed.columns:
            ev_counts = (changed["_LastStatusEvent"]
                         .fillna("—").astype(str)
                         .value_counts().rename_axis("Event").reset_index(name="Count"))
            cmap_ev = brand_map_for(ev_counts["Event"].tolist())
            fig_ev = px.bar(ev_counts, x="Event", y="Count",
                            color="Event", color_discrete_map=cmap_ev,
                            text_auto=True)
            clicked = show_chart(style_fig(fig_ev, title="Last Status Event — counts", subtitle=window_label),
                                 key="status-ev-counts", clickable=True)
            if clicked:
                sel_events = list(dict.fromkeys(clicked))  # unique order
        else:
            st.info("No _LastStatusEvent column available.")

    with right:
        if "Current Status" in changed.columns:
            st_counts = (changed["Current Status"]
                         .fillna("—").astype(str)
                         .value_counts().rename_axis("Status").reset_index(name="Count"))
            cmap_st = brand_map_for(st_counts["Status"].tolist())
            fig_st = px.bar(st_counts, x="Status", y="Count",
                            color="Status", color_discrete_map=cmap_st,
                            text_auto=True)
            clicked_s = show_chart(style_fig(fig_st, title="Current Status — counts", subtitle=window_label),
                                   key="status-cur-counts", clickable=True)
            if clicked_s:
                sel_statuses = list(dict.fromkeys(clicked_s))
        else:
            st.info("Column 'Current Status' not found.")

    # Fallback selectors if no click package
    if not HAVE_PLOTLY_EVENTS:
        col_fs1, col_fs2 = st.columns(2)
        with col_fs1:
            all_ev = sorted(changed["_LastStatusEvent"].dropna().astype(str).unique().tolist()) if "_LastStatusEvent" in changed.columns else []
            sel_events = st.multiselect("Filter by Event", all_ev, default=sel_events)
        with col_fs2:
            all_st = sorted(changed["Current Status"].dropna().astype(str).unique().tolist()) if "Current Status" in changed.columns else []
            sel_statuses = st.multiselect("Filter by Status", all_st, default=sel_statuses)

    # Apply interactive filters
    mm = np.ones(len(changed), dtype=bool)
    if sel_events and "_LastStatusEvent" in changed.columns:
        mm &= changed["_LastStatusEvent"].astype(str).isin(sel_events).to_numpy()
    if sel_statuses and "Current Status" in changed.columns:
        mm &= changed["Current Status"].astype(str).isin(sel_statuses).to_numpy()

    changed_view = changed.loc[mm].copy()

    st.markdown("##### Changes over time")
    daily = (changed_view.assign(Date=last_change.loc[changed_view.index].dt.date)
             .groupby("Date").size().reset_index(name="Count")) if len(changed_view) else pd.DataFrame(columns=["Date","Count"])
    fig_line = px.line(daily, x="Date", y="Count", markers=True)
    show_chart(style_fig(fig_line, title="Status changes per day", subtitle=window_label), key="status-line")

    # R2C by Assignee (within window/selection)
    if len(changed_view) and "_R2C_Flag" in changed_view.columns and "Assigned Team User" in changed_view.columns:
        r2c_win = changed_view[changed_view["_R2C_Flag"]==1]
        if len(r2c_win):
            vc = r2c_win["Assigned Team User"].fillna("—").astype(str).value_counts().reset_index()
            vc.columns = ["Assignee","R2C"]
            cmap = brand_map_for(vc["Assignee"].tolist())
            fig = px.bar(vc.sort_values("R2C"), x="R2C", y="Assignee", orientation="h",
                         color="Assignee", color_discrete_map=cmap, text_auto=True)
            show_chart(style_fig(fig, title="R2C in window — by Assignee"), key="status-r2c-assignee")

    st.markdown("##### Rows matching selection (highlighted by Current Status)")
    # Display table (styled)
    show_cols = [c for c in [
        "Reference ID","Project Name","Location / Reference","Location Variable (Fixed)",
        "Current Status","_LastStatusEvent","_LastStatusChangeDT",
        "Assigned Team","Assigned Team User",
        "Responding Time (Hrs)","Computed Closure Time (Hrs)","Close After Response (Hrs)",
        "_RespondedNotClosed_Flag","_R2C_Flag","_R2C_Strict_Flag",
        "URL"
    ] if c in changed_view.columns]
    if not show_cols:
        st.info("No recognizable columns to display.")
    else:
        table_df = changed_view[show_cols].sort_values("_LastStatusChangeDT", ascending=False)
        try:
            st.write(style_status_rows(table_df.head(1500)).to_html(), unsafe_allow_html=True)
        except Exception:
            st.dataframe(table_df.head(1500), use_container_width=True)
        dl = table_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download (CSV) — Status window/selection", data=dl,
                           file_name="status_window_selection.csv", mime="text/csv", key="dl-status")

# ---------- NC Table ----------
with tabs[10]:
    st.header("NC Table")
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
        "Labour Cost","Material Cost","Machinery Cost","Other Cost","Total Cost",
        "URL"
    ] if c in df_filtered.columns]

    if not display_cols:
        st.info("No known NC columns found in the dataset.")
    else:
        view = df_filtered[display_cols]
        if shade:
            try:
                st.write(style_status_rows(view.head(1500)).to_html(), unsafe_allow_html=True)
            except Exception:
                st.dataframe(view.head(1500), use_container_width=True)
        else:
            st.dataframe(view.head(1500), use_container_width=True)
        csv_data = view.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download filtered table (CSV)", data=csv_data,
                           file_name="digiqc_filtered.csv", mime="text/csv", key="dl-full-table")

st.caption("© Digital Issue Dashboard — Streamlit (SJCPL Brand)")
