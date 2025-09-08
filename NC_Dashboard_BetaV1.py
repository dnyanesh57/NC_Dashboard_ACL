# digiqc_dashboard_NC_V2.6_BRAND.py
# Digital NC Register — Streamlit (SJCPL Brand)
# - Brand-locked colors only (Blue/Black/Grey/White)
# - Unique color per series within each chart (no reuse)
# - Gradient sampling when more colors are needed
# - Uses your NEW robust file upload logic (CSV/Excel + demo fallback)
# --------------------------------------------------------------
# Run:
#   pip install -U streamlit plotly pandas numpy openpyxl
#   streamlit run digiqc_dashboard_NC_V2.6_BRAND.py

from typing import Optional, Any, Tuple, List, Sequence
import datetime as dt
import io
import re
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

BRAND_GRADIENT = [
    [0.0, WHITE],
    [0.5, BLUE],
    [1.0, BLACK],
]
BRAND_SEQ = [BLUE, GREY, BLACK]

THEMES = {
    "SJCPL": {
        "template": "plotly_white",
        "status_map": SJCPL_STATUS,
        "metric_map": SJCPL_METRICS,
        "gradient": BRAND_GRADIENT,
        "palette": BRAND_SEQ
    }
}
theme = "SJCPL"

# --- Persistent gradient titlebar (forces white text) ---
APP_TITLE = "🧭 DigiQC — NC Insights Dashboard"
APP_SUB   = "SJCPL visual theme · Roboto · Brand colors only"

HEADER_BG = f"linear-gradient(90deg, {BLACK} 0%, {BLUE} 100%)"  # uses your brand vars

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');
html, body, [class*="css"], .stApp {{
  font-family: 'Roboto', sans-serif;
}}
/* Ultra-specific + !important so nothing can override it later */
div#sj-titlebar {{
  background-image: {HEADER_BG} !important;
  background-color: {BLACK} !important;     /* fallback while gradient loads */
  color: {WHITE} !important;
  padding: 14px 18px; border-radius: 14px; margin: 6px 0 18px 0;
}}
div#sj-titlebar h1, div#sj-titlebar p {{
  color: {WHITE} !important;
  margin: 0;
}}
div#sj-titlebar p {{ margin-top: 4px; opacity: .9; }}
</style>

<div id="sj-titlebar">
  <h1>{APP_TITLE}</h1>
  <p>{APP_SUB}</p>
</div>
""", unsafe_allow_html=True)

# ---------- Brand-aware colour helpers ----------
def _hex_to_rgb(h: str) -> tuple[int,int,int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _rgb_to_hex(rgb: tuple[int,int,int]) -> str:
    return "#%02X%02X%02X" % rgb

def _mix(c1: str, c2: str, t: float) -> str:
    r1,g1,b1 = _hex_to_rgb(c1); r2,g2,b2 = _hex_to_rgb(c2)
    r = int(round(r1 + (r2 - r1) * t))
    g = int(round(g1 + (g2 - g1) * t))
    b = int(round(b1 + (b2 - b1) * t))
    return _rgb_to_hex((r, g, b))

def _text_on(bg_hex: str) -> str:
    r,g,b = _hex_to_rgb(bg_hex)
    lum = 0.2126*r + 0.7152*g + 0.0722*b
    return WHITE if lum < 140 else BLACK

def sample_brand_gradient(n: int, clamp: tuple[float,float]=(0.06, 0.94)) -> list[str]:
    if n <= 0: return []
    xs = np.linspace(clamp[0], clamp[1], n)
    out: list[str] = []
    for t in xs:
        if t <= 0.5:
            out.append(_mix(WHITE, BLUE, (t - 0.0)/0.5))
        else:
            out.append(_mix(BLUE, BLACK, (t - 0.5)/0.5))
    # ensure uniqueness
    seen = set(); uniq = []
    for i, c in enumerate(out):
        u = c.upper()
        if u in seen:
            tj = float(xs[i]) + 0.001
            tj = min(clamp[1], max(clamp[0], tj))
            c = _mix(WHITE, BLUE, (tj - 0.0)/0.5) if tj <= 0.5 else _mix(BLUE, BLACK, (tj - 0.5)/0.5)
            u = c.upper()
        uniq.append(c); seen.add(u)
    return uniq

def generate_brand_sequence(n: int) -> list[str]:
    base = BRAND_SEQ[:]
    if n <= len(base): return base[:n]
    return base + sample_brand_gradient(n - len(base))

def brand_map_for(values: Sequence) -> dict[str,str]:
    vals = [str(v) for v in values]
    cols = generate_brand_sequence(len(vals))
    return dict(zip(vals, cols))

def style_fig(fig, theme_name: str):
    fig.update_layout(
        template=THEMES[theme_name]["template"],
        font=dict(family="Roboto, Arial, sans-serif", size=12, color=BLACK),
        legend=dict(title='', orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="rgba(255,255,255,0.95)", font=dict(family="Roboto, Arial, sans-serif", color=BLACK)),
        bargap=0.22,
        margin=dict(l=10, r=10, t=80, b=30)
    )
    fig.update_traces(marker_line_width=0.5, marker_line_color="rgba(0,0,0,0.20)")
    fig.update_xaxes(showgrid=True, gridcolor="#ECEFF1", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#ECEFF1", zeroline=False)
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
    return out

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

def style_status_rows(df: pd.DataFrame, status_map: dict) -> Styler:
    def highlight(row):
        status = str(row.get("Current Status", "")).strip()
        bg = status_map.get(status, WHITE)
        txt = _text_on(bg)
        return [f"background-color: {bg}; color: {txt};"] * len(row)
    return df.style.apply(highlight, axis=1)

# ---------- Derived columns (effective closure + flags) ----------
def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in [
        "Root Cause Analysis","Correction","Corrective Action",
        "Labour Cost","Material Cost","Machinery Cost","Other Cost","Total Cost"
    ]:
        if col not in df.columns:
            df[col] = np.nan

    df["Location Variable (Fixed)"] = extract_location_variable(_safe_get(df, "Location Variable")) if "Location Variable" in df.columns else np.nan

    df["_RaisedOnDT"]     = combine_datetime(_safe_get(df, "Raised On Date"), _safe_get(df, "Raised On Time"))
    df["_DeadlineDT"]     = combine_datetime(_safe_get(df, "Deadline Date"), _safe_get(df, "Deadline Time"))
    df["_RespondedOnDT"]  = combine_datetime(_safe_get(df, "Responded On Date"), _safe_get(df, "Responded On Time"))
    df["_RejectedOnDT"]   = combine_datetime(_safe_get(df, "Rejected On Date"), _safe_get(df, "Rejected On Time"))
    df["_ClosedOnDT"]     = combine_datetime(_safe_get(df, "Closed On Date"), _safe_get(df, "Closed On Time"))

    eff = df["_ClosedOnDT"].copy()
    mask_eff = eff.isna() & df["_RespondedOnDT"].notna() & df["_RaisedOnDT"].notna() & (df["_RespondedOnDT"] > df["_RaisedOnDT"])
    eff.loc[mask_eff] = df.loc[mask_eff, "_RespondedOnDT"]
    df["_EffectiveResolutionDT"] = eff

    df["Computed Closure Time (Hrs)"] = (df["_EffectiveResolutionDT"] - df["_RaisedOnDT"]).dt.total_seconds() / 3600.0
    df["Responding Time (Hrs)"]       = (df["_RespondedOnDT"] - df["_RaisedOnDT"]).dt.total_seconds() / 3600.0

    df["_RespondedNotClosed_Flag"] = (
        df["_ClosedOnDT"].isna() &
        df["_RespondedOnDT"].notna() &
        df["_RaisedOnDT"].notna() &
        (df["_RespondedOnDT"] > df["_RaisedOnDT"])
    ).astype(int)

    mask_car = df["_EffectiveResolutionDT"].notna() & df["_RespondedOnDT"].notna() & (df["_EffectiveResolutionDT"] >= df["_RespondedOnDT"])
    df["Close After Response (Hrs)"] = np.where(
        mask_car,
        (df["_EffectiveResolutionDT"] - df["_RespondedOnDT"]).dt.total_seconds() / 3600.0,
        np.nan
    )

    df["SLA Met"] = np.where(
        df["_DeadlineDT"].notna() & df["_EffectiveResolutionDT"].notna(),
        df["_EffectiveResolutionDT"] <= df["_DeadlineDT"],
        np.nan
    )

    parts = ["Labour Cost","Material Cost","Machinery Cost","Other Cost"]
    if "Total Cost" in df.columns:
        part_sum = df[[p for p in parts if p in df.columns]].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        df["Total Cost"] = pd.to_numeric(df["Total Cost"], errors="coerce").fillna(part_sum)

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

    return df

def _to_label(x) -> str:
    if pd.isna(x): return "—"
    if isinstance(x, (list, tuple, set)):
        return ", ".join("" if pd.isna(y) else str(y).strip() for y in x)
    s = str(x).strip()
    return s if s else "—"

def bar_top_counts(df: pd.DataFrame, col: str, topn: int = 10, color_seq=None, template="plotly_white", theme_name: str="SJCPL"):
    if col not in df.columns:
        return px.bar(pd.DataFrame({col: [], "count": []}), x="count", y=col, template=template)
    labels = df[col].apply(_to_label)
    vc = labels.value_counts(dropna=False).head(topn)
    cats = [str(x) for x in vc.index.tolist()]
    counts = pd.DataFrame({col: cats, "count": vc.values})
    cmap = brand_map_for(cats)
    fig = px.bar(
        counts.sort_values("count", ascending=True),
        x="count", y=col, orientation="h",
        color=col, color_discrete_map=cmap, template=template, text_auto=True
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

# ---------- YOUR NEW LOAD LOGIC (as provided) ----------
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
            st.error("No file uploaded and demo CSV not found in working directory.")
            st.stop()
    else:
        name = getattr(file, "name", "uploaded.csv").lower()
        if name.endswith(".xlsx") or name.endswith(".xls"):
            df = pd.read_excel(file)
        else:
            # Try a few encodings for CSV
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
    # Clean & dedupe headers safely (does NOT change label wording)
    df = df.rename(columns={c: normalize_colname(c) for c in df.columns})
    df = df.loc[:, ~pd.Series(df.columns).duplicated().values]
    return df

# ---------- Sidebar (brand-locked) ----------
with st.sidebar:
    st.title("🧭 Digital NC Register")
    st.caption("SJCPL brand-locked UI (Roboto + Blue/Black/Grey/White)")
    logo_url = st.text_input("Logo URL (optional)", value="")
    st.markdown("#### Data Source")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx","xls"], key="uploader")
    st.caption("If nothing is uploaded, a demo CSV from GitHub will be loaded automatically.")

# Apply brand defaults for Plotly
px.defaults.template = THEMES[theme]["template"]
px.defaults.color_discrete_sequence = THEMES[theme]["palette"]

# ---------- Sticky Header (branding) ----------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');
    html, body, [class*="css"] {{
      font-family: 'Roboto', Arial, sans-serif !important;
      color: {BLACK};
    }}
    :root {{
      --sj-primary: {BLUE};
      --sj-surface: {WHITE};
      --sj-text: {BLACK};
      --sj-muted: {GREY};
    }}
    .sj-header {{
      position: sticky; top: 0; z-index: 1000;
      background: var(--sj-surface);
      border-bottom: 1px solid rgba(0,0,0,0.08);
      padding: 8px 12px;
      margin: -1rem -1rem 0.5rem -1rem;
    }}
    .sj-header-inner {{
      display:flex; justify-content:space-between; align-items:center;
      gap: 1rem;
    }}
    .sj-title {{
      font-weight: 700; font-size: 20px; color: var(--sj-primary);
    }}
    .sj-subtle {{
      color: var(--sj-muted); font-size: 13px;
    }}
    .sj-logo {{
      height: 32px; object-fit: contain;
    }}
    .plot-container .legend {{
      overflow-x: auto;
    }}
    </style>
    <div class="sj-header">
      <div class="sj-header-inner">
        <div>
          <div class="sj-title">Digital NC Register</div>
          <div class="sj-subtle">SJCPL — Issue Analytics Dashboard</div>
        </div>
        <div>
          {"<img class='sj-logo' src='" + logo_url + "' />" if logo_url else ""}
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- Load & preprocess ----------
try:
    df_raw = load_data(uploaded)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

@st.cache_data(show_spinner=False)
def preprocess(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df.columns = [c.strip() for c in df.columns]
    return add_derived_columns(df)

df = preprocess(df_raw)

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

# ---------- Dynamic, per-view colour maps ----------
def _uniq(col: str) -> List[str]:
    return df_filtered.get(col, pd.Series(dtype=str)).dropna().astype(str).unique().tolist()

metric_colors = THEMES[theme]["metric_map"]
status_colors = THEMES[theme]["status_map"]

STATUS_COLOR_MAP = brand_map_for(sorted(_uniq("Current Status")))
METRIC_MAP_2 = brand_map_for(["Total", "Rejected→Closed"])
METRIC_MAP_3 = brand_map_for(["Resolved", "R2C", "RespOnly"])
METRIC_MAP_4 = brand_map_for(["Total", "Resolved", "R2C", "RespOnly"])

mask_r2c_inferred   = (df_filtered["_R2C_Flag"] == 1)
mask_responly       = (df_filtered["_RespondedNotClosed_Flag"] == 1)

r2c_count_scope     = int(mask_r2c_inferred.sum())
resp_only_count     = int(mask_responly.sum())
total_ncs_scope     = len(df_filtered)

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
        color_discrete_map=METRIC_MAP_2
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
            color_discrete_sequence=[metric_colors["RespOnly"]]
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
            color_discrete_sequence=[metric_colors["RespOnly"]]
        )
        show_chart(style_fig(fig_rb, theme), key="ov-responded-not-closed-raisedby")

    if "Current Status" in df_filtered.columns:
        vc = df_filtered["Current Status"].value_counts(dropna=False).reset_index()
        vc.columns = ["Current Status","Count"]
        fig_sd = px.bar(vc, x="Current Status", y="Count", text_auto=True, title="Current Status Distribution",
                        color="Current Status", color_discrete_map=STATUS_COLOR_MAP)
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

# ---------- Project Status ----------
with tabs[1]:
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
                          color_discrete_map=METRIC_MAP_4)
        fig_proj.update_xaxes(tickangle=30, tickfont=dict(size=11))
        show_chart(style_fig(fig_proj, theme), key="tab1-project-bar")
    else:
        st.info("Column 'Project Name' not found.")

# ---------- Project Explorer ----------
with tabs[2]:
    st.header("Project Explorer")
    c1, c2 = st.columns([1,2])
    with c1:
        st.caption("Counts by Types & Tags")
        for colname, key in [("Type L0","typeL0"), ("Type L1","typeL1"), ("Type L2","typeL2"), ("Tag 1","tag1"), ("Tag 2","tag2")]:
            if colname in df_filtered.columns:
                show_chart(bar_top_counts(df_filtered, colname,
                                          color_seq=THEMES[theme]["palette"], template=THEMES[theme]["template"], theme_name=theme),
                           key=f"tab2-{key}")
    with c2:
        st.caption("Counts by Status, SLA, R→C and Responded-not-Closed")
        if "Current Status" in df_filtered.columns:
            fig_st = bar_top_counts(df_filtered, "Current Status", color_seq=None, template=THEMES[theme]["template"], theme_name=theme)
            fig_st.for_each_trace(lambda t: t.update(marker=dict(line_width=0.5)))
            show_chart(fig_st, key="tab2-status")

        if "SLA Met" in df_filtered.columns:
            work = df_filtered.copy()
            work["SLA State"] = df_filtered["SLA Met"].map({True: "Met", False: "Missed"}).fillna("Unknown")
            show_chart(bar_top_counts(work, "SLA State",
                                      color_seq=THEMES[theme]["palette"], template=THEMES[theme]["template"], theme_name=theme),
                       key="tab2-sla")

        if "Assigned Team User" in df_filtered.columns and r2c_count_scope > 0:
            counts = (df_filtered.loc[mask_r2c_inferred, "Assigned Team User"]
                      .fillna("—").astype(str)
                      .value_counts().rename_axis("Assignee").reset_index(name="Rejected→Closed"))
            fig_r2c_scope = px.bar(counts.sort_values("Rejected→Closed"),
                                   x="Rejected→Closed", y="Assignee", orientation="h",
                                   title="Rejected → Closed — by Assignee (inferred, scope)",
                                   color_discrete_sequence=[metric_colors["R2C"]])
            show_chart(style_fig(fig_r2c_scope, theme), key="tab2-r2c-assignee")

        if "Assigned Team User" in df_filtered.columns and resp_only_count > 0:
            resp_counts = (df_filtered.loc[mask_responly, "Assigned Team User"]
                           .fillna("—").astype(str)
                           .value_counts().rename_axis("Assignee").reset_index(name="Responded not Closed"))
            fig_resp_scope = px.bar(resp_counts.sort_values("Responded not Closed"),
                                    x="Responded not Closed", y="Assignee", orientation="h",
                                    title="Responded but NOT Closed — by Assignee (scope)",
                                    color_discrete_sequence=[metric_colors["RespOnly"]])
            show_chart(style_fig(fig_resp_scope, theme), key="tab2-responly-assignee")

# ---------- Tower-Wise ----------
with tabs[3]:
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
                           color_discrete_map=METRIC_MAP_4)
        fig_tower.update_xaxes(tickangle=30, tickfont=dict(size=11))
        show_chart(style_fig(fig_tower, theme), key="tab3-tower-group")
    else:
        st.info("Column 'Location L1' not found.")

# ---------- User-Wise ----------
with tabs[4]:
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
                           title="User — Resolved vs Rejected-not-Closed vs Responded-not-Closed (Top N)",
                           color_discrete_map=METRIC_MAP_3, text_auto=True)
        fig_u_grp.update_yaxes(categoryorder="total ascending")
        show_chart(style_fig(fig_u_grp, theme), key="tab4-u-group")

        resp_all = usr[usr["RespOnly"] > 1].sort_values("RespOnly")
        if len(resp_all):
            fig_u_resp_only = px.bar(resp_all,
                                     x="RespOnly", y="Assigned Team User", orientation="h",
                                     title="Responded but NOT Closed — Counts (ALL users, count > 1)",
                                     color_discrete_sequence=[metric_colors["RespOnly"]])
            show_chart(style_fig(fig_u_resp_only, theme), key="tab4-u-resp-only")
        else:
            st.info("No Responded-not-Closed users with count > 1.")

        c1, c2 = st.columns(2)
        with c1:
            fig_u_total = px.bar(usr.sort_values("Total", ascending=False).head(25),
                                 x="Assigned Team User", y="Total", title="Top Assignees by Total",
                                 color_discrete_sequence=[metric_colors["Total"]])
            fig_u_total.update_xaxes(tickangle=30)
            show_chart(style_fig(fig_u_total, theme), key="tab4-u-total")
        with c2:
            fig_u_res = px.bar(usr.sort_values("Resolved", ascending=False).head(25),
                               x="Assigned Team User", y="Resolved", title="Top Assignees by Resolved",
                               color_discrete_sequence=[metric_colors["Resolved"]])
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
                               color_discrete_sequence=[metric_colors["Resolved"]])
            show_chart(style_fig(fig_u_med, theme), key="tab4-u-med")
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
                color_discrete_sequence=[metric_colors["RespOnly"]]
            )
            show_chart(style_fig(fig_rb_all, theme), key="tab4-raisedby-responly")
        else:
            st.info("No Responded-not-Closed raised-by reminders with count > 1 in current filters.")

# ---------- Activity-Wise ----------
with tabs[5]:
    st.header("Activity-Wise")
    st.caption("Totals + R→C (inferred) + Responded-not-Closed; brand gradient for %R→C (White→Blue→Black).")

    def grouped_measures(by_col: str, key_prefix: str, show_ratio: bool = True, topn: int = 20):
        if by_col not in df_filtered.columns:
            return
        agg = df_filtered.groupby(by_col).agg(
            Total=("Reference ID","count") if "Reference ID" in df_filtered.columns else (by_col,"count"),
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
            color_discrete_map=METRIC_MAP_4, text_auto=True
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

    c1, c2 = st.columns(2)
    with c1:
        grouped_measures("Type L1", "tab5", show_ratio=True, topn=20)
        if "Tag 1" in df_filtered.columns:
            grouped_measures("Tag 1", "tab5-tag1", show_ratio=True, topn=20)
    with c2:
        grouped_measures("Type L2", "tab5", show_ratio=True, topn=20)
        if "Tag 2" in df_filtered.columns:
            grouped_measures("Tag 2", "tab5-tag2", show_ratio=True, topn=20)

# ---------- Timelines (light) ----------
with tabs[6]:
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
        line_cols = generate_brand_sequence(4)
        fig2.add_trace(go.Scatter(x=series["Date"], y=series["Raised"],   mode="lines", name="Raised",               line=dict(color=line_cols[0])))
        fig2.add_trace(go.Scatter(x=series["Date"], y=series["Resolved"], mode="lines", name="Resolved",             line=dict(color=line_cols[1])))
        fig2.add_trace(go.Scatter(x=series["Date"], y=series["R2C"],      mode="lines", name="Rejected→Closed",      line=dict(color=line_cols[2])))
        fig2.add_trace(go.Scatter(x=series["Date"], y=series["RespOnly"], mode="lines", name="Responded-not-Closed", line=dict(color=line_cols[3])))
        fig2.update_layout(title="Daily Flow — Raised vs Resolved vs R→C vs Responded-not-Closed")
        show_chart(style_fig(fig2, theme), key="tab6-lines")
    else:
        st.info("No Raised On timestamps available.")

# ---------- NC-View ----------
with tabs[7]:
    st.header("NC-View")

    proj_opts = sorted(df_filtered.get("Project Name", pd.Series(dtype=str)).dropna().unique().tolist())
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

        events = []
        for name, col in [
            ("Raised", "_RaisedOnDT"),
            ("Responded", "_RespondedOnDT"),
            ("Rejected", "_RejectedOnDT"),
            ("Closed", "_ClosedOnDT"),
            ("Effective", "_EffectiveResolutionDT"),
            ("Deadline", "_DeadlineDT"),
        ]:
            ts = r.get(col)
            if pd.notna(ts):
                events.append(dict(name=name, ts=pd.to_datetime(ts)))
        if events:
            event_names = [e["name"] for e in events]
            event_map = brand_map_for(event_names)
            evdf = pd.DataFrame([dict(name=e["name"], ts=e["ts"], color=event_map[e["name"]]) for e in events]).sort_values("ts")
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
            seg_map = brand_map_for([s["segment"] for s in segs])
            fig_tl = px.timeline(tdf, x_start="start", x_end="finish", y="segment",
                                 color="segment", color_discrete_map=seg_map)
            fig_tl.update_yaxes(autorange="reversed")
            fig_tl.update_layout(title="Gantt (segments)", height=250, showlegend=False)
            show_chart(style_fig(fig_tl, theme), key="nc-gantt")
        else:
            st.info("Not enough timestamps to render Gantt segments.")

        steps = []
        if pd.notna(rs): steps.append(("Raised", pd.to_datetime(rs), 0))
        if pd.notna(rp) and pd.notna(rs) and rp >= rs: steps.append(("Responded", pd.to_datetime(rp), 1))
        rj = r.get("_RejectedOnDT")
        if pd.notna(rj) and (not steps or pd.to_datetime(rj) >= steps[-1][1]): steps.append(("Rejected", pd.to_datetime(rj), 2))
        cl = r.get("_ClosedOnDT")
        if pd.notna(cl): steps.append(("Closed", pd.to_datetime(cl), 3))
        ef_ts = r.get("_EffectiveResolutionDT")
        if pd.notna(ef_ts) and (not cl or pd.to_datetime(ef_ts) != pd.to_datetime(cl)):
            steps.append(("Effective", pd.to_datetime(ef_ts), 4))
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

        rs = r.get("_RaisedOnDT"); dl = r.get("_DeadlineDT"); eff = r.get("_EffectiveResolutionDT")
        if pd.notna(rs) and (pd.notna(dl) or pd.notna(eff)):
            base = pd.to_datetime(rs)
            dl_hours = (pd.to_datetime(dl) - base).total_seconds()/3600 if pd.notna(dl) else None
            ef_hours = (pd.to_datetime(eff) - base).total_seconds()/3600 if pd.notna(eff) else None
            bars = []
            if dl_hours is not None: bars.append(("Deadline", max(0, dl_hours)))
            if ef_hours is not None: bars.append(("Actual", max(0, ef_hours)))
            if bars:
                bdf = pd.DataFrame(bars, columns=["Metric","Hours"])
                bmap = brand_map_for(bdf["Metric"].tolist())
                fig_b = px.bar(bdf, x="Hours", y="Metric", orientation="h",
                               title="SLA — Deadline vs Actual (hours)",
                               color="Metric",
                               color_discrete_map=bmap,
                               text_auto=True)
                show_chart(style_fig(fig_b, theme), key="nc-sla-bullet")
        else:
            st.info("No SLA/Effective times available for bullet chart.")
    else:
        st.info("Select a Reference ID to view details.")

# ---------- Sketch-View (Treemap) ----------
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
            palette_big = sample_brand_gradient(64)
            fig_t = px.treemap(
                agg, path=path_cols, values="Count",
                color_discrete_sequence=palette_big
            )
            fig_t.update_layout(title="Location / Reference — Treemap (aggregated)")
            show_chart(style_fig(fig_t, theme), key="sk-treemap")

        st.subheader("Issues at Selected Path")
        selected = st.text_input("Filter by path contains", "", key="sketch-filter")
        view = df_filtered[df_filtered[path_col].str.contains(selected, case=False, na=False)] if selected else df_filtered

        show_cols = [c for c in [
            "Reference ID","Project Name", path_col, "Location Variable (Fixed)",
            "Type L0","Type L1","Type L2","Tag 1","Tag 2",
            "Assigned Team","Assigned Team User","Current Status",
            "Responding Time (Hrs)","Computed Closure Time (Hrs)","Close After Response (Hrs)",
            "URL"
        ] if c in view.columns]
        if len(view):
            st.dataframe(view[show_cols].head(1500), use_container_width=True)
        else:
            st.info("No issues match the selected filter.")

# ---------- NC Table ----------
with tabs[9]:
    st.header("NC Table")
    st.caption("Styled table with row shading by Current Status (brand colours, unique per status).")
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
                st.write(style_status_rows(view.head(1500), STATUS_COLOR_MAP).to_html(), unsafe_allow_html=True)
            except Exception:
                st.dataframe(view.head(1500), use_container_width=True)
        else:
            st.dataframe(view.head(1500), use_container_width=True)
        csv_data = view.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download filtered table (CSV)", data=csv_data, file_name="digiqc_filtered.csv", mime="text/csv", key="dl-full-table")

st.caption("© Digital Issue Dashboard — Streamlit (SJCPL Brand)")
