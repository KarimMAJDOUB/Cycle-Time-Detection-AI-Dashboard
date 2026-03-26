"""
=============================================================================
  streamlit_app.py  —  Cycle-Time Detection  |  AI Dashboard
=============================================================================

  Launch:
      streamlit run streamlit_app.py

  Requirements:
      pip install streamlit plotly pandas numpy scipy scikit-learn

  Data : Data/signal_flows_raw.csv  (columns: name, time, i12, i61)
  Model: random_forest_model.pkl    (produced by test_pkl.py)
=============================================================================
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.signal import find_peaks, savgol_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Cycle-Time Detection — AI Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CUSTOM CSS   (dark premium feel)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* ── Google Font ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── Full dark background (app + header + toolbar) ── */
  .stApp                              { background: #0f172a !important; color: #e2e8f0; }
  header[data-testid="stHeader"]      { background: #0f172a !important; border-bottom: 1px solid #1e293b !important; }
  .stToolbar, [data-testid="stToolbar"] { background: #0f172a !important; }
  [data-testid="stDecoration"]        { display: none !important; }
  [data-testid="stStatusWidget"]      { color: #64748b !important; }
  .stDeployButton                     { display: none !important; }
  button[kind="header"]               { background: #1e293b !important; color: #94a3b8 !important; border:none; }
  /* Select / input widgets dark */
  div[data-baseweb="select"] > div    { background:#1e293b !important; border-color:#334155 !important; color:#e2e8f0 !important; }
  div[data-baseweb="input"] > div     { background:#1e293b !important; border-color:#334155 !important; color:#e2e8f0 !important; }
  .stSlider [data-baseweb="slider"]   { background:#334155 !important; }
  .stRadio label                      { color:#cbd5e1 !important; }
  .stDataFrame                        { background:#1e293b !important; }
  /* Metrics */
  [data-testid="stMetric"] { background:#1e293b; border-radius:10px; padding:12px 16px;
                              border:1px solid #334155; }
  [data-testid="stMetricLabel"] { color:#64748b !important; }
  [data-testid="stMetricValue"] { color:#f8fafc !important; }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
    border-right: 1px solid #334155;
  }
  section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
  section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background:#0f172a !important; border-color:#334155 !important;
  }

  /* ── KPI Cards ── */
  .kpi-grid { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:24px; }
  .kpi-card {
    flex:1; min-width:150px;
    background: linear-gradient(135deg,#1e293b,#0f172a);
    border:1px solid #334155; border-radius:14px;
    padding:18px 22px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    transition: transform .2s, box-shadow .2s;
  }
  .kpi-card:hover { transform:translateY(-3px); box-shadow: 0 8px 32px rgba(0,0,0,0.6); }
  .kpi-label { font-size:11px; font-weight:600; letter-spacing:1.2px;
               text-transform:uppercase; color:#64748b; margin-bottom:6px; }
  .kpi-value { font-size:28px; font-weight:700; color:#f8fafc; line-height:1; }
  .kpi-unit  { font-size:12px; font-weight:400; color:#94a3b8; margin-top:4px; }

  /* ── Section headings ── */
  h1 { font-size:26px !important; font-weight:700 !important;
       background: linear-gradient(90deg,#38bdf8,#818cf8);
       -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  h2 { font-size:18px !important; font-weight:600 !important; color:#e2e8f0 !important; }
  h3 { font-size:15px !important; font-weight:500 !important; color:#94a3b8 !important; }

  /* ── Divider ── */
  hr { border:none; border-top:1px solid #334155; margin:24px 0; }

  /* ── Alert box ── */
  .info-box {
    background:#1e3a5f; border:1px solid #3b82f6; border-radius:10px;
    padding:14px 18px; font-size:13px; color:#bfdbfe; margin-bottom:16px;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR PALETTE  (matching Analysis.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

C = {
    "raw":    "rgba(100,149,237,0.35)",
    "smooth": "#38bdf8",
    "peak":   "#f43f5e",
    "proba":  "#fb923c",
    "tp":     "#4ade80",
    "fn":     "#a78bfa",
    "bg":     "#0f172a",
    "grid":   "#1e293b",
    "text":   "#e2e8f0",
}

LAYOUT = dict(
    paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
    font=dict(family="Inter, Arial", size=12, color=C["text"]),
    legend=dict(bgcolor="#1e293b", bordercolor="#334155", borderwidth=1),
)

# ─────────────────────────────────────────────────────────────────────────────
#  SIGNAL PROCESSING HELPERS  (identical to AI_model.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_fs(ts_ns: np.ndarray) -> float:
    ts_s   = ts_ns / 1e9
    dt_med = float(np.median(np.diff(ts_s)))
    return 1.0 / dt_med if dt_med > 0 else 1.0


def smooth_signal(signal: np.ndarray, fs: float,
                  window_s: float = 11.0, polyorder: int = 2) -> np.ndarray:
    w = max(max(5, polyorder + 2), int(round(window_s * fs)))
    if w % 2 == 0:
        w += 1
    return savgol_filter(signal, window_length=w, polyorder=polyorder)


def detect_peaks_mad(smoothed: np.ndarray, fs: float,
                     prominence_k: float = 1.5,
                     min_dist_s: float = 5.0) -> np.ndarray:
    med  = np.median(smoothed)
    mad  = np.median(np.abs(smoothed - med)) + 1e-12
    prom = prominence_k * 1.4826 * mad
    dist = max(1, int(round(min_dist_s * fs)))
    peaks, _ = find_peaks(smoothed, prominence=prom, distance=dist)
    return peaks


def build_features(signal: np.ndarray, fs: float, win_s: float = 15.0) -> pd.DataFrame:
    w = int(max(3, round(win_s * fs)))
    x = pd.Series(signal.astype(float))
    feat = pd.DataFrame()
    feat["I"]          = x
    feat["dI"]         = x.diff().fillna(0)
    feat["ddI"]        = feat["dI"].diff().fillna(0)
    feat["roll_mean"]  = x.rolling(w, center=True, min_periods=1).mean()
    feat["roll_std"]   = x.rolling(w, center=True, min_periods=1).std().fillna(0)
    feat["roll_min"]   = x.rolling(w, center=True, min_periods=1).min()
    feat["roll_max"]   = x.rolling(w, center=True, min_periods=1).max()
    feat["roll_range"] = feat["roll_max"] - feat["roll_min"]
    feat["z_local"]    = (x - feat["roll_mean"]) / (feat["roll_std"] + 1e-6)

    feat = feat.replace([np.inf, -np.inf], 0).fillna(0)
    return feat


def build_labels(n: int, peaks: np.ndarray, tol: int = 0) -> np.ndarray:
    y = np.zeros(n, dtype=int)
    for p in peaks:
        y[max(0, p - tol): min(n, p + tol + 1)] = 1
    return y


def build_cycles(ts_ns: np.ndarray, peak_indices: np.ndarray,
                 smoothed: np.ndarray) -> pd.DataFrame:
    if len(peak_indices) < 2:
        return pd.DataFrame()
    ts_s = ts_ns / 1e9
    peak_ts = ts_s[peak_indices]
    return pd.DataFrame({
        "cycle_index":    np.arange(1, len(peak_ts), dtype=int),
        "cycle_start_ts": peak_ts[:-1],
        "cycle_end_ts":   peak_ts[1:],
        "period_s":       peak_ts[1:] - peak_ts[:-1],
        "peak_amplitude": smoothed[peak_indices[:-1]],
    })

# ─────────────────────────────────────────────────────────────────────────────
#  CACHED DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["i12"]  = pd.to_numeric(df["i12"],  errors="coerce")
    df["i61"]  = pd.to_numeric(df["i61"],  errors="coerce")
    df = df.dropna(subset=["time", "i12", "i61"])
    # ✅ Filtre global : i12 > 5A ET i61 > 5A
    df = df[(df["i12"] > 5.0) & (df["i61"] > 5.0)]

    df = df.sort_values(["name", "time"]).reset_index(drop=True)
    return df


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
#  FIXED PATHS
# ─────────────────────────────────────────────────────────────────────────────

CSV_PATH   = "Data/signal_flows_raw.csv"
MODEL_PATH = "random_forest_model.pkl"

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD DATA EARLY (needed for gauge list in sidebar)
# ─────────────────────────────────────────────────────────────────────────────

if not Path(CSV_PATH).exists():
    st.error(f"❌ CSV not found: `{CSV_PATH}`")
    st.stop()

with st.spinner("Loading data…"):
    df_raw = load_raw(CSV_PATH)

gauges = sorted(df_raw["name"].unique())

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ Cycle-Time Detection")
    st.markdown("---")

    st.markdown("### 🏭 eGauge Selection")
    gauge = st.selectbox("eGauge", gauges, index=0)

    st.markdown("---")
    st.markdown("### 📡 Signal")
    sig_col = st.radio("Current channel", ["i12", "i61"], horizontal=True)

    st.markdown("---")
    st.markdown("### 🔧 Signal Parameters")
    smooth_win   = st.slider("Smoothing window (s)",  3.0, 30.0, 11.0, 0.5)
    peak_prom_k  = st.slider("Prominence k (σ_MAD)²", 0.5, 4.0,  1.5,  0.1)
    min_dist_s   = st.slider("Min peak distance (s)", 1.0, 20.0, 5.0,  0.5)
    threshold    = st.slider("P(peak) threshold",     0.05, 0.95, 0.50, 0.05)

    st.markdown("---")
    model_path = MODEL_PATH
    st.caption("DP — Digital Practice")

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOAD
# ─────────────────────────────────────────────────────────────────────────────

model_ok = Path(model_path).exists()

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# ⚡ Cycle-Time Detection — AI Dashboard")
st.markdown(
    '<div class="info-box">Analyses electrical signals using a '
    '<b>Savitzky-Golay → MAD Peak Detection → Random Forest</b> pipeline. '
    'Select an eGauge and channel in the sidebar to explore results.</div>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────
#  SIGNAL PROCESSING FOR SELECTED GAUGE
# ─────────────────────────────────────────────────────────────────────────────

df_g  = df_raw[df_raw["name"] == gauge].reset_index(drop=True)
ts_ns = df_g["time"].to_numpy()
ts_s  = ts_ns / 1e9
t_rel = ts_s - ts_s[0]                       # elapsed time (s)
raw   = df_g[sig_col].to_numpy(dtype=float)
fs    = estimate_fs(ts_ns)

smoothed     = smooth_signal(raw, fs, window_s=smooth_win)
peak_indices = detect_peaks_mad(smoothed, fs,
                                prominence_k=peak_prom_k,
                                min_dist_s=min_dist_s)
cycles       = build_cycles(ts_ns, peak_indices, smoothed)
feat_df      = build_features(raw, fs)
labels       = build_labels(len(raw), peak_indices)

# ─────────────────────────────────────────────────────────────────────────────
#  RANDOM FOREST INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

if model_ok:
    model  = load_model(model_path)
    proba  = model.predict_proba(feat_df)[:, 1]
    y_pred = (proba >= threshold).astype(int)
else:
    st.warning("⚠️ Model not found — run `python test_pkl.py` first. Showing rule-based peaks only.")
    proba  = np.zeros(len(raw))
    y_pred = np.zeros(len(raw), dtype=int)

# ─────────────────────────────────────────────────────────────────────────────
#  KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────

n_cycles   = len(cycles)
mean_T     = cycles["period_s"].mean() if not cycles.empty else 0
std_T      = cycles["period_s"].std()  if not cycles.empty else 0
min_T      = cycles["period_s"].min()  if not cycles.empty else 0
max_T      = cycles["period_s"].max()  if not cycles.empty else 0

kpis = [
    ("SAMPLES",    f"{len(raw):,}",         ""),
    ("SAMPLING fs",f"{fs:.2f}",             "Hz"),
    ("PEAKS",      f"{len(peak_indices)}",  "detected"),
    ("CYCLES",     f"{n_cycles}",           ""),
    ("MEAN T",     f"{mean_T:.2f}",         "s"),
    ("STD T",      f"{std_T:.2f}",          "s"),
    ("MIN T",      f"{min_T:.2f}",          "s"),
    ("MAX T",      f"{max_T:.2f}",          "s"),
]

cards_html = '<div class="kpi-grid">'
for label, val, unit in kpis:
    cards_html += f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{val}</div>
      <div class="kpi-unit">{unit}</div>
    </div>"""
cards_html += "</div>"
st.markdown(cards_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CHART 1: Signal + Rule-based Peaks  &  RF Probability  (dual panel)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(f"## 📈 Signal & Peak Detection")

tp_mask = (labels == 1) & (y_pred == 1)
fn_mask = (labels == 1) & (y_pred == 0)

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.60, 0.40],
    vertical_spacing=0.06,
    subplot_titles=[
        f"<b>Signal & Rule-based Peaks</b> — {gauge} / {sig_col}",
        "<b>Random Forest Peak Probability</b>",
    ],
)

# Raw signal
fig.add_trace(go.Scatter(
    x=t_rel, y=raw, mode="lines", name="Raw signal",
    line=dict(color=C["raw"], width=1),
    hovertemplate="t=%{x:.2f} s<br>I=%{y:.4f} A<extra>Raw</extra>",
), row=1, col=1)

# Smoothed signal
fig.add_trace(go.Scatter(
    x=t_rel, y=smoothed, mode="lines", name="Smoothed signal",
    line=dict(color=C["smooth"], width=2),
    hovertemplate="t=%{x:.2f} s<br>I=%{y:.4f} A<extra>Smoothed</extra>",
), row=1, col=1)

# Confirmed peaks (TP)
if tp_mask.any():
    idx = np.where(tp_mask)[0]
    fig.add_trace(go.Scatter(
        x=t_rel[idx], y=smoothed[idx], mode="markers",
        name="Peak (confirmed)", marker=dict(size=11, color=C["tp"],
        symbol="triangle-up", line=dict(color="white", width=1)),
        hovertemplate="t=%{x:.2f} s<br>I=%{y:.4f} A<extra>Confirmed</extra>",
    ), row=1, col=1)

# Missed peaks (FN)
if fn_mask.any():
    idx = np.where(fn_mask)[0]
    fig.add_trace(go.Scatter(
        x=t_rel[idx], y=smoothed[idx], mode="markers",
        name="Peak (missed)", marker=dict(size=11, color=C["fn"],
        symbol="x", line=dict(color="white", width=1)),
        hovertemplate="t=%{x:.2f} s<br>I=%{y:.4f} A<extra>Missed</extra>",
    ), row=1, col=1)

# P(peak) area chart
fig.add_hrect(y0=threshold, y1=1.0,
              fillcolor="rgba(244,63,94,0.06)", line_width=0, row=2, col=1)

fig.add_trace(go.Scatter(
    x=t_rel, y=proba, mode="lines", name="P(peak)",
    line=dict(color=C["proba"], width=2),
    fill="tozeroy", fillcolor="rgba(251,146,60,0.15)",
    hovertemplate="t=%{x:.2f} s<br>P(peak)=%{y:.3f}<extra></extra>",
), row=2, col=1)

fig.add_hline(y=threshold, row=2, col=1,
              line=dict(color=C["peak"], dash="dash", width=1.5),
              annotation_text=f"  Threshold = {threshold:.2f}",
              annotation_font_color=C["peak"])

fig.update_layout(
    **LAYOUT,
    height=600,
    hovermode="x unified",
    margin=dict(l=60, r=40, t=80, b=60),
    xaxis2=dict(title="Elapsed time (s)", gridcolor=C["grid"]),
    yaxis=dict(title="Current (A)", gridcolor=C["grid"]),
    yaxis2=dict(title="P(peak)", range=[0, 1.05], gridcolor=C["grid"]),
)
st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CHART 2: Cycle period distribution
# ─────────────────────────────────────────────────────────────────────────────

if not cycles.empty:
    st.markdown("---")
    st.markdown("## 🔄 Cycle Period Distribution")

    col1, col2 = st.columns([2, 1])

    with col1:
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=cycles["period_s"],
            nbinsx=40,
            marker=dict(color="#38bdf8", line=dict(color="#0f172a", width=1)),
            name="Cycle period",
            hovertemplate="Period: %{x:.2f} s<br>Count: %{y}<extra></extra>",
        ))
        fig2.add_vline(x=mean_T, line=dict(color="#f43f5e", dash="dash", width=2),
                       annotation_text=f"  Mean = {mean_T:.2f} s",
                       annotation_font_color="#f43f5e")
        fig2.update_layout(
            **LAYOUT, height=320, margin=dict(l=60, r=40, t=50, b=60),
            title="<b>Histogram of Cycle Periods</b>",
            xaxis=dict(title="Period (s)", gridcolor=C["grid"]),
            yaxis=dict(title="Count",      gridcolor=C["grid"]),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("### 📋 Cycle Summary")
        summary_data = {
            "Metric": ["Cycles", "Mean T (s)", "Std T (s)", "Min T (s)", "Max T (s)", "Frequency (Hz)"],
            "Value":  [
                str(n_cycles),
                f"{mean_T:.3f}",
                f"{std_T:.3f}",
                f"{min_T:.3f}",
                f"{max_T:.3f}",
                f"{1/mean_T:.4f}" if mean_T > 0 else "—",
            ]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#475569; font-size:12px;">'
    'Cycle-Time Detection — AI Dashboard &nbsp;|&nbsp; Digital Petroleum &nbsp;|&nbsp; '
    f'eGauge: <b style="color:#64748b">{gauge}</b> &nbsp;|&nbsp; Channel: <b style="color:#64748b">{sig_col}</b>'
    '</div>',
    unsafe_allow_html=True,
)
