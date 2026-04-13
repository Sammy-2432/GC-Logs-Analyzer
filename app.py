import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import time
from datetime import datetime, timedelta
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GC Observability Dashboard",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark base */
.stApp {
    background: #0b0e1a;
    color: #e2e8f0;
}

.block-container {
    padding: 1.5rem 2rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #131929 0%, #1a2235 100%);
    border: 1px solid #2a3a5c;
    border-radius: 12px;
    padding: 18px 22px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #38bdf8, #6366f1);
}
.metric-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 26px;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1;
}
.metric-sub {
    font-size: 12px;
    color: #94a3b8;
    margin-top: 4px;
}
.metric-card.danger::before { background: linear-gradient(90deg, #f43f5e, #fb923c); }
.metric-card.warn::before   { background: linear-gradient(90deg, #f59e0b, #fcd34d); }
.metric-card.ok::before     { background: linear-gradient(90deg, #34d399, #10b981); }

/* Section headers */
.section-header {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #38bdf8;
    margin: 28px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1e3a5f, transparent);
}

/* Upload zone */
.upload-zone {
    background: #0f1623;
    border: 2px dashed #1e3a5f;
    border-radius: 12px;
    padding: 28px;
    text-align: center;
    transition: border-color 0.2s;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #080c16 !important;
    border-right: 1px solid #1a2540;
}
section[data-testid="stSidebar"] .block-container {
    padding: 1rem;
}

/* Tables */
.dataframe { font-family: 'JetBrains Mono', monospace; font-size: 12px; }

/* Correlation badge */
.corr-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.corr-high  { background: #7f1d1d; color: #fca5a5; }
.corr-med   { background: #78350f; color: #fcd34d; }
.corr-low   { background: #14532d; color: #6ee7b7; }

/* Alert boxes */
.alert-box {
    background: #1a1428;
    border-left: 4px solid #f43f5e;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 13px;
}
.alert-box.warn { border-color: #f59e0b; }
.alert-box.info { border-color: #38bdf8; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# GC LOG PARSER  (streaming + regex-optimised)
# ─────────────────────────────────────────────

# Pre-compiled patterns (compiled once at module level)
_RE_UNIFIED = re.compile(
    r'\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)[^\]]*\]'   # ISO timestamp
    r'.*?\[(\w+)\s+\(([^)]+)\)\]'                               # GC type + cause
    r'.*?(\d+)M->(\d+)M\((\d+)M\)'                             # heap before->after(total)
    r'.*?([\d.]+)ms',                                            # pause ms
    re.DOTALL
)
_RE_G1_PAUSE = re.compile(
    r'\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)[^\]]*\]'
    r'.*?GC\((\d+)\).*?(Pause\s+\w+(?:\s+\w+)?)'
    r'.*?(\d+)M->(\d+)M\((\d+)M\)'
    r'.*?([\d.]+)ms',
    re.DOTALL
)
_RE_SIMPLE = re.compile(
    r'(\d+\.\d+):\s+\[(?:GC|Full GC)'
    r'(?:\s+\([^)]+\))?\s+'
    r'(?:\w+:\s+)?(\d+)K->(\d+)K\((\d+)K\),'
    r'\s+([\d.]+)\s+secs'
)
_RE_TIMESTAMP_ISO = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}')


def _parse_line_unified(line: str) -> dict | None:
    m = _RE_UNIFIED.search(line)
    if m:
        ts_str, gc_type, cause, before, after, total, pause = m.groups()
        try:
            ts = datetime.fromisoformat(ts_str)
        except ValueError:
            return None
        return {
            "timestamp": ts,
            "gc_type": gc_type.strip(),
            "cause": cause.strip(),
            "heap_before_mb": int(before),
            "heap_after_mb": int(after),
            "heap_total_mb": int(total),
            "pause_ms": float(pause),
            "reclaimed_mb": int(before) - int(after),
        }
    return None


def _parse_line_simple(line: str) -> dict | None:
    m = _RE_SIMPLE.search(line)
    if m:
        elapsed, before_k, after_k, total_k, secs = m.groups()
        gc_type = "Full GC" if "Full GC" in line else "Minor GC"
        return {
            "timestamp": float(elapsed),          # seconds elapsed (no wall clock)
            "gc_type": gc_type,
            "cause": "unknown",
            "heap_before_mb": int(before_k) // 1024,
            "heap_after_mb": int(after_k) // 1024,
            "heap_total_mb": int(total_k) // 1024,
            "pause_ms": float(secs) * 1000,
            "reclaimed_mb": (int(before_k) - int(after_k)) // 1024,
        }
    return None


def parse_gc_log_streaming(file_obj) -> pd.DataFrame:
    """Stream-parse a GC log file without loading everything into memory."""
    records = []
    text_wrapper = io.TextIOWrapper(file_obj, encoding="utf-8", errors="replace")
    for line in text_wrapper:
        r = _parse_line_unified(line) or _parse_line_simple(line)
        if r:
            records.append(r)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # If timestamp is numeric (elapsed seconds), convert to datetime offset from epoch
    if pd.api.types.is_float_dtype(df["timestamp"]):
        base = datetime(2024, 1, 1)
        df["timestamp"] = df["timestamp"].apply(
            lambda s: base + timedelta(seconds=s)
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────
# JTL PARSER
# ─────────────────────────────────────────────

def parse_jtl(file_obj) -> pd.DataFrame:
    """Parse JMeter JTL (CSV) file."""
    df = pd.read_csv(file_obj)
    # Normalise column names (JMeter can produce varying names)
    col_map = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in ("timestamp", "ts"):
            col_map[c] = "timestamp"
        elif lc in ("elapsed", "response_time", "latency"):
            col_map[c] = "response_time_ms"
        elif lc in ("label", "transaction", "name"):
            col_map[c] = "label"
        elif lc in ("success", "status"):
            col_map[c] = "success"
        elif lc in ("bytes", "sentbytes"):
            col_map[c] = "bytes"
        elif lc in ("connect", "connect_time"):
            col_map[c] = "connect_ms"
    df.rename(columns=col_map, inplace=True)

    if "timestamp" not in df.columns:
        raise ValueError("JTL file has no recognisable timestamp column.")

    # JMeter timestamps are epoch-milliseconds
    if df["timestamp"].dtype in (np.int64, np.float64):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    if "response_time_ms" not in df.columns:
        df["response_time_ms"] = np.nan

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────
# CORRELATION ENGINE
# ─────────────────────────────────────────────

def build_time_series(gc_df: pd.DataFrame, jtl_df: pd.DataFrame,
                      bucket_secs: int = 10) -> pd.DataFrame:
    """
    Align GC events and JTL response times onto a common time bucket grid.
    Returns a merged dataframe suitable for correlation analysis.
    """
    freq = f"{bucket_secs}s"

    # ── GC buckets ──
    gc_ts = gc_df.set_index("timestamp")
    gc_agg = gc_ts.resample(freq).agg(
        gc_count=("pause_ms", "count"),
        gc_total_pause_ms=("pause_ms", "sum"),
        gc_max_pause_ms=("pause_ms", "max"),
        gc_avg_pause_ms=("pause_ms", "mean"),
        heap_reclaimed_mb=("reclaimed_mb", "sum"),
        heap_after_mb=("heap_after_mb", "last"),
    ).fillna(0)

    # ── JTL buckets ──
    if not jtl_df.empty and "response_time_ms" in jtl_df.columns:
        jtl_ts = jtl_df.set_index("timestamp")
        jtl_agg = jtl_ts.resample(freq).agg(
            req_count=("response_time_ms", "count"),
            p50_ms=("response_time_ms", lambda x: np.percentile(x.dropna(), 50) if len(x.dropna()) else np.nan),
            p95_ms=("response_time_ms", lambda x: np.percentile(x.dropna(), 95) if len(x.dropna()) else np.nan),
            p99_ms=("response_time_ms", lambda x: np.percentile(x.dropna(), 99) if len(x.dropna()) else np.nan),
            avg_ms=("response_time_ms", "mean"),
            err_rate=("success", lambda x: (x.astype(str).str.upper() == "FALSE").mean() if len(x) else 0),
        ).fillna(method="ffill")
    else:
        jtl_agg = pd.DataFrame()

    if jtl_agg.empty:
        merged = gc_agg.reset_index()
    else:
        merged = gc_agg.join(jtl_agg, how="outer").fillna(0).reset_index()
    merged.rename(columns={"index": "timestamp"}, inplace=True)
    if "timestamp" not in merged.columns and merged.index.name == "timestamp":
        merged = merged.reset_index()
    return merged


def compute_correlations(ts_df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation between GC metrics and latency metrics."""
    gc_cols = ["gc_total_pause_ms", "gc_max_pause_ms", "gc_count", "heap_after_mb"]
    lat_cols = [c for c in ["p95_ms", "p99_ms", "avg_ms", "err_rate"] if c in ts_df.columns]
    rows = []
    for gc_c in gc_cols:
        for lat_c in lat_cols:
            if gc_c in ts_df.columns and lat_c in ts_df.columns:
                valid = ts_df[[gc_c, lat_c]].dropna()
                if len(valid) > 5:
                    corr = valid[gc_c].corr(valid[lat_c])
                    rows.append({"gc_metric": gc_c, "latency_metric": lat_c, "pearson_r": round(corr, 4)})
    return pd.DataFrame(rows)


def detect_gc_storms(gc_df: pd.DataFrame, window_secs: int = 60,
                     min_events: int = 5) -> pd.DataFrame:
    """Find time windows with abnormally high GC frequency."""
    if gc_df.empty:
        return pd.DataFrame()
    gc_df = gc_df.set_index("timestamp").sort_index()
    rolling = gc_df["pause_ms"].rolling(f"{window_secs}s").count()
    storms = rolling[rolling >= min_events]
    if storms.empty:
        return pd.DataFrame()
    storm_df = storms.reset_index()
    storm_df.columns = ["timestamp", "gc_events_in_window"]
    return storm_df


def detect_latency_spikes(ts_df: pd.DataFrame,
                           col: str = "p95_ms",
                           sigma: float = 2.0) -> pd.DataFrame:
    if col not in ts_df.columns:
        return pd.DataFrame()
    mu, std = ts_df[col].mean(), ts_df[col].std()
    if std == 0:
        return pd.DataFrame()
    spikes = ts_df[ts_df[col] > mu + sigma * std][["timestamp", col]].copy()
    spikes["z_score"] = ((spikes[col] - mu) / std).round(2)
    return spikes


# ─────────────────────────────────────────────
# SYNTHETIC DEMO DATA
# ─────────────────────────────────────────────

def generate_demo_gc() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    base = datetime(2024, 3, 15, 10, 0, 0)
    records = []
    heap = 512
    t = base
    for _ in range(500):
        t += timedelta(seconds=rng.integers(3, 25))
        is_full = rng.random() < 0.08
        gc_type = "Full GC" if is_full else "G1 Young"
        pause = rng.normal(200 if is_full else 35, 50 if is_full else 12)
        pause = max(5.0, pause)
        reclaim = rng.integers(30, 200) if is_full else rng.integers(5, 60)
        records.append({
            "timestamp": t,
            "gc_type": gc_type,
            "cause": rng.choice(["Allocation Failure", "G1 Humongous Allocation",
                                  "System.gc()", "Ergonomics"]),
            "heap_before_mb": heap,
            "heap_after_mb": max(100, heap - reclaim),
            "heap_total_mb": 2048,
            "pause_ms": round(pause, 2),
            "reclaimed_mb": reclaim,
        })
        heap = max(200, min(1900, heap + rng.integers(-20, 80)))
    return pd.DataFrame(records)


def generate_demo_jtl(gc_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    base = gc_df["timestamp"].min()
    end  = gc_df["timestamp"].max()
    total_secs = int((end - base).total_seconds())
    n = 8000
    ts = [base + timedelta(seconds=rng.integers(0, total_secs)) for _ in range(n)]
    ts.sort()

    # Inject latency spikes near Full GC events
    full_gc_times = gc_df[gc_df["gc_type"] == "Full GC"]["timestamp"].tolist()
    rows = []
    for t in ts:
        base_rt = rng.normal(120, 30)
        for fg in full_gc_times:
            delta = abs((t - fg).total_seconds())
            if delta < 15:
                base_rt += rng.normal(300, 80) * (1 - delta / 15)
        base_rt = max(5.0, base_rt)
        success = "true" if rng.random() > 0.02 else "false"
        rows.append({
            "timestamp": t,
            "response_time_ms": round(base_rt, 1),
            "label": rng.choice(["GET /api/search", "POST /api/order", "GET /api/user"]),
            "success": success,
            "bytes": rng.integers(500, 50000),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# PLOTLY CHART HELPERS
# ─────────────────────────────────────────────

COLORS = {
    "minor_gc":  "#38bdf8",
    "full_gc":   "#f43f5e",
    "p50":       "#34d399",
    "p95":       "#fbbf24",
    "p99":       "#f97316",
    "heap":      "#818cf8",
    "pause":     "#f43f5e",
    "storm":     "rgba(244,63,94,0.15)",
}

LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", color="#94a3b8", size=11),
    xaxis=dict(gridcolor="#1e2d47", zerolinecolor="#1e2d47", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#1e2d47", zerolinecolor="#1e2d47", tickfont=dict(size=10)),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                font=dict(size=10, color="#94a3b8")),
    margin=dict(l=50, r=20, t=40, b=40),
    hovermode="x unified",
)


def fig_gc_timeline(gc_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for gtype, color in [("Full GC", COLORS["full_gc"]),
                          ("G1 Young", COLORS["minor_gc"]),
                          ("Minor GC", COLORS["minor_gc"])]:
        sub = gc_df[gc_df["gc_type"] == gtype]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["timestamp"], y=sub["pause_ms"],
            mode="markers",
            name=gtype,
            marker=dict(color=color, size=sub["pause_ms"].clip(4, 20) / 4,
                        opacity=0.8, line=dict(width=0)),
            hovertemplate=(
                "<b>%{x|%H:%M:%S}</b><br>"
                "Pause: %{y:.1f} ms<br>"
                f"Type: {gtype}<extra></extra>"
            )
        ))
    fig.update_layout(**LAYOUT, title=dict(text="GC Pause Duration Over Time",
                                            font=dict(color="#e2e8f0", size=13),
                                            x=0.01))
    fig.update_yaxes(title_text="Pause (ms)")
    return fig


def fig_correlated(ts_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         vertical_spacing=0.06,
                         subplot_titles=("Response Latency (ms)", "GC Total Pause (ms)"))
    if "p50_ms" in ts_df.columns:
        fig.add_trace(go.Scatter(x=ts_df["timestamp"], y=ts_df["p50_ms"],
                                  name="P50", line=dict(color=COLORS["p50"], width=1.5)),
                       row=1, col=1)
    if "p95_ms" in ts_df.columns:
        fig.add_trace(go.Scatter(x=ts_df["timestamp"], y=ts_df["p95_ms"],
                                  name="P95", line=dict(color=COLORS["p95"], width=1.5)),
                       row=1, col=1)
    if "p99_ms" in ts_df.columns:
        fig.add_trace(go.Scatter(x=ts_df["timestamp"], y=ts_df["p99_ms"],
                                  name="P99", line=dict(color=COLORS["p99"], width=1.5,
                                                          dash="dot")),
                       row=1, col=1)
    if "gc_total_pause_ms" in ts_df.columns:
        fig.add_trace(go.Bar(x=ts_df["timestamp"], y=ts_df["gc_total_pause_ms"],
                              name="GC Pause", marker_color=COLORS["pause"],
                              marker_opacity=0.7),
                       row=2, col=1)
    fig.update_layout(**LAYOUT, height=420,
                       legend=dict(orientation="h", y=1.06, x=0, bgcolor="rgba(0,0,0,0)"))
    fig.update_annotations(font=dict(color="#64748b", size=11))
    for r in [1, 2]:
        fig.update_xaxes(gridcolor="#1e2d47", row=r, col=1)
        fig.update_yaxes(gridcolor="#1e2d47", row=r, col=1)
    return fig


def fig_heap_usage(gc_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gc_df["timestamp"], y=gc_df["heap_before_mb"],
        name="Heap Before", line=dict(color="#818cf8", width=1.5),
        fill="tozeroy", fillcolor="rgba(129,140,248,0.08)"
    ))
    fig.add_trace(go.Scatter(
        x=gc_df["timestamp"], y=gc_df["heap_after_mb"],
        name="Heap After", line=dict(color="#34d399", width=1.5)
    ))
    fig.update_layout(**LAYOUT,
                       title=dict(text="Heap Usage (MB)", font=dict(color="#e2e8f0", size=13), x=0.01),
                       height=260)
    fig.update_yaxes(title_text="MB")
    return fig


def fig_gc_cause_dist(gc_df: pd.DataFrame) -> go.Figure:
    counts = gc_df["cause"].value_counts().reset_index()
    counts.columns = ["cause", "count"]
    colors = px.colors.sequential.Blues_r[:len(counts)]
    fig = go.Figure(go.Bar(
        x=counts["count"], y=counts["cause"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=counts["count"], textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
    ))
    fig.update_layout(**LAYOUT,
                       title=dict(text="GC Cause Distribution",
                                   font=dict(color="#e2e8f0", size=13), x=0.01),
                       height=260)
    fig.update_xaxes(title_text="Events")
    return fig


def fig_pause_histogram(gc_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for gtype, color in [("Full GC", COLORS["full_gc"]), ("G1 Young", COLORS["minor_gc"]),
                          ("Minor GC", COLORS["minor_gc"])]:
        sub = gc_df[gc_df["gc_type"] == gtype]
        if sub.empty: continue
        fig.add_trace(go.Histogram(
            x=sub["pause_ms"], name=gtype,
            marker_color=color, opacity=0.7,
            xbins=dict(size=10),
        ))
    fig.update_layout(**LAYOUT,
                       title=dict(text="Pause Duration Distribution",
                                   font=dict(color="#e2e8f0", size=13), x=0.01),
                       barmode="overlay", height=260)
    fig.update_xaxes(title_text="Pause (ms)")
    fig.update_yaxes(title_text="Count")
    return fig


def fig_corr_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    pivot = corr_df.pivot(index="gc_metric", columns="latency_metric", values="pearson_r")
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0, "#0f4c81"], [0.5, "#1e293b"], [1, "#7f1d1d"]],
        zmin=-1, zmax=1,
        text=np.round(pivot.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=13, family="JetBrains Mono"),
        hoverongaps=False,
    ))
    fig.update_layout(**LAYOUT,
                       title=dict(text="GC ↔ Latency Pearson Correlation",
                                   font=dict(color="#e2e8f0", size=13), x=0.01),
                       height=280)
    return fig


def fig_throughput(ts_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "req_count" in ts_df.columns:
        fig.add_trace(go.Bar(
            x=ts_df["timestamp"], y=ts_df["req_count"],
            name="Requests / bucket",
            marker_color="#38bdf8", marker_opacity=0.6,
        ))
    if "err_rate" in ts_df.columns:
        fig.add_trace(go.Scatter(
            x=ts_df["timestamp"], y=ts_df["err_rate"] * 100,
            name="Error %",
            line=dict(color=COLORS["full_gc"], width=2),
            yaxis="y2",
        ))
    fig.update_layout(
        **LAYOUT,
        title=dict(text="Request Throughput & Error Rate",
                    font=dict(color="#e2e8f0", size=13), x=0.01),
        height=260,
        yaxis2=dict(overlaying="y", side="right", title_text="Error %",
                     gridcolor="transparent", tickfont=dict(size=10)),
    )
    return fig


# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────

def metric_card(label: str, value: str, sub: str = "", state: str = ""):
    cls = f"metric-card {state}"
    st.markdown(f"""
    <div class="{cls}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
    </div>
    """, unsafe_allow_html=True)


def section(title: str, icon: str = ""):
    st.markdown(f'<div class="section-header">{icon}&nbsp;{title}</div>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0 20px'>
        <div style='font-size:30px'>🧹</div>
        <div style='font-family:JetBrains Mono,monospace; font-size:14px;
                    font-weight:700; color:#38bdf8; letter-spacing:0.06em'>
            GC OBSERVATORY
        </div>
        <div style='font-size:11px; color:#475569; margin-top:4px'>
            JVM · Performance · Correlation
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 📂 Upload Files")

    gc_file  = st.file_uploader("GC Log (.log / .txt)", type=["log", "txt"],
                                 help="Supports G1GC, CMS, Parallel GC unified log formats")
    jtl_file = st.file_uploader("JTL / CSV (JMeter)", type=["jtl", "csv"],
                                 help="JMeter results CSV with timestamp + elapsed columns")

    st.markdown("---")
    st.markdown("##### ⚙️ Analysis Settings")
    bucket_secs  = st.slider("Time bucket (seconds)", 5, 120, 10, 5)
    storm_window = st.slider("GC Storm window (s)",   30, 300, 60, 30)
    storm_min    = st.slider("Storm min events",       3,  20,  5,  1)
    sigma_thresh = st.slider("Spike sigma threshold", 1.5, 4.0, 2.0, 0.5)

    st.markdown("---")
    use_demo = st.checkbox("🎲 Use synthetic demo data", value=(gc_file is None))

    st.markdown("""
    <div style='font-size:10px; color:#334155; padding-top:16px; line-height:1.6'>
        Supported GC log formats:<br>
        • JDK 9+ Unified Logging (-Xlog:gc*)<br>
        • G1GC, CMS, Parallel, ZGC<br>
        • JMeter JTL / CSV export
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_gc(file_bytes: bytes) -> pd.DataFrame:
    return parse_gc_log_streaming(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def load_jtl(file_bytes: bytes) -> pd.DataFrame:
    return parse_jtl(io.BytesIO(file_bytes))


with st.spinner("⚙️ Parsing logs..."):
    if use_demo or gc_file is None:
        gc_df  = generate_demo_gc()
        jtl_df = generate_demo_jtl(gc_df)
        data_source = "🎲 Demo"
    else:
        gc_df = load_gc(gc_file.read())
        if gc_df.empty:
            st.error("❌ Could not parse GC log. Ensure it's a valid JDK unified or classic GC log.")
            st.stop()
        jtl_df = load_jtl(jtl_file.read()) if jtl_file else pd.DataFrame()
        data_source = f"📁 {gc_file.name}"

ts_df   = build_time_series(gc_df, jtl_df, bucket_secs)
corr_df = compute_correlations(ts_df)
storms  = detect_gc_storms(gc_df, storm_window, storm_min)
spikes  = detect_latency_spikes(ts_df, sigma=sigma_thresh)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown(f"""
<div style='display:flex; align-items:center; justify-content:space-between;
            border-bottom:1px solid #1e2d47; padding-bottom:16px; margin-bottom:20px'>
    <div>
        <div style='font-size:22px; font-weight:800; color:#f1f5f9;
                    font-family:JetBrains Mono,monospace; letter-spacing:-0.02em'>
            JVM GC Observability Dashboard
        </div>
        <div style='font-size:12px; color:#475569; margin-top:3px'>
            {data_source} &nbsp;·&nbsp; {len(gc_df):,} GC events &nbsp;·&nbsp;
            {len(jtl_df):,} requests &nbsp;·&nbsp;
            Bucket: {bucket_secs}s
        </div>
    </div>
    <div style='font-size:11px; color:#334155; font-family:JetBrains Mono,monospace;
                text-align:right'>
        {gc_df["timestamp"].min().strftime('%Y-%m-%d %H:%M') if not gc_df.empty else "—"}<br>
        → {gc_df["timestamp"].max().strftime('%Y-%m-%d %H:%M') if not gc_df.empty else "—"}
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────

total_pause   = gc_df["pause_ms"].sum()
avg_pause     = gc_df["pause_ms"].mean()
max_pause     = gc_df["pause_ms"].max()
full_gc_count = (gc_df["gc_type"] == "Full GC").sum()
duration_min  = (gc_df["timestamp"].max() - gc_df["timestamp"].min()).total_seconds() / 60
gc_overhead   = (total_pause / 1000) / max(1, duration_min * 60) * 100

p99_val = ts_df["p99_ms"].mean() if "p99_ms" in ts_df.columns else None
top_corr = corr_df["pearson_r"].abs().max() if not corr_df.empty else 0

cols = st.columns(6)
with cols[0]:
    metric_card("Total GC Events", f"{len(gc_df):,}", f"{duration_min:.0f} min window")
with cols[1]:
    state = "danger" if max_pause > 500 else "warn" if max_pause > 200 else "ok"
    metric_card("Max Pause", f"{max_pause:.0f}ms", "worst STW event", state)
with cols[2]:
    metric_card("Avg Pause", f"{avg_pause:.1f}ms", "all GC events")
with cols[3]:
    state = "danger" if full_gc_count > 10 else "warn" if full_gc_count > 3 else "ok"
    metric_card("Full GC Count", str(full_gc_count), "stop-the-world events", state)
with cols[4]:
    state = "danger" if gc_overhead > 10 else "warn" if gc_overhead > 3 else "ok"
    metric_card("GC Overhead", f"{gc_overhead:.2f}%", "of total wall time", state)
with cols[5]:
    if p99_val:
        state = "danger" if p99_val > 500 else "warn" if p99_val > 200 else "ok"
        metric_card("Avg P99 Latency", f"{p99_val:.0f}ms", "request response time", state)
    else:
        metric_card("Correlation Peak", f"{top_corr:.2f}", "GC ↔ latency Pearson r",
                    "danger" if top_corr > 0.7 else "warn" if top_corr > 0.4 else "ok")


# ─────────────────────────────────────────────
# ALERTS
# ─────────────────────────────────────────────

alerts = []
if max_pause > 1000:
    alerts.append(("danger", f"🔴 Extreme STW pause detected: {max_pause:.0f}ms — potential application freeze"))
elif max_pause > 500:
    alerts.append(("warn",   f"🟡 High STW pause: {max_pause:.0f}ms — review heap sizing and GC policy"))
if full_gc_count > 10:
    alerts.append(("danger", f"🔴 {full_gc_count} Full GC events — likely memory pressure or promotion failure"))
if gc_overhead > 5:
    alerts.append(("warn",   f"🟡 GC overhead {gc_overhead:.1f}% exceeds 5% threshold — tune or upgrade heap"))
if not storms.empty:
    alerts.append(("danger", f"🔴 GC Storm detected: {len(storms)} windows with ≥{storm_min} events in {storm_window}s"))
if top_corr > 0.7:
    alerts.append(("danger", f"🔴 Strong GC→latency correlation ({top_corr:.2f}) — GC pauses are causing request slowdowns"))
if not spikes.empty and "p95_ms" in spikes.columns:
    alerts.append(("warn", f"🟡 {len(spikes)} latency spike windows detected (>{sigma_thresh}σ above mean)"))

if alerts:
    section("Alerts & Anomalies", "⚠️")
    for level, msg in alerts:
        st.markdown(f'<div class="alert-box {level}">{msg}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN CHARTS
# ─────────────────────────────────────────────

section("GC Pause Timeline", "📈")
st.plotly_chart(fig_gc_timeline(gc_df), use_container_width=True)

if not jtl_df.empty:
    section("Correlated: Latency vs GC Pressure", "🔗")
    st.plotly_chart(fig_correlated(ts_df), use_container_width=True)

col_a, col_b = st.columns(2)
with col_a:
    section("Heap Usage", "💾")
    st.plotly_chart(fig_heap_usage(gc_df), use_container_width=True)
with col_b:
    section("GC Cause Breakdown", "🔍")
    st.plotly_chart(fig_gc_cause_dist(gc_df), use_container_width=True)

col_c, col_d = st.columns(2)
with col_c:
    section("Pause Distribution", "📊")
    st.plotly_chart(fig_pause_histogram(gc_df), use_container_width=True)
with col_d:
    if not jtl_df.empty:
        section("Throughput & Errors", "🚦")
        st.plotly_chart(fig_throughput(ts_df), use_container_width=True)
    elif not corr_df.empty:
        section("Correlation Heatmap", "🌡️")
        st.plotly_chart(fig_corr_heatmap(corr_df), use_container_width=True)


# ─────────────────────────────────────────────
# CORRELATION HEATMAP (full row if JTL present)
# ─────────────────────────────────────────────

if not corr_df.empty and not jtl_df.empty:
    section("GC ↔ Latency Correlation Matrix", "🌡️")
    col_h, col_t = st.columns([2, 1])
    with col_h:
        st.plotly_chart(fig_corr_heatmap(corr_df), use_container_width=True)
    with col_t:
        st.markdown("<br>", unsafe_allow_html=True)
        for _, row in corr_df.iterrows():
            r = row["pearson_r"]
            badge_cls = "corr-high" if abs(r) > 0.7 else "corr-med" if abs(r) > 0.4 else "corr-low"
            st.markdown(
                f'<div style="margin:4px 0; font-size:12px; color:#94a3b8">'
                f'{row["gc_metric"]} → {row["latency_metric"]}&nbsp;&nbsp;'
                f'<span class="corr-badge {badge_cls}">{r:+.3f}</span></div>',
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────
# DATA TABLES
# ─────────────────────────────────────────────

section("Raw GC Events", "📋")
tab1, tab2, tab3 = st.tabs(["GC Events", "Time-Series Buckets", "GC Storms"])

with tab1:
    show_cols = [c for c in ["timestamp", "gc_type", "cause", "pause_ms",
                               "heap_before_mb", "heap_after_mb", "reclaimed_mb"]
                 if c in gc_df.columns]
    st.dataframe(
        gc_df[show_cols].sort_values("timestamp", ascending=False).head(200),
        use_container_width=True, height=300,
        column_config={
            "timestamp":     st.column_config.DatetimeColumn("Timestamp", format="HH:mm:ss.SSS"),
            "pause_ms":      st.column_config.NumberColumn("Pause (ms)", format="%.2f"),
            "reclaimed_mb":  st.column_config.NumberColumn("Reclaimed MB"),
        }
    )

with tab2:
    st.dataframe(ts_df.sort_values("timestamp", ascending=False).head(200),
                 use_container_width=True, height=300)

with tab3:
    if storms.empty:
        st.info("No GC storms detected with current thresholds.")
    else:
        st.dataframe(storms, use_container_width=True, height=300)


# ─────────────────────────────────────────────
# BOTTLENECK SUMMARY
# ─────────────────────────────────────────────

section("Bottleneck Analysis Summary", "🩺")
st.markdown(f"""
<div style='display:grid; grid-template-columns:1fr 1fr; gap:16px'>

<div style='background:#0f1826; border:1px solid #1e3a5f; border-radius:10px; padding:16px'>
    <div style='font-size:11px; font-weight:700; color:#38bdf8; letter-spacing:0.1em;
                text-transform:uppercase; margin-bottom:10px'>GC Health</div>
    <table style='width:100%; font-size:12px; font-family:JetBrains Mono,monospace; border-collapse:collapse'>
        <tr><td style='color:#64748b; padding:3px 0'>Total events</td>
            <td style='color:#e2e8f0; text-align:right'>{len(gc_df):,}</td></tr>
        <tr><td style='color:#64748b; padding:3px 0'>Full GC events</td>
            <td style='color:{"#f43f5e" if full_gc_count > 5 else "#34d399"}; text-align:right'>{full_gc_count}</td></tr>
        <tr><td style='color:#64748b; padding:3px 0'>Max pause</td>
            <td style='color:{"#f43f5e" if max_pause > 500 else "#fbbf24" if max_pause > 200 else "#34d399"}; text-align:right'>{max_pause:.1f} ms</td></tr>
        <tr><td style='color:#64748b; padding:3px 0'>GC overhead</td>
            <td style='color:{"#f43f5e" if gc_overhead > 5 else "#fbbf24" if gc_overhead > 2 else "#34d399"}; text-align:right'>{gc_overhead:.2f}%</td></tr>
        <tr><td style='color:#64748b; padding:3px 0'>Storms detected</td>
            <td style='color:{"#f43f5e" if not storms.empty else "#34d399"}; text-align:right'>{"Yes — "+str(len(storms)) if not storms.empty else "None"}</td></tr>
    </table>
</div>

<div style='background:#0f1826; border:1px solid #1e3a5f; border-radius:10px; padding:16px'>
    <div style='font-size:11px; font-weight:700; color:#818cf8; letter-spacing:0.1em;
                text-transform:uppercase; margin-bottom:10px'>Recommendations</div>
    <ul style='font-size:12px; color:#94a3b8; padding-left:16px; margin:0; line-height:2'>
        {"<li style='color:#f43f5e'>Increase heap (-Xmx) — frequent Full GCs indicate memory pressure</li>" if full_gc_count > 5 else ""}
        {"<li style='color:#fbbf24'>Tune GC pause targets (-XX:MaxGCPauseMillis)</li>" if max_pause > 200 else ""}
        {"<li style='color:#fbbf24'>Investigate GC storm root cause — check allocation hotspots</li>" if not storms.empty else ""}
        {"<li style='color:#f43f5e'>GC pauses are directly impacting request latency (r>{:.2f})</li>".format(top_corr) if top_corr > 0.6 else ""}
        {"<li>Heap sizing looks healthy — no Full GC pressure</li>" if full_gc_count == 0 else ""}
        <li>Enable GC logging: <code>-Xlog:gc*:file=gc.log:time,uptime:filecount=5,filesize=20m</code></li>
    </ul>
</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
